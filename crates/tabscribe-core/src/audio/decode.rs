//! Decoding audio files into sample buffers.
//!
//! `tabscribe` accepts WAV, FLAC, MP3, OGG/Vorbis and AAC/M4A. Decoding is
//! delegated to `symphonia`, which is a pure-Rust implementation and therefore
//! needs no system libraries.
//!
//! The decoder always produces `f32` samples, whatever the source format, and
//! remembers the original channel layout so that the caller can decide
//! whether to downmix or to analyse the channels separately.

use std::fs::File;
use std::path::Path;

use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::{MediaSource, MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;

use crate::error::{Error, Result};

/// A block of decoded audio.
#[derive(Debug, Clone, Default)]
pub struct AudioBuffer {
    /// The samples. Interleaved when [`AudioBuffer::channels`] is greater than
    /// one.
    pub samples: Vec<f32>,
    /// The sample rate in Hertz.
    pub sample_rate: u32,
    /// The number of channels.
    pub channels: usize,
}

impl AudioBuffer {
    /// The number of frames (samples per channel).
    #[must_use]
    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / self.channels
        }
    }

    /// The duration of the buffer.
    #[must_use]
    pub fn duration(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.frames() as f64 / self.sample_rate as f64
        }
    }

    /// Mix the channels down to a single mono signal.
    ///
    /// The mix is a plain average, which preserves the amplitude of a
    /// correlated stereo signal and attenuates uncorrelated noise.
    #[must_use]
    pub fn to_mono(&self) -> Vec<f32> {
        if self.channels <= 1 {
            return self.samples.clone();
        }
        let frames = self.frames();
        let mut out = vec![0.0f32; frames];
        for channel in 0..self.channels {
            for frame in 0..frames {
                out[frame] += self.samples[frame * self.channels + channel];
            }
        }
        let scale = 1.0 / self.channels as f32;
        for sample in &mut out {
            *sample *= scale;
        }
        out
    }
}

/// Decode an audio file from disk.
///
/// # Errors
///
/// Returns [`Error::Io`] when the file cannot be opened,
/// [`Error::UnsupportedFormat`] when the container is not recognised, and
/// [`Error::Decode`] when the audio data itself cannot be decoded.
pub fn decode_file(path: &Path) -> Result<AudioBuffer> {
    let file = File::open(path).map_err(|source| Error::io(path.display().to_string(), source))?;
    let extension = path
        .extension()
        .map(|e| e.to_string_lossy().to_string())
        .unwrap_or_default();
    decode_source(Box::new(file), Some(&extension))
}

/// Decode audio from any media source.
///
/// # Errors
///
/// See [`decode_file`].
pub fn decode_source(source: Box<dyn MediaSource>, extension: Option<&str>) -> Result<AudioBuffer> {
    let mss = MediaSourceStream::new(source, MediaSourceStreamOptions::default());

    let mut hint = Hint::new();
    if let Some(ext) = extension {
        if !ext.is_empty() {
            hint.with_extension(ext);
        }
    }

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| Error::UnsupportedFormat(format!("could not identify the file: {e}")))?;

    let track = format
        .default_track(TrackType::Audio)
        .ok_or(Error::NoAudioStream)?;
    let track_id = track.id;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or(Error::NoAudioStream)?
        .clone();

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .map_err(|e| Error::UnsupportedFormat(format!("unsupported audio codec: {e}")))?;

    let mut buffer = AudioBuffer::default();
    let mut scratch: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(e) => {
                // A truncated file is still usable: keep what we decoded.
                if buffer.samples.is_empty() {
                    return Err(Error::Decode(format!("error reading the audio stream: {e}")));
                }
                tracing::warn!("stopped decoding early: {e}");
                break;
            }
        };

        if packet.track_id != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = decoded.spec();
                let channels = spec.channels().count().max(1);
                let frames = decoded.frames();
                let needed = frames * channels;
                if scratch.len() < needed {
                    scratch.resize(needed, 0.0);
                }
                decoded.copy_to_slice_interleaved::<f32, _>(&mut scratch[..needed]);

                if buffer.channels == 0 {
                    buffer.channels = channels;
                    buffer.sample_rate = spec.rate();
                }
                buffer.samples.extend_from_slice(&scratch[..needed]);
            }
            Err(e) => {
                tracing::debug!("skipping an undecodable packet: {e}");
            }
        }
    }

    if buffer.samples.is_empty() {
        return Err(Error::NoAudioStream);
    }

    // Some decoders report a sample rate of zero for exotic files; fall back
    // to a sane value rather than dividing by zero later.
    if buffer.sample_rate == 0 {
        buffer.sample_rate = 44_100;
    }
    if buffer.channels == 0 {
        buffer.channels = 1;
    }

    Ok(buffer)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn mono_mix_averages_the_channels() {
        let buffer = AudioBuffer {
            samples: vec![1.0, -1.0, 0.5, 0.5],
            sample_rate: 44_100,
            channels: 2,
        };
        assert_eq!(buffer.frames(), 2);
        assert_eq!(buffer.to_mono(), vec![0.0, 0.5]);
    }

    #[test]
    fn decoding_a_non_audio_file_fails_cleanly() {
        let error = decode_source(
            Box::new(std::io::Cursor::new(b"definitely not audio".to_vec())),
            Some("mp3"),
        )
        .expect_err("should fail");
        assert!(matches!(
            error,
            Error::UnsupportedFormat(_) | Error::Decode(_) | Error::NoAudioStream
        ));
    }

    #[test]
    fn decoding_empty_input_fails_cleanly() {
        let error = decode_source(Box::new(std::io::Cursor::new(Vec::new())), Some("wav"))
            .expect_err("should fail");
        assert!(matches!(
            error,
            Error::UnsupportedFormat(_) | Error::Decode(_) | Error::NoAudioStream
        ));
    }
}
