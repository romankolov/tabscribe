//! Onset-gated pitch detection using frame-level YIN analysis.
//!
//! Detects onsets first, then looks up the frame-level YIN pitch at each
//! onset. This avoids the HMM smoothing in the full DSP pipeline that causes
//! pitch "stickiness" (e.g. E2 persisting across note changes), and avoids
//! the HPS harmonic-ambiguity problem where a note's harmonic is mistaken
//! for a separate fundamental.

use crate::music::{MidiNote, Seconds};
use crate::transcribe::dsp::{self, YinConfig};
use crate::transcribe::rhythm::{self, detect_onsets, onset_envelope, SpectrumConfig};
use crate::transcription::NoteEvent;
use crate::transcribe::{TranscribeOptions, Transcriber};

/// An onset-gated transcriber: detects onsets, then reads the frame-level
/// YIN pitch at each onset to find the newly attacked note's pitch.
#[derive(Debug, Clone, Default)]
pub struct OnsetGatedTranscriber {}

impl OnsetGatedTranscriber {
    /// Create a new onset-gated transcriber.
    #[must_use]
    pub fn new() -> Self {
        OnsetGatedTranscriber {}
    }

    /// Detect onsets and read the YIN pitch at each one.
    fn onset_gated_notes(
        &self,
        signal: &[f32],
        sample_rate: u32,
    ) -> Vec<NoteEvent> {
        let config = SpectrumConfig::for_sample_rate(sample_rate);
        let envelope = onset_envelope(signal, sample_rate, &config);
        let onsets = detect_onsets(&envelope, 0.3);

        if onsets.is_empty() {
            return Vec::new();
        }

        // Run frame-level YIN analysis on the full signal.
        // We use a small window so that each frame's pitch is local and not
        // smeared across note boundaries.
        let yin_config = YinConfig {
            window_size: 1024,
            hop_size: 256,
            ..YinConfig::from_options(sample_rate, &Default::default())
        };
        let analysis = dsp::analyze_frames(signal, &yin_config);

        if analysis.frames.is_empty() {
            return Vec::new();
        }

        let mut notes = Vec::new();
        let min_gap_ms = 80.0f64;

        for (i, &onset) in onsets.iter().enumerate() {
            // Skip onsets that are too close together
            if i > 0 && onset - onsets[i - 1] < min_gap_ms / 1000.0 {
                continue;
            }

            let end = if i + 1 < onsets.len() {
                onsets[i + 1]
            } else {
                onset + 1.0
            };
            let note_dur = (end - onset).max(0.1);

            // Find the frame closest to the onset time (within 50ms)
            let best_frame = analysis.frames.iter().find(|f| {
                (f.time - onset).abs() <= 0.05 && f.probability > 0.4
            });

            if let Some(frame) = best_frame {
                if let Some(pitch) = MidiNote::from_hz(frame.frequency) {
                    notes.push(NoteEvent {
                        pitch,
                        start: Seconds::from_secs(onset),
                        end: Seconds::from_secs(onset + note_dur),
                        velocity: 100,
                        confidence: frame.probability as f32,
                    });
                }
            }
        }

        notes
    }
}

impl Transcriber for OnsetGatedTranscriber {
    fn name(&self) -> &'static str {
        "onset-gated"
    }

    fn transcribe(
        &self,
        signal: &[f32],
        sample_rate: u32,
        _options: &TranscribeOptions,
    ) -> crate::error::Result<Vec<NoteEvent>> {
        Ok(self.onset_gated_notes(signal, sample_rate))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_signal() {
        let t = OnsetGatedTranscriber::new();
        let signal = vec![0.0f32; 44100];
        let notes = t.onset_gated_notes(&signal, 44100);
        assert!(notes.is_empty());
    }
}
