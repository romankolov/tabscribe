//! The transcription stages: the traits that describe them and the options
//! that control them.
//!
//! Every analysis stage is a trait with a plain-data signature, so that the
//! neural and classical engines can be swapped without the rest of the
//! pipeline noticing.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::transcribe::dsp::DspTranscriber;
use crate::transcription::NoteEvent;

pub mod cleanup;
pub mod dsp;
pub mod harmony;
pub mod poly;
pub mod rhythm;

/// The options that control a transcription run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscribeOptions {
    /// How permissive the note detection should be, from `0.0` (conservative)
    /// to `1.0` (permissive).
    pub sensitivity: f32,
    /// The lowest frequency to look for, in Hertz.
    pub min_frequency: f64,
    /// The highest frequency to look for, in Hertz.
    pub max_frequency: f64,
    /// The shortest note that is reported, in seconds.
    pub min_note_seconds: f64,
    /// The tempo used to convert between seconds and beats.
    pub tempo: crate::music::Tempo,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        TranscribeOptions {
            sensitivity: 0.5,
            min_frequency: 60.0,
            max_frequency: 1_300.0,
            min_note_seconds: 0.05,
            tempo: crate::music::Tempo::DEFAULT,
        }
    }
}

impl TranscribeOptions {
    /// The options with the sensitivity applied.
    ///
    /// The sensitivity shifts the detection thresholds: a higher value accepts
    /// weaker notes and shorter gaps.
    #[must_use]
    pub fn with_sensitivity(mut self, sensitivity: f32) -> Self {
        self.sensitivity = sensitivity.clamp(0.0, 1.0);
        self
    }

    /// The frequency range as MIDI note numbers.
    #[must_use]
    pub fn midi_range(&self) -> (crate::music::MidiNote, crate::music::MidiNote) {
        let low = crate::music::MidiNote::from_hz(self.min_frequency)
            .unwrap_or(crate::music::MidiNote::new(24).unwrap_or(crate::music::MidiNote::MIN));
        let high = crate::music::MidiNote::from_hz(self.max_frequency)
            .unwrap_or(crate::music::MidiNote::new(96).unwrap_or(crate::music::MidiNote::MAX));
        (low, high)
    }
}

/// A pitch-detection engine.
///
/// The trait is deliberately narrow: a mono signal in, a list of note events
/// out. Everything else (tempo, harmony, tablature) is the responsibility of
/// the later stages.
pub trait Transcriber {
    /// The name of the engine, as shown in the summary.
    fn name(&self) -> &'static str;

    /// Transcribe a mono signal.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Config`] when the options are inconsistent.
    fn transcribe(
        &self,
        signal: &[f32],
        sample_rate: u32,
        options: &TranscribeOptions,
    ) -> Result<Vec<NoteEvent>>;
}

/// Which engine to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Engine {
    /// Pick the best engine for the material.
    Auto,
    /// The neural polyphonic engine.
    Neural,
    /// The classical DSP engine.
    Dsp,
    /// The polyphonic spectral engine.
    Polyphonic,
}

impl Engine {
    /// The canonical lowercase name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Engine::Auto => "auto",
            Engine::Neural => "neural",
            Engine::Dsp => "dsp",
            Engine::Polyphonic => "polyphonic",
        }
    }
}

/// The engine that picks the best analyser for the material.
///
/// The choice is made once, from the periodicity of the whole signal: a
/// signal that is well explained by a single period is transcribed with the
/// monophonic detector, which is more accurate; a signal that is not is
/// transcribed with the polyphonic one.
#[derive(Debug, Clone)]
pub struct AutoTranscriber {
    /// The monophonic detector.
    pub monophonic: DspTranscriber,
    /// The polyphonic detector.
    pub polyphonic: crate::transcribe::poly::PolyphonicTranscriber,
}

impl Default for AutoTranscriber {
    fn default() -> Self {
        AutoTranscriber {
            monophonic: DspTranscriber::new(),
            polyphonic: crate::transcribe::poly::PolyphonicTranscriber::new(),
        }
    }
}

impl AutoTranscriber {
    /// Create the engine with the default configuration.
    #[must_use]
    pub fn new() -> Self {
        AutoTranscriber::default()
    }
}

impl Transcriber for AutoTranscriber {
    fn name(&self) -> &'static str {
        "auto"
    }

    fn transcribe(
        &self,
        signal: &[f32],
        sample_rate: u32,
        options: &TranscribeOptions,
    ) -> Result<Vec<NoteEvent>> {
        match estimate_material(signal) {
            Material::Monophonic => self.monophonic.transcribe(signal, sample_rate, options),
            Material::Polyphonic => self.polyphonic.transcribe(signal, sample_rate, options),
        }
    }
}

/// How polyphonic the material appears to be.
///
/// The `auto` engine uses this to decide between the monophonic and the
/// polyphonic strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Material {
    /// One note at a time: a single melodic line.
    Monophonic,
    /// Several notes at a time: chords or a full band mix.
    Polyphonic,
}

/// Estimate how polyphonic a signal is.
///
/// The estimate uses the **periodicity confidence** of the YIN analysis: the
/// normalised difference function tells how well a single lag explains the
/// signal. A single sustained note — even a rich one with many harmonics —
/// has a very low difference at its period, while a chord has no such lag and
/// the difference stays high everywhere.
///
/// The confidence is measured over the whole signal, because a chord is
/// polyphonic throughout, not just at its attack.
///
/// # Panics
///
/// This function does not panic; an empty signal is reported as monophonic.
#[must_use]
pub fn estimate_material(signal: &[f32]) -> Material {
    let sample_rate = 22_050;
    let config = dsp::YinConfig::from_options(sample_rate, &TranscribeOptions::default());
    let analysis = dsp::analyze_frames(signal, &config);
    if analysis.frames.is_empty() {
        return Material::Monophonic;
    }

    let mut confidences: Vec<f64> = analysis
        .frames
        .iter()
        .map(|frame| frame.probability)
        .collect();
    confidences.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = confidences[confidences.len() / 2];

    // A single note scores above 0.95; a chord of three or more notes scores
    // below 0.85. The threshold sits in between, closer to the chord, because
    // treating a monophonic line as polyphonic is the more damaging mistake.
    if median < 0.90 {
        Material::Polyphonic
    } else {
        Material::Monophonic
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn a_pure_tone_is_monophonic() {
        let sample_rate = 22_050;
        let signal: Vec<f32> = (0..sample_rate)
            .map(|i| {
                (2.0 * std::f64::consts::PI * 440.0 * i as f64 / sample_rate as f64).sin() as f32
            })
            .collect();
        assert_eq!(estimate_material(&signal), Material::Monophonic);
    }

    #[test]
    fn a_dense_chord_is_polyphonic() {
        let sample_rate = 22_050;
        let signal: Vec<f32> = (0..sample_rate)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                let value = (2.0 * std::f64::consts::PI * 220.0 * t).sin()
                    + (2.0 * std::f64::consts::PI * 277.0 * t).sin()
                    + (2.0 * std::f64::consts::PI * 330.0 * t).sin()
                    + (2.0 * std::f64::consts::PI * 415.0 * t).sin();
                (value / 4.0) as f32
            })
            .collect();
        assert_eq!(estimate_material(&signal), Material::Polyphonic);
    }

    #[test]
    fn midi_range_is_derived_from_the_frequency_range() {
        let options = TranscribeOptions::default();
        let (low, high) = options.midi_range();
        assert!(low.as_u8() < high.as_u8());
        assert!(low.as_u8() <= 40 && high.as_u8() >= 80);
    }
}
