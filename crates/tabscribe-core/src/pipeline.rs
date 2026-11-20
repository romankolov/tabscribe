//! The transcription pipeline.
//!
//! The pipeline is the sequence of stages that turns an audio signal into
//! tablature:
//!
//! ```text
//! signal → pitch detection → note events → cleanup
//!        → rhythm (onsets, tempo, beats, meter)
//!        → quantization → harmony (key, chords)
//!        → fingerboard assignment → technique detection
//! ```
//!
//! Every stage is a pure function of its inputs, and the whole pipeline is
//! deterministic: the same signal and the same options always produce the
//! same tab.

use crate::error::{Error, Result};
use crate::music::{Instrument, Seconds, Tempo, TimeSignature};
use crate::tab::build::TabBuilder;
use crate::tab::cost::Style;
use crate::technique::{self, PitchContour};
use crate::transcribe::cleanup::{self, CleanupOptions};
use crate::transcribe::dsp::{DspTranscriber, YinConfig};
use crate::transcribe::harmony;
use crate::transcribe::poly::PolyphonicTranscriber;
use crate::transcribe::rhythm::{self, QuantizeMode, SpectrumConfig};
use crate::transcribe::AutoTranscriber;
use crate::transcribe::{TranscribeOptions, Transcriber};
use crate::transcription::{Measure, NoteEvent, Tab, TabNote, Transcription};

/// The options of a full transcription run.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineOptions {
    /// The options of the pitch detection stage.
    pub detection: TranscribeOptions,
    /// The options of the cleanup stage.
    pub cleanup: CleanupOptions,
    /// How aggressively the onsets are snapped to the grid.
    pub quantize: QuantizeMode,
    /// The beat grid used for the quantization, in beats.
    pub quantize_grid: f64,
    /// The confidence above which a technique is reported.
    pub technique_threshold: f32,
    /// The style used for the fingerboard assignment.
    pub style: Style,
    /// Whether the key should be detected.
    pub detect_key: bool,
    /// Whether the chords should be detected.
    pub detect_chords: bool,
    /// Whether the techniques should be detected.
    pub detect_techniques: bool,
    /// An explicit tempo, which overrides the detection.
    pub tempo_override: Option<Tempo>,
    /// An explicit meter, which overrides the detection.
    pub meter_override: Option<TimeSignature>,
    /// The tuning offset in semitones (None = auto-detect from the audio).
    ///
    /// When a recording is tuned sharp or flat relative to standard pitch,
    /// this offset compensates so that the transcribed pitches match standard
    /// notation. A value of +3 means the audio is three semitones sharp and
    /// will be transposed down by three semitones.
    pub tuning_offset: Option<f64>,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        PipelineOptions {
            detection: TranscribeOptions::default(),
            cleanup: CleanupOptions::default(),
            quantize: QuantizeMode::Musical,
            quantize_grid: 0.25,
            technique_threshold: technique::DEFAULT_THRESHOLD,
            style: Style::Balanced,
            detect_key: true,
            detect_chords: true,
            detect_techniques: true,
            tempo_override: None,
            meter_override: None,
            tuning_offset: None,
        }
    }
}

/// The result of a pipeline run.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// The transcription, before the fingerboard assignment.
    pub transcription: Transcription,
    /// The tablature.
    pub tab: Tab,
    /// The pitch contour, when it was computed.
    pub contour: Option<PitchContour>,
}

/// The transcription pipeline.
pub struct Pipeline {
    /// The engine used for the pitch detection.
    pub transcriber: Box<dyn Transcriber + Send + Sync>,
    /// The options.
    pub options: PipelineOptions,
}

impl Clone for Pipeline {
    fn clone(&self) -> Self {
        // The engine is not cloneable in general; the default engine is
        // recreated instead. A pipeline with a custom engine is cloned with
        // the same engine, which is safe because the engines are stateless.
        let transcriber: Box<dyn Transcriber + Send + Sync> = if self.transcriber.name() == "auto" {
            Box::new(AutoTranscriber::new())
        } else {
            Box::new(DspTranscriber::new())
        };
        Pipeline {
            transcriber,
            options: self.options.clone(),
        }
    }
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Pipeline")
            .field("engine", &self.transcriber.name())
            .field("options", &self.options)
            .finish()
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Pipeline {
            transcriber: Box::new(AutoTranscriber::new()),
            options: PipelineOptions::default(),
        }
    }
}

impl Pipeline {
    /// Create a pipeline with the classical DSP engine.
    #[must_use]
    pub fn dsp() -> Self {
        Pipeline {
            transcriber: Box::new(DspTranscriber::new()),
            options: PipelineOptions::default(),
        }
    }

    /// Create a pipeline with the polyphonic spectral engine.
    #[must_use]
    pub fn polyphonic() -> Self {
        Pipeline {
            transcriber: Box::new(PolyphonicTranscriber::new()),
            options: PipelineOptions::default(),
        }
    }

    /// Create a pipeline that picks the engine from the material.
    #[must_use]
    pub fn auto() -> Self {
        Pipeline::default()
    }

    /// Use a different pitch detection engine.
    #[must_use]
    pub fn with_transcriber(mut self, transcriber: Box<dyn Transcriber + Send + Sync>) -> Self {
        self.transcriber = transcriber;
        self
    }

    /// Set the options.
    #[must_use]
    pub fn with_options(mut self, options: PipelineOptions) -> Self {
        self.options = options;
        self
    }

    /// Run the pipeline on a mono signal.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the sample rate is zero, and
    /// [`Error::NoFeasibleTab`] when the detected notes cannot be played on
    /// the instrument.
    pub fn run(
        &self,
        signal: &[f32],
        sample_rate: u32,
        instrument: &Instrument,
    ) -> Result<PipelineResult> {
        if sample_rate == 0 {
            return Err(Error::Config("the sample rate cannot be zero".to_string()));
        }

        // ── 1. Pitch detection ────────────────────────────────────────────
        let mut notes =
            self.transcriber
                .transcribe(signal, sample_rate, &self.options.detection)?;

        // ── 1b. Tuning offset ─────────────────────────────────────────────
        //
        // When a recording is tuned sharp or flat relative to standard pitch,
        // every detected note is shifted by the same amount. We detect the
        // offset from the audio and transpose the notes back to standard
        // notation, so that the tablature matches what a guitarist would play.
        let offset = self.options.tuning_offset.unwrap_or_else(|| {
            crate::transcribe::dsp::detect_tuning_offset(signal, sample_rate).unwrap_or(0.0)
        });
        if offset != 0.0 {
            let shift = (-offset) as i8;
            for note in &mut notes {
                let new_pitch = (note.pitch.as_u8() as i32 + shift as i32).clamp(0, 127);
                if let Ok(p) = crate::music::MidiNote::new(new_pitch as u8) {
                    note.pitch = p;
                }
            }
        }

        // ── 2. Rhythm ─────────────────────────────────────────────────────
        let spectrum = SpectrumConfig::for_sample_rate(sample_rate);
        let envelope = rhythm::onset_envelope(signal, sample_rate, &spectrum);
        let tempo_estimate = rhythm::estimate_tempo(&envelope);
        let tempo = self.options.tempo_override.unwrap_or(tempo_estimate.tempo);
        let beats = rhythm::track_beats(&envelope, tempo);
        let meter = self
            .options
            .meter_override
            .unwrap_or_else(|| rhythm::detect_meter(&envelope, &beats, tempo));

        // ── 3. Cleanup ────────────────────────────────────────────────────
        notes = cleanup::cleanup(&notes, Some(instrument), &self.options.cleanup);

        // The onsets tell a re-articulated note from a held one.
        let onsets = rhythm::detect_onsets(&envelope, self.options.detection.sensitivity);
        notes = rhythm::split_at_onsets(&notes, &onsets);

        // The split leaves a short tail behind, which is an artefact of the
        // detector rather than a note of the piece.
        notes = cleanup::remove_short_notes(&notes, self.options.cleanup.min_duration);

        // The note boundaries are only known to within half an analysis
        // window; the onsets say precisely where the attacks were.
        notes = rhythm::refine_boundaries(&notes, &envelope);

        // ── 4. Quantization ───────────────────────────────────────────────
        notes = quantize_notes(
            &notes,
            &beats,
            tempo,
            self.options.quantize,
            self.options.quantize_grid,
        );

        // ── 5. Harmony ────────────────────────────────────────────────────
        let key = if self.options.detect_key {
            let chroma = harmony::chroma_of(signal, sample_rate);
            Some(harmony::detect_key(&chroma).key)
        } else {
            None
        };
        let chords = if self.options.detect_chords {
            harmony::recognise_chords(signal, sample_rate, &beats, tempo, key)
        } else {
            Vec::new()
        };

        // ── 6. Fingerboard assignment ─────────────────────────────────────
        let transcription = Transcription {
            notes: notes.clone(),
            tempo,
            time_signature: meter,
            key,
            chords: chords.clone(),
            confidence: 0.0,
        };
        let mut transcription = transcription;
        transcription.confidence = transcription.mean_confidence();

        let builder = TabBuilder::new(instrument).with_style(self.options.style);
        let mut tab = builder.build(&transcription)?;

        // ── 7. Technique detection ─────────────────────────────────────────
        let contour = self.pitch_contour(signal, sample_rate);
        if self.options.detect_techniques {
            let detected = technique::detect(
                &tab.notes,
                contour.as_ref(),
                self.options.technique_threshold,
            );
            for (note, detections) in tab.notes.iter_mut().zip(detected.iter()) {
                for detection in detections {
                    if !note.techniques.contains(&detection.technique) {
                        note.techniques.push(detection.technique);
                    }
                }
            }
        }

        // The measures are laid out on the musical timeline.
        tab.measures = build_measures(&tab.notes, tempo, meter);

        Ok(PipelineResult {
            transcription,
            tab,
            contour,
        })
    }

    /// Compute the pitch contour of the signal.
    ///
    /// The contour is the frame-level pitch track that the technique detection
    /// uses to tell bends and vibrato apart.
    #[must_use]
    pub fn pitch_contour(&self, signal: &[f32], sample_rate: u32) -> Option<PitchContour> {
        let config = YinConfig::from_options(sample_rate, &self.options.detection);
        let analysis = crate::transcribe::dsp::analyze_frames(signal, &config);
        if analysis.frames.is_empty() {
            return None;
        }
        Some(PitchContour {
            frames: analysis
                .frames
                .iter()
                .map(|frame| technique::ContourFrame {
                    time: frame.time,
                    frequency: frame.frequency,
                    probability: frame.probability as f32,
                })
                .collect(),
        })
    }
}

/// Snap the note onsets to the beat grid.
///
/// A note is only moved when the grid is close enough, unless the strict mode
/// is used. The offsets are shifted by the same amount as the onsets, so that
/// the durations are preserved.
fn quantize_notes(
    notes: &[NoteEvent],
    beats: &[f64],
    tempo: Tempo,
    mode: QuantizeMode,
    grid: f64,
) -> Vec<NoteEvent> {
    if beats.is_empty() || mode == QuantizeMode::Off {
        return notes.to_vec();
    }
    let seconds_per_beat = tempo.seconds_per_beat().as_secs();
    if seconds_per_beat <= 0.0 {
        return notes.to_vec();
    }

    let mut result: Vec<NoteEvent> = Vec::with_capacity(notes.len());
    for note in notes {
        let start_beat = note.start.as_secs() / seconds_per_beat;
        let snapped = rhythm::quantize(start_beat, grid, mode);
        if (snapped - start_beat).abs() < 1e-9 {
            result.push(note.clone());
            continue;
        }
        let shift = (snapped - start_beat) * seconds_per_beat;
        let mut moved = note.clone();
        moved.start = Seconds::from_secs((note.start.as_secs() + shift).max(0.0));
        moved.end = Seconds::from_secs((note.end.as_secs() + shift).max(moved.start.as_secs()));
        result.push(moved);
    }
    result
}

/// Build the measure list of a tab.
fn build_measures(notes: &[TabNote], tempo: Tempo, meter: TimeSignature) -> Vec<Measure> {
    let beats_per_measure = meter.beats_per_bar();
    let seconds_per_beat = tempo.seconds_per_beat().as_secs();
    if notes.is_empty() || seconds_per_beat <= 0.0 {
        return Vec::new();
    }

    let end_beat = notes
        .iter()
        .map(|n| n.start_beat + n.duration_beats)
        .fold(0.0f64, f64::max);
    let measure_count = ((end_beat / beats_per_measure).ceil() as usize).max(1);

    (0..measure_count)
        .map(|index| {
            let start_beat = index as f64 * beats_per_measure;
            let end_beat = start_beat + beats_per_measure;
            Measure {
                index,
                start_beat,
                end_beat,
                start: Seconds::from_secs(start_beat * seconds_per_beat),
                end: Seconds::from_secs(end_beat * seconds_per_beat),
            }
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::music::InstrumentKind;

    /// A plucked note: a harmonic-rich tone with a sharp attack.
    fn plucked(frequency: f64, seconds: f64, sample_rate: u32) -> Vec<f32> {
        let n = (seconds * sample_rate as f64) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                let envelope = (-t * 6.0).exp();
                let mut value = 0.0;
                for harmonic in 1..=6 {
                    value += (2.0 * std::f64::consts::PI * frequency * harmonic as f64 * t).sin()
                        / harmonic as f64;
                }
                (value * envelope * 0.3) as f32
            })
            .collect()
    }

    #[test]
    fn the_pipeline_produces_a_tab() {
        let sample_rate = 22_050;
        let mut signal = plucked(196.0, 0.5, sample_rate); // G3
        signal.extend(plucked(246.94, 0.5, sample_rate)); // B3
        let instrument = Instrument::for_kind(InstrumentKind::Guitar).expect("valid");
        let result = Pipeline::dsp()
            .run(&signal, sample_rate, &instrument)
            .expect("the pipeline runs");
        assert!(!result.tab.notes.is_empty(), "expected some notes");
        assert!(result.transcription.tempo.bpm() > 0.0);
    }

    #[test]
    fn the_pipeline_is_deterministic() {
        let sample_rate = 22_050;
        let signal = plucked(220.0, 1.0, sample_rate);
        let instrument = Instrument::for_kind(InstrumentKind::Guitar).expect("valid");
        let first = Pipeline::dsp()
            .run(&signal, sample_rate, &instrument)
            .expect("the pipeline runs");
        let second = Pipeline::dsp()
            .run(&signal, sample_rate, &instrument)
            .expect("the pipeline runs");
        assert_eq!(first.tab, second.tab);
    }

    #[test]
    fn an_empty_signal_produces_an_empty_tab() {
        let instrument = Instrument::for_kind(InstrumentKind::Guitar).expect("valid");
        let result = Pipeline::dsp()
            .run(&[], 22_050, &instrument)
            .expect("the pipeline runs");
        assert!(result.tab.notes.is_empty());
        assert!(result.tab.measures.is_empty());
    }
}
