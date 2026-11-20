//! The rich JSON report.
//!
//! The report is meant to be read by other programs, so it carries everything
//! the pipeline found: the notes with their confidence, the rhythm, the key,
//! the chords and the fingering alternatives. The format is versioned, so a
//! consumer can rely on the shape of the document.

use serde::{Deserialize, Serialize};

use crate::music::{ChordEvent, Key, Tempo, TimeSignature};
use crate::transcription::{Fingering, Tab, TabNote};

/// The version of the report format.
pub const REPORT_VERSION: u32 = 1;

/// A complete report about one transcription.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Report {
    /// The version of the report format.
    pub version: u32,
    /// The tool that produced the report.
    pub generator: String,
    /// The name of the source file, when it is known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// The engine that performed the analysis.
    pub engine: String,
    /// The overall confidence of the transcription, from 0 to 1.
    pub confidence: f32,
    /// The musical context.
    pub musical: MusicalContext,
    /// The detected notes.
    pub notes: Vec<ReportNote>,
    /// The measures of the piece.
    pub measures: Vec<ReportMeasure>,
    /// The detected chords.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chords: Vec<ChordEvent>,
    /// The warnings the run produced.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

/// The musical context of a transcription.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicalContext {
    /// The tempo.
    pub tempo: Tempo,
    /// The time signature.
    pub time_signature: TimeSignature,
    /// The key, when one was detected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<Key>,
    /// The name of the instrument.
    pub instrument: String,
    /// The open string notes, from the lowest string to the highest.
    pub tuning: Vec<u8>,
    /// The capo position.
    pub capo: u8,
    /// The duration of the piece in seconds.
    pub duration: f64,
}

/// One note of the report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReportNote {
    /// The sounding pitch as a MIDI note number.
    pub pitch: u8,
    /// The name of the pitch.
    pub pitch_name: String,
    /// The string the note is played on, counting from the top.
    pub string: u8,
    /// The fret.
    pub fret: u8,
    /// The onset in seconds.
    pub start: f64,
    /// The offset in seconds.
    pub end: f64,
    /// The onset in beats.
    pub start_beat: f64,
    /// The duration in beats.
    pub duration_beats: f64,
    /// The velocity.
    pub velocity: u8,
    /// The confidence in the note, from 0 to 1.
    pub confidence: f32,
    /// The techniques applied to the note.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub techniques: Vec<String>,
    /// The alternative fingerings for the note.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub alternatives: Vec<Fingering>,
}

/// One measure of the report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReportMeasure {
    /// The zero-based measure number.
    pub index: usize,
    /// The start of the measure in beats.
    pub start_beat: f64,
    /// The end of the measure in beats.
    pub end_beat: f64,
    /// The number of notes in the measure.
    pub note_count: usize,
}

impl Report {
    /// Build a report from a tab.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    #[must_use]
    pub fn from_tab(tab: &Tab, engine: &str) -> Self {
        let notes = tab.notes.iter().map(ReportNote::from).collect();
        let measures = tab
            .measures
            .iter()
            .map(|measure| {
                let count = tab
                    .notes
                    .iter()
                    .filter(|note| {
                        note.start_beat >= measure.start_beat && note.start_beat < measure.end_beat
                    })
                    .count();
                ReportMeasure {
                    index: measure.index,
                    start_beat: measure.start_beat,
                    end_beat: measure.end_beat,
                    note_count: count,
                }
            })
            .collect();

        Report {
            version: REPORT_VERSION,
            generator: format!("tabscribe {}", crate::VERSION),
            source: None,
            engine: engine.to_string(),
            confidence: tab.confidence,
            musical: MusicalContext {
                tempo: tab.tempo,
                time_signature: tab.time_signature,
                key: tab.key,
                instrument: tab.instrument.describe(),
                tuning: tab
                    .instrument
                    .tuning()
                    .strings()
                    .iter()
                    .map(|s| s.open.as_u8())
                    .collect(),
                capo: tab.instrument.capo(),
                duration: tab.duration().as_secs(),
            },
            notes,
            measures,
            chords: tab.chords.clone(),
            warnings: Vec::new(),
        }
    }

    /// Attach the name of the source file.
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Attach a warning.
    pub fn warn(&mut self, message: impl Into<String>) {
        self.warnings.push(message.into());
    }

    /// Serialize the report as pretty-printed JSON.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Serde`] when the report cannot be serialized,
    /// which cannot happen for the current schema but is reported rather than
    /// hidden.
    pub fn to_json(&self) -> crate::error::Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| crate::error::Error::Serde {
            what: "JSON report".to_string(),
            reason: e.to_string(),
        })
    }

    /// Serialize the report as compact JSON.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Serde`] when the report cannot be serialized.
    pub fn to_json_compact(&self) -> crate::error::Result<String> {
        serde_json::to_string(self).map_err(|e| crate::error::Error::Serde {
            what: "JSON report".to_string(),
            reason: e.to_string(),
        })
    }
}

impl From<&TabNote> for ReportNote {
    fn from(note: &TabNote) -> Self {
        ReportNote {
            pitch: note.pitch.as_u8(),
            pitch_name: note.pitch.name(),
            string: note.string.as_u8(),
            fret: note.fret.as_u8(),
            start: note.start.as_secs(),
            end: note.end.as_secs(),
            start_beat: note.start_beat,
            duration_beats: note.duration_beats,
            velocity: 80,
            confidence: note.confidence,
            techniques: note
                .techniques
                .iter()
                .map(|t| format!("{t:?}").to_lowercase())
                .collect(),
            alternatives: note.alternatives.clone(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::music::{Instrument, InstrumentKind};

    #[test]
    fn a_report_round_trips_through_json() {
        let instrument = Instrument::for_kind(InstrumentKind::Guitar).expect("valid");
        let tab = Tab {
            instrument,
            tempo: Tempo::DEFAULT,
            time_signature: TimeSignature::COMMON,
            key: Some(Key::C_MAJOR),
            measures: Vec::new(),
            notes: Vec::new(),
            chords: Vec::new(),
            confidence: 0.8,
        };
        let report = Report::from_tab(&tab, "dsp").with_source("song.wav");
        let json = report.to_json().expect("serializes");
        assert!(json.contains("\"version\": 1"));
        assert!(json.contains("\"source\": \"song.wav\""));
        let parsed: Report = serde_json::from_str(&json).expect("parses");
        assert_eq!(parsed, report);
    }

    #[test]
    fn the_report_is_deterministic() {
        let instrument = Instrument::for_kind(InstrumentKind::Guitar).expect("valid");
        let tab = Tab {
            instrument,
            tempo: Tempo::DEFAULT,
            time_signature: TimeSignature::COMMON,
            key: None,
            measures: Vec::new(),
            notes: Vec::new(),
            chords: Vec::new(),
            confidence: 0.0,
        };
        let first = Report::from_tab(&tab, "dsp").to_json().expect("serializes");
        let second = Report::from_tab(&tab, "dsp").to_json().expect("serializes");
        assert_eq!(first, second);
    }
}
