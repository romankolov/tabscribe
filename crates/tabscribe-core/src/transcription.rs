//! The data types that flow between pipeline stages.
//!
//! # The two clocks
//!
//! Every stage of the pipeline agrees on two representations of time:
//!
//! * **Musical time** ([`Beats`]) — position in a notated timeline. This is
//!   what quantization, beat tracking and the renderers work in.
//! * **Wall-clock time** ([`Seconds`]) — position in the audio file. This is
//!   what the audio analysis stages work in.
//!
//! [`TempoMap`] converts between the two and is the only place where the
//! conversion happens, so a tempo change can never silently corrupt timings.

use serde::{Deserialize, Serialize};

use crate::music::{
    ChordEvent, Fret, Instrument, Key, MidiNote, Seconds, StringIdx, Tempo, TimeSignature,
};

/// A sounding note with its timing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoteEvent {
    /// The sounding pitch.
    pub pitch: MidiNote,
    /// When the note starts.
    pub start: Seconds,
    /// When the note ends.
    pub end: Seconds,
    /// The note velocity, `0..=127`.
    pub velocity: u8,
    /// The detector's confidence in this note, `0.0..=1.0`.
    pub confidence: f32,
}

impl NoteEvent {
    /// Construct a note event.
    #[must_use]
    pub fn new(pitch: MidiNote, start: Seconds, end: Seconds) -> Self {
        NoteEvent {
            pitch,
            start,
            end,
            velocity: 80,
            confidence: 1.0,
        }
    }

    /// The duration of the note.
    #[must_use]
    pub fn duration(&self) -> Seconds {
        Seconds((self.end.as_secs() - self.start.as_secs()).max(0.0))
    }

    /// Whether two notes sound at the same time.
    #[must_use]
    pub fn overlaps(&self, other: &NoteEvent) -> bool {
        self.start < other.end && other.start < self.end
    }
}

/// A technique applied to a note.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Technique {
    /// Hammer-on.
    HammerOn,
    /// Pull-off.
    PullOff,
    /// Slide up.
    SlideUp,
    /// Slide down.
    SlideDown,
    /// Pitch bend.
    Bend,
    /// Bend release.
    Release,
    /// Vibrato.
    Vibrato,
    /// Palm mute.
    PalmMute,
    /// Dead note (muted string).
    DeadNote,
    /// Let ring.
    LetRing,
}

impl Technique {
    /// The symbol used in ASCII tablature.
    #[must_use]
    pub fn symbol(self) -> char {
        match self {
            Technique::HammerOn => 'h',
            Technique::PullOff => 'p',
            Technique::SlideUp => '/',
            Technique::SlideDown => '\\',
            Technique::Bend => 'b',
            Technique::Release => 'r',
            Technique::Vibrato => '~',
            Technique::PalmMute => 'x',
            Technique::DeadNote => 'x',
            Technique::LetRing => '.',
        }
    }

    /// Whether the technique is drawn to the *left* of the fret number.
    ///
    /// Slashes and hammer-ons precede the note; most others follow it.
    #[must_use]
    pub fn is_prefix(self) -> bool {
        matches!(self, Technique::SlideUp | Technique::SlideDown)
    }
}

/// A note placed on the fingerboard.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TabNote {
    /// The pitch this note sounds.
    pub pitch: MidiNote,
    /// The string the note is played on.
    pub string: StringIdx,
    /// The fret the note is played at.
    pub fret: Fret,
    /// When the note starts.
    pub start: Seconds,
    /// When the note ends.
    pub end: Seconds,
    /// The position of the note in musical time, in beats from the start.
    pub start_beat: f64,
    /// The length of the note in beats.
    pub duration_beats: f64,
    /// The detector's confidence in this note, `0.0..=1.0`.
    pub confidence: f32,
    /// The techniques applied to this note.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub techniques: Vec<Technique>,
    /// Alternative fingerings for this note, best first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub alternatives: Vec<Fingering>,
}

/// A `(string, fret)` pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fingering {
    /// The string index.
    pub string: u8,
    /// The fret number.
    pub fret: u8,
}

impl Fingering {
    /// Construct a fingering.
    #[must_use]
    pub const fn new(string: StringIdx, fret: Fret) -> Self {
        Fingering {
            string: string.as_u8(),
            fret: fret.as_u8(),
        }
    }
}

/// One bar of the tablature.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Measure {
    /// The zero-based measure number.
    pub index: usize,
    /// The start of the measure in musical time, in beats.
    pub start_beat: f64,
    /// The end of the measure in musical time, in beats.
    pub end_beat: f64,
    /// The start of the measure in wall-clock time.
    pub start: Seconds,
    /// The end of the measure in wall-clock time.
    pub end: Seconds,
}

impl Measure {
    /// The length of the measure in beats.
    #[must_use]
    pub fn length_beats(&self) -> f64 {
        (self.end_beat - self.start_beat).max(0.0)
    }
}

/// The result of a transcription, before tablature assignment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Transcription {
    /// The detected notes, sorted by start time.
    pub notes: Vec<NoteEvent>,
    /// The detected or overridden tempo.
    pub tempo: Tempo,
    /// The detected or overridden time signature.
    pub time_signature: TimeSignature,
    /// The detected key, if one could be determined.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<Key>,
    /// The detected chords.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chords: Vec<ChordEvent>,
    /// A note about the overall confidence of the transcription.
    #[serde(default)]
    pub confidence: f32,
}

impl Transcription {
    /// Construct an empty transcription with the given meter.
    #[must_use]
    pub fn new(tempo: Tempo, time_signature: TimeSignature) -> Self {
        Transcription {
            notes: Vec::new(),
            tempo,
            time_signature,
            key: None,
            chords: Vec::new(),
            confidence: 0.0,
        }
    }

    /// The mean confidence of all notes.
    #[must_use]
    pub fn mean_confidence(&self) -> f32 {
        if self.notes.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.notes.iter().map(|n| n.confidence).sum();
        sum / self.notes.len() as f32
    }

    /// The lowest and highest sounding pitch.
    #[must_use]
    pub fn pitch_range(&self) -> Option<(MidiNote, MidiNote)> {
        let min = self.notes.iter().map(|n| n.pitch.as_u8()).min()?;
        let max = self.notes.iter().map(|n| n.pitch.as_u8()).max()?;
        Some((MidiNote::clamped(min as u16), MidiNote::clamped(max as u16)))
    }
}

/// The final tablature: notes placed on the fingerboard plus musical context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tab {
    /// The instrument the tab was written for.
    pub instrument: Instrument,
    /// The tempo of the piece.
    pub tempo: Tempo,
    /// The time signature of the piece.
    pub time_signature: TimeSignature,
    /// The key of the piece, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<Key>,
    /// The measures of the piece.
    pub measures: Vec<Measure>,
    /// The notes, sorted by start time.
    pub notes: Vec<TabNote>,
    /// The chord annotations.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chords: Vec<ChordEvent>,
    /// The overall confidence of the tabulation.
    #[serde(default)]
    pub confidence: f32,
}

impl Tab {
    /// The total length of the piece in wall-clock time.
    #[must_use]
    pub fn duration(&self) -> Seconds {
        let end = self
            .notes
            .iter()
            .map(|n| n.end.as_secs())
            .fold(0.0_f64, f64::max);
        Seconds::from_secs(end)
    }
}
