//! Music theory primitives: pitches, tunings, instruments, time and harmony.
//!
//! This module is intentionally free of any audio processing or I/O so that it
//! can be unit-tested exhaustively and reused by the tablature optimizer, the
//! renderers and the evaluation harness alike.

pub mod chord;
pub mod instrument;
pub mod key;
pub mod note;
pub mod tempo_map;
pub mod time;
pub mod tuning;

pub use chord::{Chord, ChordEvent, ChordKind};
pub use instrument::{Instrument, InstrumentKind};
pub use key::{Key, Mode};
pub use note::{Accidental, Interval, MidiNote, PitchClass};
pub use tempo_map::{TempoChange, TempoMap};
pub use time::{Beats, MidiTicks, Seconds, Tempo, TimeSignature};
pub use tuning::{Fret, StringIdx, Tuning, TuningPreset};
