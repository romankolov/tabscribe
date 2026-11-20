//! Pitch primitives: MIDI note numbers, pitch classes and note names.
//!
//! # Unit safety
//!
//! Pitch is represented by [`MidiNote`], a validated MIDI note number in
//! `0..=127`. Conversions to and from frequency are explicit
//! ([`MidiNote::to_hz`] / [`MidiNote::from_hz`]) so that a raw `f32` can never be
//! silently mistaken for a note number.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// A MIDI note number in the range `0..=127`.
///
/// Note 60 is middle C (C4) in the scientific pitch notation used throughout
/// `tabscribe`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MidiNote(u8);

impl MidiNote {
    /// The lowest representable MIDI note (C-1).
    pub const MIN: MidiNote = MidiNote(0);
    /// The highest representable MIDI note (G9).
    pub const MAX: MidiNote = MidiNote(127);
    /// Middle C, used as the reference for equal temperament conversions.
    pub const A4: MidiNote = MidiNote(69);
    /// The reference frequency of A4 in Hertz, for A4 = 440 Hz tuning.
    pub const A4_HZ: f64 = 440.0;

    /// Construct a note from a raw MIDI number.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] if `value` is greater than 127.
    pub fn new(value: u8) -> Result<Self> {
        if value > 127 {
            return Err(Error::OutOfRange {
                pitch: value.to_string(),
                range: "0..=127".to_string(),
            });
        }
        Ok(MidiNote(value))
    }

    /// Construct a note from a MIDI number, clamping into range.
    ///
    /// Useful when the input originates from a lossy source such as a pitch
    /// estimator, where out-of-range values are an artifact rather than a
    /// user error.
    #[must_use]
    pub fn clamped(value: u16) -> Self {
        MidiNote(value.min(127) as u8)
    }

    /// The raw MIDI number.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self.0
    }

    /// The pitch class of this note (`0..=11`, where 0 is C).
    #[must_use]
    pub const fn pitch_class(self) -> PitchClass {
        PitchClass(self.0 % 12)
    }

    /// The octave of this note in scientific pitch notation.
    ///
    /// Middle C (MIDI 60) is in octave 4; notes below MIDI 12 are in octave 0
    /// or lower.
    #[must_use]
    pub const fn octave(self) -> i32 {
        (self.0 as i32 / 12) - 1
    }

    /// The equal-tempered frequency of this note in Hertz.
    #[must_use]
    pub fn to_hz(self) -> f64 {
        Self::A4_HZ * 2f64.powf((self.0 as f64 - Self::A4.0 as f64) / 12.0)
    }

    /// The nearest equal-tempered note to `hz`.
    ///
    /// Returns `None` for non-finite or non-positive input.
    #[must_use]
    pub fn from_hz(hz: f64) -> Option<Self> {
        if !hz.is_finite() || hz <= 0.0 {
            return None;
        }
        let semitones = 12.0 * (hz / Self::A4_HZ).log2();
        let rounded = semitones.round() + Self::A4.0 as f64;
        if !rounded.is_finite() {
            return None;
        }
        Some(MidiNote(rounded.clamp(0.0, 127.0) as u8))
    }

    /// The pitch class of this note with the given accidental applied
    /// (sharp = +1 semitone, flat = -1 semitone).
    #[must_use]
    pub fn with_accidental(self, accidental: Accidental) -> Option<Self> {
        let delta = match accidental {
            Accidental::Sharp => 1,
            Accidental::Flat => -1,
            Accidental::Natural => 0,
        };
        let value = self.0 as i32 + delta;
        if (0..=127).contains(&value) {
            Some(MidiNote(value as u8))
        } else {
            None
        }
    }

    /// Build a note from a pitch class and octave.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when the resulting MIDI number is outside
    /// `0..=127`.
    pub fn from_pitch_class(pitch_class: PitchClass, octave: i32) -> Result<Self> {
        let value = (octave + 1) * 12 + pitch_class.as_u8() as i32;
        if !(0..=127).contains(&value) {
            return Err(Error::OutOfRange {
                pitch: format!("{pitch_class}{octave}"),
                range: "MIDI 0..=127".to_string(),
            });
        }
        Ok(MidiNote(value as u8))
    }

    /// A display name using sharps, e.g. `C#4`.
    #[must_use]
    pub fn name(self) -> String {
        let pitch_class = self.pitch_class();
        format!(
            "{}{}{}",
            pitch_class.base_letter(),
            pitch_class.accidental(),
            self.octave()
        )
    }
}

/// A pitch class: the twelve semitone positions, `0..=11`, where 0 is C.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PitchClass(u8);

impl PitchClass {
    /// Pitch class of C.
    pub const C: PitchClass = PitchClass(0);
    /// Pitch class of C# / Db.
    pub const CS: PitchClass = PitchClass(1);
    /// Pitch class of D.
    pub const D: PitchClass = PitchClass(2);
    /// Pitch class of D# / Eb.
    pub const DS: PitchClass = PitchClass(3);
    /// Pitch class of E.
    pub const E: PitchClass = PitchClass(4);
    /// Pitch class of F.
    pub const F: PitchClass = PitchClass(5);
    /// Pitch class of F# / Gb.
    pub const FS: PitchClass = PitchClass(6);
    /// Pitch class of G.
    pub const G: PitchClass = PitchClass(7);
    /// Pitch class of G# / Ab.
    pub const GS: PitchClass = PitchClass(8);
    /// Pitch class of A.
    pub const A: PitchClass = PitchClass(9);
    /// Pitch class of A# / Bb.
    pub const AS: PitchClass = PitchClass(10);
    /// Pitch class of B.
    pub const B: PitchClass = PitchClass(11);

    /// Construct a pitch class from a `0..=11` value.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when `value > 11`.
    pub fn new(value: u8) -> Result<Self> {
        if value > 11 {
            return Err(Error::OutOfRange {
                pitch: value.to_string(),
                range: "0..=11".to_string(),
            });
        }
        Ok(PitchClass(value))
    }

    /// The raw semitone index.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self.0
    }

    /// The natural letter (A-G) this pitch class is derived from.
    #[must_use]
    pub const fn base_letter(self) -> char {
        match self.0 {
            0 => 'C',
            2 => 'D',
            4 => 'E',
            5 => 'F',
            7 => 'G',
            9 => 'A',
            11 => 'B',
            1 => 'C',
            3 => 'D',
            6 => 'F',
            8 => 'G',
            _ => 'A',
        }
    }

    /// The accidental required to spell this pitch class with sharps.
    #[must_use]
    pub const fn accidental(self) -> &'static str {
        match self.0 {
            1 | 3 | 6 | 8 | 10 => "#",
            _ => "",
        }
    }

    /// Transpose this pitch class by `semitones`, wrapping around the octave.
    #[must_use]
    pub const fn transposed(self, semitones: i32) -> PitchClass {
        PitchClass(((self.0 as i32 + semitones).rem_euclid(12)) as u8)
    }

    /// The interval in semitones from `self` to `other`, always in `0..12`.
    #[must_use]
    pub const fn interval_to(self, other: PitchClass) -> u8 {
        (other.0 + 12 - self.0) % 12
    }

    /// The diatonic note name for a major scale, e.g. `0 => "C"`.
    #[must_use]
    pub const fn major_scale_name(self) -> &'static str {
        const NAMES: [&str; 12] = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];
        NAMES[self.0 as usize]
    }

    /// The name of this pitch class spelled with a flat.
    #[must_use]
    pub const fn flat_name(self) -> &'static str {
        const NAMES: [&str; 12] = [
            "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B",
        ];
        NAMES[self.0 as usize]
    }

    /// The name of this pitch class spelled the way chord charts usually do.
    ///
    /// The four roots that are conventionally written with a flat (Db, Eb,
    /// Ab and Bb) use it; the rest use sharps, so that `F#m` stays `F#m`
    /// rather than becoming the rarely used `Gbm`.
    #[must_use]
    pub const fn chord_name(self) -> &'static str {
        const NAMES: [&str; 12] = [
            "C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
        ];
        NAMES[self.0 as usize]
    }
}

impl fmt::Display for PitchClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.base_letter(), self.accidental())
    }
}

impl FromStr for PitchClass {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return Err(Error::InvalidTuning {
                input: s.to_string(),
                reason: "empty pitch class".to_string(),
            });
        }
        let (letter, rest) = trimmed.split_at(1);
        let letter_upper = letter.to_ascii_uppercase();
        let base = match letter_upper.as_str() {
            "C" => 0,
            "D" => 2,
            "E" => 4,
            "F" => 5,
            "G" => 7,
            "A" => 9,
            "B" => 11,
            other => {
                return Err(Error::InvalidTuning {
                    input: s.to_string(),
                    reason: format!("unknown note letter '{other}'"),
                })
            }
        };
        let accidental_count = rest.chars().take_while(|c| *c == '#' || *c == 'b').count();
        let suffix = &rest[accidental_count..];
        if !suffix.is_empty() && !suffix.chars().all(|c| c.is_ascii_digit()) {
            return Err(Error::InvalidTuning {
                input: s.to_string(),
                reason: format!("unexpected trailing characters '{suffix}'"),
            });
        }
        let semitone = match accidental_count {
            0 => base,
            1 => {
                let c = rest.as_bytes()[0];
                if c == b'#' {
                    (base + 1) % 12
                } else {
                    (base + 11) % 12
                }
            }
            n => {
                let c = rest.as_bytes()[0];
                let dir = if c == b'#' { 1 } else { -1 };
                ((base as i32 + dir * n as i32).rem_euclid(12)) as u8
            }
        };
        PitchClass::new(semitone)
    }
}

/// The accidental applied to a note when it is spelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Accidental {
    /// A sharp.
    Sharp,
    /// A flat.
    Flat,
    /// An explicit natural.
    Natural,
}

impl fmt::Display for Accidental {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Accidental::Sharp => "#",
            Accidental::Flat => "b",
            Accidental::Natural => "",
        })
    }
}

/// A signed interval measured in semitones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Interval(i16);

impl Interval {
    /// A perfect unison.
    pub const UNISON: Interval = Interval(0);
    /// A minor second.
    pub const MINOR_SECOND: Interval = Interval(1);
    /// A major second.
    pub const MAJOR_SECOND: Interval = Interval(2);
    /// A perfect fifth.
    pub const PERFECT_FIFTH: Interval = Interval(7);
    /// An octave.
    pub const OCTAVE: Interval = Interval(12);

    /// Construct an interval from a semitone count.
    #[must_use]
    pub const fn new(semitones: i16) -> Self {
        Interval(semitones)
    }

    /// The signed number of semitones.
    #[must_use]
    pub const fn as_i16(self) -> i16 {
        self.0
    }

    /// The absolute number of semitones.
    #[must_use]
    pub const fn abs(self) -> i16 {
        self.0.abs()
    }
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} semitones", self.0)
    }
}

/// A note number with an explicit spelling, used for chord and key names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpelledNote {
    /// The sounding pitch class.
    pub pitch_class: PitchClass,
    /// The accidental used in the spelling.
    pub accidental: Accidental,
    /// The octave.
    pub octave: i32,
}

impl fmt::Display for SpelledNote {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}{}", self.pitch_class.base_letter(), self.accidental, self.octave)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn midi_note_range_is_validated() {
        assert!(MidiNote::new(127).is_ok());
        assert!(MidiNote::new(128).is_err());
    }

    #[test]
    fn hz_round_trip_is_exact_for_equal_temperament() {
        for midi in 0..=127u8 {
            let note = MidiNote::new(midi).expect("in range");
            let round_tripped = MidiNote::from_hz(note.to_hz()).expect("finite");
            assert_eq!(note, round_tripped, "midi {midi}");
        }
    }

    #[test]
    fn a4_is_440_hz() {
        assert!((MidiNote::A4.to_hz() - 440.0).abs() < 1e-9);
    }

    #[test]
    fn octave_follows_scientific_pitch_notation() {
        assert_eq!(MidiNote::new(60).expect("in range").octave(), 4);
        assert_eq!(MidiNote::new(21).expect("in range").octave(), 0);
        assert_eq!(MidiNote::new(45).expect("in range").octave(), 2);
        assert_eq!(MidiNote::new(0).expect("in range").octave(), -1);
        assert_eq!(MidiNote::new(69).expect("in range").octave(), 4);
    }

    #[test]
    fn pitch_class_parses_spellings() {
        assert_eq!("C".parse::<PitchClass>().expect("valid"), PitchClass::C);
        assert_eq!("C#".parse::<PitchClass>().expect("valid"), PitchClass::CS);
        assert_eq!("Db".parse::<PitchClass>().expect("valid"), PitchClass::CS);
        assert_eq!("Bb".parse::<PitchClass>().expect("valid"), PitchClass::AS);
        assert!("H".parse::<PitchClass>().is_err());
        assert!("".parse::<PitchClass>().is_err());
    }

    #[test]
    fn transposition_wraps_around_the_octave() {
        assert_eq!(PitchClass::B.transposed(1), PitchClass::C);
        assert_eq!(PitchClass::C.transposed(-1), PitchClass::B);
        assert_eq!(PitchClass::C.transposed(12), PitchClass::C);
    }

    #[test]
    fn note_names_use_scientific_pitch_notation() {
        assert_eq!(MidiNote::new(60).expect("in range").name(), "C4");
        assert_eq!(MidiNote::new(69).expect("in range").name(), "A4");
        assert_eq!(MidiNote::new(61).expect("in range").name(), "C#4");
    }
}
