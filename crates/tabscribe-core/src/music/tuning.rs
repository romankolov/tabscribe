//! Tunings, string layouts and capo handling.
//!
//! A [`Tuning`] describes the open strings of an instrument in **string
//! order**, where index 0 is string 1: the string at the top of an ASCII tab
//! and the highest-numbered string in conventional tablature notation.
//!
//! String order is *not* the same as pitch order. On a ukulele in standard
//! GCEA tuning, string 4 (G4) sounds higher than string 3 (C4), so a
//! pitch-sorted representation would print the tab upside down. Every
//! instrument `tabscribe` supports numbers its strings the way a tablature
//! player expects, so the renderer can print them top to bottom directly.
//!
//! The [`Instrument`] type layers a tuning together with a capo position and a
//! fret range to describe a concrete, playable configuration.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use super::note::MidiNote;
use crate::error::{Error, Result};

/// A zero-based index of a string within a [`Tuning`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StringIdx(u8);

impl StringIdx {
    /// The lowest string index.
    pub const LOWEST: StringIdx = StringIdx(0);

    /// Construct a string index.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when `value > 127`.
    pub fn new(value: u8) -> Result<Self> {
        if value > 127 {
            return Err(Error::OutOfRange {
                pitch: format!("string {value}"),
                range: "0..=127".to_string(),
            });
        }
        Ok(StringIdx(value))
    }

    /// The raw index.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self.0
    }

    /// The index of the next-higher-pitched string, if any.
    #[must_use]
    pub const fn next(self) -> Option<StringIdx> {
        if self.0 == u8::MAX {
            None
        } else {
            Some(StringIdx(self.0 + 1))
        }
    }
}

impl fmt::Display for StringIdx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "string {}", self.0)
    }
}

/// A fret number, where 0 is an open string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Fret(u8);

impl Fret {
    /// An open string.
    pub const OPEN: Fret = Fret(0);

    /// Construct a fret number.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when `value > 127`.
    pub fn new(value: u8) -> Result<Self> {
        if value > 127 {
            return Err(Error::OutOfRange {
                pitch: format!("fret {value}"),
                range: "0..=127".to_string(),
            });
        }
        Ok(Fret(value))
    }

    /// The raw fret number.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self.0
    }

    /// The number of semitones above the open string.
    #[must_use]
    pub const fn semitones(self) -> i32 {
        self.0 as i32
    }

    /// The fret `n` semitones above this one, if it stays in range.
    #[must_use]
    pub fn offset(self, n: i32) -> Option<Fret> {
        let value = self.0 as i32 + n;
        if (0..=127).contains(&value) {
            Some(Fret(value as u8))
        } else {
            None
        }
    }
}

impl fmt::Display for Fret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// One string of a [`Tuning`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StringSpec {
    /// Position of this string within the tuning (0 = lowest pitch).
    pub index: StringIdx,
    /// The note sounded by the open (unfretted) string.
    pub open: MidiNote,
    /// A human readable label, e.g. `E2`.
    pub label: String,
}

/// The set of open strings of an instrument.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tuning {
    strings: Vec<StringSpec>,
}

impl Tuning {
    /// Build a tuning from open-string MIDI notes in string order.
    ///
    /// `notes[0]` is string 1 (the top line of an ASCII tab).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInstrument`] when the list is empty.
    pub fn from_open_notes(notes: &[MidiNote]) -> Result<Self> {
        if notes.is_empty() {
            return Err(Error::InvalidInstrument(
                "a tuning must have at least one string".to_string(),
            ));
        }
        let strings = notes
            .iter()
            .enumerate()
            .map(|(i, &open)| StringSpec {
                index: StringIdx(i as u8),
                open,
                label: open.name(),
            })
            .collect();
        Ok(Tuning { strings })
    }

    /// Build a tuning from a list of open strings given as `MIDI` numbers.
    ///
    /// # Errors
    ///
    /// Propagates [`Tuning::from_open_notes`].
    pub fn from_midi_notes(notes: &[u8]) -> Result<Self> {
        let parsed: Vec<MidiNote> = notes
            .iter()
            .map(|&n| MidiNote::new(n))
            .collect::<Result<Vec<_>>>()?;
        Tuning::from_open_notes(&parsed)
    }

    /// The strings of this tuning, lowest pitch first.
    #[must_use]
    pub fn strings(&self) -> &[StringSpec] {
        &self.strings
    }

    /// The number of strings.
    #[must_use]
    pub fn string_count(&self) -> usize {
        self.strings.len()
    }

    /// The open note of the string at `index`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when `index` is beyond the last string.
    pub fn open_note(&self, index: StringIdx) -> Result<MidiNote> {
        self.strings
            .get(index.as_u8() as usize)
            .map(|s| s.open)
            .ok_or_else(|| Error::OutOfRange {
                pitch: format!("string index {}", index.as_u8()),
                range: format!("0..{}", self.string_count().saturating_sub(1)),
            })
    }

    /// The pitch produced by a string fretted at `fret`, ignoring capo.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when the resulting pitch leaves the MIDI
    /// range.
    pub fn pitch_at(&self, index: StringIdx, fret: Fret) -> Result<MidiNote> {
        let open = self.open_note(index)?.as_u8() as i32;
        let value = open + fret.semitones();
        if !(0..=127).contains(&value) {
            return Err(Error::OutOfRange {
                pitch: format!("string {index} fret {fret}"),
                range: "MIDI 0..=127".to_string(),
            });
        }
        MidiNote::new(value as u8).map_err(|_| Error::OutOfRange {
            pitch: format!("string {index} fret {fret}"),
            range: "MIDI 0..=127".to_string(),
        })
    }

    /// The lowest pitch this tuning can produce (all strings open).
    #[must_use]
    pub fn lowest_open(&self) -> MidiNote {
        self.strings
            .iter()
            .map(|s| s.open)
            .min_by_key(|n| n.as_u8())
            .unwrap_or(MidiNote::MIN)
    }

    /// The highest pitch this tuning can produce (all strings open).
    #[must_use]
    pub fn highest_open(&self) -> MidiNote {
        self.strings
            .iter()
            .map(|s| s.open)
            .max_by_key(|n| n.as_u8())
            .unwrap_or(MidiNote::MIN)
    }

    /// A human readable description such as `E2 A2 D3 G3 B3 E4`.
    #[must_use]
    pub fn describe(&self) -> String {
        self.strings
            .iter()
            .map(|s| s.label.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// A named, well-known tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TuningPreset {
    /// Standard guitar tuning: E2 A2 D3 G3 B3 E4.
    Standard,
    /// Drop D guitar tuning: D2 A2 D3 G3 B3 E4.
    DropD,
    /// Drop C guitar tuning: C2 G2 C3 F3 A3 D4.
    DropC,
    /// Half-step-down guitar tuning.
    HalfStepDown,
    /// DADGAD guitar tuning: D2 A2 D3 G3 A3 D4.
    Dadgad,
    /// Open G guitar tuning: D2 G2 D3 G3 B3 D4.
    OpenG,
    /// Open D guitar tuning: D2 A2 D3 F3 A3 D4.
    OpenD,
    /// Open E guitar tuning: E2 B2 E3 G#3 B3 E4.
    OpenE,
    /// Four-string bass in standard tuning: E1 A1 D2 G2.
    Bass4,
    /// Five-string bass in standard tuning: B0 E1 A1 D2 G2.
    Bass5,
    /// Six-string bass in standard tuning: B0 E1 A1 D2 G2 C3.
    Bass6,
    /// Drop D bass tuning: D1 A1 D2 G2.
    BassDropD,
    /// Standard ukulele tuning: G4 C4 E4 A4.
    Ukulele,
    /// Baritone ukulele tuning: D4 G4 B3 E4.
    UkuleleBaritone,
}

impl TuningPreset {
    /// All presets, in a stable order suitable for `--help` output.
    pub const ALL: &'static [TuningPreset] = &[
        TuningPreset::Standard,
        TuningPreset::DropD,
        TuningPreset::DropC,
        TuningPreset::HalfStepDown,
        TuningPreset::Dadgad,
        TuningPreset::OpenG,
        TuningPreset::OpenD,
        TuningPreset::OpenE,
        TuningPreset::Bass4,
        TuningPreset::Bass5,
        TuningPreset::Bass6,
        TuningPreset::BassDropD,
        TuningPreset::Ukulele,
        TuningPreset::UkuleleBaritone,
    ];

    /// The canonical name of this preset.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            TuningPreset::Standard => "standard",
            TuningPreset::DropD => "drop-d",
            TuningPreset::DropC => "drop-c",
            TuningPreset::HalfStepDown => "half-step-down",
            TuningPreset::Dadgad => "dadgad",
            TuningPreset::OpenG => "open-g",
            TuningPreset::OpenD => "open-d",
            TuningPreset::OpenE => "open-e",
            TuningPreset::Bass4 => "bass-4",
            TuningPreset::Bass5 => "bass-5",
            TuningPreset::Bass6 => "bass-6",
            TuningPreset::BassDropD => "bass-drop-d",
            TuningPreset::Ukulele => "ukulele",
            TuningPreset::UkuleleBaritone => "ukulele-baritone",
        }
    }

    /// The open-string MIDI notes of this preset.
    #[must_use]
    pub fn open_notes(self) -> Vec<MidiNote> {
        let raw: &[u8] = match self {
            // String order: index 0 is string 1 (top line of the tab).
            TuningPreset::Standard => &[64, 59, 55, 50, 45, 40],
            TuningPreset::DropD => &[64, 59, 55, 50, 45, 38],
            TuningPreset::DropC => &[62, 57, 53, 48, 43, 36],
            TuningPreset::HalfStepDown => &[63, 58, 54, 49, 44, 39],
            TuningPreset::Dadgad => &[62, 57, 55, 50, 45, 38],
            TuningPreset::OpenG => &[62, 59, 55, 50, 43, 38],
            TuningPreset::OpenD => &[62, 57, 53, 50, 45, 38],
            TuningPreset::OpenE => &[64, 59, 56, 52, 47, 40],
            TuningPreset::Bass4 => &[43, 38, 33, 28],
            TuningPreset::Bass5 => &[43, 38, 33, 28, 23],
            TuningPreset::Bass6 => &[48, 43, 38, 33, 28, 23],
            TuningPreset::BassDropD => &[43, 38, 33, 26],
            TuningPreset::Ukulele => &[69, 64, 60, 67],
            TuningPreset::UkuleleBaritone => &[69, 64, 60, 55],
        };
        raw.iter().filter_map(|&n| MidiNote::new(n).ok()).collect()
    }

    /// Materialize this preset as a [`Tuning`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInstrument`] if the preset data is inconsistent.
    pub fn tuning(self) -> Result<Tuning> {
        Tuning::from_open_notes(&self.open_notes())
    }
}

impl fmt::Display for TuningPreset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for TuningPreset {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        let normalized = s.trim().to_ascii_lowercase().replace('_', "-");
        TuningPreset::ALL
            .iter()
            .copied()
            .find(|p| p.name() == normalized)
            .ok_or_else(|| Error::InvalidTuning {
                input: s.to_string(),
                reason: format!(
                    "unknown preset; available presets: {}",
                    TuningPreset::ALL
                        .iter()
                        .map(|p| p.name())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            })
    }
}

/// Parse a tuning given either as a preset name or as a comma separated list
/// of open strings such as `E4,B3,G3,D3,A2,E2`.
///
/// The list is given in **string order**: the first entry is string 1, the top
/// line of an ASCII tab. `E2,A2,D3,G3,B3,E4` (low pitch first) is also
/// accepted and is detected as reversed automatically, because that ordering is
/// what most people write down.
///
/// # Errors
///
/// Returns [`Error::InvalidTuning`] when the input matches neither form.
pub fn parse_tuning(spec: &str) -> Result<Tuning> {
    let trimmed = spec.trim();
    if trimmed.is_empty() {
        return Err(Error::InvalidTuning {
            input: spec.to_string(),
            reason: "empty tuning specification".to_string(),
        });
    }
    if trimmed.contains(',') {
        let mut notes = Vec::new();
        for part in trimmed.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            notes.push(parse_open_note(part)?);
        }
        // Accept either spelling order: if the list ascends in pitch it is
        // almost certainly written low-to-high, so reverse it into string
        // order. A list that is neither ascending nor descending is left
        // alone (this is the ukulele case, where string order is genuinely
        // non-monotonic).
        if notes.len() > 1 && notes.windows(2).all(|w| w[0] < w[1]) {
            notes.reverse();
        }
        return Tuning::from_open_notes(&notes);
    }
    let preset: TuningPreset = trimmed.parse()?;
    preset.tuning()
}

/// Parse a single open-string label such as `E2`, `A#1` or `Bb3`.
///
/// # Errors
///
/// Returns [`Error::InvalidTuning`] when the label is not a note plus octave.
pub fn parse_open_note(label: &str) -> Result<MidiNote> {
    let label = label.trim();
    let split = label
        .find(|c: char| c.is_ascii_digit())
        .ok_or_else(|| Error::InvalidTuning {
            input: label.to_string(),
            reason: "missing octave number".to_string(),
        })?;
    let (name, octave_str) = label.split_at(split);
    let octave: i32 = octave_str.parse().map_err(|_| Error::InvalidTuning {
        input: label.to_string(),
        reason: format!("invalid octave '{octave_str}'"),
    })?;
    let pitch_class: super::note::PitchClass =
        name.parse().map_err(|_| Error::InvalidTuning {
            input: label.to_string(),
            reason: format!("invalid note name '{name}'"),
        })?;
    MidiNote::from_pitch_class(pitch_class, octave)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn standard_guitar_is_numbered_from_the_high_string() {
        let tuning = TuningPreset::Standard.tuning().expect("valid preset");
        assert_eq!(tuning.string_count(), 6);
        // String 1 is the high E.
        assert_eq!(
            tuning.open_note(StringIdx::new(0).expect("valid")).expect("valid"),
            MidiNote::new(64).expect("valid")
        );
        // String 6 is the low E.
        assert_eq!(
            tuning.open_note(StringIdx::new(5).expect("valid")).expect("valid"),
            MidiNote::new(40).expect("valid")
        );
        assert_eq!(tuning.lowest_open(), MidiNote::new(40).expect("valid"));
        assert_eq!(tuning.highest_open(), MidiNote::new(64).expect("valid"));
    }

    #[test]
    fn tuning_spec_parses_open_strings_in_either_order() {
        let tuning = parse_tuning("E2,A2,D3,G3,B3,E4").expect("valid");
        assert_eq!(tuning.string_count(), 6);
        // Normalised into string order: high E first.
        assert_eq!(tuning.describe(), "E4 B3 G3 D3 A2 E2");

        let explicit = parse_tuning("E4,B3,G3,D3,A2,E2").expect("valid");
        assert_eq!(explicit, tuning);
    }

    #[test]
    fn tuning_spec_parses_presets() {
        let drop_d = parse_tuning("drop-d").expect("valid");
        assert_eq!(drop_d.lowest_open(), MidiNote::new(38).expect("valid"));
        assert!(parse_tuning("not-a-tuning").is_err());
    }

    #[test]
    fn empty_tunings_are_rejected() {
        assert!(Tuning::from_open_notes(&[]).is_err());
    }

    #[test]
    fn pitch_at_composes_open_string_and_fret() {
        let tuning = TuningPreset::Standard.tuning().expect("valid preset");
        let third = StringIdx::new(2).expect("valid");
        // String 3 of a standard guitar is G3 = MIDI 55, fret 3 = Bb3 = 58.
        assert_eq!(
            tuning.pitch_at(third, Fret::new(3).expect("valid")).expect("valid"),
            MidiNote::new(58).expect("valid")
        );
    }

    #[test]
    fn ukulele_string_order_is_not_pitch_order() {
        let tuning = TuningPreset::Ukulele.tuning().expect("valid preset");
        assert_eq!(tuning.string_count(), 4);
        // String order is A4, E4, C4, G4: the G string is the lowest but
        // sounds higher than the C string.
        let names: Vec<String> = (0..4)
            .map(|i| {
                tuning
                    .open_note(StringIdx::new(i).expect("valid"))
                    .expect("valid")
                    .name()
            })
            .collect();
        assert_eq!(names, vec!["A4", "E4", "C4", "G4"]);
        assert_eq!(tuning.lowest_open(), MidiNote::new(60).expect("valid"));
        assert_eq!(tuning.highest_open(), MidiNote::new(69).expect("valid"));
    }
}
