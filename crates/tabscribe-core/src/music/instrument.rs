//! Instrument definitions: kind, tuning, capo and playable range.
//!
//! An [`Instrument`] is the single source of truth for what a human can
//! physically play. The tablature optimizer only ever considers positions
//! returned by [`Instrument::positions_for`], which is what makes the
//! "playable by a human" guarantee structural rather than a post-hoc check.

use std::fmt;

use serde::{Deserialize, Serialize};

use super::note::MidiNote;
use super::tuning::{Fret, StringIdx, Tuning, TuningPreset};
use crate::error::{Error, Result};

/// The broad family an instrument belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InstrumentKind {
    /// A six-string guitar.
    Guitar,
    /// A bass guitar (4, 5 or 6 strings).
    Bass,
    /// A ukulele.
    Ukulele,
    /// A user-defined string layout.
    Custom,
}

impl InstrumentKind {
    /// All kinds, in a stable order suitable for `--help` output.
    pub const ALL: &'static [InstrumentKind] = &[
        InstrumentKind::Guitar,
        InstrumentKind::Bass,
        InstrumentKind::Ukulele,
        InstrumentKind::Custom,
    ];

    /// The canonical lowercase name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            InstrumentKind::Guitar => "guitar",
            InstrumentKind::Bass => "bass",
            InstrumentKind::Ukulele => "ukulele",
            InstrumentKind::Custom => "custom",
        }
    }

    /// The default tuning for this kind.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInstrument`] if the default tuning is malformed.
    pub fn default_tuning(self) -> Result<Tuning> {
        match self {
            InstrumentKind::Guitar => TuningPreset::Standard.tuning(),
            InstrumentKind::Bass => TuningPreset::Bass4.tuning(),
            InstrumentKind::Ukulele => TuningPreset::Ukulele.tuning(),
            InstrumentKind::Custom => Err(Error::InvalidInstrument(
                "custom instruments require an explicit --tuning".to_string(),
            )),
        }
    }

    /// The General MIDI program number used when rendering MIDI.
    #[must_use]
    pub fn gm_program(self) -> u8 {
        match self {
            // 24 = Acoustic Guitar (steel), 25 = Acoustic Guitar (nylon).
            InstrumentKind::Guitar => 25,
            // 33 = Electric Bass (finger).
            InstrumentKind::Bass => 33,
            // 24 = Acoustic Guitar (steel) is the usual ukulele stand-in.
            InstrumentKind::Ukulele => 24,
            InstrumentKind::Custom => 25,
        }
    }

    /// The General MIDI channel used when rendering MIDI.
    #[must_use]
    pub fn gm_channel(self) -> u8 {
        match self {
            InstrumentKind::Bass => 1,
            _ => 0,
        }
    }
}

impl fmt::Display for InstrumentKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A concrete, playable instrument configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Instrument {
    kind: InstrumentKind,
    tuning: Tuning,
    capo: u8,
    max_fret: u8,
}

/// The default number of frets assumed when none is specified.
pub const DEFAULT_MAX_FRET: u8 = 22;

/// The maximum fret a capo may be placed at.
pub const MAX_CAPO: u8 = 12;

impl Instrument {
    /// Build an instrument from its parts.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInstrument`] when the capo is out of range or the
    /// fret range is empty.
    pub fn new(kind: InstrumentKind, tuning: Tuning, capo: u8, max_fret: u8) -> Result<Self> {
        if capo > MAX_CAPO {
            return Err(Error::InvalidInstrument(format!(
                "capo {capo} exceeds the maximum of {MAX_CAPO}"
            )));
        }
        if max_fret == 0 {
            return Err(Error::InvalidInstrument(
                "the instrument must have at least one fret".to_string(),
            ));
        }
        Ok(Instrument { kind, tuning, capo, max_fret })
    }

    /// The default instrument for a given kind.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInstrument`] if the default tuning is malformed.
    pub fn for_kind(kind: InstrumentKind) -> Result<Self> {
        let tuning = kind.default_tuning()?;
        Instrument::new(kind, tuning, 0, DEFAULT_MAX_FRET)
    }

    /// A custom instrument with an explicit tuning.
    ///
    /// # Errors
    ///
    /// Propagates [`Instrument::new`].
    pub fn custom(tuning: Tuning, capo: u8, max_fret: u8) -> Result<Self> {
        Instrument::new(InstrumentKind::Custom, tuning, capo, max_fret)
    }

    /// The instrument family.
    #[must_use]
    pub fn kind(&self) -> InstrumentKind {
        self.kind
    }

    /// The tuning.
    #[must_use]
    pub fn tuning(&self) -> &Tuning {
        &self.tuning
    }

    /// The capo position (0 = no capo).
    #[must_use]
    pub fn capo(&self) -> u8 {
        self.capo
    }

    /// The highest fret available.
    #[must_use]
    pub fn max_fret(&self) -> u8 {
        self.max_fret
    }

    /// The number of strings.
    #[must_use]
    pub fn string_count(&self) -> usize {
        self.tuning.string_count()
    }

    /// The pitch a string sounds at when fretted, *ignoring* the capo.
    ///
    /// This is the pitch that is written in the tablature.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when the pitch leaves the MIDI range.
    pub fn written_note(&self, string: StringIdx, fret: Fret) -> Result<MidiNote> {
        self.tuning.pitch_at(string, fret)
    }

    /// The pitch a string actually sounds at, including the capo.
    ///
    /// This is the pitch a transcriber or MIDI file refers to.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when the pitch leaves the MIDI range.
    pub fn sounding_note(&self, string: StringIdx, fret: Fret) -> Result<MidiNote> {
        let written = self.written_note(string, fret)?;
        let value = written.as_u8() as i32 + self.capo as i32;
        if !(0..=127).contains(&value) {
            return Err(Error::OutOfRange {
                pitch: format!("string {string} fret {fret} with capo {}", self.capo),
                range: "MIDI 0..=127".to_string(),
            });
        }
        MidiNote::new(value as u8).map_err(|_| Error::OutOfRange {
            pitch: format!("string {string} fret {fret} with capo {}", self.capo),
            range: "MIDI 0..=127".to_string(),
        })
    }

    /// The lowest and highest sounding pitches of the instrument.
    #[must_use]
    pub fn range(&self) -> (MidiNote, MidiNote) {
        let lowest = self.tuning.lowest_open().as_u8() as i32 + self.capo as i32;
        let highest = self.tuning.highest_open().as_u8() as i32
            + self.capo as i32
            + self.max_fret as i32;
        (
            MidiNote::clamped(lowest.max(0) as u16),
            MidiNote::clamped(highest.min(127) as u16),
        )
    }

    /// Whether a sounding pitch can be produced by this instrument.
    #[must_use]
    pub fn can_play(&self, pitch: MidiNote) -> bool {
        !self.positions_for(pitch).is_empty()
    }

    /// Every `(string, fret)` pair that produces the given sounding pitch.
    ///
    /// The result is ordered from the lowest string to the highest, which is the
    /// order the optimizer explores.
    ///
    /// # Errors
    ///
    /// Never fails, but returns an empty vector when the pitch is out of range.
    #[must_use]
    pub fn positions_for(&self, pitch: MidiNote) -> Vec<(StringIdx, Fret)> {
        let target = pitch.as_u8() as i32 - self.capo as i32;
        let mut out = Vec::new();
        for spec in self.tuning.strings() {
            let delta = target - spec.open.as_u8() as i32;
            if delta < 0 || delta > self.max_fret as i32 {
                continue;
            }
            if let Ok(fret) = Fret::new(delta as u8) {
                out.push((spec.index, fret));
            }
        }
        out
    }

    /// The fret that produces `pitch` on `string`, if any.
    ///
    /// # Errors
    ///
    /// Never fails; returns `None` when the pitch is unreachable on that string.
    #[must_use]
    pub fn fret_for(&self, string: StringIdx, pitch: MidiNote) -> Option<Fret> {
        let target = pitch.as_u8() as i32 - self.capo as i32;
        let open = self.tuning.open_note(string).ok()?.as_u8() as i32;
        let delta = target - open;
        if delta < 0 || delta > self.max_fret as i32 {
            return None;
        }
        Fret::new(delta as u8).ok()
    }

    /// A one line description used in headers, e.g.
    /// `Tuning: E A D G B E   Capo: 2`.
    ///
    /// The tuning is listed from the lowest-pitched string to the highest,
    /// which is how tunings are conventionally written. The ASCII tab body
    /// uses the opposite (string) order.
    #[must_use]
    pub fn describe(&self) -> String {
        // Sort numerically by pitch rather than lexicographically.
        let mut sorted: Vec<(u8, String)> = self
            .tuning
            .strings()
            .iter()
            .map(|s| (s.open.as_u8(), s.open.name()))
            .collect();
        sorted.sort_by_key(|(midi, _)| *midi);
        let letters: Vec<String> = sorted
            .iter()
            .map(|(_, name)| {
                name.trim_end_matches(|c: char| c.is_ascii_digit())
                    .to_string()
            })
            .collect();
        format!("Tuning: {}   Capo: {}", letters.join(" "), self.capo)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn capo_shifts_sounding_but_not_written_pitch() {
        let instrument = Instrument::new(
            InstrumentKind::Guitar,
            TuningPreset::Standard.tuning().expect("valid"),
            2,
            22,
        )
        .expect("valid instrument");
        // String 1 of a standard guitar is the high E.
        let high_e = StringIdx::new(0).expect("valid");
        let written = instrument.written_note(high_e, Fret::OPEN).expect("valid");
        let sounding = instrument.sounding_note(high_e, Fret::OPEN).expect("valid");
        assert_eq!(written.as_u8(), 64);
        assert_eq!(sounding.as_u8(), 66);
    }

    #[test]
    fn positions_for_enumerates_every_reachable_position() {
        let instrument = Instrument::for_kind(InstrumentKind::Guitar).expect("valid");
        // E4 (64) is reachable on string 1 open, string 2 fret 5, string 3
        // fret 9, string 4 fret 14 and string 5 fret 19 (standard tuning,
        // 22 frets).
        let positions = instrument.positions_for(MidiNote::new(64).expect("valid"));
        assert_eq!(positions.len(), 5);
        assert!(positions.contains(&(StringIdx::new(0).expect("valid"), Fret::OPEN)));
        assert!(positions.contains(&(StringIdx::new(1).expect("valid"), Fret::new(5).expect("valid"))));
        assert!(positions.contains(&(StringIdx::new(2).expect("valid"), Fret::new(9).expect("valid"))));
        assert!(positions.contains(&(StringIdx::new(3).expect("valid"), Fret::new(14).expect("valid"))));
        assert!(positions.contains(&(StringIdx::new(4).expect("valid"), Fret::new(19).expect("valid"))));
    }

    #[test]
    fn positions_respect_the_max_fret() {
        let instrument = Instrument::new(
            InstrumentKind::Guitar,
            TuningPreset::Standard.tuning().expect("valid"),
            0,
            5,
        )
        .expect("valid");
        let positions = instrument.positions_for(MidiNote::new(64).expect("valid"));
        assert!(positions.iter().all(|(_, fret)| fret.as_u8() <= 5));
        assert_eq!(positions.len(), 2);
    }

    #[test]
    fn header_describes_the_tuning_low_pitch_first() {
        let instrument = Instrument::for_kind(InstrumentKind::Guitar).expect("valid");
        assert_eq!(instrument.describe(), "Tuning: E A D G B E   Capo: 0");
    }

    #[test]
    fn out_of_range_pitches_have_no_positions() {
        let instrument = Instrument::for_kind(InstrumentKind::Guitar).expect("valid");
        assert!(instrument.positions_for(MidiNote::new(0).expect("valid")).is_empty());
    }
}
