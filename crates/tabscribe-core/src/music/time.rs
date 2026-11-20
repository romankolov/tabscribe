//! Time, tempo and meter primitives.
//!
//! Three distinct notions of time appear throughout the pipeline and mixing
//! them up is the single most common source of subtle bugs, so each gets its
//! own newtype:
//!
//! * [`Seconds`] — wall-clock time in the audio file.
//! * [`Beats`] — musical time measured in quarter-note beats.
//! * [`MidiTicks`] — absolute MIDI tick positions used in `.mid` files.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// A duration or position in seconds.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Seconds(pub f64);

impl Seconds {
    /// Zero seconds.
    pub const ZERO: Seconds = Seconds(0.0);

    /// Construct a duration from a raw number of seconds.
    #[must_use]
    pub const fn from_secs(secs: f64) -> Self {
        Seconds(secs)
    }

    /// Construct a duration from a number of milliseconds.
    #[must_use]
    pub fn from_millis(ms: f64) -> Self {
        Seconds(ms / 1000.0)
    }

    /// The raw value in seconds.
    #[must_use]
    pub const fn as_secs(self) -> f64 {
        self.0
    }

    /// The raw value in milliseconds.
    #[must_use]
    pub fn as_millis(self) -> f64 {
        self.0 * 1000.0
    }

    /// Whether the value is finite and non-negative.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.0.is_finite() && self.0 >= 0.0
    }

    /// The absolute value.
    #[must_use]
    pub fn abs(self) -> Seconds {
        Seconds(self.0.abs())
    }
}

impl fmt::Display for Seconds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.3}s", self.0)
    }
}

/// A musical duration measured in quarter-note beats.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Beats(pub f64);

impl Beats {
    /// Zero beats.
    pub const ZERO: Beats = Beats(0.0);

    /// Construct a duration from a raw number of beats.
    #[must_use]
    pub const fn from_beats(beats: f64) -> Self {
        Beats(beats)
    }

    /// The raw value in beats.
    #[must_use]
    pub const fn as_beats(self) -> f64 {
        self.0
    }

    /// Whether the value is finite and non-negative.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.0.is_finite() && self.0 >= 0.0
    }
}

impl fmt::Display for Beats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.3} beats", self.0)
    }
}

/// An absolute position in a MIDI file, measured in ticks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MidiTicks(pub u32);

impl MidiTicks {
    /// Tick zero.
    pub const ZERO: MidiTicks = MidiTicks(0);

    /// The raw tick value.
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Convert ticks to seconds given the tick resolution and tempo.
    #[must_use]
    pub fn to_seconds(self, ticks_per_quarter: u32, tempo: Tempo) -> Seconds {
        let quarter_secs = tempo.seconds_per_beat().as_secs();
        Seconds((self.0 as f64 / ticks_per_quarter as f64) * quarter_secs)
    }

    /// Convert seconds to ticks given the tick resolution and tempo.
    #[must_use]
    pub fn from_seconds(secs: Seconds, ticks_per_quarter: u32, tempo: Tempo) -> MidiTicks {
        let quarter_secs = tempo.seconds_per_beat().as_secs();
        if quarter_secs <= 0.0 {
            return MidiTicks(0);
        }
        let ticks = (secs.as_secs() / quarter_secs) * ticks_per_quarter as f64;
        MidiTicks(ticks.max(0.0).round() as u32)
    }
}

/// A tempo, expressed in quarter-note beats per minute.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Tempo {
    bpm: f64,
}

impl Tempo {
    /// The default tempo used when nothing can be detected.
    pub const DEFAULT: Tempo = Tempo { bpm: 120.0 };

    /// Construct a tempo from a BPM value.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the value is not a finite positive number.
    pub fn from_bpm(bpm: f64) -> Result<Self> {
        if !bpm.is_finite() || bpm <= 0.0 {
            return Err(Error::Config(format!(
                "tempo must be a positive finite number of BPM, got {bpm}"
            )));
        }
        Ok(Tempo { bpm })
    }

    /// The tempo in quarter-note beats per minute.
    #[must_use]
    pub fn bpm(self) -> f64 {
        self.bpm
    }

    /// The duration of one quarter-note beat.
    #[must_use]
    pub fn seconds_per_beat(self) -> Seconds {
        Seconds(60.0 / self.bpm)
    }

    /// Convert a tempo to the microseconds-per-quarter-note used by MIDI files.
    #[must_use]
    pub fn to_microseconds_per_quarter(self) -> u32 {
        (60_000_000.0 / self.bpm).round() as u32
    }

    /// Build a tempo from a MIDI microseconds-per-quarter-note value.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] for a zero denominator.
    pub fn from_microseconds_per_quarter(us: u32) -> Result<Self> {
        if us == 0 {
            return Err(Error::Config(
                "MIDI tempo of 0 microseconds per quarter note is invalid".to_string(),
            ));
        }
        Tempo::from_bpm(60_000_000.0 / us as f64)
    }

    /// Scale the tempo by a factor, e.g. to move between half and double time.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the resulting tempo is invalid.
    pub fn scaled(self, factor: f64) -> Result<Self> {
        Tempo::from_bpm(self.bpm * factor)
    }
}

impl fmt::Display for Tempo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1} BPM", self.bpm)
    }
}

/// A time signature, e.g. `4/4` or `6/8`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeSignature {
    /// Number of beats in a bar.
    pub numerator: u8,
    /// The note value that gets one beat.
    pub denominator: u8,
}

impl TimeSignature {
    /// Common time.
    pub const COMMON: TimeSignature = TimeSignature { numerator: 4, denominator: 4 };

    /// Construct a time signature.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the numerator is zero or the denominator
    /// is not a power of two.
    pub fn new(numerator: u8, denominator: u8) -> Result<Self> {
        if numerator == 0 {
            return Err(Error::Config(
                "time signature numerator must be greater than zero".to_string(),
            ));
        }
        if denominator == 0 || !denominator.is_power_of_two() {
            return Err(Error::Config(format!(
                "time signature denominator {denominator} must be a power of two"
            )));
        }
        Ok(TimeSignature { numerator, denominator })
    }

    /// The number of quarter-note beats in one bar.
    #[must_use]
    pub fn beats_per_bar(self) -> f64 {
        self.numerator as f64 * 4.0 / self.denominator as f64
    }

    /// The duration of one bar.
    #[must_use]
    pub fn bar_duration(self, tempo: Tempo) -> Seconds {
        Seconds(tempo.seconds_per_beat().as_secs() * self.beats_per_bar())
    }

    /// Whether the signature is a compound meter (e.g. 6/8, 9/8, 12/8).
    #[must_use]
    pub fn is_compound(self) -> bool {
        self.denominator >= 8 && self.numerator % 3 == 0 && self.numerator > 3
    }

    /// The grouping of beats within a bar, used by the beat tracker.
    ///
    /// For `4/4` this is `[1, 1, 1, 1]`; for `6/8` it is `[3, 3]` because the
    /// bar is felt in two dotted-quarter pulses.
    #[must_use]
    pub fn beat_grouping(self) -> Vec<u8> {
        if self.is_compound() {
            let groups = self.numerator / 3;
            vec![3; groups as usize]
        } else {
            vec![1; self.numerator as usize]
        }
    }
}

impl fmt::Display for TimeSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.numerator, self.denominator)
    }
}

impl FromStr for TimeSignature {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        let trimmed = s.trim();
        let (num, den) = trimmed.split_once('/').ok_or_else(|| Error::Config(format!(
            "time signature '{trimmed}' must be written as N/D, for example 4/4"
        )))?;
        let numerator: u8 = num.trim().parse().map_err(|_| {
            Error::Config(format!("invalid time signature numerator '{num}'"))
        })?;
        let denominator: u8 = den.trim().parse().map_err(|_| {
            Error::Config(format!("invalid time signature denominator '{den}'"))
        })?;
        TimeSignature::new(numerator, denominator)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn tempo_round_trips_through_midi_representation() {
        let tempo = Tempo::from_bpm(96.0).expect("valid");
        let us = tempo.to_microseconds_per_quarter();
        let back = Tempo::from_microseconds_per_quarter(us).expect("valid");
        assert!((back.bpm() - 96.0).abs() < 0.01);
    }

    #[test]
    fn invalid_tempos_are_rejected() {
        assert!(Tempo::from_bpm(0.0).is_err());
        assert!(Tempo::from_bpm(f64::NAN).is_err());
        assert!(Tempo::from_bpm(-30.0).is_err());
    }

    #[test]
    fn time_signature_parsing_rejects_garbage() {
        assert_eq!(
            "4/4".parse::<TimeSignature>().expect("valid"),
            TimeSignature::COMMON
        );
        assert_eq!(
            "6/8".parse::<TimeSignature>().expect("valid").numerator,
            6
        );
        assert!("4".parse::<TimeSignature>().is_err());
        assert!("4/5".parse::<TimeSignature>().is_err());
        assert!("0/4".parse::<TimeSignature>().is_err());
    }

    #[test]
    fn compound_meters_group_beats() {
        let six_eight = TimeSignature::new(6, 8).expect("valid");
        assert!(six_eight.is_compound());
        assert_eq!(six_eight.beat_grouping(), vec![3, 3]);
        let four_four = TimeSignature::COMMON;
        assert!(!four_four.is_compound());
        assert_eq!(four_four.beat_grouping(), vec![1, 1, 1, 1]);
    }

    #[test]
    fn ticks_and_seconds_round_trip() {
        let tempo = Tempo::from_bpm(120.0).expect("valid");
        // At 120 BPM a second is two beats, i.e. 960 ticks.
        let ticks = MidiTicks::from_seconds(Seconds::from_secs(1.0), 480, tempo);
        assert_eq!(ticks.as_u32(), 960);
        let secs = ticks.to_seconds(480, tempo);
        assert!((secs.as_secs() - 1.0).abs() < 1e-9);
    }
}
