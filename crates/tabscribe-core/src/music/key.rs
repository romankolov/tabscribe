//! Keys, modes and key signatures.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use super::note::PitchClass;
use crate::error::{Error, Result};

/// The mode of a key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Mode {
    /// Major mode.
    Major,
    /// Minor mode.
    Minor,
}

impl Mode {
    /// The suffix used in key names, e.g. `m` for minor.
    #[must_use]
    pub fn suffix(self) -> &'static str {
        match self {
            Mode::Major => "",
            Mode::Minor => "m",
        }
    }
}

/// A musical key: a tonic pitch class plus a mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Key {
    /// The tonic.
    pub tonic: PitchClass,
    /// The mode.
    pub mode: Mode,
}

impl Key {
    /// The key of C major.
    pub const C_MAJOR: Key = Key {
        tonic: PitchClass::C,
        mode: Mode::Major,
    };

    /// Construct a key.
    #[must_use]
    pub const fn new(tonic: PitchClass, mode: Mode) -> Self {
        Key { tonic, mode }
    }

    /// The pitch classes of the diatonic scale, ascending from the tonic.
    #[must_use]
    pub fn scale(&self) -> Vec<PitchClass> {
        let offsets: &[u8] = match self.mode {
            Mode::Major => &[0, 2, 4, 5, 7, 9, 11],
            Mode::Minor => &[0, 2, 3, 5, 7, 8, 10],
        };
        offsets
            .iter()
            .map(|&o| self.tonic.transposed(o as i32))
            .collect()
    }

    /// The number of sharps (positive) or flats (negative) in the key signature.
    ///
    /// Used by the MusicXML exporter to emit a `<key>` element.
    #[must_use]
    pub fn fifths(self) -> i8 {
        // The minor key signature equals that of its relative major, which is
        // three semitones above the minor tonic.
        let major_tonic = match self.mode {
            Mode::Major => self.tonic,
            Mode::Minor => self.tonic.transposed(3),
        };
        match major_tonic.as_u8() {
            0 => 0,   // C
            7 => 1,   // G
            2 => 2,   // D
            9 => 3,   // A
            4 => 4,   // E
            11 => 5,  // B
            6 => 6,   // F#
            1 => 7,   // C#
            5 => -1,  // F
            10 => -2, // Bb
            3 => -3,  // Eb
            8 => -4,  // Ab
            _ => -5,  // Db
        }
    }

    /// The name of the key, e.g. `Am`, `C`, `F#m`.
    ///
    /// Flat keys are spelled with flats, which is how they are written down.
    #[must_use]
    pub fn name(self) -> String {
        let tonic = if self.fifths() < 0 {
            self.tonic.flat_name()
        } else {
            self.tonic.major_scale_name()
        };
        format!("{tonic}{}", self.mode.suffix())
    }

    /// The relative major of this key.
    #[must_use]
    pub fn relative_major(self) -> Key {
        match self.mode {
            Mode::Major => self,
            Mode::Minor => Key::new(self.tonic.transposed(3), Mode::Major),
        }
    }
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name())
    }
}

impl FromStr for Key {
    type Err = Error;

    /// Parse a key name such as `Am`, `C`, `F#m`, `bb`, `D`.
    fn from_str(s: &str) -> Result<Self> {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return Err(Error::Config("empty key name".to_string()));
        }
        let (name, mode) = match trimmed.chars().last() {
            Some('m') if trimmed.len() > 1 => (&trimmed[..trimmed.len() - 1], Mode::Minor),
            Some('M') if trimmed.len() > 1 => (&trimmed[..trimmed.len() - 1], Mode::Major),
            _ => (trimmed, Mode::Major),
        };
        let tonic: PitchClass = name.parse().map_err(|_| {
            Error::Config(format!(
                "invalid key '{trimmed}'; expected something like 'Am', 'C' or 'F#m'"
            ))
        })?;
        Ok(Key { tonic, mode })
    }
}

/// The Krumhansl-Schmuckler key profiles, used for key detection.
///
/// The major profile correlates with the major scale and the minor profile
/// with the natural minor scale. These are the classic Temperley/Krumhansl
/// values, which remain a strong, cheap baseline.
pub struct KeyProfiles;

impl KeyProfiles {
    /// The major-scale correlation profile.
    pub const MAJOR: [f64; 12] = [
        6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
    ];

    /// The minor-scale correlation profile.
    pub const MINOR: [f64; 12] = [
        6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
    ];

    /// The rotated correlation score of a chroma vector against a profile.
    ///
    /// `chroma` must have 12 entries, one per pitch class, starting at C. The
    /// profile is indexed by scale degree, so the pitch class `i` is matched
    /// against the degree `(i - tonic) mod 12`.
    #[must_use]
    pub fn correlation(chroma: &[f64; 12], profile: &[f64; 12], tonic: PitchClass) -> f64 {
        let mut sum = 0.0;
        for (i, &value) in chroma.iter().enumerate() {
            let degree = (i as i32 - tonic.as_u8() as i32).rem_euclid(12);
            sum += value * profile[degree as usize];
        }
        sum
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn keys_parse_and_render() {
        assert_eq!("Am".parse::<Key>().expect("valid").name(), "Am");
        assert_eq!("C".parse::<Key>().expect("valid").name(), "C");
        assert_eq!("F#m".parse::<Key>().expect("valid").name(), "F#m");
        assert_eq!("Bb".parse::<Key>().expect("valid").name(), "Bb");
        assert!("H".parse::<Key>().is_err());
    }

    #[test]
    fn minor_keys_parse_case_sensitively() {
        let lower = "am".parse::<Key>().expect("valid");
        assert_eq!(lower.mode, Mode::Minor);
        let upper = "AM".parse::<Key>().expect("valid");
        assert_eq!(upper.mode, Mode::Major);
    }

    #[test]
    fn fifths_match_standard_key_signatures() {
        assert_eq!(Key::new(PitchClass::C, Mode::Major).fifths(), 0);
        assert_eq!(Key::new(PitchClass::G, Mode::Major).fifths(), 1);
        assert_eq!(Key::new(PitchClass::F, Mode::Major).fifths(), -1);
        assert_eq!(Key::new(PitchClass::A, Mode::Minor).fifths(), 0);
        assert_eq!(Key::new(PitchClass::E, Mode::Minor).fifths(), 1);
    }

    #[test]
    fn scales_contain_seven_notes() {
        let a_minor = Key::new(PitchClass::A, Mode::Minor);
        let scale = a_minor.scale();
        assert_eq!(scale.len(), 7);
        assert_eq!(scale[0], PitchClass::A);
        assert_eq!(scale[6], PitchClass::G);
    }
}
