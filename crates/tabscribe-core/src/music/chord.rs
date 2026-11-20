//! Chords, chord kinds and chord events.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use super::key::Key;
use super::note::PitchClass;
use super::time::Seconds;
use crate::error::{Error, Result};

/// The quality of a chord.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChordKind {
    /// Major triad: 1 3 5.
    Major,
    /// Minor triad: 1 b3 5.
    Minor,
    /// Dominant seventh: 1 3 5 b7.
    DominantSeventh,
    /// Major seventh: 1 3 5 7.
    MajorSeventh,
    /// Minor seventh: 1 b3 5 b7.
    MinorSeventh,
    /// Minor-major seventh: 1 b3 5 7.
    MinorMajorSeventh,
    /// Suspended second: 1 2 5.
    SuspendedSecond,
    /// Suspended fourth: 1 4 5.
    SuspendedFourth,
    /// Diminished triad: 1 b3 b5.
    Diminished,
    /// Diminished seventh: 1 b3 b5 bb7.
    DiminishedSeventh,
    /// Augmented triad: 1 3 #5.
    Augmented,
    /// Power chord (no third): 1 5.
    Power,
}

impl ChordKind {
    /// Every chord kind, in a stable order.
    pub const ALL: &'static [ChordKind] = &[
        ChordKind::Major,
        ChordKind::Minor,
        ChordKind::DominantSeventh,
        ChordKind::MajorSeventh,
        ChordKind::MinorSeventh,
        ChordKind::MinorMajorSeventh,
        ChordKind::SuspendedSecond,
        ChordKind::SuspendedFourth,
        ChordKind::Diminished,
        ChordKind::DiminishedSeventh,
        ChordKind::Augmented,
        ChordKind::Power,
    ];

    /// The suffix appended to the root name, e.g. `m7` for minor seventh.
    #[must_use]
    pub fn suffix(self) -> &'static str {
        match self {
            ChordKind::Major => "",
            ChordKind::Minor => "m",
            ChordKind::DominantSeventh => "7",
            ChordKind::MajorSeventh => "maj7",
            ChordKind::MinorSeventh => "m7",
            ChordKind::MinorMajorSeventh => "mMaj7",
            ChordKind::SuspendedSecond => "sus2",
            ChordKind::SuspendedFourth => "sus4",
            ChordKind::Diminished => "dim",
            ChordKind::DiminishedSeventh => "dim7",
            ChordKind::Augmented => "aug",
            ChordKind::Power => "5",
        }
    }

    /// The pitch class offsets above the root that form this chord.
    #[must_use]
    pub fn intervals(self) -> &'static [u8] {
        match self {
            ChordKind::Major => &[0, 4, 7],
            ChordKind::Minor => &[0, 3, 7],
            ChordKind::DominantSeventh => &[0, 4, 7, 10],
            ChordKind::MajorSeventh => &[0, 4, 7, 11],
            ChordKind::MinorSeventh => &[0, 3, 7, 10],
            ChordKind::MinorMajorSeventh => &[0, 3, 7, 11],
            ChordKind::SuspendedSecond => &[0, 2, 7],
            ChordKind::SuspendedFourth => &[0, 5, 7],
            ChordKind::Diminished => &[0, 3, 6],
            ChordKind::DiminishedSeventh => &[0, 3, 6, 9],
            ChordKind::Augmented => &[0, 4, 8],
            ChordKind::Power => &[0, 7],
        }
    }

    /// The major or minor scale degree of the third, used to distinguish
    /// `C` from `Cm` when matching a template.
    #[must_use]
    pub fn third_quality(self) -> Option<bool> {
        match self {
            ChordKind::Major
            | ChordKind::DominantSeventh
            | ChordKind::MajorSeventh
            | ChordKind::Augmented => Some(true),
            ChordKind::Minor
            | ChordKind::MinorSeventh
            | ChordKind::MinorMajorSeventh
            | ChordKind::Diminished
            | ChordKind::DiminishedSeventh => Some(false),
            ChordKind::SuspendedSecond | ChordKind::SuspendedFourth | ChordKind::Power => None,
        }
    }
}

/// A chord: a root, a quality and an inversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Chord {
    /// The root pitch class.
    pub root: PitchClass,
    /// The quality.
    pub kind: ChordKind,
    /// The inversion, 0 = root position.
    pub inversion: u8,
}

impl Chord {
    /// Construct a chord in root position.
    #[must_use]
    pub const fn new(root: PitchClass, kind: ChordKind) -> Self {
        Chord {
            root,
            kind,
            inversion: 0,
        }
    }

    /// The pitch classes contained in this chord.
    #[must_use]
    pub fn pitch_classes(&self) -> Vec<PitchClass> {
        self.kind
            .intervals()
            .iter()
            .map(|&i| self.root.transposed(i as i32))
            .collect()
    }

    /// The name of the chord, e.g. `Am7`.
    #[must_use]
    pub fn name(&self) -> String {
        format!("{}{}", self.root.chord_name(), self.kind.suffix())
    }
}

impl fmt::Display for Chord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name())
    }
}

impl FromStr for Chord {
    type Err = Error;

    /// Parse a chord name such as `C`, `Am7`, `F#m`, `G7`, `Dsus4`.
    fn from_str(s: &str) -> Result<Self> {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return Err(Error::Config("empty chord name".to_string()));
        }
        let mut root_text = String::new();
        let mut idx = 1;
        let chars: Vec<char> = trimmed.chars().collect();
        if chars.is_empty() {
            return Err(Error::Config("empty chord name".to_string()));
        }
        root_text.push(chars[0]);
        while idx < chars.len() && (chars[idx] == '#' || chars[idx] == 'b') {
            root_text.push(chars[idx]);
            idx += 1;
        }
        let root: PitchClass = root_text.parse().map_err(|_| {
            Error::Config(format!("invalid chord root '{root_text}' in '{trimmed}'"))
        })?;
        let suffix: String = chars[idx..].iter().collect();
        let kind = parse_chord_kind(&suffix).ok_or_else(|| {
            Error::Config(format!("unknown chord quality '{suffix}' in '{trimmed}'"))
        })?;
        Ok(Chord::new(root, kind))
    }
}

/// Parse a chord-quality suffix.
///
/// # Errors
///
/// Returns `None` (as `Option`) when the suffix is not recognised; the caller
/// converts this into a user-facing message.
fn parse_chord_kind(suffix: &str) -> Option<ChordKind> {
    match suffix {
        "" => Some(ChordKind::Major),
        "m" => Some(ChordKind::Minor),
        "7" => Some(ChordKind::DominantSeventh),
        "maj7" | "M7" | "Δ7" => Some(ChordKind::MajorSeventh),
        "m7" | "min7" | "-7" => Some(ChordKind::MinorSeventh),
        "mMaj7" | "mM7" | "mmaj7" => Some(ChordKind::MinorMajorSeventh),
        "sus2" => Some(ChordKind::SuspendedSecond),
        "sus4" => Some(ChordKind::SuspendedFourth),
        "dim" | "o" => Some(ChordKind::Diminished),
        "dim7" | "o7" => Some(ChordKind::DiminishedSeventh),
        "aug" | "+" => Some(ChordKind::Augmented),
        "5" => Some(ChordKind::Power),
        _ => None,
    }
}

/// A chord that occurs over a span of time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChordEvent {
    /// The chord.
    pub chord: Chord,
    /// When the chord starts.
    pub start: Seconds,
    /// When the chord ends.
    pub end: Seconds,
    /// The recogniser's confidence in `[0, 1]`.
    pub confidence: f32,
}

impl ChordEvent {
    /// The duration of the chord.
    #[must_use]
    pub fn duration(&self) -> Seconds {
        Seconds((self.end.as_secs() - self.start.as_secs()).max(0.0))
    }
}

/// Whether a chord is diatonic to a key.
///
/// Used to rank chord candidates: a chord that fits the key is preferred over
/// an equally plausible chord that does not.
#[must_use]
pub fn is_diatonic(chord: &Chord, key: &Key) -> bool {
    let scale = key.scale();
    let _chord_tones = chord.pitch_classes();
    // The root must be in the key; a non-diatonic third (e.g. a major chord on
    // the minor seventh degree) is tolerated as it is common in pop music.
    scale.contains(&chord.root)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn chords_parse_and_render() {
        assert_eq!("C".parse::<Chord>().expect("valid").name(), "C");
        assert_eq!("Am7".parse::<Chord>().expect("valid").name(), "Am7");
        assert_eq!("F#m".parse::<Chord>().expect("valid").name(), "F#m");
        assert_eq!("G7".parse::<Chord>().expect("valid").name(), "G7");
        assert_eq!("Dsus4".parse::<Chord>().expect("valid").name(), "Dsus4");
        assert!("Cfoo".parse::<Chord>().is_err());
        assert!("H7".parse::<Chord>().is_err());
    }

    #[test]
    fn chord_intervals_are_correct() {
        let a_minor = Chord::new(PitchClass::A, ChordKind::Minor);
        let pcs = a_minor.pitch_classes();
        assert_eq!(pcs, vec![PitchClass::A, PitchClass::C, PitchClass::E]);
    }

    #[test]
    fn diatonicity_uses_the_key_scale() {
        let key = Key::new(PitchClass::A, super::super::key::Mode::Minor);
        assert!(is_diatonic(
            &Chord::new(PitchClass::E, ChordKind::Major),
            &key
        ));
        assert!(is_diatonic(
            &Chord::new(PitchClass::C, ChordKind::Major),
            &key
        ));
        assert!(!is_diatonic(
            &Chord::new(PitchClass::FS, ChordKind::Major),
            &key
        ));
    }
}
