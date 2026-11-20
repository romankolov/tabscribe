//! The cost function of the tablature optimizer.
//!
//! # What "good" tablature means
//!
//! A pitch-correct fingering is not necessarily a *playable* one. The cost
//! function encodes the things a guitarist actually cares about:
//!
//! * the hand should not travel far between notes,
//! * a chord should fit under the fingers (a small fret span),
//! * one note per string,
//! * open strings are cheap to play,
//! * familiar chord shapes are easier to find on the neck than invented ones.
//!
//! Every term is weighted, and [`Style`] bundles a set of weights that suit a
//! particular playing style. The weights are deliberately exposed so that the
//! evaluation harness can tune them against ground truth.

use serde::{Deserialize, Serialize};

/// A `(string, fret)` pair, the fundamental unit of a fingering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Position {
    /// The string index (0 = string 1, the top line of a tab).
    pub string: u8,
    /// The fret number (0 = open).
    pub fret: u8,
}

impl Position {
    /// Construct a position.
    #[must_use]
    pub const fn new(string: u8, fret: u8) -> Self {
        Position { string, fret }
    }
}

/// The weights of the tablature cost function.
///
/// All weights are non-negative except the bonuses, which are negative. The
/// absolute scale is irrelevant; only the ratios matter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostWeights {
    /// Cost per semitone of fret movement between consecutive notes.
    pub fret_movement: f64,
    /// Cost per string crossed between consecutive notes.
    pub string_movement: f64,
    /// Extra cost when more than one string is crossed (string skipping).
    pub string_skip: f64,
    /// Cost per semitone of fret span inside a chord.
    pub span: f64,
    /// Bonus (negative) for playing an open string.
    pub open_string: f64,
    /// Cost per fret of the average position of the chord, favouring lower
    /// positions on the neck.
    pub low_position: f64,
    /// Bonus (negative) for a fingering that matches a well-known chord shape.
    pub shape_bonus: f64,
    /// Bonus (negative) for staying on the same string when the note is
    /// played legato (hammer-on, pull-off or slide).
    pub same_string_legato: f64,
    /// Cost when a note in one chord has no counterpart in the previous chord,
    /// i.e. the voice moved rather than the finger.
    pub unmatched_voice: f64,
}

impl Default for CostWeights {
    fn default() -> Self {
        CostWeights {
            fret_movement: 1.0,
            string_movement: 1.5,
            string_skip: 1.0,
            span: 0.35,
            open_string: -0.6,
            low_position: 0.02,
            shape_bonus: -2.0,
            same_string_legato: -1.5,
            unmatched_voice: 2.0,
        }
    }
}

/// A playing style, which determines the cost weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Style {
    /// Balance movement, span and shape preferences.
    Balanced,
    /// Keep the hand as low on the neck as possible.
    LowPosition,
    /// Keep the hand as high on the neck as possible.
    HighPosition,
    /// Prefer open strings and simple shapes.
    OpenChords,
    /// Favour smooth, single-string lines for lead playing.
    Lead,
}

impl Style {
    /// All styles, in a stable order suitable for `--help` output.
    pub const ALL: &'static [Style] = &[
        Style::Balanced,
        Style::LowPosition,
        Style::HighPosition,
        Style::OpenChords,
        Style::Lead,
    ];

    /// The canonical lowercase name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Style::Balanced => "balanced",
            Style::LowPosition => "low-position",
            Style::HighPosition => "high-position",
            Style::OpenChords => "open-chords",
            Style::Lead => "lead",
        }
    }

    /// The cost weights that implement this style.
    #[must_use]
    pub fn weights(self) -> CostWeights {
        let base = CostWeights::default();
        match self {
            Style::Balanced => base,
            Style::LowPosition => CostWeights {
                low_position: 0.12,
                open_string: -1.0,
                ..base
            },
            Style::HighPosition => CostWeights {
                low_position: -0.06,
                fret_movement: 0.8,
                ..base
            },
            Style::OpenChords => CostWeights {
                open_string: -2.5,
                low_position: 0.06,
                span: 0.6,
                ..base
            },
            Style::Lead => CostWeights {
                fret_movement: 1.6,
                string_movement: 3.0,
                string_skip: 2.0,
                same_string_legato: -3.0,
                low_position: 0.0,
                ..base
            },
        }
    }
}

impl std::fmt::Display for Style {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn every_style_produces_usable_weights() {
        for style in Style::ALL {
            let w = style.weights();
            assert!(w.fret_movement >= 0.0);
            assert!(w.string_movement >= 0.0);
            assert!(w.span >= 0.0);
            assert!(w.open_string <= 0.0);
        }
    }

    #[test]
    fn lead_style_prefers_string_changes_less_than_balanced() {
        let lead = Style::Lead.weights();
        let balanced = Style::Balanced.weights();
        assert!(lead.string_movement > balanced.string_movement);
        assert!(lead.same_string_legato < balanced.same_string_legato);
    }
}
