//! The chord-shape dictionary.
//!
//! A guitarist does not invent fingerings: they reach for the shapes they
//! know. The dictionary stores, for every chord in a fixed vocabulary, the
//! fingerings that the optimizer itself considers best for the instrument at
//! hand — so the preferences are always consistent with the rest of the cost
//! function, and the dictionary is rebuilt automatically for any tuning.
//!
//! Building the dictionary is cheap (a few hundred tiny optimization
//! problems) and happens once per transcription.

use std::collections::HashMap;

use crate::music::{Chord, ChordKind, Instrument, MidiNote, PitchClass};
use crate::tab::assign::TabOptimizer;
use crate::tab::cost::{CostWeights, Position};

/// A set of well-known fingerings, keyed by chord name.
#[derive(Debug, Clone, Default)]
pub struct ShapeDictionary {
    shapes: HashMap<String, Vec<Vec<Position>>>,
}

impl ShapeDictionary {
    /// Build the dictionary for an instrument.
    ///
    /// The fingerings are produced by the optimizer itself, so the dictionary
    /// always agrees with the style the user asked for.
    #[must_use]
    pub fn build(instrument: &Instrument, weights: &CostWeights) -> Self {
        let optimizer = TabOptimizer::new(instrument, weights.clone());
        let mut shapes: HashMap<String, Vec<Vec<Position>>> = HashMap::new();

        for &kind in ChordKind::ALL {
            for root in 0u8..12 {
                let Ok(pitch_class) = PitchClass::new(root) else {
                    continue;
                };
                let chord = Chord::new(pitch_class, kind);
                let mut candidates: Vec<Vec<Position>> = Vec::new();

                for shift in [-24i32, -12, 0, 12] {
                    let Some(pitches) = voicing(instrument, &chord, shift) else {
                        continue;
                    };
                    let Ok(positions) = optimizer.best_single(&pitches) else {
                        continue;
                    };
                    if !is_root_position(instrument, &pitches, &positions) {
                        continue;
                    }
                    if !candidates.contains(&positions) {
                        candidates.push(positions);
                    }
                }

                candidates.truncate(3);
                if !candidates.is_empty() {
                    shapes.insert(chord.name(), candidates);
                }
            }
        }

        ShapeDictionary { shapes }
    }

    /// The known fingerings of a chord, best first.
    #[must_use]
    pub fn get(&self, chord: &Chord) -> Option<&Vec<Vec<Position>>> {
        self.shapes.get(&chord.name())
    }

    /// Whether a fingering matches one of the known shapes of a chord.
    #[must_use]
    pub fn matches(&self, chord: &Chord, state: &[Position]) -> bool {
        self.get(chord).is_some_and(|shapes| {
            let key = shape_key(state);
            shapes.iter().any(|s| shape_key(s) == key)
        })
    }

    /// The number of chords in the dictionary.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shapes.len()
    }

    /// Whether the dictionary is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shapes.is_empty()
    }
}

/// A canonical, order-independent key for a fingering.
fn shape_key(state: &[Position]) -> Vec<(u8, u8)> {
    let mut key: Vec<(u8, u8)> = state.iter().map(|p| (p.string, p.fret)).collect();
    key.sort_unstable();
    key
}

/// Build a playable voicing of a chord near the middle of the instrument's
/// range.
fn voicing(instrument: &Instrument, chord: &Chord, shift: i32) -> Option<Vec<MidiNote>> {
    let (lo, hi) = instrument.range();
    let mid = (lo.as_u8() as i32 + hi.as_u8() as i32) / 2 + shift;

    let mut pitches = Vec::with_capacity(chord.kind.intervals().len());
    for &interval in chord.kind.intervals() {
        let pitch_class = (chord.root.as_u8() as i32 + interval as i32).rem_euclid(12) as u8;
        let mut best: Option<i32> = None;
        let mut best_distance = i32::MAX;
        for midi in 0..=127i32 {
            if (midi % 12) as u8 != pitch_class {
                continue;
            }
            let distance = (midi - mid).abs();
            if distance < best_distance {
                best_distance = distance;
                best = Some(midi);
            }
        }
        pitches.push(MidiNote::clamped(best? as u16));
    }
    Some(pitches)
}

/// Whether the lowest sounding note of a fingering is the chord root.
fn is_root_position(
    instrument: &Instrument,
    pitches: &[MidiNote],
    positions: &[Position],
) -> bool {
    let mut lowest: Option<(u8, u8)> = None;
    for (pitch, position) in pitches.iter().zip(positions.iter()) {
        let sounding = instrument
            .sounding_note(
                crate::music::StringIdx::new(position.string).unwrap_or(crate::music::StringIdx::LOWEST),
                crate::music::Fret::new(position.fret).unwrap_or(crate::music::Fret::OPEN),
            )
            .map(|n| n.as_u8())
            .unwrap_or(pitch.as_u8());
        let candidate = (sounding, pitch.pitch_class().as_u8());
        match lowest {
            Some((current, _)) if current <= sounding => {}
            _ => lowest = Some(candidate),
        }
    }
    let Some((_, bass_class)) = lowest else {
        return false;
    };
    bass_class == chord_root_class(positions, pitches)
}

/// The pitch class of the chord root, recovered from the voicing.
fn chord_root_class(_positions: &[Position], pitches: &[MidiNote]) -> u8 {
    // The root is the lowest note of the voicing when the voicing was built
    // root-first; `voicing` always lists the root first because the root
    // interval of every chord kind is zero.
    pitches.first().map_or(0, |p| p.pitch_class().as_u8())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::music::InstrumentKind;

    #[test]
    fn dictionary_is_built_for_standard_guitar() {
        let instrument = Instrument::for_kind(InstrumentKind::Guitar).expect("valid");
        let dictionary = ShapeDictionary::build(&instrument, &CostWeights::default());
        assert!(dictionary.len() >= 30, "expected many chords, got {}", dictionary.len());
        let c_major = Chord::new(PitchClass::C, ChordKind::Major);
        assert!(dictionary.get(&c_major).is_some());
    }

    #[test]
    fn matching_is_order_independent() {
        let instrument = Instrument::for_kind(InstrumentKind::Guitar).expect("valid");
        let dictionary = ShapeDictionary::build(&instrument, &CostWeights::default());
        let c_major = Chord::new(PitchClass::C, ChordKind::Major);
        let shape = dictionary.get(&c_major).expect("known chord").first().cloned().expect("shape");
        let mut shuffled = shape.clone();
        shuffled.reverse();
        assert!(dictionary.matches(&c_major, &shuffled));
        assert!(!dictionary.matches(&c_major, &[]));
    }
}
