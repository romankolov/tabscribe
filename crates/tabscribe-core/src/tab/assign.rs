//! The tablature assignment optimizer.
//!
//! # Formulation
//!
//! Assigning fingerings to a transcribed melody is a sequence labelling
//! problem. The notes are grouped into *chords* (sets of notes that sound
//! together); for every group the optimizer chooses a *state*, which is an
//! assignment of a `(string, fret)` to each note in the group. The state space
//! is constrained by the instrument:
//!
//! * a pitch may only be played where the instrument can reach it,
//! * no two notes in a group may share a string,
//! * the fret span inside a group may not exceed `max_span`.
//!
//! Between two consecutive groups the cost is a sum of terms describing how
//! hard the move is, and the whole sequence is minimised with a dynamic
//! program (a Viterbi pass over the state graph).
//!
//! Because the DP is a shortest-path computation over a layered DAG, the
//! forward and backward passes can be combined to recover the *k* best
//! alternatives for any single group, which is what the JSON exporter reports
//! for ambiguous passages.

use crate::error::{Error, Result};
use crate::music::{Chord, Instrument, MidiNote, Seconds};
use crate::tab::cost::{CostWeights, Position};
use crate::tab::shapes::ShapeDictionary;

/// A set of notes that sound simultaneously.
#[derive(Debug, Clone)]
pub struct Group {
    /// The pitches sounding together.
    pub pitches: Vec<MidiNote>,
    /// The onset of the group.
    pub start: Seconds,
    /// Whether the notes are played legato from the previous group.
    pub legato: bool,
    /// The chord this group realises, if a chord was detected.
    pub chord: Option<Chord>,
}

impl Group {
    /// Construct a group of simultaneously sounding pitches.
    #[must_use]
    pub fn new(pitches: Vec<MidiNote>, start: Seconds) -> Self {
        Group {
            pitches,
            start,
            legato: false,
            chord: None,
        }
    }

    /// Mark the group as legato from the previous one.
    #[must_use]
    pub fn legato(mut self, legato: bool) -> Self {
        self.legato = legato;
        self
    }

    /// Attach a chord label.
    #[must_use]
    pub fn with_chord(mut self, chord: Option<Chord>) -> Self {
        self.chord = chord;
        self
    }
}

/// The result of an assignment: one fingering per note of every group.
#[derive(Debug, Clone)]
pub struct Assignment {
    /// The fingerings, one per group, each holding one position per note.
    pub fingerings: Vec<Vec<Position>>,
    /// The total cost of the chosen path.
    pub cost: f64,
    /// The alternative fingerings for each group, best first.
    ///
    /// Each entry is a complete alternative state of the group (one position
    /// per note), not a single position.
    pub alternatives: Vec<Vec<Vec<Position>>>,
}

/// Assigns fingerings to groups of notes.
///
/// The optimizer is cheap to clone and holds only a reference to the
/// instrument, so a single instance can be reused across a whole song.
#[derive(Debug, Clone)]
pub struct TabOptimizer<'a> {
    instrument: &'a Instrument,
    weights: CostWeights,
    max_span: u8,
    max_states_per_group: usize,
    shapes: Option<&'a ShapeDictionary>,
}

impl<'a> TabOptimizer<'a> {
    /// Create an optimizer for an instrument.
    #[must_use]
    pub fn new(instrument: &'a Instrument, weights: CostWeights) -> Self {
        TabOptimizer {
            instrument,
            weights,
            max_span: default_max_span(instrument),
            max_states_per_group: 256,
            shapes: None,
        }
    }

    /// Use a chord-shape dictionary to reward familiar fingerings.
    #[must_use]
    pub fn with_shapes(mut self, shapes: &'a ShapeDictionary) -> Self {
        self.shapes = Some(shapes);
        self
    }

    /// Limit the maximum fret span inside a chord.
    #[must_use]
    pub fn with_max_span(mut self, max_span: u8) -> Self {
        self.max_span = max_span;
        self
    }

    /// Limit how many candidate states are kept per group.
    ///
    /// This bounds the memory and time of the dynamic program. Groups with
    /// more candidates than this are pruned by the local cost, keeping the
    /// most compact and lowest fingerings.
    #[must_use]
    pub fn with_max_states(mut self, max_states: usize) -> Self {
        self.max_states_per_group = max_states.max(1);
        self
    }

    /// The weights in use.
    #[must_use]
    pub fn weights(&self) -> &CostWeights {
        &self.weights
    }

    /// Assign fingerings to a sequence of groups.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when a note cannot be played on the
    /// instrument at all, and [`Error::NoFeasibleTab`] when the constraints
    /// admit no assignment (for example a chord with more notes than strings).
    pub fn assign(&self, groups: &[Group]) -> Result<Assignment> {
        let states = self.enumerate_all(groups)?;
        if states.is_empty() {
            return Err(Error::NoFeasibleTab(
                "the piece contains no notes to assign".to_string(),
            ));
        }

        let n_groups = states.len();
        let mut forward: Vec<Vec<f64>> = Vec::with_capacity(n_groups);
        // `backward[g][s]` is the cheapest cost of *leaving* the state `s` of
        // group `g`: the transition to the next group plus everything after
        // it. The local cost of group `g` itself is not included, because it
        // is already counted by the forward pass.
        let mut backward: Vec<Vec<f64>> = vec![Vec::new(); n_groups];

        // Forward pass: the cheapest cost of reaching every state.
        for (g, group_states) in states.iter().enumerate() {
            let chord = groups[g].chord.as_ref();
            let mut costs = Vec::with_capacity(group_states.len());
            for state in group_states {
                let local = self.local_cost(state, chord);
                if g == 0 {
                    costs.push(local);
                    continue;
                }
                let prev_states = &states[g - 1];
                let mut best = f64::INFINITY;
                for (p, prev) in prev_states.iter().enumerate() {
                    let prev_cost = forward[g - 1][p];
                    if !prev_cost.is_finite() {
                        continue;
                    }
                    let transition = self.transition_cost(prev, state, groups[g].legato);
                    best = best.min(prev_cost + transition);
                }
                costs.push(best + local);
            }
            forward.push(costs);
        }

        // Backward pass: the cheapest cost of leaving every state.
        for g in (0..n_groups).rev() {
            let _chord = groups[g].chord.as_ref();
            let group_states = &states[g];
            let mut costs = vec![f64::INFINITY; group_states.len()];
            for (s, state) in group_states.iter().enumerate() {
                if g + 1 == n_groups {
                    // The last group has nothing to leave to.
                    costs[s] = 0.0;
                    continue;
                }
                let next_states = &states[g + 1];
                let next_chord = groups[g + 1].chord.as_ref();
                let mut best = f64::INFINITY;
                for (n, next) in next_states.iter().enumerate() {
                    let next_cost = backward[g + 1][n];
                    if !next_cost.is_finite() {
                        continue;
                    }
                    let transition = self.transition_cost(state, next, groups[g + 1].legato);
                    let local_next = self.local_cost(next, next_chord);
                    best = best.min(transition + local_next + next_cost);
                }
                costs[s] = best;
            }
            backward[g] = costs;
        }

        // Recover the best path and, for every group, the best alternatives.
        let mut fingerings = Vec::with_capacity(n_groups);
        let mut alternatives = Vec::with_capacity(n_groups);
        let mut total_cost = f64::INFINITY;

        for (g, group_states) in states.iter().enumerate() {
            let mut best_s = 0usize;
            let mut best_c = f64::INFINITY;
            for (s, _) in group_states.iter().enumerate() {
                let c = forward[g][s] + backward[g][s];
                if c < best_c {
                    best_c = c;
                    best_s = s;
                }
            }
            if g == 0 {
                total_cost = best_c;
            }
            fingerings.push(group_states[best_s].clone());

            // Alternatives: the best other states for this group, measured by
            // the cost of the complete path that goes through them.
            let mut alts: Vec<(f64, Vec<Position>)> = group_states
                .iter()
                .enumerate()
                .filter(|(s, _)| *s != best_s)
                .map(|(s, state)| (forward[g][s] + backward[g][s], state.clone()))
                .collect();
            alts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            alts.truncate(3);
            alternatives.push(alts.into_iter().map(|(_, state)| state).collect());
        }

        Ok(Assignment {
            fingerings,
            cost: total_cost,
            alternatives,
        })
    }

    /// The notes of a group that can be played together.
    ///
    /// A transcription is not guaranteed to be playable: two detectors can
    /// report the same pitch twice, and a chord can be wider than the hand
    /// can reach. Rather than refusing to write the piece at all, the
    /// offending notes are found and dropped.
    ///
    /// The notes the instrument cannot reach are dropped first. If the group
    /// is still impossible, the note with the fewest ways of playing it is
    /// dropped, again and again, until the rest fits.
    ///
    /// The returned indices point into `pitches`. An empty result means that
    /// not one of the notes can be played.
    #[must_use]
    pub fn playable_subset(&self, pitches: &[MidiNote]) -> Vec<usize> {
        let mut keep: Vec<usize> = (0..pitches.len())
            .filter(|&index| !self.instrument.positions_for(pitches[index]).is_empty())
            .collect();
        if keep.is_empty() {
            return keep;
        }

        while keep.len() > 1 {
            let subset: Vec<MidiNote> = keep.iter().map(|&index| pitches[index]).collect();
            if self.enumerate_group(&subset).is_ok() {
                return keep;
            }

            // Drop the note that has the fewest ways of being played; the
            // highest pitch breaks the tie, because the low notes of a chord
            // are the ones a guitarist is least likely to drop.
            let victim = keep
                .iter()
                .copied()
                .min_by_key(|&index| {
                    (
                        self.instrument.positions_for(pitches[index]).len(),
                        u8::MAX - pitches[index].as_u8(),
                    )
                })
                .unwrap_or(keep[0]);
            keep.retain(|&index| index != victim);
        }

        keep
    }

    /// The best single fingering for a set of pitches.
    ///
    /// This is used to build the chord-shape dictionary and to pick a
    /// reasonable fallback when a group has to be resolved in isolation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when a pitch is unplayable.
    pub fn best_single(&self, pitches: &[MidiNote]) -> Result<Vec<Position>> {
        let states = self.enumerate_group(pitches)?;
        states
            .into_iter()
            .min_by(|a, b| {
                let la = self.local_cost(a, None);
                let lb = self.local_cost(b, None);
                la.partial_cmp(&lb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| {
                Error::NoFeasibleTab(format!(
                    "no playable fingering for {:?}",
                    pitches.iter().map(|p| p.name()).collect::<Vec<_>>()
                ))
            })
    }

    /// Enumerate the candidate states of every group.
    fn enumerate_all(&self, groups: &[Group]) -> Result<Vec<Vec<Vec<Position>>>> {
        let mut all = Vec::with_capacity(groups.len());
        for group in groups {
            all.push(self.enumerate_group(&group.pitches)?);
        }
        Ok(all)
    }

    /// Enumerate the candidate states of a single group.
    fn enumerate_group(&self, pitches: &[MidiNote]) -> Result<Vec<Vec<Position>>> {
        if pitches.is_empty() {
            return Ok(Vec::new());
        }
        if pitches.len() > self.instrument.string_count() {
            return Err(Error::NoFeasibleTab(format!(
                "{} notes sound together but the instrument only has {} strings",
                pitches.len(),
                self.instrument.string_count()
            )));
        }

        // The candidate positions of each note.
        let options: Vec<Vec<Position>> = pitches
            .iter()
            .map(|&pitch| {
                let positions = self.instrument.positions_for(pitch);
                if positions.is_empty() {
                    return Err(Error::OutOfRange {
                        pitch: pitch.name(),
                        range: format!(
                            "{} .. {} ({} strings, capo {})",
                            self.instrument.range().0.name(),
                            self.instrument.range().1.name(),
                            self.instrument.string_count(),
                            self.instrument.capo()
                        ),
                    });
                }
                Ok(positions
                    .into_iter()
                    .map(|(string, fret)| Position::new(string.as_u8(), fret.as_u8()))
                    .collect())
            })
            .collect::<Result<Vec<_>>>()?;

        let mut states: Vec<Vec<Position>> = Vec::new();
        let mut used_strings: Vec<u8> = Vec::with_capacity(pitches.len());
        let mut current: Vec<Position> = Vec::with_capacity(pitches.len());
        self.search(
            0,
            &options,
            &mut current,
            &mut used_strings,
            None,
            None,
            &mut states,
        );

        if states.is_empty() {
            return Err(Error::NoFeasibleTab(format!(
                "no fingering satisfies the constraints for {:?} (max span {} frets)",
                pitches.iter().map(|p| p.name()).collect::<Vec<_>>(),
                self.max_span
            )));
        }

        // Prune to the most promising states.
        if states.len() > self.max_states_per_group {
            states.sort_by(|a, b| {
                self.local_cost(a, None)
                    .partial_cmp(&self.local_cost(b, None))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            states.truncate(self.max_states_per_group);
        }
        Ok(states)
    }

    /// Recursive enumeration of the valid fingerings of a group.
    #[allow(clippy::too_many_arguments)]
    fn search(
        &self,
        index: usize,
        options: &[Vec<Position>],
        current: &mut Vec<Position>,
        used_strings: &mut Vec<u8>,
        min_fret: Option<u8>,
        max_fret: Option<u8>,
        out: &mut Vec<Vec<Position>>,
    ) {
        if index == options.len() {
            out.push(current.clone());
            return;
        }
        for candidate in &options[index] {
            if used_strings.contains(&candidate.string) {
                continue;
            }
            let new_min = match min_fret {
                Some(m) => m.min(candidate.fret),
                None => candidate.fret,
            };
            let new_max = match max_fret {
                Some(m) => m.max(candidate.fret),
                None => candidate.fret,
            };
            if new_max.saturating_sub(new_min) > self.max_span {
                continue;
            }
            used_strings.push(candidate.string);
            current.push(*candidate);
            self.search(
                index + 1,
                options,
                current,
                used_strings,
                Some(new_min),
                Some(new_max),
                out,
            );
            current.pop();
            used_strings.pop();
        }
    }

    /// The cost of a single state, independent of its neighbours.
    fn local_cost(&self, state: &[Position], chord: Option<&Chord>) -> f64 {
        if state.is_empty() {
            return 0.0;
        }
        let fret_sum: usize = state.iter().map(|p| p.fret as usize).sum();
        let mean_fret = fret_sum as f64 / state.len() as f64;
        let open_count = state.iter().filter(|p| p.fret == 0).count() as f64;
        let span = state.iter().map(|p| p.fret).max().unwrap_or(0) as i32
            - state.iter().map(|p| p.fret).min().unwrap_or(0) as i32;

        let mut cost = self.weights.low_position * mean_fret
            + self.weights.open_string * open_count
            + self.weights.span * span as f64;

        if let (Some(shapes), Some(chord)) = (self.shapes, chord) {
            if shapes.matches(chord, state) {
                cost += self.weights.shape_bonus;
            }
        }
        cost
    }

    /// The cost of moving from one state to the next.
    fn transition_cost(&self, prev: &[Position], next: &[Position], legato: bool) -> f64 {
        match_cost(prev, next, &self.weights, legato)
    }
}

/// The default maximum fret span inside a chord.
#[must_use]
pub fn default_max_span(instrument: &Instrument) -> u8 {
    match instrument.kind() {
        crate::music::InstrumentKind::Guitar => 5,
        crate::music::InstrumentKind::Bass => 4,
        crate::music::InstrumentKind::Ukulele => 4,
        crate::music::InstrumentKind::Custom => 5,
    }
}

/// The cost of matching the notes of two consecutive groups.
///
/// Notes are paired optimally (a minimum-cost bipartite matching, solved with
/// a subset dynamic program), and notes without a counterpart are charged an
/// `unmatched_voice` penalty because the picking hand has to move anyway.
fn match_cost(prev: &[Position], next: &[Position], weights: &CostWeights, legato: bool) -> f64 {
    if prev.is_empty() || next.is_empty() {
        return 0.0;
    }
    let n = next.len();
    let size = 1usize << n;
    let mut dp = vec![f64::INFINITY; size];
    dp[0] = 0.0;

    for prev_pos in prev {
        let mut ndp = vec![f64::INFINITY; size];
        for (mask, &value) in dp.iter().enumerate() {
            if !value.is_finite() {
                continue;
            }
            // Leave this note unmatched: the voice disappeared.
            ndp[mask] = ndp[mask].min(value + weights.unmatched_voice);
            for (j, next_pos) in next.iter().enumerate() {
                if mask & (1 << j) != 0 {
                    continue;
                }
                let cost = pair_cost(*prev_pos, *next_pos, weights, legato);
                let target = mask | (1 << j);
                ndp[target] = ndp[target].min(value + cost);
            }
        }
        dp = ndp;
    }

    // Any note of the new group that did not receive a partner is a new
    // voice entering the texture.
    let mut best = f64::INFINITY;
    for (mask, &value) in dp.iter().enumerate() {
        if !value.is_finite() {
            continue;
        }
        let unmatched_next = n - mask.count_ones() as usize;
        best = best.min(value + weights.unmatched_voice * unmatched_next as f64);
    }
    best
}

/// The cost of moving the hand from one position to another.
fn pair_cost(a: Position, b: Position, weights: &CostWeights, legato: bool) -> f64 {
    let fret_delta = (a.fret as f64 - b.fret as f64).abs();
    let string_delta = (a.string as f64 - b.string as f64).abs();
    let mut cost = weights.fret_movement * fret_delta + weights.string_movement * string_delta;
    if string_delta > 1.0 {
        cost += weights.string_skip * (string_delta - 1.0);
    }
    if legato && a.string == b.string && a.fret != b.fret {
        // A hammer-on, pull-off or slide keeps the finger on the same string.
        cost += weights.same_string_legato;
    }
    cost
}
