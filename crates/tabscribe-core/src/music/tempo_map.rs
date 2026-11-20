//! The tempo map: the bridge between musical time and wall-clock time.
//!
//! A piece may change tempo. Everything that needs to convert between beats
//! and seconds goes through a [`TempoMap`], so that a tempo change is handled
//! in exactly one place.
//!
//! The map is defined over MIDI ticks (the natural unit of a MIDI file) and
//! exposes conversions to and from both ticks and quarter-note beats.

use serde::{Deserialize, Serialize};

use super::time::{MidiTicks, Seconds, Tempo};

/// A tempo change at a given tick.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TempoChange {
    /// The tick at which the new tempo takes effect.
    pub tick: u32,
    /// The new tempo.
    pub tempo: Tempo,
}

/// A piecewise-constant tempo map.
#[derive(Debug, Clone, PartialEq)]
pub struct TempoMap {
    changes: Vec<TempoChange>,
    ticks_per_quarter: u32,
}

impl TempoMap {
    /// Build a tempo map from a list of changes.
    ///
    /// The list is sorted by tick; a change at tick 0 is inserted when it is
    /// missing so that the map is always total.
    #[must_use]
    pub fn new(mut changes: Vec<(u32, Tempo)>) -> Self {
        changes.sort_by_key(|(tick, _)| *tick);
        changes.dedup_by_key(|(tick, _)| *tick);
        if changes.first().map(|(tick, _)| *tick) != Some(0) {
            let initial = changes
                .first()
                .map_or(Tempo::DEFAULT, |(_, tempo)| *tempo);
            changes.insert(0, (0, initial));
        }
        TempoMap {
            changes: changes
                .into_iter()
                .map(|(tick, tempo)| TempoChange { tick, tempo })
                .collect(),
            ticks_per_quarter: crate::io::midi::TICKS_PER_QUARTER as u32,
        }
    }

    /// Build a tempo map with a constant tempo.
    #[must_use]
    pub fn constant(tempo: Tempo) -> Self {
        TempoMap::new(vec![(0, tempo)])
    }

    /// Set the tick resolution.
    #[must_use]
    pub fn with_ticks_per_quarter(mut self, ticks_per_quarter: u32) -> Self {
        self.ticks_per_quarter = ticks_per_quarter.max(1);
        self
    }

    /// The tick resolution.
    #[must_use]
    pub fn ticks_per_quarter(&self) -> u32 {
        self.ticks_per_quarter
    }

    /// The tempo changes of the map.
    #[must_use]
    pub fn changes(&self) -> &[TempoChange] {
        &self.changes
    }

    /// The tempo in effect at a given tick.
    #[must_use]
    pub fn tempo_at_tick(&self, tick: u32) -> Tempo {
        let mut current = self.changes[0].tempo;
        for change in &self.changes {
            if change.tick > tick {
                break;
            }
            current = change.tempo;
        }
        current
    }

    /// The wall-clock position of a tick.
    #[must_use]
    pub fn seconds_at_tick(&self, tick: u32) -> Seconds {
        let mut seconds = 0.0_f64;
        let mut cursor = 0u32;
        for (i, change) in self.changes.iter().enumerate() {
            if change.tick > tick {
                break;
            }
            let segment_end = self
                .changes
                .get(i + 1)
                .map_or(tick, |next| next.tick.min(tick));
            let segment_ticks = segment_end.saturating_sub(cursor);
            seconds += tick_seconds(segment_ticks, change.tempo, self.ticks_per_quarter);
            cursor = segment_end;
            if i + 1 == self.changes.len() {
                // The final segment runs to the requested tick.
                break;
            }
        }
        Seconds::from_secs(seconds)
    }

    /// The tick at a given wall-clock position.
    #[must_use]
    pub fn tick_at_seconds(&self, seconds: Seconds) -> u32 {
        let mut remaining = seconds.as_secs().max(0.0);
        let mut cursor = 0u32;
        for (i, change) in self.changes.iter().enumerate() {
            let next_tick = self.changes.get(i + 1).map_or(u32::MAX, |n| n.tick);
            let segment_ticks = next_tick.saturating_sub(cursor);
            let segment_seconds =
                tick_seconds(segment_ticks, change.tempo, self.ticks_per_quarter);
            if remaining < segment_seconds || next_tick == u32::MAX {
                let fraction = if segment_seconds > 0.0 {
                    remaining / segment_seconds
                } else {
                    0.0
                };
                let ticks = (segment_ticks as f64 * fraction).round() as u32;
                return cursor.saturating_add(ticks);
            }
            remaining -= segment_seconds;
            cursor = next_tick;
        }
        cursor
    }

    /// The wall-clock position of a musical position, in quarter-note beats.
    #[must_use]
    pub fn seconds_at_beat(&self, beat: f64) -> Seconds {
        let tick = (beat * self.ticks_per_quarter as f64).max(0.0).round() as u32;
        self.seconds_at_tick(tick)
    }

    /// The musical position, in quarter-note beats, of a wall-clock position.
    #[must_use]
    pub fn beat_at_seconds(&self, seconds: Seconds) -> f64 {
        self.tick_at_seconds(seconds) as f64 / self.ticks_per_quarter as f64
    }

    /// The wall-clock position of a musical position, given in beats.
    #[must_use]
    pub fn midi_ticks_at_beat(&self, beat: f64) -> MidiTicks {
        MidiTicks((beat * self.ticks_per_quarter as f64).max(0.0).round() as u32)
    }
}

/// The duration of a number of ticks at a given tempo.
fn tick_seconds(ticks: u32, tempo: Tempo, ticks_per_quarter: u32) -> f64 {
    let quarters = ticks as f64 / ticks_per_quarter.max(1) as f64;
    quarters * tempo.seconds_per_beat().as_secs()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn constant_tempo_maps_seconds_linearly() {
        let map = TempoMap::constant(Tempo::from_bpm(120.0).expect("valid"));
        // 480 ticks is one quarter note, which is half a second at 120 BPM.
        assert!((map.seconds_at_tick(480).as_secs() - 0.5).abs() < 1e-9);
        assert!((map.seconds_at_tick(240).as_secs() - 0.25).abs() < 1e-9);
        assert_eq!(map.tick_at_seconds(Seconds::from_secs(3.0)), 2880);
    }

    #[test]
    fn tempo_changes_are_honoured() {
        let map = TempoMap::new(vec![
            (0, Tempo::from_bpm(120.0).expect("valid")),
            (480, Tempo::from_bpm(60.0).expect("valid")),
        ]);
        // One beat at 120 BPM is half a second, then one beat at 60 BPM is a
        // full second.
        assert!((map.seconds_at_tick(480).as_secs() - 0.5).abs() < 1e-9);
        assert!((map.seconds_at_tick(960).as_secs() - 1.5).abs() < 1e-9);
        assert!((map.tempo_at_tick(480).bpm() - 60.0).abs() < 1e-9);
        assert!((map.tempo_at_tick(479).bpm() - 120.0).abs() < 1e-9);
    }

    #[test]
    fn beats_and_seconds_round_trip() {
        let map = TempoMap::constant(Tempo::from_bpm(137.0).expect("valid"));
        for beat in [0.0, 1.0, 2.5, 7.25] {
            let seconds = map.seconds_at_beat(beat);
            let back = map.beat_at_seconds(seconds);
            assert!((back - beat).abs() < 0.02, "beat {beat} came back as {back}");
        }
    }
}
