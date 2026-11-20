//! Turning a [`Transcription`] into a [`Tab`].
//!
//! This is the bridge between the analysis stages and the renderers: it groups
//! the detected notes into chords, runs the assignment optimizer over them and
//! lays the result out on the musical timeline.

use crate::error::{Error, Result};
use crate::music::{Instrument, MidiNote, Seconds, Tempo, TimeSignature};
use crate::tab::assign::{Group, TabOptimizer};
use crate::tab::cost::Style;
use crate::tab::shapes::ShapeDictionary;
use crate::transcription::{Fingering, Measure, NoteEvent, Tab, TabNote, Transcription};

/// The tolerance used when grouping notes that sound together.
const GROUP_TOLERANCE: f64 = 0.020;

/// Builds a [`Tab`] from a [`Transcription`].
#[derive(Debug, Clone)]
pub struct TabBuilder<'a> {
    /// The instrument to write for.
    pub instrument: &'a Instrument,
    /// The preferred playing style.
    pub style: Style,
    /// The maximum fret span inside a chord.
    pub max_span: Option<u8>,
    /// The maximum fret to consider.
    pub max_fret: Option<u8>,
}

impl<'a> TabBuilder<'a> {
    /// Create a builder for an instrument.
    #[must_use]
    pub fn new(instrument: &'a Instrument) -> Self {
        TabBuilder {
            instrument,
            style: Style::Balanced,
            max_span: None,
            max_fret: None,
        }
    }

    /// Set the playing style.
    #[must_use]
    pub fn with_style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }

    /// Limit the fret span inside a chord.
    #[must_use]
    pub fn with_max_span(mut self, max_span: u8) -> Self {
        self.max_span = Some(max_span);
        self
    }

    /// Limit the highest fret that may be used.
    #[must_use]
    pub fn with_max_fret(mut self, max_fret: u8) -> Self {
        self.max_fret = Some(max_fret);
        self
    }

    /// Build the tab.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when a detected note cannot be played on
    /// the instrument and [`Error::NoFeasibleTab`] when no fingering exists.
    pub fn build(&self, transcription: &Transcription) -> Result<Tab> {
        let groups = group_notes(&transcription.notes);
        if groups.is_empty() {
            return Ok(Tab {
                instrument: self.instrument.clone(),
                tempo: transcription.tempo,
                time_signature: transcription.time_signature,
                key: transcription.key,
                measures: Vec::new(),
                notes: Vec::new(),
                chords: transcription.chords.clone(),
                confidence: transcription.confidence,
            });
        }

        let weights = self.style.weights();
        let shapes = ShapeDictionary::build(self.instrument, &weights);

        // A restricted fret range means a different instrument, so the
        // optimizer has to run against that one.
        let limited = match self.max_fret {
            Some(max_fret) => Some(limited_instrument(self.instrument, max_fret)?),
            None => None,
        };
        let effective = limited.as_ref().unwrap_or(self.instrument);

        let mut optimizer = TabOptimizer::new(effective, weights).with_shapes(&shapes);
        if let Some(max_span) = self.max_span {
            optimizer = optimizer.with_max_span(max_span);
        }

        // Work out which notes of each group can actually be played.
        //
        // A transcription is not guaranteed to be playable: two detectors can
        // report the same pitch twice, and a chord can be wider than the hand
        // can reach. The notes that cannot be played are dropped, so that one
        // impossible chord does not cost the whole piece.
        let mut playable: Vec<(usize, Vec<usize>)> = Vec::with_capacity(groups.len());
        for (index, group) in groups.iter().enumerate() {
            let keep = optimizer.playable_subset(&group.pitches);
            if !keep.is_empty() {
                playable.push((index, keep));
            }
        }
        if playable.is_empty() {
            return Ok(Tab {
                instrument: self.instrument.clone(),
                tempo: transcription.tempo,
                time_signature: transcription.time_signature,
                key: transcription.key,
                measures: Vec::new(),
                notes: Vec::new(),
                chords: transcription.chords.clone(),
                confidence: transcription.confidence,
            });
        }

        // Attach the chord that each group realises, so that the optimizer can
        // prefer familiar shapes.
        let mut labelled_groups: Vec<Group> = Vec::with_capacity(playable.len());
        for (position, (index, keep)) in playable.iter().enumerate() {
            let group = &groups[*index];
            let pitches: Vec<MidiNote> = keep.iter().map(|&i| group.pitches[i]).collect();
            let chord = chord_at(&transcription.chords, group.start);
            let legato = position > 0
                && {
                    let previous = &groups[playable[position - 1].0];
                    previous.end >= group.start.as_secs() - 1e-6
                };
            labelled_groups.push(
                Group::new(pitches, group.start)
                    .legato(legato)
                    .with_chord(chord),
            );
        }

        let assignment = optimizer.assign(&labelled_groups)?;

        // Lay the notes out on the musical timeline.
        let tempo = transcription.tempo;
        let seconds_per_beat = tempo.seconds_per_beat().as_secs();
        let mut notes: Vec<TabNote> = Vec::new();

        for (position, (index, keep)) in playable.iter().enumerate() {
            let group = &groups[*index];
            let positions = &assignment.fingerings[position];
            let alternatives = &assignment.alternatives[position];
            let start_beat = group.start.as_secs() / seconds_per_beat;

            for (slot, &note_index) in keep.iter().enumerate() {
                let note = &group.notes[note_index];
                let Some(position) = positions.get(slot) else {
                    continue;
                };
                let string = crate::music::StringIdx::new(position.string)
                    .map_err(|_| Error::NoFeasibleTab("invalid string index".to_string()))?;
                let fret = crate::music::Fret::new(position.fret)
                    .map_err(|_| Error::NoFeasibleTab("invalid fret number".to_string()))?;
                let duration_beats = note.duration().as_secs() / seconds_per_beat;

                let alts: Vec<Fingering> = alternatives
                    .iter()
                    .filter_map(|state| state.get(slot))
                    .map(|p| {
                        Fingering::new(
                            crate::music::StringIdx::new(p.string)
                                .unwrap_or(crate::music::StringIdx::LOWEST),
                            crate::music::Fret::new(p.fret).unwrap_or(crate::music::Fret::OPEN),
                        )
                    })
                    .collect();

                notes.push(TabNote {
                    pitch: note.pitch,
                    string,
                    fret,
                    start: note.start,
                    end: note.end,
                    start_beat,
                    duration_beats,
                    confidence: note.confidence,
                    techniques: Vec::new(),
                    alternatives: alts,
                });
            }
        }

        notes.sort_by(|a, b| {
            a.start
                .as_secs()
                .partial_cmp(&b.start.as_secs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let measures = build_measures(
            tempo,
            transcription.time_signature,
            notes.last().map(|n| n.end).unwrap_or(Seconds::ZERO),
        );

        Ok(Tab {
            instrument: self.instrument.clone(),
            tempo,
            time_signature: transcription.time_signature,
            key: transcription.key,
            measures,
            notes,
            chords: transcription.chords.clone(),
            confidence: transcription.confidence,
        })
    }
}

/// A group of notes that sound together.
#[derive(Debug, Clone)]
pub struct NoteGroup {
    /// The notes of the group.
    pub notes: Vec<NoteEvent>,
    /// The pitches of the group.
    pub pitches: Vec<MidiNote>,
    /// The onset of the group.
    pub start: Seconds,
    /// The end of the group.
    pub end: f64,
}

/// Group the notes that sound simultaneously.
///
/// Notes are grouped by onset; a small tolerance absorbs the jitter of
/// slightly staggered detections of the same chord.
#[must_use]
pub fn group_notes(notes: &[NoteEvent]) -> Vec<NoteGroup> {
    let mut sorted: Vec<&NoteEvent> = notes.iter().collect();
    sorted.sort_by(|a, b| {
        a.start
            .as_secs()
            .partial_cmp(&b.start.as_secs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut groups: Vec<NoteGroup> = Vec::new();
    for note in sorted {
        let onset = note.start.as_secs();
        if let Some(last) = groups.last_mut() {
            if onset - last.start.as_secs() <= GROUP_TOLERANCE {
                last.notes.push(note.clone());
                last.pitches.push(note.pitch);
                last.end = last.end.max(note.end.as_secs());
                continue;
            }
        }
        groups.push(NoteGroup {
            notes: vec![note.clone()],
            pitches: vec![note.pitch],
            start: note.start,
            end: note.end.as_secs(),
        });
    }
    groups
}

/// The chord that is sounding at a given time.
fn chord_at(chords: &[crate::music::ChordEvent], at: Seconds) -> Option<crate::music::Chord> {
    chords
        .iter()
        .find(|c| c.start.as_secs() <= at.as_secs() && at.as_secs() < c.end.as_secs())
        .map(|c| c.chord)
}

/// Build the measure list of a piece.
fn build_measures(tempo: Tempo, time_signature: TimeSignature, end: Seconds) -> Vec<Measure> {
    let beats_per_bar = time_signature.beats_per_bar();
    let seconds_per_beat = tempo.seconds_per_beat().as_secs();
    let total_beats = end.as_secs() / seconds_per_beat;
    let count = (total_beats / beats_per_bar).ceil().max(1.0) as usize;

    (0..count)
        .map(|index| {
            let start_beat = index as f64 * beats_per_bar;
            let end_beat = start_beat + beats_per_bar;
            Measure {
                index,
                start_beat,
                end_beat,
                start: Seconds::from_secs(start_beat * seconds_per_beat),
                end: Seconds::from_secs(end_beat * seconds_per_beat),
            }
        })
        .collect()
}

/// An instrument with a restricted fret range.
fn limited_instrument(instrument: &Instrument, max_fret: u8) -> Result<Instrument> {
    Instrument::new(
        instrument.kind(),
        instrument.tuning().clone(),
        instrument.capo(),
        max_fret,
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::music::InstrumentKind;

    #[test]
    fn notes_with_the_same_onset_are_grouped() {
        let notes = vec![
            NoteEvent::new(MidiNote::new(64).expect("valid"), Seconds::from_secs(0.0), Seconds::from_secs(1.0)),
            NoteEvent::new(MidiNote::new(60).expect("valid"), Seconds::from_secs(0.005), Seconds::from_secs(1.0)),
            NoteEvent::new(MidiNote::new(67).expect("valid"), Seconds::from_secs(0.5), Seconds::from_secs(1.0)),
        ];
        let groups = group_notes(&notes);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].notes.len(), 2);
    }

    #[test]
    fn measures_cover_the_whole_piece() {
        let measures = build_measures(
            Tempo::from_bpm(120.0).expect("valid"),
            TimeSignature::COMMON,
            Seconds::from_secs(4.0),
        );
        // Four seconds at 120 BPM is eight beats, i.e. two 4/4 measures.
        assert_eq!(measures.len(), 2);
        assert_eq!(measures[0].start_beat, 0.0);
        assert_eq!(measures[1].start_beat, 4.0);
    }
}
