//! Note cleanup.
//!
//! The raw output of a pitch detector is noisy: the same note can be reported
//! twice, a note can be split into fragments, and the detector sometimes
//! reports notes that the instrument cannot play. This stage removes those
//! artefacts.
//!
//! The cleanup is deliberately conservative. Removing a real note is worse
//! than keeping a spurious one, because the user can see a spurious note but
//! not a missing one.

use crate::music::{Instrument, Seconds};
use crate::transcription::NoteEvent;

/// The options of the cleanup stage.
#[derive(Debug, Clone, PartialEq)]
pub struct CleanupOptions {
    /// Notes shorter than this are removed.
    pub min_duration: f64,
    /// Two notes of the same pitch that overlap by less than this are merged.
    pub merge_overlap: f64,
    /// Notes that start less than this far apart on the same pitch are merged.
    pub merge_gap: f64,
    /// Whether notes outside the instrument's range are removed.
    pub enforce_range: bool,
}

impl Default for CleanupOptions {
    fn default() -> Self {
        CleanupOptions {
            min_duration: 0.045,
            merge_overlap: 0.030,
            merge_gap: 0.020,
            enforce_range: true,
        }
    }
}

/// Clean up a list of notes.
///
/// The notes are sorted by onset and then processed in order.
///
/// # Panics
///
/// This function does not panic.
#[must_use]
pub fn cleanup(
    notes: &[NoteEvent],
    instrument: Option<&Instrument>,
    options: &CleanupOptions,
) -> Vec<NoteEvent> {
    let mut sorted: Vec<NoteEvent> = notes.to_vec();
    sorted.sort_by(|a, b| {
        a.start
            .as_secs()
            .partial_cmp(&b.start.as_secs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut result: Vec<NoteEvent> = Vec::with_capacity(sorted.len());
    for note in sorted {
        // Drop the notes the instrument cannot play.
        if options.enforce_range {
            if let Some(instrument) = instrument {
                if instrument.positions_for(note.pitch).is_empty() {
                    continue;
                }
            }
        }

        // Merge with the previous note of the same pitch when they overlap or
        // almost touch: this is what a detector reports when it splits a note
        // into two.
        if let Some(last) = result.last_mut() {
            if last.pitch == note.pitch {
                let gap = note.start.as_secs() - last.end.as_secs();
                if gap <= options.merge_gap {
                    last.end = Seconds::from_secs(last.end.as_secs().max(note.end.as_secs()));
                    last.confidence = last.confidence.max(note.confidence);
                    continue;
                }
            }
        }

        result.push(note);
    }

    // Remove the notes that are too short to be real.
    result.retain(|note| note.duration().as_secs() >= options.min_duration);

    // Limit the number of simultaneous notes to the number of strings.
    if let Some(instrument) = instrument {
        result = limit_polyphony(&result, instrument.string_count());
    }

    result
}

/// Remove the notes that are too short to be real.
///
/// This is used after the notes have been split at the onsets: the split
/// leaves a short tail of the note that was cut, and that tail is an
/// artefact of the detector rather than a note of the piece.
///
/// # Panics
///
/// This function does not panic.
#[must_use]
pub fn remove_short_notes(notes: &[NoteEvent], min_duration: f64) -> Vec<NoteEvent> {
    notes
        .iter()
        .filter(|note| note.duration().as_secs() >= min_duration)
        .cloned()
        .collect()
}

/// Keep at most `max_notes` simultaneous notes.
///
/// When too many notes sound together, the ones with the lowest confidence
/// are dropped first.
fn limit_polyphony(notes: &[NoteEvent], max_notes: usize) -> Vec<NoteEvent> {
    if max_notes == 0 || notes.is_empty() {
        return Vec::new();
    }
    let mut kept: Vec<NoteEvent> = Vec::with_capacity(notes.len());
    // A simple sweep: keep a list of the notes that are currently sounding.
    let mut active: Vec<(usize, f64)> = Vec::new();

    for note in notes {
        let start = note.start.as_secs();
        active.retain(|(_, end)| *end > start);
        active.push((kept.len(), note.end.as_secs()));
        kept.push(note.clone());
        if active.len() > max_notes {
            // Drop the least confident of the sounding notes.
            let mut worst = 0usize;
            for (index, (position, _)) in active.iter().enumerate() {
                if kept[*position].confidence < kept[active[worst].0].confidence {
                    worst = index;
                }
            }
            let position = active.remove(worst).0;
            kept[position].confidence = -1.0;
        }
    }

    kept.into_iter().filter(|n| n.confidence >= 0.0).collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::music::{InstrumentKind, MidiNote};

    #[test]
    fn short_notes_are_removed() {
        let notes = vec![
            NoteEvent::new(MidiNote::new(60).expect("valid"), Seconds::from_secs(0.0), Seconds::from_secs(0.01)),
            NoteEvent::new(MidiNote::new(62).expect("valid"), Seconds::from_secs(1.0), Seconds::from_secs(2.0)),
        ];
        let cleaned = cleanup(&notes, None, &CleanupOptions::default());
        assert_eq!(cleaned.len(), 1);
        assert_eq!(cleaned[0].pitch.as_u8(), 62);
    }

    #[test]
    fn fragmented_notes_are_merged() {
        let notes = vec![
            NoteEvent::new(MidiNote::new(60).expect("valid"), Seconds::from_secs(0.0), Seconds::from_secs(0.5)),
            NoteEvent::new(MidiNote::new(60).expect("valid"), Seconds::from_secs(0.51), Seconds::from_secs(1.0)),
        ];
        let cleaned = cleanup(&notes, None, &CleanupOptions::default());
        assert_eq!(cleaned.len(), 1);
        assert!((cleaned[0].end.as_secs() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn unplayable_notes_are_removed() {
        let instrument = Instrument::for_kind(InstrumentKind::Guitar).expect("valid");
        let notes = vec![
            NoteEvent::new(MidiNote::new(60).expect("valid"), Seconds::from_secs(0.0), Seconds::from_secs(1.0)),
            NoteEvent::new(MidiNote::new(20).expect("valid"), Seconds::from_secs(0.0), Seconds::from_secs(1.0)),
        ];
        let cleaned = cleanup(&notes, Some(&instrument), &CleanupOptions::default());
        assert_eq!(cleaned.len(), 1);
        assert_eq!(cleaned[0].pitch.as_u8(), 60);
    }

    #[test]
    fn too_many_simultaneous_notes_are_reduced() {
        let instrument = Instrument::for_kind(InstrumentKind::Guitar).expect("valid");
        let notes: Vec<NoteEvent> = (0..8)
            .map(|i| {
                let mut note = NoteEvent::new(
                    MidiNote::new(60 + i).expect("valid"),
                    Seconds::from_secs(0.0),
                    Seconds::from_secs(1.0),
                );
                note.confidence = 0.5 + i as f32 * 0.01;
                note
            })
            .collect();
        let cleaned = cleanup(&notes, Some(&instrument), &CleanupOptions::default());
        assert!(cleaned.len() <= instrument.string_count());
    }
}
