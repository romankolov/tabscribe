//! The ASCII tablature renderer.
//!
//! # Layout
//!
//! The tab is laid out on a rhythmic grid. The grid resolution is one
//! sixteenth note, and every grid step occupies a fixed number of columns, so
//! that two notes a sixteenth apart are always two columns apart and a note
//! can never be drawn on top of another.
//!
//! ```text
//! Tuning: E A D G B E   Capo: 0   Tempo: 96 BPM   Key: Am   Time: 4/4
//!
//!        Am                  F
//! e|-----0-----0---|-----1-----1---|
//! B|---1-----1-----|---1-----1-----|
//! G|-2-----2-------|---2-----2-------|
//! D|---------------|---------------|
//! A|---------------|---------------|
//! E|---------------|---------------|
//! ```
//!
//! Two-digit frets occupy two columns, so the renderer reserves the space and
//! nudges the following note to the right when necessary. A measure that needs
//! more room than the nominal width gets it; measures are then packed into
//! lines that respect the configured width.

use std::collections::HashMap;
use std::fmt::Write as _;

use crate::music::{Fret, StringIdx};
use crate::transcription::{Tab, Technique};

/// The options of the ASCII renderer.
#[derive(Debug, Clone)]
pub struct AsciiTabOptions {
    /// The maximum width of a line, in columns.
    pub width: u16,
    /// Whether to draw the rhythm line under the tablature.
    pub show_rhythm: bool,
    /// Whether to draw the chord names above the tablature.
    pub show_chords: bool,
    /// Whether to number the measures.
    pub show_measure_numbers: bool,
    /// How many columns one sixteenth-note step occupies.
    pub columns_per_step: u16,
}

impl Default for AsciiTabOptions {
    fn default() -> Self {
        AsciiTabOptions {
            width: 80,
            show_rhythm: false,
            show_chords: true,
            show_measure_numbers: false,
            columns_per_step: 2,
        }
    }
}

/// The nominal width of one grid step, in columns.
const GRID_STEPS_PER_BEAT: f64 = 4.0;

/// Render a tab as ASCII.
#[must_use]
pub fn render_tab(tab: &Tab, options: &AsciiTabOptions) -> String {
    let mut out = String::new();
    let labels = string_labels(tab);

    // Header.
    let key_text = tab.key.map_or_else(|| "-".to_string(), |k| k.name());
    let _ = writeln!(
        out,
        "{}   Tempo: {:.0} BPM   Key: {}   Time: {}",
        tab.instrument.describe(),
        tab.tempo.bpm(),
        key_text,
        tab.time_signature
    );
    out.push('\n');

    if tab.measures.is_empty() {
        return out;
    }

    let beats_per_bar = tab.time_signature.beats_per_bar();
    let nominal_width = (beats_per_bar * GRID_STEPS_PER_BEAT * f64::from(options.columns_per_step))
        .round()
        .max(4.0) as usize;

    // Group the notes by measure.
    let mut notes_by_measure: HashMap<usize, Vec<&crate::transcription::TabNote>> = HashMap::new();
    for note in &tab.notes {
        let index = measure_index_of(&tab.measures, note.start_beat);
        notes_by_measure.entry(index).or_default().push(note);
    }

    // Lay the measures out into lines.
    let lines = pack_measures(&tab.measures, nominal_width, options.width, labels.len());

    let mut measure_cursor = 0usize;
    for line in lines {
        let line_measures: Vec<usize> = line;
        let mut line_width = 0usize;
        for &m in &line_measures {
            line_width += measure_width_for(
                tab,
                m,
                nominal_width,
                options.columns_per_step,
                &notes_by_measure,
            ) + 1;
        }
        let content_width = line_width.saturating_sub(1);

        // The chord row.
        //
        // The row starts at the same column as the tab itself, so that a
        // chord name sits above the note it belongs to.
        if options.show_chords {
            let prefix = labels.first().map_or(0, |label| label.chars().count()) + 1;
            let mut row = vec![' '; content_width + prefix];
            let mut placed: Vec<(usize, String)> = Vec::new();
            let mut previous: Option<String> = None;
            for chord in &tab.chords {
                let text = chord.chord.name();
                // Only a change of chord is worth a name; a chord that is
                // repeated on every beat would hide the tab underneath it.
                if previous.as_deref() == Some(text.as_str()) {
                    continue;
                }
                previous = Some(text.clone());

                let chord_beat = chord_beat_of(tab, chord.start.as_secs());
                let Some((m, _)) = locate(&tab.measures, chord_beat) else {
                    continue;
                };
                if !line_measures.contains(&m) {
                    continue;
                }
                let mut column = 0usize;
                for &other in &line_measures {
                    if other == m {
                        break;
                    }
                    column += measure_width_for(
                        tab,
                        other,
                        nominal_width,
                        options.columns_per_step,
                        &notes_by_measure,
                    ) + 1;
                }
                column += column_of(tab, m, chord_beat, options.columns_per_step, nominal_width);
                placed.push((column + prefix, text));
            }
            placed.sort_by_key(|(c, _)| *c);
            let mut last_end = 0usize;
            for (column, text) in placed {
                let start = column.max(last_end);
                if start >= row.len() {
                    continue;
                }
                let end = (start + text.len()).min(row.len());
                for (offset, ch) in text.chars().enumerate() {
                    if start + offset < end {
                        row[start + offset] = ch;
                    }
                }
                last_end = end;
            }
            let row: String = row.into_iter().collect();
            out.push_str(row.trim_end());
            out.push('\n');
        }

        // The tab itself.
        for (string_index, label) in labels.iter().enumerate() {
            let mut line_text = format!("{label}|");
            for &m in &line_measures {
                let width = measure_width_for(
                    tab,
                    m,
                    nominal_width,
                    options.columns_per_step,
                    &notes_by_measure,
                );
                let mut cells = vec![' '; width];
                if let Some(notes) = notes_by_measure.get(&m) {
                    let mut sorted: Vec<&&crate::transcription::TabNote> = notes.iter().collect();
                    sorted.sort_by(|a, b| {
                        a.start_beat
                            .partial_cmp(&b.start_beat)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then(a.string.as_u8().cmp(&b.string.as_u8()))
                    });
                    let mut cursor = 0usize;
                    for note in sorted {
                        if note.string.as_u8() as usize != string_index {
                            continue;
                        }
                        let target = column_of(
                            tab,
                            m,
                            note.start_beat,
                            options.columns_per_step,
                            nominal_width,
                        );
                        let text = note_text(note);
                        let mut column = target.max(cursor);
                        if column + text.chars().count() > width {
                            // The note does not fit the nominal width; the
                            // measure was widened to make room for it.
                            column = cursor;
                        }
                        for (offset, ch) in text.chars().enumerate() {
                            if column + offset < cells.len() {
                                cells[column + offset] = ch;
                            }
                        }
                        cursor = column + text.chars().count();
                    }
                }
                line_text.push_str(&cells.into_iter().collect::<String>());
                line_text.push('|');
            }
            out.push_str(line_text.trim_end());
            out.push('\n');
        }

        // The rhythm line.
        if options.show_rhythm {
            let mut rhythm = format!("{}|", " ".repeat(labels.len()));
            for &m in &line_measures {
                let width = measure_width_for(
                    tab,
                    m,
                    nominal_width,
                    options.columns_per_step,
                    &notes_by_measure,
                );
                rhythm.push_str(&rhythm_line(tab, m, width, options.columns_per_step));
                rhythm.push('|');
            }
            out.push_str(rhythm.trim_end());
            out.push('\n');
        }

        measure_cursor += line_measures.len();
        let _ = measure_cursor;
    }

    out
}

/// The text drawn for a single note, including its technique symbols.
fn note_text(note: &crate::transcription::TabNote) -> String {
    let mut text = String::new();
    for technique in &note.techniques {
        if technique.is_prefix() {
            text.push(technique.symbol());
        }
    }
    text.push_str(&note.fret.as_u8().to_string());
    for technique in &note.techniques {
        if !technique.is_prefix() {
            text.push(technique.symbol());
        }
    }
    text
}

/// The label of every string, from string 1 to the last one.
///
/// When two strings share a note name (the two E strings of a guitar), the
/// higher-pitched one is written in lower case, following the convention of
/// printed tablature.
fn string_labels(tab: &Tab) -> Vec<String> {
    let strings = tab.instrument.tuning().strings();
    let mut counts: HashMap<String, usize> = HashMap::new();
    for spec in strings {
        let letter = spec
            .label
            .trim_end_matches(|c: char| c.is_ascii_digit())
            .to_string();
        *counts.entry(letter).or_insert(0) += 1;
    }

    let mut seen: HashMap<String, usize> = HashMap::new();
    strings
        .iter()
        .map(|spec| {
            let letter = spec
                .label
                .trim_end_matches(|c: char| c.is_ascii_digit())
                .to_string();
            let occurrence = seen.entry(letter.clone()).or_insert(0);
            *occurrence += 1;
            if counts.get(&letter).copied().unwrap_or(0) > 1 && *occurrence == 1 {
                letter.to_lowercase()
            } else {
                letter
            }
        })
        .collect()
}

/// The index of the measure that contains a beat.
fn measure_index_of(measures: &[crate::transcription::Measure], beat: f64) -> usize {
    measures
        .iter()
        .position(|m| beat >= m.start_beat && beat < m.end_beat)
        .unwrap_or_else(|| measures.len().saturating_sub(1))
}

/// The musical position of a chord, derived from the tempo map of the tab.
///
/// The chord is snapped to the beat, because that is where the chord
/// recogniser puts it and because the notes are snapped to the grid: a chord
/// name that sits between two columns does not sit above the note it
/// belongs to.
fn chord_beat_of(tab: &Tab, seconds: f64) -> f64 {
    let beat = seconds / tab.tempo.seconds_per_beat().as_secs().max(1e-9);
    beat.round()
}

/// Locate a musical position inside the measures.
fn locate(measures: &[crate::transcription::Measure], beat: f64) -> Option<(usize, f64)> {
    let index = measure_index_of(measures, beat);
    let measure = measures.get(index)?;
    Some((index, beat - measure.start_beat))
}

/// The column at which a note starts inside its measure.
fn column_of(
    tab: &Tab,
    measure_index: usize,
    beat: f64,
    columns_per_step: u16,
    nominal_width: usize,
) -> usize {
    let Some(measure) = tab.measures.get(measure_index) else {
        return 0;
    };
    let offset = (beat - measure.start_beat).max(0.0);
    let steps = offset * GRID_STEPS_PER_BEAT;
    let column = (steps * f64::from(columns_per_step)).round() as usize;
    column.min(nominal_width.saturating_sub(1))
}

/// The width a measure needs, in columns.
fn measure_width_for(
    tab: &Tab,
    measure_index: usize,
    nominal_width: usize,
    columns_per_step: u16,
    notes_by_measure: &HashMap<usize, Vec<&crate::transcription::TabNote>>,
) -> usize {
    let mut width = nominal_width;
    if let Some(notes) = notes_by_measure.get(&measure_index) {
        // Simulate the placement to find the required width.
        let mut cursor = 0usize;
        let mut sorted: Vec<&&crate::transcription::TabNote> = notes.iter().collect();
        sorted.sort_by(|a, b| {
            a.start_beat
                .partial_cmp(&b.start_beat)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.string.as_u8().cmp(&b.string.as_u8()))
        });
        for note in sorted {
            let target = column_of(
                tab,
                measure_index,
                note.start_beat,
                columns_per_step,
                nominal_width,
            );
            let text_width = note_text(note).chars().count();
            let column = target.max(cursor);
            cursor = column + text_width;
            if cursor > width {
                width = cursor;
            }
        }
    }
    width
}

/// The rhythm line of a measure.
fn rhythm_line(tab: &Tab, measure_index: usize, width: usize, columns_per_step: u16) -> String {
    let mut cells = vec![' '; width];
    let Some(measure) = tab.measures.get(measure_index) else {
        return cells.into_iter().collect();
    };
    let mut notes: Vec<&crate::transcription::TabNote> = tab
        .notes
        .iter()
        .filter(|n| n.start_beat >= measure.start_beat && n.start_beat < measure.end_beat)
        .collect();
    notes.sort_by(|a, b| {
        a.start_beat
            .partial_cmp(&b.start_beat)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for note in notes {
        let offset = (note.start_beat - measure.start_beat).max(0.0);
        let steps = (offset * GRID_STEPS_PER_BEAT).round() as usize;
        let column = (steps * columns_per_step as usize).min(width.saturating_sub(1));
        let marker = duration_marker(note.duration_beats);
        if column < cells.len() {
            cells[column] = marker;
        }
    }
    cells.into_iter().collect()
}

/// The letter used for a note duration in the rhythm line.
fn duration_marker(beats: f64) -> char {
    let steps = (beats * GRID_STEPS_PER_BEAT).round() as usize;
    match steps {
        0 | 1 => 'e',
        2 => 'q',
        3 => 'q',
        4 => 'h',
        5..=6 => 'h',
        7 | 8 => 'w',
        9..=12 => 'w',
        13..=15 => 'w',
        _ => 'w',
    }
}

/// Pack the measures into lines that respect the width budget.
fn pack_measures(
    measures: &[crate::transcription::Measure],
    nominal_width: usize,
    width: u16,
    label_width: usize,
) -> Vec<Vec<usize>> {
    let budget = (width as usize).saturating_sub(label_width + 1).max(1);
    let mut lines: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut used = 0usize;

    for (index, _) in measures.iter().enumerate() {
        let measure_width = nominal_width + 1;
        if used + measure_width > budget && !current.is_empty() {
            lines.push(std::mem::take(&mut current));
            used = 0;
        }
        current.push(index);
        used += measure_width;
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

/// The fret number of a note, as it appears in the tab.
#[must_use]
pub fn fret_of(note: &crate::transcription::TabNote) -> Fret {
    note.fret
}

/// The string of a note, as it appears in the tab.
#[must_use]
pub fn string_of(note: &crate::transcription::TabNote) -> StringIdx {
    note.string
}

/// The technique symbols drawn after a note.
#[must_use]
pub fn technique_symbols(techniques: &[Technique]) -> String {
    techniques
        .iter()
        .filter(|t| !t.is_prefix())
        .map(|t| t.symbol())
        .collect()
}
