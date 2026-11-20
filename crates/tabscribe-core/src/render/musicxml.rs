//! The MusicXML exporter.
//!
//! The exporter writes a `score-partwise` document with two staves per part:
//! the standard notation and the tablature. The tablature staff uses the
//! `TAB` clef on line 6, which is what MuseScore, Sibelius and Guitar Pro
//! expect.
//!
//! The document follows the MusicXML 4.0 schema. The details that matter for
//! a reader to lay the file out correctly are:
//!
//! * the divisions, the key, the time signature, the number of staves and the
//!   clefs live in the `<attributes>` of the first measure, and nowhere else;
//! * the two staves are written one after the other, and a `<backup>` moves
//!   the cursor back to the start of the measure before the second staff is
//!   written;
//! * notes that sound together form a chord, which MusicXML writes as a
//!   `<note>` with a `<chord/>` element;
//! * the string and the fret of a tablature note live in a `<technical>`
//!   element inside the `<notations>` of the note, not directly in the note;
//! * a note that crosses a bar line is written as two notes joined by a tie.
//!
//! The document is written with `quick-xml`, which escapes the text nodes
//! properly, so a piece with an ampersand in a chord name still produces
//! valid XML.

use quick_xml::events::{BytesEnd, BytesStart, BytesText, Event};
use quick_xml::Writer;

use crate::music::{ChordKind, MidiNote, Tempo, TimeSignature};
use crate::transcription::{Tab, TabNote};

/// The divisions used by the document: the number of ticks per quarter note.
pub const DIVISIONS: u32 = 480;

/// The options of the exporter.
#[derive(Debug, Clone, PartialEq)]
pub struct MusicXmlOptions {
    /// The title of the piece.
    pub title: String,
    /// The composer of the piece.
    pub composer: String,
    /// Whether the tablature staff is written.
    pub include_tablature: bool,
    /// Whether the chord symbols are written.
    pub include_harmony: bool,
}

impl Default for MusicXmlOptions {
    fn default() -> Self {
        MusicXmlOptions {
            title: "Untitled".to_string(),
            composer: String::new(),
            include_tablature: true,
            include_harmony: true,
        }
    }
}

/// The step of a pitch: `C`, `D`, … and the accidentals that go with it.
fn pitch_step(pitch: MidiNote) -> (char, i32) {
    const STEPS: [char; 7] = ['C', 'D', 'E', 'F', 'G', 'A', 'B'];
    // The natural spelling of the white keys.
    const NATURAL: [u8; 7] = [0, 2, 4, 5, 7, 9, 11];
    let value = pitch.as_u8() as i32;
    let class = value.rem_euclid(12);

    // Find the closest natural key, preferring the flat side.
    let mut best = 0usize;
    let mut best_distance = 100;
    for (index, natural) in NATURAL.iter().enumerate() {
        let distance = (class - *natural as i32).abs();
        if distance < best_distance {
            best_distance = distance;
            best = index;
        }
    }
    let accidental = class - NATURAL[best] as i32;
    (STEPS[best], accidental)
}

/// The number of the octave, as MusicXML wants it.
fn octave_of(pitch: MidiNote) -> i32 {
    pitch.as_u8() as i32 / 12 - 1
}

/// The accidental element for a pitch.
fn accidental_element(alter: i32) -> Option<&'static str> {
    match alter {
        -2 => Some("flat-flat"),
        -1 => Some("flat"),
        0 => None,
        1 => Some("sharp"),
        2 => Some("double-sharp"),
        _ => None,
    }
}

/// The MusicXML `kind` value of a chord.
///
/// The MusicXML vocabulary has its own names for the chord kinds; they are
/// not the same as the suffixes a guitarist reads (`m`, `7`, `maj7`, …).
fn musicxml_kind(kind: ChordKind) -> &'static str {
    match kind {
        ChordKind::Major => "major",
        ChordKind::Minor => "minor",
        ChordKind::DominantSeventh => "dominant",
        ChordKind::MajorSeventh => "major-seventh",
        ChordKind::MinorSeventh => "minor-seventh",
        ChordKind::MinorMajorSeventh => "major-minor",
        ChordKind::SuspendedSecond => "suspended-second",
        ChordKind::SuspendedFourth => "suspended-fourth",
        ChordKind::Diminished => "diminished",
        ChordKind::DiminishedSeventh => "diminished-seventh",
        ChordKind::Augmented => "augmented",
        ChordKind::Power => "power",
    }
}

/// Write the tab as a MusicXML document.
///
/// # Panics
///
/// This function does not panic; the writer is a `quick_xml` writer, whose
/// errors are turned into an empty document rather than a panic.
#[must_use]
pub fn render(tab: &Tab, options: &MusicXmlOptions) -> String {
    let mut writer = Writer::new_with_indent(Vec::new(), b' ', 2);
    let mut ok = true;

    let mut write = |writer: &mut Writer<Vec<u8>>, event: Event<'_>| {
        if let Err(error) = writer.write_event(event) {
            // A `quick_xml` writer only fails on an I/O error, and the writer
            // here writes into a vector, so this cannot happen in practice.
            eprintln!("tabscribe: could not write the MusicXML document: {error}");
            ok = false;
        }
    };

    write(
        &mut writer,
        Event::Decl(quick_xml::events::BytesDecl::new("1.0", Some("UTF-8"), None)),
    );

    let mut score = BytesStart::new("score-partwise");
    score.push_attribute(("version", "4.0"));
    write(&mut writer, Event::Start(score));

    // ── Work ──────────────────────────────────────────────────────────────
    let work = BytesStart::new("work");
    write(&mut writer, Event::Start(work));
    write_text(&mut writer, "work-title", &options.title);
    write(&mut writer, Event::End(BytesEnd::new("work")));

    // ── Identification ────────────────────────────────────────────────────
    let identification = BytesStart::new("identification");
    write(&mut writer, Event::Start(identification));
    if !options.composer.is_empty() {
        let mut creator = BytesStart::new("creator");
        creator.push_attribute(("type", "composer"));
        write(&mut writer, Event::Start(creator));
        write(&mut writer, Event::Text(BytesText::new(&options.composer)));
        write(&mut writer, Event::End(BytesEnd::new("creator")));
    }
    let mut encoding = BytesStart::new("encoding");
    write(&mut writer, Event::Start(encoding));
    write_text(
        &mut writer,
        "software",
        &format!("tabscribe {}", crate::VERSION),
    );
    write(&mut writer, Event::End(BytesEnd::new("encoding")));
    write(&mut writer, Event::End(BytesEnd::new("identification")));

    // ── Part list ─────────────────────────────────────────────────────────
    let mut part_list = BytesStart::new("part-list");
    write(&mut writer, Event::Start(part_list));
    let mut score_part = BytesStart::new("score-part");
    score_part.push_attribute(("id", "P1"));
    write(&mut writer, Event::Start(score_part));
    write_text(&mut writer, "part-name", &tab.instrument.describe());
    write(&mut writer, Event::End(BytesEnd::new("score-part")));
    write(&mut writer, Event::End(BytesEnd::new("part-list")));

    // ── The part ──────────────────────────────────────────────────────────
    let mut part = BytesStart::new("part");
    part.push_attribute(("id", "P1"));
    write(&mut writer, Event::Start(part));

    let measures = build_measures(tab);
    let last = measures.len().saturating_sub(1);
    for (index, measure) in measures.iter().enumerate() {
        write_measure(&mut writer, measure, index, index == last, tab, options);
    }

    write(&mut writer, Event::End(BytesEnd::new("part")));
    write(&mut writer, Event::End(BytesEnd::new("score-partwise")));

    if !ok {
        return String::new();
    }

    let bytes = writer.into_inner();
    String::from_utf8(bytes).unwrap_or_default()
}

/// Write an empty element, e.g. `<chord/>`.
fn write_empty(writer: &mut Writer<Vec<u8>>, name: &str) {
    let _ = writer.write_event(Event::Empty(BytesStart::new(name)));
}

/// Write a text element with content.
fn write_text(writer: &mut Writer<Vec<u8>>, name: &str, content: &str) {
    let start = BytesStart::new(name);
    if writer.write_event(Event::Start(start)).is_err() {
        return;
    }
    if writer.write_event(Event::Text(BytesText::new(content))).is_err() {
        return;
    }
    let _ = writer.write_event(Event::End(BytesEnd::new(name)));
}

/// The divisions element.
fn write_divisions(writer: &mut Writer<Vec<u8>>) {
    write_text(writer, "divisions", &DIVISIONS.to_string());
}

/// The key element.
fn write_key(writer: &mut Writer<Vec<u8>>, tab: &Tab) {
    let start = BytesStart::new("key");
    if writer.write_event(Event::Start(start)).is_err() {
        return;
    }
    let fifths = tab.key.map(|k| k.fifths()).unwrap_or(0);
    write_text(writer, "fifths", &fifths.to_string());
    if let Some(key) = tab.key {
        write_text(
            writer,
            "mode",
            if key.mode == crate::music::Mode::Minor {
                "minor"
            } else {
                "major"
            },
        );
    }
    let _ = writer.write_event(Event::End(BytesEnd::new("key")));
}

/// The time element.
fn write_time(writer: &mut Writer<Vec<u8>>, meter: TimeSignature) {
    let start = BytesStart::new("time");
    if writer.write_event(Event::Start(start)).is_err() {
        return;
    }
    write_text(writer, "beats", &meter.numerator.to_string());
    write_text(writer, "beat-type", &meter.denominator.to_string());
    let _ = writer.write_event(Event::End(BytesEnd::new("time")));
}

/// The staves element.
fn write_staves(writer: &mut Writer<Vec<u8>>, options: &MusicXmlOptions) {
    let count = if options.include_tablature { 2 } else { 1 };
    write_text(writer, "staves", &count.to_string());
}

/// The treble clef.
fn write_clef(writer: &mut Writer<Vec<u8>>) {
    let start = BytesStart::new("clef");
    if writer.write_event(Event::Start(start)).is_err() {
        return;
    }
    write_text(writer, "sign", "G");
    write_text(writer, "line", "2");
    let _ = writer.write_event(Event::End(BytesEnd::new("clef")));
}

/// The tablature clef: the `TAB` sign on the sixth line.
fn write_tab_clef(writer: &mut Writer<Vec<u8>>) {
    let mut start = BytesStart::new("clef");
    start.push_attribute(("number", "2"));
    if writer.write_event(Event::Start(start)).is_err() {
        return;
    }
    write_text(writer, "sign", "TAB");
    write_text(writer, "line", "6");
    let _ = writer.write_event(Event::End(BytesEnd::new("clef")));
}

/// A note, placed inside a measure.
#[derive(Debug, Clone)]
struct PlacedNote {
    /// The note itself.
    note: TabNote,
    /// Where the note starts, in beats from the start of the measure.
    start: f64,
    /// How long the note lasts inside the measure, in beats.
    duration: f64,
    /// Whether the note is tied from the previous measure.
    tie_stop: bool,
    /// Whether the note is tied into the next measure.
    tie_start: bool,
}

/// One measure of the tab.
#[derive(Debug, Clone)]
struct BuiltMeasure {
    /// The notes of the measure, in onset order.
    notes: Vec<PlacedNote>,
    /// The start of the measure in beats.
    start_beat: f64,
    /// The end of the measure in beats.
    end_beat: f64,
}

/// Split the tab into measures.
///
/// A note that crosses a bar line is cut in two, and the two halves are
/// joined by a tie, which is how MusicXML expects a note that spans a bar
/// line to be written.
fn build_measures(tab: &Tab) -> Vec<BuiltMeasure> {
    let beats_per_measure = tab.time_signature.beats_per_bar();
    if tab.notes.is_empty() {
        return vec![BuiltMeasure {
            notes: Vec::new(),
            start_beat: 0.0,
            end_beat: beats_per_measure,
        }];
    }

    let end_beat = tab
        .notes
        .iter()
        .map(|n| n.start_beat + n.duration_beats)
        .fold(0.0f64, f64::max);
    let count = ((end_beat / beats_per_measure).ceil() as usize).max(1);

    (0..count)
        .map(|index| {
            let start_beat = index as f64 * beats_per_measure;
            let end_beat = start_beat + beats_per_measure;

            let mut notes: Vec<PlacedNote> = tab
                .notes
                .iter()
                .filter_map(|note| {
                    let note_start = note.start_beat;
                    let note_end = note.start_beat + note.duration_beats;
                    if note_end <= start_beat + 1e-9 || note_start >= end_beat - 1e-9 {
                        return None;
                    }
                    let local_start = note_start.max(start_beat) - start_beat;
                    let local_end = note_end.min(end_beat) - start_beat;
                    Some(PlacedNote {
                        note: note.clone(),
                        start: local_start,
                        duration: (local_end - local_start).max(0.0),
                        tie_stop: note_start < start_beat - 1e-9,
                        tie_start: note_end > end_beat + 1e-9,
                    })
                })
                .collect();

            notes.sort_by(|a, b| {
                a.start
                    .partial_cmp(&b.start)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        a.note
                            .pitch
                            .as_u8()
                            .cmp(&b.note.pitch.as_u8())
                    })
            });

            BuiltMeasure {
                notes,
                start_beat,
                end_beat,
            }
        })
        .collect()
}

/// The duration of a note in divisions.
fn beats_to_divisions(beats: f64) -> u32 {
    let value = (beats.max(0.0) * f64::from(DIVISIONS)).round();
    if value <= 0.0 {
        1
    } else {
        value as u32
    }
}

/// The written note type of a duration, and the number of dots.
///
/// The types are listed from the longest to the shortest, so the first entry
/// that fits the duration is the one that is used.
const NOTE_TYPES: [(f64, &str, u8); 11] = [
    (4.0, "whole", 0),
    (3.0, "half", 1),
    (2.0, "half", 0),
    (1.5, "quarter", 1),
    (1.0, "quarter", 0),
    (0.75, "eighth", 1),
    (0.5, "eighth", 0),
    (0.375, "16th", 1),
    (0.25, "16th", 0),
    (0.1875, "32nd", 1),
    (0.125, "32nd", 0),
];

/// The note type of a duration in beats.
fn note_type(beats: f64) -> (&'static str, u8) {
    for (value, name, dots) in NOTE_TYPES {
        if beats >= value - 1e-6 {
            return (name, dots);
        }
    }
    ("32nd", 0)
}

/// Write one measure.
fn write_measure(
    writer: &mut Writer<Vec<u8>>,
    measure: &BuiltMeasure,
    index: usize,
    is_last: bool,
    tab: &Tab,
    options: &MusicXmlOptions,
) {
    let number = index + 1;
    let mut start = BytesStart::new("measure");
    start.push_attribute(("number", number.to_string().as_str()));
    if writer.write_event(Event::Start(start)).is_err() {
        return;
    }

    // The attributes of the piece belong to the first measure.
    if index == 0 {
        let mut attributes = BytesStart::new("attributes");
        if writer.write_event(Event::Start(attributes)).is_err() {
            return;
        }
        write_divisions(writer);
        write_key(writer, tab);
        write_time(writer, tab.time_signature);
        write_staves(writer, options);
        write_clef(writer);
        if options.include_tablature {
            write_tab_clef(writer);
        }
        let _ = writer.write_event(Event::End(BytesEnd::new("attributes")));
    }

    // The chord symbols, each written just before the first note it applies
    // to.
    if options.include_harmony {
        write_harmony(writer, measure, tab);
    }

    // The standard notation.
    let elapsed = write_staff(writer, measure, tab, options, 1);

    // The tablature, after a backup that returns the cursor to the start of
    // the measure.
    if options.include_tablature {
        write_backup(writer, elapsed);
        write_staff(writer, measure, tab, options, 2);
    }

    // A measure with no notes still needs a rest, or the file is invalid.
    if measure.notes.is_empty() {
        write_rest(writer, measure.end_beat - measure.start_beat, 1);
    }

    if is_last {
        write_final_barline(writer);
    }

    let _ = writer.write_event(Event::End(BytesEnd::new("measure")));
}

/// The chord symbols of a measure.
fn write_harmony(writer: &mut Writer<Vec<u8>>, measure: &BuiltMeasure, tab: &Tab) {
    let seconds_per_beat = tab.tempo.seconds_per_beat().as_secs().max(1e-9);
    for chord in &tab.chords {
        let start_beat = chord.start.as_secs() / seconds_per_beat;
        if start_beat < measure.start_beat - 1e-9 || start_beat >= measure.end_beat - 1e-9 {
            continue;
        }

        let mut harmony = BytesStart::new("harmony");
        harmony.push_attribute(("print-frame", "no"));
        if writer.write_event(Event::Start(harmony)).is_err() {
            return;
        }

        let mut root = BytesStart::new("root");
        let (step, alter) = chord_step(chord.chord.root.as_u8());
        if writer.write_event(Event::Start(root)).is_err() {
            return;
        }
        write_text(writer, "root-step", &step.to_string());
        if alter != 0 {
            write_text(writer, "root-alter", &alter.to_string());
        }
        let _ = writer.write_event(Event::End(BytesEnd::new("root")));

        let mut kind = BytesStart::new("kind");
        kind.push_attribute(("text", chord.chord.kind.suffix()));
        if writer.write_event(Event::Start(kind)).is_err() {
            return;
        }
        if writer
            .write_event(Event::Text(BytesText::new(musicxml_kind(chord.chord.kind))))
            .is_err()
        {
            return;
        }
        let _ = writer.write_event(Event::End(BytesEnd::new("kind")));

        let _ = writer.write_event(Event::End(BytesEnd::new("harmony")));
    }
}

/// Write the notes of a measure on one staff.
///
/// The notes that sound together are written as a chord: the first one
/// normally, the others with a `<chord/>` element. The gaps between the
/// chords are filled with rests, so that the staff always covers the whole
/// measure.
///
/// The total length of the staff, in divisions, is returned.
fn write_staff(
    writer: &mut Writer<Vec<u8>>,
    measure: &BuiltMeasure,
    tab: &Tab,
    options: &MusicXmlOptions,
    staff: u8,
) -> u32 {
    let measure_length = measure.end_beat - measure.start_beat;
    let mut elapsed = 0u32;
    let mut cursor = 0.0f64;

    // The notes, grouped by the beat they start on.
    let mut groups: Vec<Vec<&PlacedNote>> = Vec::new();
    for note in &measure.notes {
        match groups.last_mut() {
            Some(group) if (group[0].start - note.start).abs() < 1e-6 => group.push(note),
            _ => groups.push(vec![note]),
        }
    }

    for group in groups {
        let start = group[0].start;
        if start > cursor + 1e-6 {
            // The gap before this chord is a rest.
            let gap = start - cursor;
            elapsed += write_rest(writer, gap, staff);
            cursor = start;
        }

        let duration = group[0].duration;
        let divisions = beats_to_divisions(duration);

        for (position, note) in group.iter().enumerate() {
            if staff == 1 {
                write_note(writer, note, divisions, staff, position > 0);
            } else {
                write_tab_note(writer, note, tab, divisions, staff, position > 0);
            }
        }

        elapsed += divisions;
        cursor += duration;
    }

    // Whatever is left of the measure is a rest.
    if cursor < measure_length - 1e-6 {
        elapsed += write_rest(writer, measure_length - cursor, staff);
    }

    let _ = options;
    elapsed
}

/// Write a backup that returns the cursor to the start of the measure.
fn write_backup(writer: &mut Writer<Vec<u8>>, duration: u32) {
    let mut start = BytesStart::new("backup");
    if writer.write_event(Event::Start(start)).is_err() {
        return;
    }
    write_text(writer, "duration", &duration.to_string());
    let _ = writer.write_event(Event::End(BytesEnd::new("backup")));
}

/// Write a rest that fills a gap.
fn write_rest(writer: &mut Writer<Vec<u8>>, beats: f64, staff: u8) -> u32 {
    let divisions = beats_to_divisions(beats);
    let start = BytesStart::new("note");
    if writer.write_event(Event::Start(start)).is_err() {
        return 0;
    }
    write_empty(writer, "rest");
    write_text(writer, "duration", &divisions.to_string());
    write_text(writer, "voice", &voice_of(staff));
    if staff > 1 {
        write_text(writer, "staff", &staff.to_string());
    }
    let _ = writer.write_event(Event::End(BytesEnd::new("note")));
    divisions
}

/// The voice of a staff.
///
/// The tablature staff has its own voice, so that the two staves never mix.
fn voice_of(staff: u8) -> String {
    if staff > 1 {
        "5".to_string()
    } else {
        "1".to_string()
    }
}

/// Write a note on the standard notation staff.
fn write_note(
    writer: &mut Writer<Vec<u8>>,
    placed: &PlacedNote,
    divisions: u32,
    staff: u8,
    in_chord: bool,
) {
    let start = BytesStart::new("note");
    if writer.write_event(Event::Start(start)).is_err() {
        return;
    }

    if in_chord {
        write_empty(writer, "chord");
    }

    let pitch = placed.note.pitch;
    let (step, alter) = pitch_step(pitch);
    let octave = octave_of(pitch);

    let mut pitch_element = BytesStart::new("pitch");
    if writer.write_event(Event::Start(pitch_element)).is_err() {
        return;
    }
    write_text(writer, "step", &step.to_string());
    if alter != 0 {
        write_text(writer, "alter", &alter.to_string());
    }
    write_text(writer, "octave", &octave.to_string());
    let _ = writer.write_event(Event::End(BytesEnd::new("pitch")));

    write_text(writer, "duration", &divisions.to_string());

    // The tie says how the note sounds, the tied says how it is written.
    if placed.tie_stop {
        write_tie(writer, "tie", "stop");
    }
    if placed.tie_start {
        write_tie(writer, "tie", "start");
    }

    write_text(writer, "voice", &voice_of(staff));

    let (name, dots) = note_type(placed.duration);
    write_text(writer, "type", name);
    for _ in 0..dots {
        write_empty(writer, "dot");
    }

    if let Some(accidental) = accidental_element(alter) {
        write_text(writer, "accidental", accidental);
    }

    if staff > 1 {
        write_text(writer, "staff", &staff.to_string());
    }

    if placed.tie_stop || placed.tie_start {
        let mut notations = BytesStart::new("notations");
        if writer.write_event(Event::Start(notations)).is_err() {
            return;
        }
        if placed.tie_stop {
            write_tie(writer, "tied", "stop");
        }
        if placed.tie_start {
            write_tie(writer, "tied", "start");
        }
        let _ = writer.write_event(Event::End(BytesEnd::new("notations")));
    }

    let _ = writer.write_event(Event::End(BytesEnd::new("note")));
}

/// Write a tie or a tied element.
fn write_tie(writer: &mut Writer<Vec<u8>>, name: &str, kind: &str) {
    let mut start = BytesStart::new(name);
    start.push_attribute(("type", kind));
    let _ = writer.write_event(Event::Empty(start));
}

/// Write a note on the tablature staff.
///
/// The note carries its pitch as well as the string and the fret, because the
/// pitch is what gives the note its place in the rhythm of the staff.
fn write_tab_note(
    writer: &mut Writer<Vec<u8>>,
    placed: &PlacedNote,
    _tab: &Tab,
    divisions: u32,
    staff: u8,
    in_chord: bool,
) {
    let start = BytesStart::new("note");
    if writer.write_event(Event::Start(start)).is_err() {
        return;
    }

    if in_chord {
        write_empty(writer, "chord");
    }

    let pitch = placed.note.pitch;
    let (step, alter) = pitch_step(pitch);
    let octave = octave_of(pitch);

    let mut pitch_element = BytesStart::new("pitch");
    if writer.write_event(Event::Start(pitch_element)).is_err() {
        return;
    }
    write_text(writer, "step", &step.to_string());
    if alter != 0 {
        write_text(writer, "alter", &alter.to_string());
    }
    write_text(writer, "octave", &octave.to_string());
    let _ = writer.write_event(Event::End(BytesEnd::new("pitch")));

    write_text(writer, "duration", &divisions.to_string());

    if placed.tie_stop {
        write_tie(writer, "tie", "stop");
    }
    if placed.tie_start {
        write_tie(writer, "tie", "start");
    }

    write_text(writer, "voice", &voice_of(staff));

    let (name, dots) = note_type(placed.duration);
    write_text(writer, "type", name);
    for _ in 0..dots {
        write_empty(writer, "dot");
    }

    if staff > 1 {
        write_text(writer, "staff", &staff.to_string());
    }

    let mut notations = BytesStart::new("notations");
    if writer.write_event(Event::Start(notations)).is_err() {
        return;
    }

    if placed.tie_stop {
        write_tie(writer, "tied", "stop");
    }
    if placed.tie_start {
        write_tie(writer, "tied", "start");
    }

    // The string and the fret.
    //
    // MusicXML numbers the strings from the highest-pitched one, which is
    // the same order as the internal numbering.
    let string_number = (placed.note.string.as_u8() as usize + 1).clamp(1, 255);

    let mut technical = BytesStart::new("technical");
    if writer.write_event(Event::Start(technical)).is_err() {
        return;
    }
    write_text(writer, "string", &string_number.to_string());
    write_text(writer, "fret", &placed.note.fret.as_u8().to_string());
    let _ = writer.write_event(Event::End(BytesEnd::new("technical")));

    let _ = writer.write_event(Event::End(BytesEnd::new("notations")));
    let _ = writer.write_event(Event::End(BytesEnd::new("note")));
}

/// The step and alteration of a pitch class.
fn chord_step(pitch_class: u8) -> (char, i32) {
    const STEPS: [char; 7] = ['C', 'D', 'E', 'F', 'G', 'A', 'B'];
    const NATURAL: [u8; 7] = [0, 2, 4, 5, 7, 9, 11];
    let mut best = 0usize;
    let mut best_distance = 100;
    for (index, natural) in NATURAL.iter().enumerate() {
        let distance = (pitch_class as i32 - *natural as i32).abs();
        if distance < best_distance {
            best_distance = distance;
            best = index;
        }
    }
    (STEPS[best], pitch_class as i32 - NATURAL[best] as i32)
}

/// Write the bar line that closes the piece.
fn write_final_barline(writer: &mut Writer<Vec<u8>>) {
    let mut start = BytesStart::new("barline");
    start.push_attribute(("location", "right"));
    if writer.write_event(Event::Start(start)).is_err() {
        return;
    }
    write_text(writer, "bar-style", "light-heavy");
    let _ = writer.write_event(Event::End(BytesEnd::new("barline")));
}

/// The tempo in microseconds per quarter note, for the report.
#[must_use]
pub fn tempo_microseconds(tempo: Tempo) -> u32 {
    tempo.to_microseconds_per_quarter()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::music::{Instrument, InstrumentKind, Key};

    fn instrument() -> Instrument {
        Instrument::for_kind(InstrumentKind::Guitar).expect("valid")
    }

    fn simple_tab() -> Tab {
        Tab {
            instrument: instrument(),
            tempo: Tempo::DEFAULT,
            time_signature: TimeSignature::COMMON,
            key: Some(Key::C_MAJOR),
            measures: Vec::new(),
            notes: Vec::new(),
            chords: Vec::new(),
            confidence: 1.0,
        }
    }

    fn note(pitch: u8, string: u8, fret: u8, start: f64, duration: f64) -> TabNote {
        TabNote {
            pitch: MidiNote::new(pitch).expect("valid"),
            string: crate::music::StringIdx::new(string).expect("valid"),
            fret: crate::music::Fret::new(fret).expect("valid"),
            start: crate::music::Seconds::from_secs(0.0),
            end: crate::music::Seconds::from_secs(duration),
            start_beat: start,
            duration_beats: duration,
            confidence: 1.0,
            techniques: Vec::new(),
            alternatives: Vec::new(),
        }
    }

    #[test]
    fn an_empty_tab_produces_valid_xml() {
        let xml = render(&simple_tab(), &MusicXmlOptions::default());
        assert!(xml.starts_with("<?xml"));
        assert!(xml.contains("<score-partwise"));
        assert!(xml.contains("</score-partwise>"));
    }

    #[test]
    fn the_document_is_deterministic() {
        let options = MusicXmlOptions::default();
        let first = render(&simple_tab(), &options);
        let second = render(&simple_tab(), &options);
        assert_eq!(first, second);
    }

    #[test]
    fn the_title_is_escaped() {
        let options = MusicXmlOptions {
            title: "A & B <test>".to_string(),
            ..MusicXmlOptions::default()
        };
        let xml = render(&simple_tab(), &options);
        assert!(xml.contains("&amp;"));
        assert!(xml.contains("&lt;test&gt;"));
        assert!(!xml.contains("<test>"));
    }

    #[test]
    fn the_pitch_step_matches_the_pitch() {
        assert_eq!(pitch_step(MidiNote::new(60).expect("valid")), ('C', 0));
        assert_eq!(pitch_step(MidiNote::new(61).expect("valid")), ('C', 1));
        assert_eq!(pitch_step(MidiNote::new(59).expect("valid")), ('B', 0));
        assert_eq!(octave_of(MidiNote::new(60).expect("valid")), 4);
    }

    #[test]
    fn the_attributes_are_only_written_in_the_first_measure() {
        let mut tab = simple_tab();
        tab.notes = vec![note(60, 2, 0, 0.0, 4.0)];
        let xml = render(&tab, &MusicXmlOptions::default());
        assert_eq!(xml.matches("<divisions>").count(), 1);
        assert_eq!(xml.matches("<staves>").count(), 1);
        assert_eq!(xml.matches("<key>").count(), 1);
        assert_eq!(xml.matches("<time>").count(), 1);
    }

    #[test]
    fn the_tablature_staff_is_written_with_the_tab_clef() {
        let mut tab = simple_tab();
        tab.notes = vec![note(60, 2, 0, 0.0, 4.0)];
        let xml = render(&tab, &MusicXmlOptions::default());
        assert!(xml.contains("<sign>TAB</sign>"));
        assert!(xml.contains("<line>6</line>"));
        assert!(xml.contains("<staves>2</staves>"));
        assert!(xml.contains("<staff>2</staff>"));
    }

    #[test]
    fn the_tablature_can_be_left_out() {
        let mut tab = simple_tab();
        tab.notes = vec![note(60, 2, 0, 0.0, 4.0)];
        let options = MusicXmlOptions {
            include_tablature: false,
            ..MusicXmlOptions::default()
        };
        let xml = render(&tab, &options);
        assert!(!xml.contains("<sign>TAB</sign>"));
        assert!(!xml.contains("<staff>2</staff>"));
        assert!(xml.contains("<staves>1</staves>"));
    }

    #[test]
    fn the_staves_are_separated_by_a_backup() {
        let mut tab = simple_tab();
        tab.notes = vec![note(60, 2, 0, 0.0, 4.0)];
        let xml = render(&tab, &MusicXmlOptions::default());
        assert!(xml.contains("<backup>"));
        // The backup has to come after the standard note and before the tab
        // note.
        let standard = xml.find("<pitch>").expect("a standard note");
        let backup = xml.find("<backup>").expect("a backup");
        let tab_note = xml.find("<staff>2</staff>").expect("a tab note");
        assert!(standard < backup);
        assert!(backup < tab_note);
    }

    #[test]
    fn notes_that_sound_together_are_written_as_a_chord() {
        let mut tab = simple_tab();
        tab.notes = vec![
            note(60, 2, 0, 0.0, 4.0),
            note(64, 1, 0, 0.0, 4.0),
            note(67, 0, 0, 0.0, 4.0),
        ];
        let xml = render(&tab, &MusicXmlOptions::default());
        assert_eq!(xml.matches("<chord/>").count(), 4);
    }

    #[test]
    fn the_fret_lives_in_a_technical_element() {
        let mut tab = simple_tab();
        tab.notes = vec![note(60, 2, 5, 0.0, 4.0)];
        let xml = render(&tab, &MusicXmlOptions::default());
        assert!(xml.contains("<technical>"));
        assert!(xml.contains("<string>3</string>"));
        assert!(xml.contains("<fret>5</fret>"));
        // The technical element belongs to the notations of the note.
        let notations = xml.find("<notations>").expect("notations");
        let technical = xml.find("<technical").expect("technical");
        assert!(notations < technical);
    }

    #[test]
    fn a_note_that_crosses_a_bar_line_is_tied() {
        let mut tab = simple_tab();
        tab.notes = vec![note(60, 2, 0, 3.0, 2.0)];
        let xml = render(&tab, &MusicXmlOptions::default());
        assert!(xml.contains("<tie type=\"start\""));
        assert!(xml.contains("<tie type=\"stop\""));
    }

    #[test]
    fn the_chord_kind_is_written_in_the_musicxml_vocabulary() {
        let mut tab = simple_tab();
        tab.notes = vec![note(60, 2, 0, 0.0, 4.0)];
        tab.chords = vec![crate::music::ChordEvent {
            chord: crate::music::Chord::new(crate::music::PitchClass::C, ChordKind::MinorSeventh),
            start: crate::music::Seconds::from_secs(0.0),
            end: crate::music::Seconds::from_secs(2.0),
            confidence: 0.9,
        }];
        let xml = render(&tab, &MusicXmlOptions::default());
        assert!(xml.contains("<kind text=\"m7\">"));
        assert!(xml.contains("minor-seventh"));
    }

    #[test]
    fn the_harmony_can_be_left_out() {
        let mut tab = simple_tab();
        tab.notes = vec![note(60, 2, 0, 0.0, 4.0)];
        let options = MusicXmlOptions {
            include_harmony: false,
            ..MusicXmlOptions::default()
        };
        let xml = render(&tab, &options);
        assert!(!xml.contains("<harmony"));
    }

    #[test]
    fn the_note_type_follows_the_duration() {
        assert_eq!(note_type(4.0).0, "whole");
        assert_eq!(note_type(3.0).0, "half");
        assert_eq!(note_type(3.0).1, 1);
        assert_eq!(note_type(1.0).0, "quarter");
        assert_eq!(note_type(0.5).0, "eighth");
        assert_eq!(note_type(0.25).0, "16th");
    }
}
