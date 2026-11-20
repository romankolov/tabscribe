//! Reading and writing Standard MIDI Files.
//!
//! # The format
//!
//! `tabscribe` writes Type 1 files: a tempo map track followed by one track
//! per instrument. Every track carries its own time signature and tempo
//! events, so the file plays back correctly in any DAW even when the
//! instrument changes mid-piece.
//!
//! Ticks are used as the internal unit; the conversion to seconds happens
//! through the [`TempoMap`], never by assuming a constant tempo.

use std::collections::BTreeMap;

use midly::num::{u15, u24, u28, u4, u7};
use midly::{Format, Header, MetaMessage, MidiMessage, Timing, TrackEvent, TrackEventKind};

use crate::error::{Error, Result};
use crate::music::{MidiNote, MidiTicks, Seconds, Tempo, TimeSignature};
use crate::transcription::NoteEvent;

/// The tick resolution used in the files `tabscribe` writes.
pub const TICKS_PER_QUARTER: u16 = 480;

/// One track of a MIDI document.
#[derive(Debug, Clone)]
pub struct MidiTrack {
    /// The name of the track.
    pub name: String,
    /// The General MIDI program number.
    pub program: u8,
    /// The General MIDI channel, `0..=15`.
    pub channel: u8,
    /// The time signature of the track.
    pub time_signature: TimeSignature,
    /// The tempo of the track.
    pub tempo: Tempo,
    /// The notes of the track.
    pub notes: Vec<NoteEvent>,
}

impl MidiTrack {
    /// Construct an empty track.
    #[must_use]
    pub fn new(name: impl Into<String>, program: u8, channel: u8) -> Self {
        MidiTrack {
            name: name.into(),
            program,
            channel,
            time_signature: TimeSignature::COMMON,
            tempo: Tempo::DEFAULT,
            notes: Vec::new(),
        }
    }
}

/// A MIDI document: a set of tracks.
#[derive(Debug, Clone)]
pub struct MidiDocument {
    /// The tracks of the document.
    pub tracks: Vec<MidiTrack>,
}

impl MidiDocument {
    /// Construct a document with a single track.
    #[must_use]
    pub fn single(track: MidiTrack) -> Self {
        MidiDocument {
            tracks: vec![track],
        }
    }

    /// The notes of every track, merged and sorted by onset.
    #[must_use]
    pub fn merged_notes(&self) -> Vec<NoteEvent> {
        let mut notes: Vec<NoteEvent> = self
            .tracks
            .iter()
            .flat_map(|t| t.notes.iter().cloned())
            .collect();
        notes.sort_by(|a, b| {
            a.start
                .as_secs()
                .partial_cmp(&b.start.as_secs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        notes
    }

    /// Serialize the document to a Standard MIDI File.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Serde`] when the file cannot be assembled.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let header = Header::new(
            Format::Parallel,
            Timing::Metrical(u15::new(TICKS_PER_QUARTER)),
        );

        // The track names are borrowed by the events, so they have to outlive
        // the event lists.
        let names: Vec<Vec<u8>> = self
            .tracks
            .iter()
            .map(|t| t.name.as_bytes().to_vec())
            .collect();
        let tracks: Vec<Vec<TrackEvent<'_>>> = self
            .tracks
            .iter()
            .zip(names.iter())
            .map(|(track, name)| build_track(track, name))
            .collect();

        let mut buffer = Vec::new();
        midly::write(&header, tracks.iter(), &mut buffer).map_err(|e| Error::Serde {
            what: "MIDI file".to_string(),
            reason: e.to_string(),
        })?;
        Ok(buffer)
    }

    /// Parse a Standard MIDI File.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Serde`] when the file is malformed, and
    /// [`Error::UnsupportedFormat`] when it is not a Standard MIDI File.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let (header, track_iter) = midly::parse(bytes)
            .map_err(|e| Error::UnsupportedFormat(format!("not a readable MIDI file: {e}")))?;

        let ticks_per_quarter = match header.timing {
            Timing::Metrical(ticks) => u32::from(ticks.as_int()).max(1),
            Timing::Timecode(_, _) => u32::from(TICKS_PER_QUARTER),
        };

        // Collect the tempo map first: the note timings depend on it.
        let mut tempo_changes: Vec<(u32, Tempo)> = Vec::new();
        let mut time_signature = TimeSignature::COMMON;
        for track_result in track_iter {
            let events = track_result.map_err(|e| Error::Serde {
                what: "MIDI track".to_string(),
                reason: e.to_string(),
            })?;
            let mut tick = 0u32;
            for event_result in events {
                let event = event_result.map_err(|e| Error::Serde {
                    what: "MIDI event".to_string(),
                    reason: e.to_string(),
                })?;
                tick = tick.saturating_add(event.delta.as_int());
                if let TrackEventKind::Meta(meta) = event.kind {
                    match meta {
                        MetaMessage::Tempo(us) => {
                            if let Ok(tempo) = Tempo::from_microseconds_per_quarter(us.as_int()) {
                                tempo_changes.push((tick, tempo));
                            }
                        }
                        MetaMessage::TimeSignature(numerator, denominator, _, _) => {
                            if let Ok(sig) = TimeSignature::new(numerator, denominator) {
                                time_signature = sig;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        tempo_changes.sort_by_key(|(tick, _)| *tick);
        tempo_changes.dedup_by_key(|(tick, _)| *tick);
        if tempo_changes.is_empty() {
            tempo_changes.push((0, Tempo::DEFAULT));
        }
        let tempo_map = crate::music::TempoMap::new(tempo_changes.clone())
            .with_ticks_per_quarter(ticks_per_quarter);

        // Collect the notes of every track.
        let mut tracks: Vec<MidiTrack> = Vec::new();
        for (index, track_result) in midly::parse(bytes)
            .map_err(|e| Error::UnsupportedFormat(format!("not a readable MIDI file: {e}")))?
            .1
            .enumerate()
        {
            let events = track_result.map_err(|e| Error::Serde {
                what: "MIDI track".to_string(),
                reason: e.to_string(),
            })?;

            let mut name = format!("Track {}", index + 1);
            let mut program = 0u8;
            let channel = 0u8;
            let mut tick = 0u32;
            // (key, channel) -> (start tick, velocity)
            let mut pending: BTreeMap<(u8, u8), (u32, u8)> = BTreeMap::new();
            let mut notes: Vec<NoteEvent> = Vec::new();

            for event_result in events {
                let event = event_result.map_err(|e| Error::Serde {
                    what: "MIDI event".to_string(),
                    reason: e.to_string(),
                })?;
                tick = tick.saturating_add(event.delta.as_int());

                match event.kind {
                    TrackEventKind::Midi {
                        channel: ch,
                        message,
                    } => match message {
                        MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
                            pending.insert((key.as_int(), ch.as_int()), (tick, vel.as_int()));
                        }
                        MidiMessage::NoteOn { key, .. } | MidiMessage::NoteOff { key, .. } => {
                            if let Some((start_tick, vel)) =
                                pending.remove(&(key.as_int(), ch.as_int()))
                            {
                                let start = tempo_map.seconds_at_tick(start_tick);
                                let end = tempo_map.seconds_at_tick(tick);
                                notes.push(NoteEvent {
                                    pitch: MidiNote::clamped(u16::from(key.as_int())),
                                    start,
                                    end: Seconds::from_secs(end.as_secs().max(start.as_secs())),
                                    velocity: vel,
                                    confidence: 1.0,
                                });
                            }
                        }
                        MidiMessage::ProgramChange { program: p } if ch.as_int() == channel => {
                            program = p.as_int();
                        }
                        _ => {}
                    },
                    TrackEventKind::Meta(MetaMessage::TrackName(bytes)) => {
                        name = String::from_utf8_lossy(bytes).to_string();
                    }
                    _ => {}
                }
            }

            // Notes still held when the track ended.
            for ((key, _), (start_tick, vel)) in pending {
                let start = tempo_map.seconds_at_tick(start_tick);
                let end = tempo_map.seconds_at_tick(start_tick + ticks_per_quarter);
                notes.push(NoteEvent {
                    pitch: MidiNote::clamped(u16::from(key)),
                    start,
                    end: Seconds::from_secs(end.as_secs().max(start.as_secs())),
                    velocity: vel,
                    confidence: 1.0,
                });
            }

            notes.sort_by(|a, b| {
                a.start
                    .as_secs()
                    .partial_cmp(&b.start.as_secs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            if !notes.is_empty() {
                tracks.push(MidiTrack {
                    name,
                    program,
                    channel,
                    time_signature,
                    tempo: tempo_changes.first().map_or(Tempo::DEFAULT, |(_, t)| *t),
                    notes,
                });
            }
        }

        if tracks.is_empty() {
            return Err(Error::Serde {
                what: "MIDI file".to_string(),
                reason: "the file contains no notes".to_string(),
            });
        }

        Ok(MidiDocument { tracks })
    }
}

/// Build the event list of a track.
#[allow(clippy::vec_init_then_push)]
fn build_track<'a>(track: &MidiTrack, name: &'a [u8]) -> Vec<TrackEvent<'a>> {
    let mut out: Vec<TrackEvent<'_>> = Vec::new();

    out.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::TrackName(name)),
    });
    out.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::TimeSignature(
            track.time_signature.numerator,
            track.time_signature.denominator,
            24,
            8,
        )),
    });
    out.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::new(
            track.tempo.to_microseconds_per_quarter(),
        ))),
    });
    out.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Midi {
            channel: u4::new(track.channel),
            message: MidiMessage::ProgramChange {
                program: u7::new(track.program),
            },
        },
    });

    // Note on/off pairs, sorted by onset.
    let mut events: Vec<(u32, u32, MidiMessage)> = Vec::new();
    for note in &track.notes {
        let start = MidiTicks::from_seconds(note.start, u32::from(TICKS_PER_QUARTER), track.tempo);
        let end = MidiTicks::from_seconds(note.end, u32::from(TICKS_PER_QUARTER), track.tempo);
        let key = note.pitch.as_u8();
        events.push((
            start.as_u32(),
            1,
            MidiMessage::NoteOn {
                key: u7::new(key),
                vel: u7::new(note.velocity.max(1)),
            },
        ));
        events.push((
            end.as_u32(),
            0,
            MidiMessage::NoteOff {
                key: u7::new(key),
                vel: u7::new(64),
            },
        ));
    }
    events.sort_by_key(|(tick, priority, _)| (*tick, *priority));

    let mut previous = 0u32;
    for (tick, _, message) in events {
        out.push(TrackEvent {
            delta: u28::new(tick.saturating_sub(previous)),
            kind: TrackEventKind::Midi {
                channel: u4::new(track.channel),
                message,
            },
        });
        previous = tick;
    }

    out.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::music::MidiTicks;

    #[test]
    fn midi_round_trip_preserves_notes() {
        let track = MidiTrack {
            name: "Test".to_string(),
            program: 25,
            channel: 0,
            time_signature: TimeSignature::COMMON,
            tempo: Tempo::from_bpm(120.0).expect("valid"),
            notes: vec![
                NoteEvent::new(
                    MidiNote::new(60).expect("valid"),
                    Seconds::from_secs(0.0),
                    Seconds::from_secs(0.5),
                ),
                NoteEvent::new(
                    MidiNote::new(64).expect("valid"),
                    Seconds::from_secs(0.5),
                    Seconds::from_secs(1.0),
                ),
            ],
        };
        let document = MidiDocument::single(track);
        let bytes = document.to_bytes().expect("serializable");
        let parsed = MidiDocument::from_bytes(&bytes).expect("parseable");

        assert_eq!(parsed.tracks.len(), 1);
        assert_eq!(parsed.tracks[0].notes.len(), 2);
        assert_eq!(parsed.tracks[0].notes[0].pitch.as_u8(), 60);
        assert_eq!(parsed.tracks[0].notes[1].pitch.as_u8(), 64);
        assert!((parsed.tracks[0].notes[0].start.as_secs() - 0.0).abs() < 1e-3);
        assert!((parsed.tracks[0].notes[1].start.as_secs() - 0.5).abs() < 1e-3);
        assert_eq!(parsed.tracks[0].name, "Test");
    }

    #[test]
    fn malformed_input_is_rejected_without_panicking() {
        assert!(MidiDocument::from_bytes(b"not a midi file").is_err());
        assert!(MidiDocument::from_bytes(&[]).is_err());
    }

    #[test]
    fn ticks_per_quarter_is_used_consistently() {
        assert_eq!(TICKS_PER_QUARTER, 480);
        let ticks = MidiTicks::from_seconds(
            Seconds::from_secs(1.0),
            480,
            Tempo::from_bpm(60.0).expect("valid"),
        );
        assert_eq!(ticks.as_u32(), 480);
    }
}
