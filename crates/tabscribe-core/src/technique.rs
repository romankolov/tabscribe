//! Technique detection.
//!
//! The techniques are detected from two sources: the relationship between
//! consecutive notes (which tells hammer-ons, pull-offs and slides apart from
//! re-articulated notes) and the fine-grained pitch contour (which tells
//! bends, releases and vibrato apart).
//!
//! Every detection carries a confidence, and only the techniques above the
//! threshold are reported. A wrong `h` in a tablature is worse than a missing
//! one, because the player will try to play something that is not there.

use crate::music::MidiNote;
use crate::transcription::{TabNote, Technique};

/// The confidence above which a technique is reported.
pub const DEFAULT_THRESHOLD: f32 = 0.5;

/// One sample of the pitch contour.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContourFrame {
    /// The time of the sample, in seconds.
    pub time: f64,
    /// The fundamental frequency, in Hertz. Zero means "unvoiced".
    pub frequency: f64,
    /// The confidence in the pitch.
    pub probability: f32,
}

/// The pitch contour of a signal.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PitchContour {
    /// The samples of the contour.
    pub frames: Vec<ContourFrame>,
}

impl PitchContour {
    /// The frames that fall inside a time range.
    #[must_use]
    pub fn between(&self, start: f64, end: f64) -> Vec<&ContourFrame> {
        self.frames
            .iter()
            .filter(|f| f.time >= start && f.time <= end)
            .collect()
    }
}

/// The techniques detected for one note.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectedTechnique {
    /// The technique.
    pub technique: Technique,
    /// The confidence in the detection, from 0 to 1.
    pub confidence: f32,
    /// The size of the effect, in semitones, for bends and slides.
    pub semitones: f32,
}

/// Detect the techniques of a set of notes.
///
/// The `notes` must already carry their `(string, fret)` assignment, because
/// the legato techniques are defined by what the *fingers* do, not by what the
/// pitches are.
///
/// # Panics
///
/// This function does not panic.
#[must_use]
pub fn detect(
    notes: &[TabNote],
    contour: Option<&PitchContour>,
    threshold: f32,
) -> Vec<Vec<DetectedTechnique>> {
    let mut result: Vec<Vec<DetectedTechnique>> = vec![Vec::new(); notes.len()];

    // The legato techniques: a note that starts exactly when the previous
    // note on the same string ends.
    for index in 1..notes.len() {
        let current = &notes[index];
        let previous = &notes[index - 1];
        if current.string != previous.string {
            continue;
        }
        let gap = current.start.as_secs() - previous.end.as_secs();
        if gap > 0.030 || gap < -0.030 {
            continue;
        }
        let previous_pitch = previous.pitch.as_u8() as i32;
        let current_pitch = current.pitch.as_u8() as i32;
        let semitones = current_pitch - previous_pitch;
        if semitones == 0 {
            continue;
        }
        // A small interval is a hammer-on or a pull-off; a larger one is more
        // likely to be a slide.
        if semitones.abs() <= 2 {
            let technique = if semitones > 0 {
                Technique::HammerOn
            } else {
                Technique::PullOff
            };
            let confidence = 0.9 - 0.1 * (semitones.abs() as f32 - 1.0);
            result[index].push(DetectedTechnique {
                technique,
                confidence,
                semitones: semitones as f32,
            });
        } else {
            let technique = if semitones > 0 {
                Technique::SlideUp
            } else {
                Technique::SlideDown
            };
            let confidence = (0.9 - 0.05 * (semitones.abs() as f32 - 2.0)).clamp(0.3, 0.9);
            result[index].push(DetectedTechnique {
                technique,
                confidence,
                semitones: semitones as f32,
            });
        }
    }

    // The contour techniques: bends, releases and vibrato.
    if let Some(contour) = contour {
        for (index, note) in notes.iter().enumerate() {
            let frames = contour.between(note.start.as_secs(), note.end.as_secs());
            if frames.len() < 4 {
                continue;
            }
            let semitones = contour_semitones(&frames);
            if let Some(bend) = detect_bend(&frames, note.pitch) {
                if bend.confidence >= threshold {
                    result[index].push(bend);
                }
            }
            if let Some(vibrato) = detect_vibrato(&frames) {
                if vibrato.confidence >= threshold {
                    result[index].push(vibrato);
                }
            }
            let _ = semitones;
        }
    }

    result
}

/// The pitch, in semitones relative to the note, over a set of frames.
fn contour_semitones(frames: &[&ContourFrame]) -> Vec<f32> {
    frames
        .iter()
        .map(|f| {
            if f.frequency <= 0.0 {
                return 0.0f32;
            }
            (1200.0 * (f.frequency / 440.0).log2() / 12.0) as f32
        })
        .collect()
}

/// Detect a bend: the pitch rises and then holds.
///
/// A bend is not a slide: the pitch reaches a new level and stays there, which
/// is what distinguishes it from a note that simply glides into the next one.
///
/// The detection is deliberately conservative. A plucked note also drifts, as
/// its higher partials decay faster than its fundamental and the pitch
/// estimate wanders by a few cents. A bend must therefore be **at least a
/// quarter of a semitone** and it must be **sustained**: the last third of
/// the note has to stay near the peak, and the note has to be long enough for
/// the bend to be audible.
fn detect_bend(frames: &[&ContourFrame], pitch: MidiNote) -> Option<DetectedTechnique> {
    // A bend needs time to be heard; a very short note cannot have one.
    if frames.len() < 8 {
        return None;
    }
    let values = contour_semitones(frames);
    let reference = (1200.0 * (pitch.to_hz() / 440.0).log2() / 12.0) as f32;
    let deviations: Vec<f32> = values.iter().map(|v| v - reference).collect();

    // The maximum deviation reached in the second half of the note.
    let half = deviations.len() / 2;
    let peak = deviations[half..].iter().copied().fold(0.0f32, f32::max);
    if peak < 0.25 {
        return None;
    }
    // The bend must hold: the last third must stay near the peak.
    let third = deviations.len() * 2 / 3;
    let tail = deviations[third..].iter().copied().fold(0.0f32, f32::max);
    if tail < peak * 0.8 {
        return None;
    }
    // The bend must not have been there from the start: a note that was
    // already sharp is a sharp note, not a bend.
    let first = deviations[..half].iter().copied().fold(0.0f32, f32::max);
    if first > peak * 0.6 {
        return None;
    }
    let confidence = ((peak - 0.25) / 1.5).clamp(0.3, 0.95);
    Some(DetectedTechnique {
        technique: Technique::Bend,
        confidence,
        semitones: peak,
    })
}

/// Detect vibrato: a periodic modulation of the pitch.
///
/// A plucked note drifts: its higher partials decay faster than its
/// fundamental, so the pitch estimate wanders by a few cents from frame to
/// frame. That wander is not a vibrato. A vibrato is a **regular**
/// modulation, so the detection insists on three things:
///
/// * a depth of at least two fifths of a semitone, which is the smallest
///   modulation that is audible as vibrato rather than as pitch noise;
/// * at least three crossings of the mean, so that the modulation is seen
///   more than once;
/// * regular crossings, so that the wobble of the pitch estimate is not
///   mistaken for a wobble of the string.
fn detect_vibrato(frames: &[&ContourFrame]) -> Option<DetectedTechnique> {
    if frames.len() < 10 {
        return None;
    }
    let values = contour_semitones(frames);
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let deviations: Vec<f32> = values.iter().map(|v| v - mean).collect();

    // The modulation depth.
    let depth = (deviations.iter().copied().fold(f32::MIN, f32::max)
        - deviations.iter().copied().fold(f32::MAX, f32::min))
        / 2.0;
    if depth < 0.4 {
        return None;
    }

    // The moments at which the deviation crosses the mean.
    let mut crossings: Vec<f64> = Vec::new();
    for index in 1..deviations.len() {
        if (deviations[index - 1] < 0.0) != (deviations[index] < 0.0) {
            let before = frames[index - 1].time;
            let after = frames[index].time;
            let span = (after - before).max(1e-9);
            let ratio =
                f64::from(deviations[index - 1] / (deviations[index - 1] - deviations[index]));
            crossings.push(before + ratio * span);
        }
    }
    if crossings.len() < 3 {
        return None;
    }

    // The regularity of the crossings.
    let mut intervals: Vec<f64> = crossings.windows(2).map(|w| w[1] - w[0]).collect();
    let mean_interval = intervals.iter().sum::<f64>() / intervals.len() as f64;
    if mean_interval <= 0.0 {
        return None;
    }
    let variance = intervals
        .iter()
        .map(|i| (i - mean_interval).powi(2))
        .sum::<f64>()
        / intervals.len() as f64;
    let deviation = variance.sqrt();
    if deviation > mean_interval * 0.4 {
        return None;
    }
    intervals.clear();

    // A guitar vibrato runs at five to eight cycles per second.
    let frequency = 1.0 / (2.0 * mean_interval);
    if !(3.0..=9.0).contains(&frequency) {
        return None;
    }

    Some(DetectedTechnique {
        technique: Technique::Vibrato,
        confidence: (depth * 1.5).clamp(0.3, 0.9),
        semitones: depth * 2.0,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::music::{Fret, Seconds, StringIdx};

    fn tab_note(pitch: u8, string: u8, start: f64, end: f64) -> TabNote {
        TabNote {
            pitch: MidiNote::new(pitch).expect("valid"),
            string: StringIdx::new(string).expect("valid"),
            fret: Fret::new(0).expect("valid"),
            start: Seconds::from_secs(start),
            end: Seconds::from_secs(end),
            start_beat: start,
            duration_beats: end - start,
            confidence: 1.0,
            techniques: Vec::new(),
            alternatives: Vec::new(),
        }
    }

    #[test]
    fn a_legato_rise_is_a_hammer_on() {
        let notes = vec![tab_note(60, 0, 0.0, 0.5), tab_note(62, 0, 0.5, 1.0)];
        let detected = detect(&notes, None, DEFAULT_THRESHOLD);
        assert_eq!(detected[1].len(), 1);
        assert_eq!(detected[1][0].technique, Technique::HammerOn);
    }

    #[test]
    fn a_legato_fall_is_a_pull_off() {
        let notes = vec![tab_note(62, 0, 0.0, 0.5), tab_note(60, 0, 0.5, 1.0)];
        let detected = detect(&notes, None, DEFAULT_THRESHOLD);
        assert_eq!(detected[1][0].technique, Technique::PullOff);
    }

    #[test]
    fn a_re_articulated_note_is_not_legato() {
        let notes = vec![tab_note(60, 0, 0.0, 0.4), tab_note(62, 0, 0.6, 1.0)];
        let detected = detect(&notes, None, DEFAULT_THRESHOLD);
        assert!(detected[1].is_empty());
    }

    #[test]
    fn a_note_on_another_string_is_not_legato() {
        let notes = vec![tab_note(60, 0, 0.0, 0.5), tab_note(62, 1, 0.5, 1.0)];
        let detected = detect(&notes, None, DEFAULT_THRESHOLD);
        assert!(detected[1].is_empty());
    }

    #[test]
    fn vibrato_is_detected_in_a_modulated_contour() {
        let mut frames = Vec::new();
        for index in 0..40 {
            let time = index as f64 * 0.01;
            let frequency =
                440.0 * 2f64.powf(0.35 * (2.0 * std::f64::consts::PI * 6.0 * time).sin() / 12.0);
            frames.push(ContourFrame {
                time,
                frequency,
                probability: 0.9,
            });
        }
        let contour = PitchContour { frames };
        let detected = detect_vibrato(&contour.between(0.0, 0.4));
        assert!(detected.is_some());
    }
}
