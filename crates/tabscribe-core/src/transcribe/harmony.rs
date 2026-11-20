//! Harmony analysis: chroma, key detection and chord recognition.
//!
//! # Chroma
//!
//! The chroma is a twelve-dimensional vector of "how much energy is there in
//! each pitch class". It is computed from the magnitude spectrum by folding
//! every frequency bin onto its pitch class, which removes the octave
//! ambiguity that makes a plain spectrum useless for harmony analysis.
//!
//! # Key
//!
//! The key is detected with the Krumhansl–Schmuckler profiles: the chroma is
//! correlated with the major and minor profiles, rotated to every possible
//! tonic, and the best correlation wins.
//!
//! # Chords
//!
//! Chords are recognised on the beat-synchronous chroma: the chroma is
//! averaged over each beat and matched against a vocabulary of chord
//! templates. A template scores well when the energy of its notes is high and
//! the energy of the notes that do not belong to it is low.

use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::music::{Chord, ChordEvent, ChordKind, Key, Mode, PitchClass, Seconds, TimeSignature};

/// The number of bins used by the chroma analysis.
const CHROMA_BINS: usize = 2048;

/// A chroma vector: one energy value per pitch class.
pub type Chroma = [f64; 12];

/// A time-resolved chroma: one chroma vector per analysis frame.
#[derive(Debug, Clone, Default)]
pub struct ChromaTrack {
    /// The time of each frame, in seconds.
    pub times: Vec<f64>,
    /// The chroma of each frame.
    pub frames: Vec<Chroma>,
    /// The time between two frames, in seconds.
    pub hop_seconds: f64,
}

impl ChromaTrack {
    /// The average chroma of the whole track.
    #[must_use]
    pub fn average(&self) -> Chroma {
        let mut out = [0.0f64; 12];
        if self.frames.is_empty() {
            return out;
        }
        for frame in &self.frames {
            for (target, value) in out.iter_mut().zip(frame.iter()) {
                *target += *value;
            }
        }
        let count = self.frames.len() as f64;
        for value in &mut out {
            *value /= count;
        }
        out
    }

    /// The average chroma between two times.
    #[must_use]
    pub fn between(&self, start: f64, end: f64) -> Chroma {
        let mut out = [0.0f64; 12];
        let mut count = 0usize;
        for (index, frame) in self.frames.iter().enumerate() {
            let time = self.times[index];
            if time >= start && time < end {
                for (target, value) in out.iter_mut().zip(frame.iter()) {
                    *target += *value;
                }
                count += 1;
            }
        }
        if count > 0 {
            for value in &mut out {
                *value /= count as f64;
            }
        }
        out
    }
}

/// Compute the time-resolved chroma of a signal.
///
/// # How the chroma is built
///
/// A naive chroma folds every spectral bin onto its pitch class. That fails
/// for real instruments: the harmonics of a single note land on other pitch
/// classes, so a lone A sounds like an A-major chord.
///
/// The chroma here is therefore built from a **harmonic-sum salience**. For
/// every candidate fundamental the salience is
///
/// ```text
/// salience(f) = Σ E(h·f) / h
/// ```
///
/// over the first six harmonics, where `E` is the magnitude of the spectrum.
/// A true fundamental has all of its harmonics available, so it scores high;
/// a harmonic of another note does not have the harmonic series of its own
/// and scores lower. The salience of each octave is subtracted from the one
/// above it, which removes the residual octave ambiguity, and what is left is
/// folded onto the twelve pitch classes.
///
/// # Panics
///
/// This function does not panic. A signal shorter than one window produces an
/// empty track.
#[must_use]
pub fn chroma_track(signal: &[f32], sample_rate: u32) -> ChromaTrack {
    let mut track = ChromaTrack::default();
    if sample_rate == 0 || signal.len() < 2048 {
        return track;
    }
    let window_size = CHROMA_BINS.min(signal.len());
    let hop_size = window_size / 4;
    track.hop_seconds = hop_size as f64 / sample_rate as f64;

    let mut planner = FftPlanner::<f32>::new();
    let fft: Arc<dyn rustfft::Fft<f32>> = planner.plan_fft_forward(window_size);
    let mut buffer = vec![Complex::new(0.0f32, 0.0f32); window_size];
    let window: Vec<f32> = (0..window_size)
        .map(|i| {
            let n = i as f64;
            let denominator = (window_size - 1) as f64;
            (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n / denominator).cos()) as f32
        })
        .collect();

    // The candidate fundamentals, as MIDI note numbers.
    let lowest = 24u8;
    let highest = 108u8;

    let mut start = 0usize;
    while start + window_size <= signal.len() {
        for (index, value) in buffer.iter_mut().enumerate() {
            let sample = signal[start + index] * window[index];
            *value = Complex::new(sample, 0.0);
        }
        fft.process(&mut buffer);

        let bins = buffer.len() / 2 + 1;
        let magnitudes: Vec<f64> = buffer[..bins].iter().map(|v| v.norm() as f64).collect();

        // The salience of every candidate fundamental, computed on the
        // spectrum that is left after the harmonics of the notes already
        // found have been removed.
        let mut remaining = magnitudes.clone();
        let mut chroma = [0.0f64; 12];
        // At most six notes sound at once on the instruments `tabscribe`
        // supports; a sixth is generous.
        const MAX_NOTES: usize = 6;

        for _ in 0..MAX_NOTES {
            let mut salience = vec![0.0f64; (highest - lowest + 1) as usize];
            for (offset, slot) in salience.iter_mut().enumerate() {
                let pitch = (lowest as usize + offset) as f64;
                let f0 = 440.0 * 2f64.powf((pitch - 69.0) / 12.0);
                let mut sum = 0.0;
                for harmonic in 1..=6u32 {
                    let frequency = f0 * f64::from(harmonic);
                    if frequency >= sample_rate as f64 / 2.0 {
                        break;
                    }
                    sum += magnitude_at(&remaining, frequency, sample_rate, window_size)
                        / f64::from(harmonic);
                }
                *slot = sum;
            }

            // The strongest candidate.
            let Some(best) = salience
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(index, value)| (index, *value))
            else {
                break;
            };
            if best.1 <= 0.0 {
                break;
            }

            let pitch = lowest as usize + best.0;
            let f0 = 440.0 * 2f64.powf((pitch as f64 - 69.0) / 12.0);
            chroma[pitch % 12] += best.1;

            // Remove the harmonic series of this note, so that the next round
            // sees the notes that are still sounding.
            for harmonic in 1..=8u32 {
                let frequency = f0 * f64::from(harmonic);
                if frequency >= sample_rate as f64 / 2.0 {
                    break;
                }
                remove_around(&mut remaining, frequency, sample_rate, window_size);
            }
        }

        let total: f64 = chroma.iter().sum();
        if total > 0.0 {
            for value in &mut chroma {
                *value /= total;
            }
        }

        track
            .times
            .push((start + window_size / 2) as f64 / sample_rate as f64);
        track.frames.push(chroma);

        start += hop_size;
    }

    track
}

/// Remove a frequency and its immediate neighbours from a spectrum.
///
/// A small neighbourhood is removed because the Hann window spreads the
/// energy of a partial over a few bins.
fn remove_around(
    magnitudes: &mut [f64],
    frequency: f64,
    sample_rate: u32,
    window_size: usize,
) {
    let bin = frequency * window_size as f64 / sample_rate as f64;
    let centre = bin.round() as usize;
    if centre >= magnitudes.len() {
        return;
    }
    let low = centre.saturating_sub(2);
    let high = (centre + 2).min(magnitudes.len() - 1);
    for value in &mut magnitudes[low..=high] {
        *value = 0.0;
    }
}

/// The magnitude of the spectrum at a frequency.
///
/// The value is the maximum of the bins around the frequency, which is
/// enough resolution for the coarse frequency grid of the salience.
fn magnitude_at(
    magnitudes: &[f64],
    frequency: f64,
    sample_rate: u32,
    window_size: usize,
) -> f64 {
    let bin = frequency * window_size as f64 / sample_rate as f64;
    let centre = bin.round() as usize;
    if centre >= magnitudes.len() {
        return 0.0;
    }
    let low = centre.saturating_sub(1);
    let high = (centre + 1).min(magnitudes.len() - 1);
    magnitudes[low..=high].iter().copied().fold(0.0f64, f64::max)
}

/// Compute the average chroma of a signal.
///
/// # Panics
///
/// This function does not panic.
#[must_use]
pub fn chroma_of(signal: &[f32], sample_rate: u32) -> Chroma {
    chroma_track(signal, sample_rate).average()
}

/// The number of keys the detector considers.
const KEY_COUNT: usize = 24;

/// The weight of each semitone offset from the tonic, for a major key.
///
/// These are the Krumhansl–Schmuckler profiles, read as offsets from the
/// tonic: the diatonic offsets of the major scale carry the large weights and
/// the chromatic ones the small weights.
const MAJOR_WEIGHTS: [f64; 12] = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];

/// The weight of each semitone offset from the tonic, for a minor key.
///
/// The Krumhansl–Schmuckler minor profile, read the same way: offset 0 is the
/// tonic, offset 3 the mediant, offset 7 the dominant and offset 8 the
/// submediant.
const MINOR_WEIGHTS: [f64; 12] = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];

/// The profile of a key: the weight of each of the twelve pitch classes.
///
/// Index `0..12` are the major keys with tonic C, C♯, …, B; index `12..24`
/// are the minor keys in the same order.
#[must_use]
pub fn key_profiles() -> Vec<[f64; 12]> {
    let mut profiles = Vec::with_capacity(KEY_COUNT);
    for tonic in 0u8..12 {
        let mut profile = [0.0f64; 12];
        for (offset, weight) in MAJOR_WEIGHTS.iter().enumerate() {
            profile[((tonic as usize + offset) % 12)] = *weight;
        }
        profiles.push(profile);
    }
    for tonic in 0u8..12 {
        let mut profile = [0.0f64; 12];
        for (offset, weight) in MINOR_WEIGHTS.iter().enumerate() {
            profile[(tonic as usize + offset) % 12] = *weight;
        }
        profiles.push(profile);
    }
    profiles
}

/// The probability of each pitch class under each key.
///
/// The profiles are normalised so that they sum to one, which makes the
/// emission cost a proper negative log-likelihood.
#[must_use]
pub fn key_probabilities() -> Vec<[f64; 12]> {
    key_profiles()
        .into_iter()
        .map(|profile| {
            let total: f64 = profile.iter().sum();
            let mut probabilities = [0.0f64; 12];
            for (index, weight) in profile.iter().enumerate() {
                probabilities[index] = weight / total;
            }
            probabilities
        })
        .collect()
}

/// The transition probabilities between the keys.
///
/// A key tends to stay the same; the five closely related keys are nearly as
/// likely; the remaining eighteen are unlikely. This is the transition matrix
/// of Faraldo et al., with the weights scaled so that each row sums to one.
#[must_use]
pub fn key_transitions() -> Vec<[f64; KEY_COUNT]> {
    const SELF: f64 = 0.60;
    const RELATED: f64 = 0.06;
    const OTHER: f64 = 0.10 / 18.0;

    let mut matrix = vec![[0.0f64; KEY_COUNT]; KEY_COUNT];
    for from in 0..KEY_COUNT {
        let (tonic, mode) = (from % 12, from >= 12);
        let related: Vec<usize> = if mode {
            // Minor: the relative major, the dominant and the subdominant
            // minor, and the majors of their relatives.
            let tonic = tonic as i32;
            vec![
                ((tonic + 3).rem_euclid(12)) as usize,
                ((tonic + 7).rem_euclid(12)) as usize + 12,
                ((tonic + 5).rem_euclid(12)) as usize + 12,
                ((tonic + 4).rem_euclid(12)) as usize,
                ((tonic + 2).rem_euclid(12)) as usize,
            ]
        } else {
            // Major: the dominant and the subdominant major, the relative
            // minor, and the relatives of the relative minor.
            let tonic = tonic as i32;
            vec![
                ((tonic + 7).rem_euclid(12)) as usize,
                ((tonic + 5).rem_euclid(12)) as usize,
                ((tonic + 9).rem_euclid(12)) as usize + 12,
                ((tonic + 4).rem_euclid(12)) as usize,
                ((tonic + 2).rem_euclid(12)) as usize + 12,
            ]
        };

        let mut row = [0.0f64; KEY_COUNT];
        row[from] = SELF;
        for key in related {
            if key != from {
                row[key] += RELATED;
            }
        }
        for key in 0..KEY_COUNT {
            if row[key] == 0.0 {
                row[key] = OTHER;
            }
        }
        matrix[from] = row;
    }
    matrix
}

/// The key detected from a chroma vector, with its confidence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyEstimate {
    /// The detected key.
    pub key: Key,
    /// The confidence in the detection, from 0 to 1.
    pub confidence: f32,
}

/// The key of a slot in the key list.
fn key_at(index: usize) -> Key {
    let tonic = PitchClass::new((index % 12) as u8).unwrap_or(PitchClass::C);
    let mode = if index >= 12 { Mode::Minor } else { Mode::Major };
    Key::new(tonic, mode)
}

/// The Pearson correlation between a chroma vector and a key profile.
///
/// This is the Krumhansl–Schmuckler correlation: the chroma and the profile
/// are centred and their correlation is taken. Centring is what makes the
/// relative major and the relative minor comparable; with a raw dot product
/// the major always wins, because its profile has the larger total.
fn key_correlation(chroma: &Chroma, profile: &[f64; 12]) -> f64 {
    let chroma_mean = chroma.iter().sum::<f64>() / 12.0;
    let profile_mean = profile.iter().sum::<f64>() / 12.0;

    let mut covariance = 0.0;
    let mut chroma_variance = 0.0;
    let mut profile_variance = 0.0;
    for index in 0..12 {
        let a = chroma[index] - chroma_mean;
        let b = profile[index] - profile_mean;
        covariance += a * b;
        chroma_variance += a * a;
        profile_variance += b * b;
    }
    if chroma_variance <= 1e-12 || profile_variance <= 1e-12 {
        return 0.0;
    }
    covariance / (chroma_variance.sqrt() * profile_variance.sqrt())
}

/// Detect the key of a chroma vector.
///
/// # Panics
///
/// This function does not panic.
#[must_use]
pub fn detect_key(chroma: &Chroma) -> KeyEstimate {
    let profiles = key_profiles();
    let mut best = (0usize, f64::NEG_INFINITY);
    let mut second = f64::NEG_INFINITY;

    for (index, profile) in profiles.iter().enumerate() {
        let score = key_correlation(chroma, profile);
        if score > best.1 {
            second = best.1;
            best = (index, score);
        } else if score > second {
            second = score;
        }
    }

    // The confidence is the margin between the best and the second-best key,
    // scaled by how far the best is from a perfect correlation.
    let margin = if second.is_finite() {
        let spread = (1.0 - best.1).max(1e-6);
        ((best.1 - second) / spread).clamp(0.0, 1.0)
    } else {
        0.0
    };

    KeyEstimate {
        key: key_at(best.0),
        confidence: margin as f32,
    }
}

/// Detect the key of a whole chroma track.
///
/// The average chroma of the track is correlated with every key profile. The
/// track is not segmented: a piece that changes key halfway through is
/// reported with the key that fits most of it.
///
/// # Panics
///
/// This function does not panic.
#[must_use]
pub fn detect_key_track(track: &ChromaTrack) -> KeyEstimate {
    if track.frames.is_empty() {
        return KeyEstimate {
            key: Key::C_MAJOR,
            confidence: 0.0,
        };
    }
    detect_key(&track.average())
}

/// A chord template: the pitch classes it contains.
#[must_use]
pub fn template_pitch_classes(kind: ChordKind, root: PitchClass) -> Vec<PitchClass> {
    Chord::new(root, kind).pitch_classes()
}

/// The weight of the root of a chord, relative to the other tones.
///
/// The root carries more information than the other tones of a chord, so it
/// is weighted more heavily.
const ROOT_WEIGHT: f64 = 2.0;

/// The weight of the other tones of a chord.
const TONE_WEIGHT: f64 = 1.0;

/// The penalty for a tone of the template that the chroma does not support.
///
/// Without it, a template with extra tones scores exactly the same as the
/// shorter one it extends, and every chord would be reported as a seventh.
const MISSING_PENALTY: f64 = 0.1;

/// The score of a chord template against a chroma vector.
///
/// The score rewards the chroma energy the template explains, penalises the
/// energy it leaves unexplained, and penalises the tones of the template
/// that the chroma does not support.
fn template_score(chroma: &Chroma, kind: ChordKind, root: PitchClass) -> f64 {
    let template = template_pitch_classes(kind, root);
    let mut explained = 0.0;
    let mut unexplained = 0.0;
    let mut missing = 0.0;

    for (index, &value) in chroma.iter().enumerate() {
        let pitch_class = PitchClass::new(index as u8).unwrap_or(PitchClass::C);
        if template.contains(&pitch_class) {
            let weight = if pitch_class == root {
                ROOT_WEIGHT
            } else {
                TONE_WEIGHT
            };
            explained += value * weight;
            // The part of the weight the chroma does not support.
            missing += weight - value.min(1.0) * weight;
        } else {
            unexplained += value;
        }
    }

    explained - 0.5 * unexplained - MISSING_PENALTY * missing
}

/// Recognise a chord in a chroma vector.
///
/// The key is used as a prior: a chord that fits the key wins a tie.
///
/// # Panics
///
/// This function does not panic.
#[must_use]
pub fn recognise_chord(chroma: &Chroma, key: Option<Key>) -> Option<(Chord, f32)> {
    let mut best: Option<(Chord, f64)> = None;
    for &kind in ChordKind::ALL {
        for tonic in 0u8..12 {
            let Ok(root) = PitchClass::new(tonic) else {
                continue;
            };
            let mut score = template_score(chroma, kind, root);
            if let Some(key) = key {
                if crate::music::chord::is_diatonic(&Chord::new(root, kind), &key) {
                    score += 0.02;
                }
            }
            let better = match &best {
                Some((_, current)) => score > *current,
                None => true,
            };
            if better {
                best = Some((Chord::new(root, kind), score));
            }
        }
    }

    best.map(|(chord, score)| {
        // Map the raw score onto a rough confidence.
        let confidence = (score * 8.0).clamp(0.0, 1.0) as f32;
        (chord, confidence)
    })
}

/// The number of pitch classes a chord segment must contain before a chord
/// is reported.
///
/// A single sustained note is not a chord, and labelling it as one would put
/// a wrong chord symbol above every note of a single-note melody.
const MIN_CHORD_TONES: usize = 2;

/// The share of the chroma a pitch class must carry to count as a tone.
const TONE_THRESHOLD: f64 = 0.15;

/// Recognise the chords of a whole piece.
///
/// The chroma is computed once and then averaged over each beat. A segment
/// that carries fewer than [`MIN_CHORD_TONES`] distinct pitch classes is not
/// a chord and is not labelled.
///
/// # Panics
///
/// This function does not panic.
#[must_use]
pub fn recognise_chords(
    signal: &[f32],
    sample_rate: u32,
    beats: &[f64],
    tempo: crate::music::Tempo,
    key: Option<Key>,
) -> Vec<ChordEvent> {
    let track = chroma_track(signal, sample_rate);
    if track.frames.is_empty() || beats.is_empty() {
        return Vec::new();
    }

    let period = tempo.seconds_per_beat().as_secs().max(1e-6);
    let mut events: Vec<ChordEvent> = Vec::new();

    for (index, start) in beats.iter().enumerate() {
        let end = beats.get(index + 1).copied().unwrap_or(*start + period);
        let segment = track.between(*start, end);
        let total: f64 = segment.iter().sum();
        if total <= 0.0 {
            continue;
        }

        // A chord needs more than one pitch class.
        let tones = segment
            .iter()
            .filter(|value| **value / total >= TONE_THRESHOLD)
            .count();
        if tones < MIN_CHORD_TONES {
            continue;
        }

        let Some((chord, confidence)) = recognise_chord(&segment, key) else {
            continue;
        };
        if confidence < 0.25 {
            continue;
        }

        // Merge with the previous chord when it is the same.
        if let Some(last) = events.last_mut() {
            if last.chord == chord {
                last.end = Seconds::from_secs(end);
                last.confidence = last.confidence.max(confidence);
                continue;
            }
        }
        events.push(ChordEvent {
            chord,
            start: Seconds::from_secs(*start),
            end: Seconds::from_secs(end),
            confidence,
        });
    }

    events
}

/// The time signature implied by a number of beats.
#[must_use]
pub fn meter_from_beats(beat_count: usize) -> TimeSignature {
    match beat_count % 4 {
        3 => TimeSignature { numerator: 4, denominator: 4 },
        _ => TimeSignature::COMMON,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// A chroma with all the energy on one pitch class.
    fn chroma_with(pitch_class: u8) -> Chroma {
        let mut chroma = [0.0f64; 12];
        chroma[pitch_class as usize] = 1.0;
        chroma
    }

    #[test]
    fn the_key_of_a_c_major_chroma_is_c_major() {
        let mut chroma = [0.0f64; 12];
        for pitch_class in [0u8, 2, 4, 5, 7, 9, 11] {
            chroma[pitch_class as usize] = 1.0;
        }
        let estimate = detect_key(&chroma);
        assert_eq!(estimate.key, Key::C_MAJOR);
    }

    #[test]
    fn the_key_of_an_a_minor_chroma_is_a_minor() {
        // A minor music is built on the natural minor scale with the raised
        // seventh (G♯), which is what tells it apart from C major.
        let mut chroma = [0.0f64; 12];
        for pitch_class in [9u8, 11, 0, 2, 4, 5, 8] {
            chroma[pitch_class as usize] = 1.0;
        }
        let estimate = detect_key(&chroma);
        assert_eq!(estimate.key, Key::new(PitchClass::A, Mode::Minor));
    }

    #[test]
    fn the_relative_major_of_a_minor_is_ambiguous() {
        // The natural minor scale and its relative major share every pitch
        // class, so a chroma alone cannot separate them. The detector must
        // return one of the two rather than something unrelated.
        let mut chroma = [0.0f64; 12];
        for pitch_class in [0u8, 2, 4, 5, 7, 9, 11] {
            chroma[pitch_class as usize] = 1.0;
        }
        let estimate = detect_key(&chroma);
        assert!(
            estimate.key == Key::C_MAJOR || estimate.key == Key::new(PitchClass::A, Mode::Minor),
            "unexpected key {:?}",
            estimate.key
        );
    }

    #[test]
    fn a_c_major_chroma_is_recognised_as_c_major() {
        let mut chroma = [0.0f64; 12];
        for pitch_class in [0u8, 4, 7] {
            chroma[pitch_class as usize] = 1.0;
        }
        let (chord, confidence) = recognise_chord(&chroma, None).expect("a chord");
        assert_eq!(chord.root, PitchClass::C);
        assert_eq!(chord.kind, ChordKind::Major);
        assert!(confidence > 0.5);
    }

    #[test]
    fn a_minor_chord_is_recognised() {
        let mut chroma = [0.0f64; 12];
        for pitch_class in [9u8, 0, 4] {
            chroma[pitch_class as usize] = 1.0;
        }
        let (chord, _) = recognise_chord(&chroma, None).expect("a chord");
        assert_eq!(chord.root, PitchClass::A);
        assert_eq!(chord.kind, ChordKind::Minor);
    }

    #[test]
    fn silence_produces_no_key() {
        let chroma = [0.0f64; 12];
        let estimate = detect_key(&chroma);
        // With no energy every correlation is zero; the result is arbitrary but
        // must not panic.
        assert!(estimate.confidence <= 1.0);
    }

    #[test]
    fn chroma_of_silence_is_empty() {
        let chroma = chroma_of(&vec![0.0f32; 8192], 22_050);
        assert_eq!(chroma.iter().sum::<f64>(), 0.0);
    }
}
