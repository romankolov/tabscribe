//! Rhythm analysis: onsets, tempo, beats, meter and quantization.
//!
//! # The pipeline
//!
//! 1. The signal is transformed with an STFT and the **spectral flux** is
//!    computed: the sum of the positive changes of the magnitude spectrum.
//!    The flux is then smoothed into an **onset envelope**.
//! 2. The **tempo** is estimated from the autocorrelation of the onset
//!    envelope (the "tempogram"), which is robust and cheap.
//! 3. The **beats** are placed with a dynamic-programming beat tracker in the
//!    style of Ellis: each candidate beat is scored by the onset energy at
//!    that position plus the best score of a plausible predecessor, with a
//!    penalty for deviating from the expected beat period.
//! 4. The **meter** is chosen by testing which time signature makes the
//!    detected beats and the onset envelope agree best.
//! 5. The **quantization** snaps the note onsets to a grid, but only when the
//!    note is close enough to it: an expressive performance should not be
//!    flattened.

use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::music::{Seconds, Tempo, TimeSignature};
use crate::transcription::NoteEvent;

/// The configuration of the spectral analysis.
#[derive(Debug, Clone, Copy)]
pub struct SpectrumConfig {
    /// The size of the analysis window, in samples.
    pub window_size: usize,
    /// The distance between two windows, in samples.
    pub hop_size: usize,
}

impl SpectrumConfig {
    /// Derive a configuration from the sample rate.
    ///
    /// The window is about 93 ms and the hop about 11.6 ms, which resolves
    /// sixteenth notes at 120 BPM with two analysis frames.
    #[must_use]
    pub fn for_sample_rate(sample_rate: u32) -> Self {
        let window_size = ((sample_rate as f64) * 0.0929).round() as usize / 2 * 2;
        let window_size = window_size.clamp(512, 4096);
        let hop_size = ((sample_rate as f64) * 0.0116).round() as usize / 2 * 2;
        SpectrumConfig {
            window_size,
            hop_size: hop_size.max(1),
        }
    }
}

/// The onset envelope of a signal.
#[derive(Debug, Clone, Default)]
pub struct OnsetEnvelope {
    /// The time of each frame, in seconds.
    pub times: Vec<f64>,
    /// The onset strength of each frame.
    pub values: Vec<f32>,
    /// The time between two frames, in seconds.
    pub hop_seconds: f64,
}

impl OnsetEnvelope {
    /// The onset strength at a given time, linearly interpolated.
    #[must_use]
    pub fn at(&self, time: f64) -> f32 {
        if self.values.is_empty() {
            return 0.0;
        }
        let position = time / self.hop_seconds;
        let index = position.floor();
        if index < 0.0 {
            return self.values[0];
        }
        let index = index as usize;
        if index + 1 >= self.values.len() {
            return *self.values.last().unwrap_or(&0.0);
        }
        let fraction = (position - index as f64) as f32;
        self.values[index] * (1.0 - fraction) + self.values[index + 1] * fraction
    }

    /// The maximum onset strength.
    #[must_use]
    pub fn max(&self) -> f32 {
        self.values.iter().copied().fold(0.0f32, f32::max)
    }
}

/// The magnitude spectrum of a windowed frame.
fn magnitude_spectrum(
    frame: &[f32],
    window: &[f32],
    fft: &Arc<dyn rustfft::Fft<f32>>,
    buffer: &mut [Complex<f32>],
) -> Vec<f32> {
    for (index, value) in buffer.iter_mut().enumerate() {
        let sample = if index < frame.len() {
            frame[index] * window[index]
        } else {
            0.0
        };
        *value = Complex::new(sample, 0.0);
    }
    fft.process(buffer);
    let bins = buffer.len() / 2 + 1;
    buffer[..bins].iter().map(|value| value.norm()).collect()
}

/// Compute the onset envelope of a signal.
///
/// # Panics
///
/// This function does not panic. A signal shorter than one window produces an
/// empty envelope.
#[must_use]
pub fn onset_envelope(signal: &[f32], sample_rate: u32, config: &SpectrumConfig) -> OnsetEnvelope {
    let mut envelope = OnsetEnvelope {
        times: Vec::new(),
        values: Vec::new(),
        hop_seconds: config.hop_size as f64 / sample_rate as f64,
    };
    if signal.len() < config.window_size || sample_rate == 0 {
        return envelope;
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(config.window_size);
    let mut buffer = vec![Complex::new(0.0f32, 0.0f32); config.window_size];
    let window: Vec<f32> = (0..config.window_size)
        .map(|i| {
            let n = i as f64;
            let denominator = (config.window_size - 1) as f64;
            (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n / denominator).cos()) as f32
        })
        .collect();

    let mut previous: Vec<f32> = Vec::new();
    let mut start = 0usize;
    while start + config.window_size <= signal.len() {
        let frame = &signal[start..start + config.window_size];
        let spectrum = magnitude_spectrum(frame, &window, &fft, &mut buffer);

        let flux: f32 = if previous.is_empty() {
            0.0
        } else {
            spectrum
                .iter()
                .zip(previous.iter())
                .map(|(now, before)| (now - before).max(0.0))
                .sum()
        };

        envelope
            .times
            .push((start + config.window_size / 2) as f64 / sample_rate as f64);
        envelope.values.push(flux);
        previous = spectrum;

        start += config.hop_size;
    }

    // Smooth the flux with a short moving average: the raw flux is very
    // spiky, and the peak picker works much better on a smoothed curve.
    smooth_in_place(&mut envelope.values, 3);
    envelope
}

/// Smooth a series with a moving average of the given radius.
fn smooth_in_place(values: &mut [f32], radius: usize) {
    if radius == 0 || values.len() < 2 {
        return;
    }
    let original = values.to_vec();
    let n = values.len();
    for (index, value) in values.iter_mut().enumerate() {
        let low = index.saturating_sub(radius);
        let high = (index + radius + 1).min(n);
        let sum: f32 = original[low..high].iter().sum();
        *value = sum / (high - low) as f32;
    }
}

/// Detect the onsets of a signal.
///
/// The peak picker uses an adaptive threshold: a local mean plus a multiple of
/// the local standard deviation, so that a quiet passage is not flooded with
/// onsets and a loud one does not lose them.
///
/// # Panics
///
/// This function does not panic.
#[must_use]
pub fn detect_onsets(envelope: &OnsetEnvelope, sensitivity: f32) -> Vec<f64> {
    if envelope.values.is_empty() {
        return Vec::new();
    }
    let peak = envelope.max();
    if peak <= 0.0 {
        return Vec::new();
    }

    // A more sensitive setting lowers the threshold.
    let delta = 0.30 - 0.25 * sensitivity.clamp(0.0, 1.0);
    let window = 8usize;

    let mut onsets = Vec::new();
    let mut last_onset = f64::NEG_INFINITY;
    for index in 1..envelope.values.len().saturating_sub(1) {
        let value = envelope.values[index];
        if value < envelope.values[index - 1] || value < envelope.values[index + 1] {
            continue;
        }
        let low = index.saturating_sub(window);
        let high = (index + window + 1).min(envelope.values.len());
        let neighbourhood = &envelope.values[low..high];
        let mean = neighbourhood.iter().sum::<f32>() / neighbourhood.len() as f32;
        let variance = neighbourhood
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f32>()
            / neighbourhood.len() as f32;
        let threshold = mean + delta * variance.sqrt();

        if value < threshold || value < peak * 0.05 {
            continue;
        }
        let time = envelope.times[index];
        // Onsets closer together than 50 ms are the same event.
        if time - last_onset < 0.05 {
            continue;
        }
        last_onset = time;
        onsets.push(time);
    }
    onsets
}

/// The estimated tempo and the confidence in it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TempoEstimate {
    /// The estimated tempo.
    pub tempo: Tempo,
    /// The confidence in the estimate, from 0 to 1.
    pub confidence: f32,
    /// The phase of the first beat, in seconds.
    ///
    /// The phase is always zero: the beat tracker derives the phase from the
    /// onsets themselves, so there is no separate estimate to carry.
    pub phase: f64,
}

/// The tempo around which the prior is centred, in beats per minute.
///
/// Musicians tend to write in a band around 120 BPM, and a piece in 3/4 or
/// 4/4 has a strong periodicity at the length of the bar, which can easily
/// be stronger than the periodicity of the beat. The prior is what keeps the
/// estimate on the beat.
const TEMPO_PRIOR_CENTRE: f64 = 90.0;

/// The width of the prior, in octaves.
const TEMPO_PRIOR_WIDTH: f64 = 1.0;

/// The weight of a tempo, from a log-normal prior.
///
/// The prior is centred on [`TEMPO_PRIOR_CENTRE`] and is one octave wide, so
/// a tempo half or twice as fast is still well within the band.
///
/// # Panics
///
/// This function does not panic.
#[must_use]
pub fn tempo_prior(bpm: f64) -> f64 {
    if bpm <= 0.0 {
        return 0.0;
    }
    let octaves = (bpm / TEMPO_PRIOR_CENTRE).log2();
    (-0.5 * (octaves / TEMPO_PRIOR_WIDTH).powi(2)).exp()
}

/// Estimate the tempo of an onset envelope.
///
/// The autocorrelation of the envelope is searched for its strongest peak in
/// the range of plausible tempi. The search is restricted to 40–220 BPM, the
/// range in which music is actually written.
///
/// The peak is chosen with a prior in mind: the autocorrelation of a piece in
/// a compound meter is often stronger at the length of the bar than at the
/// length of the beat, and the prior is what keeps the estimate on the beat
/// rather than on the bar.
///
/// # Panics
///
/// This function does not panic.
#[must_use]
pub fn estimate_tempo(envelope: &OnsetEnvelope) -> TempoEstimate {
    let fallback = TempoEstimate {
        tempo: Tempo::DEFAULT,
        confidence: 0.0,
        phase: 0.0,
    };
    if envelope.values.len() < 16 || envelope.hop_seconds <= 0.0 {
        return fallback;
    }

    let min_bpm = 40.0;
    let max_bpm = 220.0;
    let min_lag = (60.0 / max_bpm / envelope.hop_seconds).floor() as usize;
    let max_lag = (60.0 / min_bpm / envelope.hop_seconds).ceil() as usize;
    if min_lag == 0 || max_lag <= min_lag {
        return fallback;
    }

    // Autocorrelation of the (mean-removed) envelope.
    let values: Vec<f64> = envelope.values.iter().map(|&v| f64::from(v)).collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let centered: Vec<f64> = values.iter().map(|v| v - mean).collect();

    let mut best_lag = min_lag;
    let mut best_score = f64::NEG_INFINITY;
    let mut best_raw = 0.0;
    let mut scores: Vec<(usize, f64)> = Vec::new();
    for lag in min_lag..=max_lag.min(centered.len() / 2) {
        let mut sum = 0.0;
        for index in 0..(centered.len() - lag) {
            sum += centered[index] * centered[index + lag];
        }
        let raw = sum / (centered.len() - lag) as f64;
        let bpm = 60.0 / (lag as f64 * envelope.hop_seconds);
        let score = raw * tempo_prior(bpm);
        scores.push((lag, raw));
        if score > best_score {
            best_score = score;
            best_raw = raw;
            best_lag = lag;
        }
    }

    // The confidence is the ratio of the best peak to the mean of the curve.
    let mean_score = scores.iter().map(|(_, s)| *s).sum::<f64>() / scores.len().max(1) as f64;
    let confidence = if mean_score.abs() > 1e-12 {
        ((best_raw / mean_score) - 1.0).clamp(0.0, 4.0) / 4.0
    } else {
        0.0
    };

    let period = best_lag as f64 * envelope.hop_seconds;
    let bpm = 60.0 / period;

    // Keep the tempo in the band in which music is actually written.
    let mut bpm = bpm;
    while bpm < 60.0 && bpm * 2.0 <= max_bpm {
        bpm *= 2.0;
    }
    while bpm > 200.0 && bpm / 2.0 >= min_bpm {
        bpm /= 2.0;
    }

    // Prefer the slower tempo when there's a strong 2:1 relationship.
    // The onset envelope often has strong periodicity at both the beat
    // level and the subdivision level (e.g. eighth notes in an arpeggio).
    // When autocorrelation finds a peak at 2x the true tempo, the prior
    // alone may not be enough to pull it back down, so we explicitly
    // check: if half the current tempo has a comparable autocorrelation
    // peak, choose the slower tempo (musical convention).
    let half_bpm = bpm / 2.0;
    if half_bpm >= min_bpm {
        let half_lag = (60.0 / half_bpm / envelope.hop_seconds).round() as usize;
        if half_lag > 0 && half_lag < centered.len() / 2 {
            let mut half_sum = 0.0;
            for index in 0..(centered.len() - half_lag) {
                half_sum += centered[index] * centered[index + half_lag];
            }
            let half_raw = half_sum / (centered.len() - half_lag) as f64;
            // If the half-tempo peak is at least 60% as strong, prefer it.
            if half_raw >= best_raw * 0.6 {
                bpm = half_bpm;
            }
        }
    }

    let tempo = Tempo::from_bpm(bpm).unwrap_or(Tempo::DEFAULT);

    TempoEstimate {
        tempo,
        confidence: confidence as f32,
        phase: 0.0,
    }
}

/// Track the beats of an onset envelope.
///
/// The tracker is the dynamic program of Ellis (2007). Every frame of the
/// onset envelope is a beat candidate. A candidate is scored by the onset
/// energy at its position plus the best score of a plausible predecessor,
/// with a penalty for departing from the expected beat period:
///
/// ```text
/// score(i) = onset(i) + max over j { score(j) − penalty · (ln(Δ/period))² }
/// ```
///
/// The predecessors are restricted to the frames between half a period and
/// two periods earlier, which is what keeps the track from collapsing or
/// doubling.
///
/// # Panics
///
/// This function does not panic.
#[must_use]
pub fn track_beats(envelope: &OnsetEnvelope, tempo: Tempo) -> Vec<f64> {
    if envelope.values.is_empty() {
        return Vec::new();
    }
    let period = tempo.seconds_per_beat().as_secs();
    if period <= 0.0 {
        return Vec::new();
    }

    // The onset weight of a candidate. The log compression keeps a very loud
    // transient from dominating the score.
    let weight = |value: f32| -> f64 { (1.0 + f64::from(value)).ln() };

    const PENALTY: f64 = 100.0;

    let n = envelope.values.len();
    let mut scores = vec![f64::NEG_INFINITY; n];
    let mut backpointers = vec![0usize; n];

    for index in 0..n {
        let onset = weight(envelope.values[index]);
        if index == 0 {
            scores[0] = onset;
            backpointers[0] = 0;
            continue;
        }

        let time = envelope.times[index];
        // The plausible predecessors are between half a period and two
        // periods earlier.
        let low_time = time - 2.0 * period;
        let high_time = time - period / 2.0;
        let mut best = f64::NEG_INFINITY;
        let mut best_index = 0usize;
        for previous in (0..index).rev() {
            let previous_time = envelope.times[previous];
            if previous_time < low_time {
                break;
            }
            if previous_time > high_time {
                continue;
            }
            if !scores[previous].is_finite() {
                continue;
            }
            let gap = time - previous_time;
            let deviation = (gap / period).ln();
            let candidate = scores[previous] - PENALTY * deviation * deviation + onset;
            if candidate > best {
                best = candidate;
                best_index = previous;
            }
        }

        if best.is_finite() {
            scores[index] = best;
            backpointers[index] = best_index;
        } else {
            // No predecessor fits: the candidate starts a new track.
            scores[index] = onset;
            backpointers[index] = index;
        }
    }

    // The best track ends at the frame with the highest score.
    let mut current = (0..n)
        .max_by(|&a, &b| {
            scores[a]
                .partial_cmp(&scores[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);

    let mut beats = Vec::new();
    while scores[current].is_finite() {
        beats.push(envelope.times[current]);
        let previous = backpointers[current];
        if previous == current {
            break;
        }
        current = previous;
    }
    beats.reverse();
    beats
}

/// The candidate time signatures, in the order they are tested.
const CANDIDATE_METERS: &[TimeSignature] = &[
    TimeSignature {
        numerator: 4,
        denominator: 4,
    },
    TimeSignature {
        numerator: 3,
        denominator: 4,
    },
    TimeSignature {
        numerator: 2,
        denominator: 4,
    },
    TimeSignature {
        numerator: 6,
        denominator: 8,
    },
];

/// Detect the time signature of a piece.
///
/// Every candidate meter is scored by how much onset energy falls on the
/// strong beats of the meter; the best one wins.
///
/// # Panics
///
/// This function does not panic.
#[must_use]
pub fn detect_meter(envelope: &OnsetEnvelope, beats: &[f64], tempo: Tempo) -> TimeSignature {
    if beats.len() < 4 {
        return TimeSignature::COMMON;
    }
    let period = tempo.seconds_per_beat().as_secs();
    if period <= 0.0 || envelope.hop_seconds <= 0.0 {
        return TimeSignature::COMMON;
    }

    let values: Vec<f64> = envelope.values.iter().map(|&v| f64::from(v)).collect();
    let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
    let centered: Vec<f64> = values.iter().map(|v| v - mean).collect();

    // The onset strength at every tracked beat.
    let accents: Vec<f64> = beats.iter().map(|&b| f64::from(envelope.at(b))).collect();

    let mut best = TimeSignature::COMMON;
    let mut best_score = f64::NEG_INFINITY;

    for meter in CANDIDATE_METERS {
        let beats_per_bar = meter.beats_per_bar();
        let length = beats_per_bar.round() as usize;
        if length < 2 {
            continue;
        }

        // The periodicity of the onsets at the length of the bar.
        let lag = ((beats_per_bar * period) / envelope.hop_seconds).round() as usize;
        if lag == 0 || lag >= centered.len() {
            continue;
        }
        let periodicity = autocorrelation(&centered, lag);

        // The downbeat has to be the strongest beat of the bar.
        let contrast = accent_contrast(&accents, length);
        if contrast <= 0.0 {
            continue;
        }

        let score = periodicity * (1.0 + contrast);
        if score > best_score {
            best_score = score;
            best = *meter;
        }
    }
    best
}

/// The autocorrelation of a series at a given lag.
///
/// The series is assumed to have its mean removed already.
fn autocorrelation(values: &[f64], lag: usize) -> f64 {
    if lag == 0 || lag >= values.len() {
        return 0.0;
    }
    let mut sum = 0.0;
    for index in 0..(values.len() - lag) {
        sum += values[index] * values[index + lag];
    }
    sum / (values.len() - lag) as f64
}

/// The contrast of the downbeat against the other beats of the bar.
///
/// The best phase is chosen, and the contrast is how much stronger the
/// downbeat is than the other beats of the bar, relative to the average
/// strength of the bar.
fn accent_contrast(accents: &[f64], length: usize) -> f64 {
    if accents.len() < 2 * length || length < 2 {
        return 0.0;
    }

    let mut best = f64::NEG_INFINITY;
    for offset in 0..length {
        let mut profile = vec![0.0f64; length];
        let mut counts = vec![0usize; length];
        for (index, value) in accents.iter().enumerate() {
            let position = (index + length - offset) % length;
            profile[position] += value;
            counts[position] += 1;
        }
        for (position, count) in counts.iter().enumerate() {
            if *count > 0 {
                profile[position] /= *count as f64;
            }
        }

        // The downbeat has to be the strongest beat of the bar; otherwise
        // this is not the right phase.
        let strongest = profile.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if profile[0] < strongest {
            continue;
        }

        let others: f64 = profile[1..].iter().sum::<f64>() / (length - 1) as f64;
        let average = profile.iter().sum::<f64>() / length as f64;
        if average <= 0.0 {
            continue;
        }
        let contrast = (profile[0] - others) / average;
        if contrast > best {
            best = contrast;
        }
    }

    if best.is_finite() {
        best
    } else {
        0.0
    }
}

/// The score of a candidate meter: the energy on the strong beats.
/// The window around a note boundary that is searched for an onset, in
/// seconds.
const REFINEMENT_WINDOW: f64 = 0.12;

/// How much stronger than its surroundings an onset has to be before it
/// moves a note boundary.
const REFINEMENT_PROMINENCE: f32 = 1.5;

/// Refine the note boundaries with the onset envelope.
///
/// The pitch detectors work on windows of a couple of hundred milliseconds,
/// so a note boundary is known only to within half a window. The onset
/// envelope says much more precisely where the attack was, so each boundary
/// is moved to the strongest onset nearby.
///
/// A boundary is only moved when there is a clear onset there. A note that
/// fades out has no attack nearby, and its boundary stays where it was.
///
/// # Panics
///
/// This function does not panic.
#[must_use]
pub fn refine_boundaries(notes: &[NoteEvent], envelope: &OnsetEnvelope) -> Vec<NoteEvent> {
    if notes.is_empty() || envelope.values.is_empty() {
        return notes.to_vec();
    }
    notes
        .iter()
        .map(|note| {
            let mut refined = note.clone();
            let start = note.start.as_secs();
            let end = note.end.as_secs();

            if let Some(onset) = strongest_onset_near(envelope, start) {
                if onset < end {
                    refined.start = Seconds::from_secs(onset);
                }
            }
            refined
        })
        .collect()
}

/// The time of the strongest onset near a moment.
///
/// The onset has to be clearly stronger than the energy around it, otherwise
/// the boundary of a fading note would be dragged to a random peak of the
/// noise.
fn strongest_onset_near(envelope: &OnsetEnvelope, time: f64) -> Option<f64> {
    let low = time - REFINEMENT_WINDOW;
    let high = time + REFINEMENT_WINDOW;

    let mut best_time = None;
    let mut best_value = f32::MIN;
    for (index, value) in envelope.values.iter().enumerate() {
        let candidate = envelope.times[index];
        if candidate < low || candidate > high {
            continue;
        }
        if *value > best_value {
            best_value = *value;
            best_time = Some(candidate);
        }
    }

    let best_value = best_value.max(0.0);
    if best_value <= 0.0 {
        return None;
    }

    // The energy around the moment, for the prominence test.
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for (index, value) in envelope.values.iter().enumerate() {
        let candidate = envelope.times[index];
        if candidate < time - 0.3 || candidate > time + 0.3 {
            continue;
        }
        sum += *value;
        count += 1;
    }
    if count == 0 {
        return None;
    }
    let surroundings = sum / count as f32;

    if best_value < surroundings * REFINEMENT_PROMINENCE {
        return None;
    }
    best_time
}

/// Split the notes at the detected onsets.
///
/// A pitch that is played again after a chord change sounds like one long
/// note to a pitch detector, because the pitch never changes. The onsets are
/// what tell the two apart: every onset inside a note is a place where the
/// note was re-articulated, and the note is cut there.
///
/// The pieces keep the original confidence, so a split note is still as
/// trustworthy as the note it came from.
///
/// # Panics
///
/// This function does not panic.
#[must_use]
pub fn split_at_onsets(notes: &[NoteEvent], onsets: &[f64]) -> Vec<NoteEvent> {
    if onsets.is_empty() || notes.is_empty() {
        return notes.to_vec();
    }
    let mut result: Vec<NoteEvent> = Vec::with_capacity(notes.len());

    for note in notes {
        let start = note.start.as_secs();
        let end = note.end.as_secs();
        let mut cuts: Vec<f64> = onsets
            .iter()
            .copied()
            .filter(|&onset| onset > start + 0.01 && onset < end - 0.01)
            .collect();
        cuts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut cursor = start;
        for cut in cuts {
            if cut <= cursor {
                continue;
            }
            let mut first = note.clone();
            first.start = Seconds::from_secs(cursor);
            first.end = Seconds::from_secs(cut);
            result.push(first);
            cursor = cut;
        }

        let mut last = note.clone();
        last.start = Seconds::from_secs(cursor);
        result.push(last);
    }

    result.sort_by(|a, b| {
        a.start
            .as_secs()
            .partial_cmp(&b.start.as_secs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    result
}

/// How aggressively the onsets are snapped to the grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizeMode {
    /// Every onset is snapped to the grid.
    Strict,
    /// Only onsets that are close to the grid are snapped.
    Musical,
    /// The onsets keep their performed timing.
    Off,
}

/// Quantize a position in beats to a grid.
///
/// The grid is given in beats per step, so a sixteenth-note grid is `0.25`.
///
/// # Panics
///
/// This function does not panic; a grid of zero leaves the position alone.
#[must_use]
pub fn quantize(beat: f64, grid: f64, mode: QuantizeMode) -> f64 {
    if grid <= 0.0 || mode == QuantizeMode::Off {
        return beat;
    }
    let snapped = (beat / grid).round() * grid;
    match mode {
        QuantizeMode::Strict => snapped,
        QuantizeMode::Musical => {
            // Only snap when the note is within a sixth of the grid step.
            let deviation = (beat - snapped).abs();
            if deviation <= grid / 6.0 {
                snapped
            } else {
                beat
            }
        }
        QuantizeMode::Off => beat,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// A click track at the given tempo.
    fn click_track(bpm: f64, seconds: f64, sample_rate: u32) -> Vec<f32> {
        let period = (60.0 / bpm * sample_rate as f64) as usize;
        let mut signal = vec![0.0f32; (seconds * sample_rate as f64) as usize];
        let mut position = 0usize;
        while position < signal.len() {
            let length = (0.005 * sample_rate as f64) as usize;
            for offset in 0..length {
                if position + offset < signal.len() {
                    let t = offset as f64 / length as f64;
                    signal[position + offset] =
                        (t * std::f64::consts::PI).sin() as f32 * (1.0 - t) as f32;
                }
            }
            position += period;
        }
        signal
    }

    #[test]
    fn onsets_are_detected_at_the_click_positions() {
        let sample_rate = 22_050;
        let signal = click_track(120.0, 4.0, sample_rate);
        let config = SpectrumConfig::for_sample_rate(sample_rate);
        let envelope = onset_envelope(&signal, sample_rate, &config);
        let onsets = detect_onsets(&envelope, 0.5);
        assert!(
            onsets.len() >= 7,
            "expected about 8 onsets, got {}",
            onsets.len()
        );
        for onset in &onsets {
            let nearest = (onset / 0.5).round() * 0.5;
            assert!(
                (onset - nearest).abs() < 0.06,
                "onset at {onset} is not on the grid"
            );
        }
    }

    #[test]
    fn the_tempo_of_a_click_track_is_estimated() {
        let sample_rate = 22_050;
        let signal = click_track(120.0, 6.0, sample_rate);
        let config = SpectrumConfig::for_sample_rate(sample_rate);
        let envelope = onset_envelope(&signal, sample_rate, &config);
        let estimate = estimate_tempo(&envelope);
        assert!(
            (estimate.tempo.bpm() - 120.0).abs() < 3.0,
            "expected 120 BPM, got {}",
            estimate.tempo.bpm()
        );
    }

    #[test]
    fn beats_are_tracked_at_the_right_rate() {
        let sample_rate = 22_050;
        let signal = click_track(90.0, 8.0, sample_rate);
        let config = SpectrumConfig::for_sample_rate(sample_rate);
        let envelope = onset_envelope(&signal, sample_rate, &config);
        let estimate = estimate_tempo(&envelope);
        let beats = track_beats(&envelope, estimate.tempo);
        assert!(
            beats.len() >= 10,
            "expected about 12 beats, got {}",
            beats.len()
        );
        let gaps: Vec<f64> = beats.windows(2).map(|w| w[1] - w[0]).collect();
        let average = gaps.iter().sum::<f64>() / gaps.len() as f64;
        assert!(
            (average - 60.0 / 90.0).abs() < 0.05,
            "expected a gap of 0.667 s, got {average}"
        );
    }

    #[test]
    fn quantization_respects_the_mode() {
        assert_eq!(quantize(0.26, 0.25, QuantizeMode::Strict), 0.25);
        assert_eq!(quantize(0.26, 0.25, QuantizeMode::Musical), 0.25);
        // Too far from the grid: the musical mode leaves it alone.
        assert_eq!(quantize(0.4, 0.25, QuantizeMode::Musical), 0.4);
        assert_eq!(quantize(0.4, 0.25, QuantizeMode::Off), 0.4);
    }
}
