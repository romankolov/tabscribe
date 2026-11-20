//! The classical DSP engine: YIN pitch detection with probabilistic smoothing
//! and HMM-based note segmentation.
//!
//! # The algorithm
//!
//! The engine follows the *probabilistic YIN* (pYIN) approach of Mauch &
//! Dixon:
//!
//! 1. The **difference function** `d(τ)` measures how well a lag `τ` explains
//!    the periodicity of a frame. It is computed through the autocorrelation
//!    (Wiener–Khinchin), which makes it `O(n log n)` rather than `O(n τ)`.
//! 2. The **cumulative mean normalized difference** `d'(τ)` removes the
//!    dependence on the signal energy, so that the threshold is meaningful
//!    across loud and quiet passages.
//! 3. Instead of taking the single best lag, pYIN samples from the posterior
//!    distribution over lags. This removes the "octave jump" artefacts of the
//!    plain YIN, because the alternative octave is part of the distribution
//!    rather than a separate decision.
//! 4. The frame-level pitch decisions are then smoothed by a **hidden Markov
//!    model** whose states are the candidate pitches plus a rest state. The
//!    transition cost penalises large pitch leaps, so the segmentation follows
//!    the melodic contour instead of flickering between neighbouring pitches.
//!
//! The sampler uses a fixed-seed generator, so the output of a given input is
//! always identical.

use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::error::{Error, Result};
use crate::music::{MidiNote, Seconds};
use crate::transcribe::{TranscribeOptions, Transcriber};
use crate::transcription::NoteEvent;

/// The configuration of the YIN analysis.
#[derive(Debug, Clone)]
pub struct YinConfig {
    /// The sample rate of the signal.
    pub sample_rate: u32,
    /// The size of the analysis window, in samples.
    pub window_size: usize,
    /// The distance between two analysis windows, in samples.
    pub hop_size: usize,
    /// The absolute threshold applied to the normalized difference, in `(0, 1]`.
    pub threshold: f64,
    /// The lowest frequency to consider, in Hertz.
    pub min_frequency: f64,
    /// The highest frequency to consider, in Hertz.
    pub max_frequency: f64,
}

impl YinConfig {
    /// Derive a configuration from the options.
    #[must_use]
    pub fn from_options(sample_rate: u32, options: &TranscribeOptions) -> Self {
        // A window of about 46 ms at 22 kHz is long enough to resolve two
        // periods of the lowest note and short enough to track fast passages.
        let window_size = ((sample_rate as f64) * 0.046).round() as usize / 2 * 2;
        let window_size = window_size.clamp(512, 4096);
        let hop_size = (window_size / 4).max(1);
        YinConfig {
            sample_rate,
            window_size,
            hop_size,
            threshold: 0.15,
            min_frequency: options.min_frequency,
            max_frequency: options.max_frequency,
        }
    }

    /// The smallest lag that is considered.
    #[must_use]
    pub fn min_lag(&self) -> usize {
        ((self.sample_rate as f64 / self.max_frequency).floor() as usize).max(2)
    }

    /// The largest lag that is considered.
    #[must_use]
    pub fn max_lag(&self) -> usize {
        let lag = (self.sample_rate as f64 / self.min_frequency).ceil() as usize;
        lag.min(self.window_size / 2)
    }
}

/// The pitch found in one frame.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    /// The time of the frame, in seconds.
    pub time: f64,
    /// The fundamental frequency, in Hertz. Zero means "unvoiced".
    pub frequency: f64,
    /// The confidence in the pitch, from 0 to 1.
    pub probability: f64,
    /// The lag that produced the pitch, in samples.
    pub lag: usize,
    /// The posterior over the plausible lags of this frame.
    pub posterior: Vec<Candidate>,
}

/// The results of the frame-level analysis.
#[derive(Debug, Clone, Default)]
pub struct FrameAnalysis {
    /// The frames, in time order.
    pub frames: Vec<Frame>,
}

/// A pair of FFT plans used for the autocorrelation.
struct Autocorrelation {
    size: usize,
    forward: Arc<dyn rustfft::Fft<f32>>,
    inverse: Arc<dyn rustfft::Fft<f32>>,
    buffer: Vec<Complex<f32>>,
}

impl Autocorrelation {
    /// Create the plans for a given window size.
    fn new(window_size: usize) -> Self {
        let size = (window_size * 2).next_power_of_two();
        let mut planner = FftPlanner::<f32>::new();
        Autocorrelation {
            size,
            forward: planner.plan_fft_forward(size),
            inverse: planner.plan_fft_inverse(size),
            buffer: vec![Complex::new(0.0f32, 0.0f32); size],
        }
    }

    /// The autocorrelation of a frame, normalised by the window length.
    fn compute(&mut self, frame: &[f32]) -> Vec<f64> {
        for (index, value) in self.buffer.iter_mut().enumerate() {
            *value = if index < frame.len() {
                Complex::new(frame[index], 0.0)
            } else {
                Complex::new(0.0, 0.0)
            };
        }
        self.forward.process(&mut self.buffer);
        for value in &mut self.buffer {
            let power = value.norm_sqr();
            *value = Complex::new(power, 0.0);
        }
        self.inverse.process(&mut self.buffer);
        let scale = 1.0 / self.size as f32;
        self.buffer
            .iter()
            .map(|value| f64::from(value.re) * f64::from(scale))
            .collect()
    }
}

/// The difference function of a frame, for every lag up to `max_lag`.
///
/// The difference function is
///
/// ```text
/// d(τ) = Σ (x[j] - x[j + τ])²
///      = Σ x[j]² + Σ x[j + τ]² - 2 Σ x[j]·x[j + τ]
/// ```
///
/// The last term is the autocorrelation, which the FFT computes for every lag
/// at once. The first two terms are running sums of the energy.
///
/// Only the first `window - τ` samples contribute to the difference, because
/// the shifted window runs off the end of the frame; this is what makes the
/// function comparable across lags.
fn difference_function(
    frame: &[f32],
    max_lag: usize,
    autocorrelation: &mut Autocorrelation,
) -> Vec<f64> {
    let correlation = autocorrelation.compute(frame);
    let window = frame.len();

    // `prefix[i]` is the energy of the first `i` samples of the frame.
    let mut prefix = vec![0.0f64; window + 1];
    for (index, sample) in frame.iter().enumerate() {
        prefix[index + 1] = prefix[index] + f64::from(*sample) * f64::from(*sample);
    }
    let total = prefix[window];

    let mut difference = vec![0.0f64; max_lag + 1];
    for lag in 0..=max_lag {
        let r = correlation.get(lag).copied().unwrap_or(0.0);
        // The overlapping part of the frame is `x[0..window - lag]`; the
        // shifted part is `x[lag..window]`.
        let head = prefix[window - lag];
        let tail = total - prefix[lag];
        difference[lag] = (head + tail - 2.0 * r).max(0.0);
    }
    difference
}

/// The cumulative mean normalized difference.
fn cumulative_mean_normalized(difference: &[f64]) -> Vec<f64> {
    let mut normalized = vec![0.0f64; difference.len()];
    normalized[0] = 1.0;
    let mut running = 0.0f64;
    for lag in 1..difference.len() {
        running += difference[lag];
        normalized[lag] = if running > 0.0 {
            difference[lag] * lag as f64 / running
        } else {
            1.0
        };
    }
    normalized
}

/// Refine a lag by parabolic interpolation around its local minimum.
fn parabolic_refine(candidate: &Candidate) -> f64 {
    let (x0, x1, x2) = (candidate.left, candidate.center, candidate.right);
    let denominator = 2.0 * (2.0 * x1 - x2 - x0);
    if denominator.abs() < 1e-12 {
        return candidate.lag as f64;
    }
    candidate.lag as f64 + (x2 - x0) / denominator
}

/// A deterministic pseudo-random generator.
///
/// The pYIN posterior is sampled rather than maximised, so the sampler has to
/// be reproducible: the same input must always produce the same output.
#[derive(Debug, Clone)]
pub struct SmallRng {
    state: u64,
}

impl SmallRng {
    /// Create a generator with a fixed seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        SmallRng { state: seed | 1 }
    }

    /// The next uniformly distributed value in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        // xorshift64*
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        let value = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (value >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// One candidate lag of a frame, with the values needed to refine it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    /// The lag, in samples.
    pub lag: usize,
    /// The probability of this lag.
    pub probability: f64,
    /// The normalized difference at `lag - 1`.
    pub left: f64,
    /// The normalized difference at `lag`.
    pub center: f64,
    /// The normalized difference at `lag + 1`.
    pub right: f64,
}

/// The posterior over the lags of one frame.
#[derive(Debug, Clone, Default)]
pub struct Posterior {
    /// The candidate lags, most probable first.
    pub candidates: Vec<Candidate>,
}

impl Posterior {
    /// Sample a lag from the posterior.
    #[must_use]
    pub fn sample(&self, rng: &mut SmallRng) -> Option<Candidate> {
        let total: f64 = self.candidates.iter().map(|c| c.probability).sum();
        if total <= 0.0 {
            return None;
        }
        let mut threshold = rng.next_f64() * total;
        for candidate in &self.candidates {
            threshold -= candidate.probability;
            if threshold <= 0.0 {
                return Some(*candidate);
            }
        }
        self.candidates.last().copied()
    }

    /// The most likely lag of the posterior.
    #[must_use]
    pub fn best(&self) -> Option<Candidate> {
        self.candidates
            .iter()
            .max_by(|a, b| {
                a.probability
                    .partial_cmp(&b.probability)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
    }
}

/// Build the posterior over the lags of a frame.
///
/// The posterior combines the likelihood `1 - d'(τ)` with a Gaussian prior in
/// the log-frequency domain, centred on the lag where the cumulative mean
/// normalized difference first crosses the threshold.
#[must_use]
pub fn posterior_for(
    normalized: &[f64],
    threshold: f64,
    min_lag: usize,
    max_lag: usize,
) -> Posterior {
    let mut candidates = Vec::new();
    if normalized.len() <= min_lag {
        return Posterior { candidates };
    }

    // The lag at which the threshold is crossed: the reference of the prior.
    //
    // The threshold is crossed by the first dip of the difference function,
    // and the first dip is the shortest period that fits the signal, which
    // is the fundamental. The global minimum is not used as the reference,
    // because a signal with a strong second harmonic often has its lowest
    // dip an octave below the true fundamental.
    let centre = (min_lag..=max_lag.min(normalized.len() - 1))
        .find(|&lag| normalized[lag] < threshold)
        .unwrap_or_else(|| {
            (min_lag..=max_lag.min(normalized.len() - 1))
                .min_by(|&a, &b| {
                    normalized[a]
                        .partial_cmp(&normalized[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(min_lag)
        });

    // All local minima below the threshold are plausible.
    for lag in min_lag..=max_lag.min(normalized.len() - 1) {
        let is_local_minimum = (lag == min_lag || normalized[lag] <= normalized[lag - 1])
            && (lag == max_lag || normalized[lag] <= normalized[lag + 1]);
        if !is_local_minimum {
            continue;
        }
        if normalized[lag] > threshold {
            continue;
        }
        // The prior is a Gaussian over the log-frequency, which makes an
        // octave error twice as far away as a semitone error.
        let log_ratio = (lag as f64 / centre as f64).ln() / std::f64::consts::LN_2;
        let prior = (-0.5 * (log_ratio / 0.5).powi(2)).exp();
        let likelihood = (1.0 - normalized[lag]).max(1e-6);
        candidates.push(Candidate {
            lag,
            probability: prior * likelihood,
            left: normalized[lag.saturating_sub(1)],
            center: normalized[lag],
            right: normalized[(lag + 1).min(normalized.len() - 1)],
        });
    }

    if candidates.is_empty() {
        // Nothing passed the threshold: fall back to the global minimum, with
        // a low probability so that the frame counts as unvoiced.
        let best = (min_lag..=max_lag.min(normalized.len() - 1))
            .min_by(|&a, &b| {
                normalized[a]
                    .partial_cmp(&normalized[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(min_lag);
        let probability = (1.0 - normalized[best]).max(0.0) * 0.5;
        candidates.push(Candidate {
            lag: best,
            probability,
            left: normalized[best.saturating_sub(1)],
            center: normalized[best],
            right: normalized[(best + 1).min(normalized.len() - 1)],
        });
    }

    Posterior { candidates }
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
    buffer[..bins]
        .iter()
        .map(|value| value.norm())
        .collect()
}

/// The harmonic salience of a frequency in a spectrum.
///
/// The salience is the sum of the strengths of the harmonics of the
/// frequency, each weighted by the inverse of its number. A pitch whose
/// fundamental is missing but whose harmonics are strong still has a high
/// salience, which is what tells a fundamental from a harmonic of another
/// note.
fn harmonic_salience(spectrum: &[f32], bin_hz: f64, frequency: f64, harmonics: usize) -> f64 {
    if frequency <= 0.0 || spectrum.is_empty() {
        return 0.0;
    }
    let nyquist = bin_hz * (spectrum.len() - 1) as f64;
    let mut sum = 0.0;
    for harmonic in 1..=harmonics {
        let partial = frequency * harmonic as f64;
        if partial > nyquist {
            break;
        }
        let bin = (partial / bin_hz).round() as usize;
        let low = bin.saturating_sub(1);
        let high = (bin + 1).min(spectrum.len() - 1);
        let peak = spectrum[low..=high].iter().copied().fold(0.0f32, f32::max) as f64;
        sum += peak / harmonic as f64;
    }
    sum
}

/// The exponent of the salience weighting.
///
/// A value of one means that the salience decides between two candidates of
/// the same probability; a larger value lets the salience overrule a
/// probability that is clearly higher.
const SALIENCE_EXPONENT: f64 = 1.0;

/// The number of harmonics used by the salience.
const SALIENCE_HARMONICS: usize = 6;

/// Run the frame-level analysis.
#[must_use]
pub fn analyze_frames(signal: &[f32], config: &YinConfig) -> FrameAnalysis {
    let mut analysis = FrameAnalysis::default();
    if signal.len() < config.window_size {
        return analysis;
    }

    let min_lag = config.min_lag();
    let max_lag = config.max_lag().min(config.window_size / 2);
    if max_lag <= min_lag {
        return analysis;
    }

    let mut autocorrelation = Autocorrelation::new(config.window_size);
    let mut frame = vec![0.0f32; config.window_size];
    let mut start = 0usize;

    // The spectrum of the frame, used to weigh the candidates of the
    // posterior by how much of the sound they actually explain.
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
    let bin_hz = config.sample_rate as f64 / config.window_size as f64;

    while start + config.window_size <= signal.len() {
        frame.copy_from_slice(&signal[start..start + config.window_size]);
        let difference = difference_function(&frame, max_lag, &mut autocorrelation);
        let normalized = cumulative_mean_normalized(&difference);

        let mut posterior = posterior_for(&normalized, config.threshold, min_lag, max_lag);
        if posterior.candidates.is_empty() {
            start += config.hop_size;
            continue;
        }

        // Weigh the candidates by the harmonic salience of their pitch.
        let spectrum = magnitude_spectrum(&frame, &window, &fft, &mut buffer);
        weigh_by_salience(&mut posterior, &spectrum, bin_hz, config);

        let Some(candidate) = posterior.best() else {
            start += config.hop_size;
            continue;
        };
        let lag = candidate.lag;
        let refined = parabolic_refine(&candidate);
        let frequency = if refined > 0.0 {
            config.sample_rate as f64 / refined
        } else {
            0.0
        };
        let probability = (1.0 - normalized[lag]).clamp(0.0, 1.0);
        let probability = if frequency > 0.0 { probability } else { 0.0 };

        let time = (start + config.window_size / 2) as f64 / config.sample_rate as f64;
        analysis.frames.push(Frame {
            time,
            frequency,
            probability,
            lag,
            posterior: posterior.candidates.clone(),
        });

        start += config.hop_size;
    }

    analysis
}

/// Weigh the candidates of a posterior by the salience of their pitch.
///
/// A candidate that explains little of the sound is unlikely to be the note
/// that is being played, even when the difference function likes its period.
/// This is what tells a fundamental apart from the harmonic of another note,
/// and it is the cure for the octave errors of the difference function.
fn weigh_by_salience(posterior: &mut Posterior, spectrum: &[f32], bin_hz: f64, config: &YinConfig) {
    let mut saliences: Vec<f64> = Vec::with_capacity(posterior.candidates.len());
    for candidate in &posterior.candidates {
        let refined = parabolic_refine(candidate);
        if refined <= 0.0 {
            saliences.push(0.0);
            continue;
        }
        let frequency = config.sample_rate as f64 / refined;
        saliences.push(harmonic_salience(
            spectrum,
            bin_hz,
            frequency,
            SALIENCE_HARMONICS,
        ));
    }

    let strongest = saliences.iter().copied().fold(0.0f64, f64::max);
    if strongest <= 0.0 {
        return;
    }

    for (candidate, salience) in posterior.candidates.iter_mut().zip(saliences.iter()) {
        let ratio = (salience / strongest).clamp(0.0, 1.0);
        candidate.probability *= ratio.powf(SALIENCE_EXPONENT);
    }
}

/// The probability of every pitch of the range, for one frame.
///
/// The candidates of the posterior are spread over the semitone scale, so
/// that a candidate that falls between two semitones shares its probability
/// between them.
fn pitch_distribution(
    candidates: &[Candidate],
    config: &YinConfig,
    low: u8,
    high: u8,
    confidence: f64,
) -> Vec<f64> {
    let width = (high - low + 1) as usize;
    let mut raw = vec![0.0f64; width];
    let mut total = 0.0f64;

    for candidate in candidates {
        let refined = parabolic_refine(candidate);
        if refined <= 0.0 {
            continue;
        }
        let frequency = config.sample_rate as f64 / refined;
        if frequency <= 0.0 {
            continue;
        }

        // The position of the candidate on the semitone scale.
        let midi = 69.0 + 12.0 * (frequency / 440.0).log2();
        if midi < f64::from(low) - 1.0 || midi > f64::from(high) + 1.0 {
            continue;
        }
        let base = midi.floor();
        let fraction = midi - base;
        let lower = base as i32;
        let upper = lower + 1;
        for (index, share) in [(lower, 1.0 - fraction), (upper, fraction)] {
            if index < i32::from(low) || index > i32::from(high) {
                continue;
            }
            raw[(index - i32::from(low)) as usize] += candidate.probability * share;
        }
        total += candidate.probability;
    }

    if total <= 0.0 {
        return vec![0.0; width];
    }
    raw.into_iter()
        .map(|value| value / total * confidence)
        .collect()
}

/// Smooth the pitch distributions over time.
///
/// A moving average over a few frames removes the isolated mistakes of a
/// single frame without blurring a real change of pitch, because a change
/// of pitch lasts for more than the width of the window.
fn smooth_distributions(distributions: &[Vec<f64>], radius: usize) -> Vec<Vec<f64>> {
    if distributions.is_empty() {
        return Vec::new();
    }
    let width = distributions[0].len();
    let mut smoothed: Vec<Vec<f64>> = Vec::with_capacity(distributions.len());
    for index in 0..distributions.len() {
        let mut total = vec![0.0f64; width];
        let mut count = 0usize;
        for offset in -(radius as isize)..=(radius as isize) {
            let other = index as isize + offset;
            if other < 0 || other as usize >= distributions.len() {
                continue;
            }
            let other = other as usize;
            for (slot, value) in total.iter_mut().zip(distributions[other].iter()) {
                *slot += value;
            }
            count += 1;
        }
        if count > 0 {
            for value in total.iter_mut() {
                *value /= count as f64;
            }
        }
        smoothed.push(total);
    }
    smoothed
}

/// The state of the note segmentation HMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// No note is sounding.
    Rest,
    /// A note is sounding.
    Note(u8),
}

/// The Viterbi decoder of the note segmentation.
///
/// The states are the candidate pitches plus a rest state. The transition
/// cost penalises large pitch leaps, which is what keeps the segmentation
/// stable when the pitch estimate flickers between neighbouring semitones.
struct Segmenter {
    states: Vec<State>,
    /// The transition cost between two states.
    transition: Vec<Vec<f64>>,
    /// The cost of entering a state from the rest state.
    _onset_cost: f64,
    /// The cost of leaving a state for the rest state.
    _offset_cost: f64,
}

impl Segmenter {
    /// Build the segmenter for a pitch range.
    fn new(low: u8, high: u8) -> Self {
        let mut states = vec![State::Rest];
        for pitch in low..=high {
            states.push(State::Note(pitch));
        }
        let n = states.len();

        // Staying in the same state is free; moving to a neighbouring pitch
        // costs a little; an octave leap costs a lot.
        let mut transition = vec![vec![0.0f64; n]; n];
        for (i, from) in states.iter().enumerate() {
            for (j, to) in states.iter().enumerate() {
                transition[i][j] = match (from, to) {
                    (State::Rest, State::Rest) => 0.0,
                    (State::Rest, State::Note(_)) => 1.0,
                    (State::Note(_), State::Rest) => 1.0,
                    (State::Note(a), State::Note(b)) => {
                        let distance = (*a as i32 - *b as i32).unsigned_abs() as f64;
                        0.4 * distance + 0.6
                    }
                };
            }
        }

        Segmenter {
            states,
            transition,
            _onset_cost: 1.0,
            _offset_cost: 1.0,
        }
    }

    /// Decode the most likely sequence of states.
    fn decode(&self, frames: &[Frame], pitch_costs: &[Vec<f64>]) -> Vec<State> {
        let n = self.states.len();
        if frames.is_empty() {
            return Vec::new();
        }

        // The cost of every state in every frame, computed on the fly.
        let mut costs = vec![vec![0.0f64; n]; frames.len()];
        let mut backpointers = vec![vec![0usize; n]; frames.len()];

        for (index, _frame) in frames.iter().enumerate() {
            for (state_index, state) in self.states.iter().enumerate() {
                let emission = match state {
                    State::Rest => pitch_costs[index][0],
                    State::Note(_) => pitch_costs[index][state_index],
                };
                if index == 0 {
                    costs[index][state_index] = emission;
                    backpointers[index][state_index] = 0;
                    continue;
                }
                let mut best = f64::INFINITY;
                let mut best_previous = 0usize;
                for previous in 0..n {
                    let candidate = costs[index - 1][previous]
                        + self.transition[previous][state_index]
                        + emission;
                    if candidate < best {
                        best = candidate;
                        best_previous = previous;
                    }
                }
                costs[index][state_index] = best;
                backpointers[index][state_index] = best_previous;
            }
        }

        // Backtrack from the cheapest final state.
        let mut current = (0..n)
            .min_by(|&a, &b| {
                costs[frames.len() - 1][a]
                    .partial_cmp(&costs[frames.len() - 1][b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);

        let mut path = vec![self.states[current]; frames.len()];
        for index in (0..frames.len()).rev() {
            path[index] = self.states[current];
            current = backpointers[index][current];
        }
        path
    }
}

/// The classical DSP transcription engine.
#[derive(Debug, Clone)]
pub struct DspTranscriber {
    /// The seed of the posterior sampler.
    pub seed: u64,
}

impl Default for DspTranscriber {
    fn default() -> Self {
        DspTranscriber {
            seed: 0x5EED_1234_ABCD_0001,
        }
    }
}

impl DspTranscriber {
    /// Create an engine with the default seed.
    #[must_use]
    pub fn new() -> Self {
        DspTranscriber::default()
    }
}

impl Transcriber for DspTranscriber {
    fn name(&self) -> &'static str {
        "dsp"
    }

    fn transcribe(
        &self,
        signal: &[f32],
        sample_rate: u32,
        options: &TranscribeOptions,
    ) -> Result<Vec<NoteEvent>> {
        if sample_rate == 0 {
            return Err(Error::Config("the sample rate cannot be zero".to_string()));
        }
        let config = YinConfig::from_options(sample_rate, options);
        let analysis = analyze_frames(signal, &config);
        if analysis.frames.is_empty() {
            return Ok(Vec::new());
        }

        let (low, high) = options.midi_range();
        let low = low.as_u8().max(24);
        let high = high.as_u8().min(120);

        // The pitch of every frame is a distribution rather than a single
        // value. The alternative octaves are part of the distribution
        // instead of a separate decision, which is what keeps the octave
        // errors out of the segmentation.
        let distributions: Vec<Vec<f64>> = analysis
            .frames
            .iter()
            .map(|frame| {
                let confidence = frame.probability.clamp(0.0, 1.0);
                pitch_distribution(&frame.posterior, &config, low, high, confidence)
            })
            .collect();
        let distributions = smooth_distributions(&distributions, 2);

        // The emission costs of the segmentation.
        let mut pitch_costs: Vec<Vec<f64>> = Vec::with_capacity(distributions.len());
        for distribution in &distributions {
            let mut row = Vec::with_capacity(distribution.len() + 1);

            // The rest state: the more likely the frame is voiced, the more
            // expensive it is to call it a rest.
            let voiced: f64 = distribution.iter().sum();
            row.push(-(1.0 - voiced.clamp(0.0, 1.0)).max(1e-9).ln());

            for probability in distribution {
                row.push(-probability.max(1e-9).ln());
            }
            pitch_costs.push(row);
        }

        let segmenter = Segmenter::new(low, high);
        let path = segmenter.decode(&analysis.frames, &pitch_costs);
        let smoothed = analysis.frames;

        // Turn the state path into note events.
        let mut notes: Vec<NoteEvent> = Vec::new();
        let frame_duration = config.hop_size as f64 / config.sample_rate as f64;
        let mut index = 0usize;
        while index < path.len() {
            let State::Note(pitch) = path[index] else {
                index += 1;
                continue;
            };
            let start_index = index;
            let mut end_index = index;
            while end_index + 1 < path.len() && path[end_index + 1] == path[index] {
                end_index += 1;
            }
            let start = smoothed[start_index].time - frame_duration / 2.0;
            let end = smoothed[end_index].time + frame_duration / 2.0;
            let duration = end - start;
            if duration >= options.min_note_seconds {
                let slot = (pitch - low) as usize;
                let confidence = distributions[start_index..=end_index]
                    .iter()
                    .filter_map(|distribution| distribution.get(slot))
                    .copied()
                    .fold(0.0f64, f64::max) as f32;
                if let Ok(pitch) = MidiNote::new(pitch) {
                    notes.push(NoteEvent {
                        pitch,
                        start: Seconds::from_secs(start.max(0.0)),
                        end: Seconds::from_secs(end.max(start)),
                        velocity: 80,
                        confidence,
                    });
                }
            }
            index = end_index + 1;
        }

        Ok(notes)
    }
}

/// Smooth the frame-level pitch estimates.
///
/// A median filter over the log-frequency removes the isolated octave errors
/// that survive the YIN threshold, without blurring real melodic movement.
fn _smooth_frequencies(frames: &[Frame], config: &YinConfig) -> Vec<Frame> {
    const MEDIAN: usize = 3;
    let mut smoothed: Vec<Frame> = frames.to_vec();
    let voiced: Vec<bool> = frames
        .iter()
        .map(|f| f.frequency > 0.0 && f.probability >= 0.35)
        .collect();

    for index in 0..frames.len() {
        if !voiced[index] {
            continue;
        }
        let mut window: Vec<f64> = Vec::with_capacity(MEDIAN * 2 + 1);
        for offset in -(MEDIAN as isize)..=(MEDIAN as isize) {
            let other = index as isize + offset;
            if other < 0 || other as usize >= frames.len() {
                continue;
            }
            let other = other as usize;
            if voiced[other] {
                window.push((frames[other].frequency / config.sample_rate as f64).ln());
            }
        }
        if window.len() < 3 {
            continue;
        }
        window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = window[window.len() / 2];
        smoothed[index].frequency = median.exp() * config.sample_rate as f64;
    }
    smoothed
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn sine(frequency: f64, sample_rate: u32, seconds: f64) -> Vec<f32> {
        let n = (sample_rate as f64 * seconds) as usize;
        (0..n)
            .map(|i| {
                (2.0 * std::f64::consts::PI * frequency * i as f64 / sample_rate as f64).sin()
                    as f32
            })
            .collect()
    }

    #[test]
    fn yin_finds_the_pitch_of_a_tone() {
        let sample_rate = 22_050;
        let signal = sine(440.0, sample_rate, 0.5);
        let options = TranscribeOptions::default();
        let config = YinConfig::from_options(sample_rate, &options);
        let analysis = analyze_frames(&signal, &config);
        assert!(!analysis.frames.is_empty());
        let detected = analysis.frames[analysis.frames.len() / 2].frequency;
        assert!(
            (detected - 440.0).abs() < 5.0,
            "expected 440 Hz, detected {detected}"
        );
    }

    #[test]
    fn the_engine_transcribes_a_simple_melody() {
        let sample_rate = 22_050;
        let mut signal = sine(440.0, sample_rate, 0.5);
        signal.extend(sine(493.88, sample_rate, 0.5));
        let options = TranscribeOptions::default();
        let notes = DspTranscriber::new()
            .transcribe(&signal, sample_rate, &options)
            .expect("transcribes");
        assert!(!notes.is_empty(), "expected at least one note");
        // A4 = 69, B4 = 71.
        assert!(notes.iter().any(|n| n.pitch.as_u8() == 69));
        assert!(notes.iter().any(|n| n.pitch.as_u8() == 71));
    }

    #[test]
    fn the_engine_is_deterministic() {
        let sample_rate = 22_050;
        let signal = sine(329.63, sample_rate, 0.4);
        let options = TranscribeOptions::default();
        let first = DspTranscriber::new()
            .transcribe(&signal, sample_rate, &options)
            .expect("transcribes");
        let second = DspTranscriber::new()
            .transcribe(&signal, sample_rate, &options)
            .expect("transcribes");
        assert_eq!(first, second);
    }

    #[test]
    fn silence_produces_no_notes() {
        let sample_rate = 22_050;
        let signal = vec![0.0f32; sample_rate as usize];
        let options = TranscribeOptions::default();
        let notes = DspTranscriber::new()
            .transcribe(&signal, sample_rate, &options)
            .expect("transcribes");
        assert!(notes.is_empty());
    }
}
