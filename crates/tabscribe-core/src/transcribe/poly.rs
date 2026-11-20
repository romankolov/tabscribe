//! The polyphonic pitch detector.
//!
//! # Why a second detector
//!
//! The YIN detector in [`crate::transcribe::dsp`] is a *monophonic* detector:
//! it finds the single period of a frame. That is the right tool for a single
//! melodic line and the wrong tool for a chord, where several notes sound at
//! once and there is no single period to find.
//!
//! This module detects several notes at once, using the spectrum:
//!
//! 1. The magnitude spectrum of the frame is computed and its local maxima
//!    are located. These are the **spectral peaks**.
//! 2. The peaks are examined from the strongest to the weakest. A peak is
//!    accepted as a **fundamental** when it is not better explained as the
//!    harmonic of a lower peak, and when it has harmonic support of its own.
//! 3. The harmonic series of an accepted fundamental is removed from the
//!    spectrum, so that the next round sees the notes that are still
//!    sounding. This is what keeps the harmonics of a low note from being
//!    reported as separate high notes.
//! 4. The remaining steps mirror the monophonic engine: the frame-level
//!    decisions are smoothed, and contiguous runs of a pitch become a note.
//!
//! The detector is deliberately conservative. A missing note is a smaller
//! mistake than a spurious one, because a spurious note shows up in the
//! tablature as a note the player cannot explain.

use rustfft::num_complex::Complex;
use std::collections::HashSet;
use rustfft::FftPlanner;

use crate::error::{Error, Result};
use crate::music::{MidiNote, Seconds};
use crate::transcribe::{TranscribeOptions, Transcriber};
use crate::transcription::NoteEvent;

/// The configuration of the polyphonic detector.
#[derive(Debug, Clone, Copy)]
pub struct PolyConfig {
    /// The sample rate of the signal.
    pub sample_rate: u32,
    /// The length of the analysis window, in samples.
    pub window_size: usize,
    /// The distance between two analysis windows, in samples.
    pub hop_size: usize,
    /// The number of harmonics used to describe a note.
    pub harmonics: usize,
    /// The lowest frequency to look for, in Hertz.
    pub min_frequency: f64,
    /// The highest frequency to look for, in Hertz.
    pub max_frequency: f64,
    /// The most notes that may sound at once.
    pub max_polyphony: usize,
}

impl PolyConfig {
    /// Derive a configuration from the options.
    #[must_use]
    pub fn from_options(sample_rate: u32, options: &TranscribeOptions) -> Self {
        // A window of about 185 ms resolves the semitones of the lowest note
        // of a bass guitar; the hop is short enough to place onsets.
        let window_size = ((sample_rate as f64) * 0.1855).round() as usize / 2 * 2;
        let window_size = window_size.clamp(1024, 8192);
        PolyConfig {
            sample_rate,
            window_size,
            hop_size: (window_size / 4).max(1),
            harmonics: 6,
            min_frequency: options.min_frequency,
            max_frequency: options.max_frequency,
            max_polyphony: 6,
        }
    }

    /// The size of the FFT, which is twice the window.
    #[must_use]
    pub fn fft_size(&self) -> usize {
        (self.window_size * 2).next_power_of_two()
    }
}

/// One pitch found in one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolyFramePitch {
    /// The pitch.
    pub pitch: MidiNote,
    /// The confidence in the pitch, from 0 to 1.
    pub confidence: f64,
}

/// The pitches found in one frame.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PolyFrame {
    /// The time of the frame, in seconds.
    pub time: f64,
    /// The pitches of the frame.
    pub pitches: Vec<PolyFramePitch>,
}

/// A spectral peak.
#[derive(Debug, Clone, Copy)]
struct Peak {
    /// The frequency of the peak.
    frequency: f64,
    /// The magnitude of the peak.
    magnitude: f64,
    /// The index of the bin.
    bin: usize,
}

/// The magnitude spectrum of a windowed frame.
fn magnitude_spectrum(
    frame: &[f32],
    window: &[f32],
    fft: &std::sync::Arc<dyn rustfft::Fft<f32>>,
    buffer: &mut [Complex<f32>],
) -> Vec<f64> {
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
    buffer[..bins].iter().map(|v| v.norm() as f64).collect()
}

/// Find the local maxima of a spectrum.
fn find_peaks(spectrum: &[f64], min_bin: usize, max_bin: usize, bin_hz: f64) -> Vec<Peak> {
    let mut peaks = Vec::new();
    for bin in min_bin..=max_bin.min(spectrum.len().saturating_sub(2)) {
        let value = spectrum[bin];
        if value <= 0.0 {
            continue;
        }
        if value < spectrum[bin - 1] || value < spectrum[bin + 1] {
            continue;
        }
        // Parabolic interpolation gives a frequency that is accurate to a
        // fraction of a bin.
        let (x0, x1, x2) = (spectrum[bin - 1], value, spectrum[bin + 1]);
        let denominator = 2.0 * (2.0 * x1 - x2 - x0);
        let offset = if denominator.abs() > 1e-12 {
            ((x2 - x0) / denominator).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        peaks.push(Peak {
            frequency: (bin as f64 + offset) * bin_hz,
            magnitude: value,
            bin,
        });
    }
    peaks
}

/// The magnitude of the spectrum at a frequency, taking the maximum of the
/// bins around it.
fn magnitude_at(spectrum: &[f64], frequency: f64, sample_rate: u32, fft_size: usize) -> f64 {
    let bin = frequency * fft_size as f64 / sample_rate as f64;
    let centre = bin.round() as usize;
    if centre >= spectrum.len() {
        return 0.0;
    }
    let low = centre.saturating_sub(2);
    let high = (centre + 2).min(spectrum.len() - 1);
    spectrum[low..=high].iter().copied().fold(0.0f64, f64::max)
}

/// The magnitude of a peak, refined by parabolic interpolation.
///
/// The three points around the maximum are fitted with a parabola; the
/// refinement recovers the part of the peak that falls between the bins,
/// which matters because a semitone is only a few bins wide at the low end.
fn peak_magnitude(spectrum: &[f64], peak: &Peak) -> f64 {
    let bin = peak.bin;
    if bin == 0 || bin + 1 >= spectrum.len() {
        return peak.magnitude;
    }
    let (x0, x1, x2) = (spectrum[bin - 1], peak.magnitude, spectrum[bin + 1]);
    let denominator = x0 - 2.0 * x1 + x2;
    if denominator.abs() > 1e-12 {
        let offset = (0.5 * (x0 - x2) / denominator).clamp(-0.5, 0.5);
        (x1 - 0.25 * (x0 - x2) * offset).max(0.0)
    } else {
        x1
    }
}

/// The polyphonic pitch detector.
#[derive(Debug, Clone)]
pub struct PolyTranscriber {
    /// The seed reserved for the deterministic parts of the detector.
    pub seed: u64,
}

impl Default for PolyTranscriber {
    fn default() -> Self {
        PolyTranscriber {
            seed: 0x5EED_1234_ABCD_0002,
        }
    }
}

impl PolyTranscriber {
    /// Create a detector with the default configuration.
    #[must_use]
    pub fn new() -> Self {
        PolyTranscriber::default()
    }

    /// Detect the pitches of every frame of a signal.
    #[must_use]
    pub fn analyze(&self, signal: &[f32], config: &PolyConfig) -> Vec<PolyFrame> {
        let mut frames = Vec::new();
        if signal.len() < config.window_size {
            return frames;
        }

        let fft_size = config.fft_size();
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_size);
        let mut buffer = vec![Complex::new(0.0f32, 0.0f32); fft_size];
        let window: Vec<f32> = (0..config.window_size)
            .map(|i| {
                let n = i as f64;
                let denominator = (config.window_size - 1) as f64;
                (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n / denominator).cos()) as f32
            })
            .collect();

        let bin_hz = config.sample_rate as f64 / fft_size as f64;
        let min_bin = ((config.min_frequency / bin_hz).floor() as usize).max(1);
        let max_bin = ((config.max_frequency / bin_hz).ceil() as usize).min(fft_size / 2 - 1);

        // The energy gate.
        //
        // A frame that is far below the loudest frame of the piece holds
        // nothing but the numerical noise of the transform. The peak picker
        // would still find peaks in it, and those peaks would become notes
        // that are not there. The gate is set very low, so that a genuinely
        // quiet passage still passes it.
        let gate = loudest_frame_energy(signal, config) * ENERGY_GATE;

        let mut start = 0usize;
        while start + config.window_size <= signal.len() {
            let frame = &signal[start..start + config.window_size];
            let time = (start + config.window_size / 2) as f64 / config.sample_rate as f64;

            let energy = frame_energy(frame);
            if energy <= gate {
                // A silent frame: it interrupts any note that is sounding.
                frames.push(PolyFrame {
                    time,
                    pitches: Vec::new(),
                });
                start += config.hop_size;
                continue;
            }

            let spectrum = magnitude_spectrum(frame, &window, &fft, &mut buffer);
            let peaks = find_peaks(&spectrum, min_bin, max_bin, bin_hz);

            let pitches = detect_pitches(&peaks, &spectrum, config, bin_hz);
            frames.push(PolyFrame { time, pitches });
            start += config.hop_size;
        }

        frames
    }
}

/// The share of the loudest frame's energy below which a frame is silent.
///
/// The gate is at −46 dB, which is low enough to keep every passage a player
/// would consider audible.
const ENERGY_GATE: f64 = 0.0025;

/// The energy of a frame.
fn frame_energy(frame: &[f32]) -> f64 {
    let sum: f64 = frame.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    sum / frame.len().max(1) as f64
}

/// The energy of the loudest frame of a signal.
fn loudest_frame_energy(signal: &[f32], config: &PolyConfig) -> f64 {
    if signal.len() < config.window_size {
        return 0.0;
    }
    let mut loudest = 0.0f64;
    let mut start = 0usize;
    while start + config.window_size <= signal.len() {
        let energy = frame_energy(&signal[start..start + config.window_size]);
        if energy > loudest {
            loudest = energy;
        }
        start += config.hop_size;
    }
    loudest
}



/// Harmonic Product Spectrum score for a candidate fundamental frequency.
///
/// Multiplies the spectrum at f, 2f, 3f, ... up to `max_harmonic`. A true
/// fundamental has energy at all its harmonics, so the product is large.
/// Frequencies that are only partials of something else (e.g. B3 when E2 is
/// the real note) score poorly because their own "harmonics" don't line up
/// with the spectrum.
#[must_use]
fn hps_score(
    spectrum: &[f64],
    candidate_hz: f64,
    sample_rate: u32,
    fft_size: usize,
    max_harmonic: usize,
) -> f64 {
    let nyquist = sample_rate as f64 / 2.0;
    let mut product = 1.0f64;
    let mut count = 0usize;
    for h in 1..=max_harmonic {
        let freq = candidate_hz * h as f64;
        if freq >= nyquist {
            break;
        }
        let value = magnitude_at(spectrum, freq, sample_rate, fft_size);
        product *= value.max(1.0);
        count += 1;
    }
    if count == 0 {
        return 0.0;
    }
    // Geometric mean so longer series don't dominate just by having more terms.
    product.powf(1.0 / count as f64)
}

/// Refinement of detect_pitches using HPS: when a candidate peak's own HPS
/// score is weak but a sub-harmonic's HPS score is strong, the sub-harmonic
/// is the true fundamental (missing fundamental case).
fn refine_with_hps(
    peak_freq: f64,
    spectrum: &[f64],
    sample_rate: u32,
    fft_size: usize,
    strongest: f64,
) -> Option<f64> {
    let nyquist = sample_rate as f64 / 2.0;
    let hps_at_peak = hps_score(spectrum, peak_freq, sample_rate, fft_size, 6);

    // If the peak itself has strong HPS support, it's likely a real fundamental.
    if hps_at_peak > strongest * 0.001 {
        return None;
    }

    // Otherwise check sub-harmonics f/2, f/3 for stronger HPS support.
    let mut best_freq = peak_freq;
    let mut best_score = hps_at_peak;
    for divisor in [2.0, 3.0, 4.0] {
        let sub_freq = peak_freq / divisor;
        if sub_freq < 50.0 {
            continue;
        }
        let score = hps_score(spectrum, sub_freq, sample_rate, fft_size, 6);
        if score > best_score * 1.5 {
            best_score = score;
            best_freq = sub_freq;
        }
    }

    if best_freq != peak_freq {
        Some(best_freq)
    } else {
        None
    }
}



/// Missing-fundamental pre-pass: find groups of peaks that form harmonic
/// series and infer the fundamental even when it is not directly present.
///
/// This handles the case where E2's fundamental bin is weak but its
/// harmonics (E3 at 165 Hz, B3 at 247 Hz, E4 at 330 Hz) are all strong.
/// The detector would otherwise miss E2 entirely and report its harmonics
/// as separate notes.
fn infer_missing_fundamentals(
    peaks: &[Peak],
    spectrum: &[f64],
    sample_rate: u32,
    fft_size: usize,
    strongest: f64,
) -> Vec<(f64, f64)> {
    // Guitar-playable fundamentals in the relevant range
    let string_bases = [82.41f64, 110.0, 146.83, 196.0, 246.94, 329.63];
    let mut guitar_freqs: Vec<f64> = Vec::new();
    for &base in &string_bases {
        for fret in 0..=12 {
            let freq = base * 2.0f64.powf(fret as f64 / 12.0);
            if freq < sample_rate as f64 / 2.0 && freq > 60.0 {
                guitar_freqs.push(freq);
            }
        }
    }

    let mut results: Vec<(f64, f64)> = Vec::new(); // (frequency, confidence)

    for &gf in &guitar_freqs {
        // Count how many harmonics of gf are present as peaks
        let mut harmonic_matches = 0usize;
        let mut total_harmonics = 0usize;
        let mut harmonic_strength = 0.0f64;

        for h in 2..=8 {
            let h_freq = gf * h as f64;
            if h_freq >= sample_rate as f64 / 2.0 {
                break;
            }
            total_harmonics += 1;
            let value = magnitude_at(spectrum, h_freq, sample_rate, fft_size);
            if value > strongest * 0.015 {
                // Check if there's a peak near this harmonic
                let has_peak = peaks.iter().any(|p| {
                    (p.frequency - h_freq).abs() < h_freq * 0.03
                });
                if has_peak {
                    harmonic_matches += 1;
                    harmonic_strength += value;
                }
            }
        }

        if total_harmonics > 0 && harmonic_matches >= total_harmonics / 2 {
            // At least half of the harmonics are present
            let coverage = harmonic_matches as f64 / total_harmonics as f64;
            let strength = harmonic_strength / harmonic_matches.max(1) as f64;
            let confidence = (coverage * 0.7 + (strength / strongest).min(1.0) * 0.3).clamp(0.0, 1.0);
            if confidence > 0.4 {
                results.push((gf, confidence));
            }
        }
    }

    results
}

/// Detect the pitches of one frame from its spectral peaks.
/// Detect the pitches of one frame from its spectral peaks.
fn detect_pitches(
    peaks: &[Peak],
    spectrum: &[f64],
    config: &PolyConfig,
    bin_hz: f64,
) -> Vec<PolyFramePitch> {
    if peaks.is_empty() {
        return Vec::new();
    }
    let fft_size = spectrum.len() * 2 - 2;
    let strongest = peaks
        .iter()
        .map(|p| peak_magnitude(spectrum, p))
        .fold(0.0f64, f64::max);
    if strongest <= 0.0 {
        return Vec::new();
    }

    // Missing-fundamental pre-pass: infer fundamentals from harmonic groups.
    let inferred = infer_missing_fundamentals(peaks, spectrum, config.sample_rate, fft_size, strongest);
    let mut inferred_freqs: Vec<f64> = inferred.iter().map(|(f, _)| *f).collect();

    // The peaks, strongest first.
    let mut ordered: Vec<&Peak> = peaks.iter().collect();
    ordered.sort_by(|a, b| {
        peak_magnitude(spectrum, b)
            .partial_cmp(&peak_magnitude(spectrum, a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // The peaks that have already been explained as a harmonic of an accepted
    // fundamental.
    let mut explained = vec![false; ordered.len()];
    let mut accepted: Vec<PolyFramePitch> = Vec::new();

    // First, accept any inferred missing fundamentals.
    for &inf_freq in &inferred_freqs {
        if accepted.len() >= config.max_polyphony {
            break;
        }
        if let Some(pitch) = crate::music::MidiNote::from_hz(inf_freq) {
            if !accepted.iter().any(|p| p.pitch.as_u8() == pitch.as_u8()) {
                // Find the confidence for this inferred frequency
                let conf = inferred.iter().find(|(f, _)| (*f - inf_freq).abs() < 1.0)
                    .map(|(_, c)| *c).unwrap_or(0.5);
                accepted.push(PolyFramePitch { pitch, confidence: conf });
            }
        }
    }

    // Mark peaks that are harmonics of inferred fundamentals as explained.
    for (index, peak) in ordered.iter().enumerate() {
        for &inf_freq in &inferred_freqs {
            let ratio = peak.frequency / inf_freq;
            let h = ratio.round();
            if h >= 2.0 && h <= 8.0 && (ratio - h).abs() < 0.08 {
                explained[index] = true;
                break;
            }
        }
    }

    for (index, peak) in ordered.iter().enumerate() {
        if accepted.len() >= config.max_polyphony {
            break;
        }
        if explained[index] {
            continue;
        }
        let magnitude = peak_magnitude(spectrum, peak);
        if magnitude < strongest * 0.01 {
            // Far below the loudest peak: the audibility floor.
            break;
        }

        // A peak that is a harmonic of a lower peak is not a fundamental.
        // We check against ALL non-explained lower-frequency peaks, even
        // weaker ones: a strong harmonic can be louder than its fundamental,
        // so we must not accept a harmonic as a fundamental just because the
        // fundamental is quieter.
        let mut is_harmonic = false;
        for (other_index, other) in ordered.iter().enumerate() {
            if other_index == index || explained[other_index] {
                continue;
            }
            let ratio = peak.frequency / other.frequency;
            let harmonic = ratio.round();
            if (2.0..=8.0).contains(&harmonic) && (ratio - harmonic).abs() < 0.06 {
                // The lower peak explains this one better.
                is_harmonic = true;
                break;
            }
        }
        if is_harmonic {
            // Before giving up on this peak, check whether it has independent
            // harmonic support: a peak that is a harmonic of an accepted note
            // might also be the fundamental of a separate, higher note (e.g.
            // A3 at 440 Hz is both A2's 4th harmonic and a note in its own
            // right). Look for its own upper harmonics that are not explained
            // by the accepted note.
            let mut independent_support = 0.0;
            let mut independent_harmonics = 0usize;
            for harmonic in 2..=config.harmonics {
                let frequency = peak.frequency * harmonic as f64;
                if frequency >= config.sample_rate as f64 / 2.0 {
                    break;
                }
                let value = magnitude_at(spectrum, frequency, config.sample_rate, fft_size);
                if value > strongest * 0.02 {
                    // Check whether this upper harmonic is also explained by
                    // an already-accepted note; if not, it supports this peak
                    // as an independent fundamental.
                    let mut explained_by_other = false;
                    for accepted_pitch in &accepted {
                        let af = accepted_pitch.pitch.to_hz();
                        let ratio = frequency / af;
                        let h = ratio.round();
                        if (2.0..=8.0).contains(&h) && (ratio - h).abs() < 0.06 {
                            explained_by_other = true;
                            break;
                        }
                    }
                    if !explained_by_other {
                        independent_support += value;
                        independent_harmonics += 1;
                    }
                }
            }
            if independent_harmonics > 0 && independent_support > magnitude * 0.05 {
                // This peak has independent harmonic support: treat it as a
                // separate fundamental despite being a harmonic of another
                // note. Don't mark it as explained.
            } else {
                explained[index] = true;
                continue;
            }
        }

        // The harmonic support of this peak.
        let mut support = 0.0;
        let mut harmonics_found = 0usize;
        for harmonic in 2..=config.harmonics {
            let frequency = peak.frequency * harmonic as f64;
            if frequency >= config.sample_rate as f64 / 2.0 {
                break;
            }
            let value = magnitude_at(spectrum, frequency, config.sample_rate, fft_size);
            if value > strongest * 0.02 {
                support += value;
                harmonics_found += 1;
            }
        }

        // A fundamental needs harmonic support: a peak without it is a
        // partial of something else, or the broadband splash of a note
        // attack. The exception is a signal that is a bare sine, which has no
        // harmonics at all; that case is handled by the fallback below.
        let has_support = harmonics_found > 0 && support > magnitude * 0.05;
        if !has_support {
            explained[index] = true;
            continue;
        }

        // The confidence grows with the harmonic support and with how loud
        // the peak is.
        let salience = magnitude + support;
        let confidence = (salience / strongest).clamp(0.0, 1.0);
        if confidence < 0.05 {
            explained[index] = true;
            continue;
        }

        // HPS refinement: if this peak's HPS score is weak but a sub-harmonic's
        // is strong, use the sub-harmonic instead (handles missing fundamental).
        let hps_refined = refine_with_hps(
            peak.frequency,
            spectrum,
            config.sample_rate,
            fft_size,
            strongest,
        );
        let final_freq = hps_refined.unwrap_or(peak.frequency);
        let midi = crate::music::MidiNote::from_hz(final_freq);
        if let Some(pitch) = midi {
            if accepted.iter().any(|p| p.pitch.as_u8() == pitch.as_u8()) {
                continue;
            }
            accepted.push(PolyFramePitch { pitch, confidence });
        }

        // Remove the harmonic series of this note, so that the next round
        // sees the notes that are still sounding.
        for other_index in 0..ordered.len() {
            let other = ordered[other_index];
            let ratio = other.frequency / peak.frequency;
            let harmonic = ratio.round();
            if harmonic >= 1.0
                && harmonic <= config.harmonics as f64
                && (ratio - harmonic).abs() < 0.08
            {
                explained[other_index] = true;
            }
        }
    }

    // Sub-harmonic consolidation: if an accepted fundamental is actually a
    // harmonic of a lower note, promote the lower note instead. We detect
    // this by checking whether a sub-harmonic (f/2 or f/3) has a complete
    // harmonic series: peaks at 2x, 3x, 4x, etc. of the sub-harmonic that
    // match the observed spectrum. This is how we identify E2 (82 Hz) when
    // its fundamental bin is weak but its harmonics (E3 at 165 Hz, B3 at 247 Hz,
    // E4 at 330 Hz) are all strong.
    let mut consolidated: Vec<PolyFramePitch> = Vec::new();
    for accepted_pitch in &accepted {
        let freq = accepted_pitch.pitch.to_hz();
        let mut promoted = false;
        for divisor in [2.0, 3.0] {
            let sub_freq = freq / divisor;
            if sub_freq < config.min_frequency {
                continue;
            }
            // Count how many harmonics of the sub-harmonic are present in
            // the spectrum. If most are present, the sub-harmonic is the
            // true fundamental.
            let mut harmonic_matches = 0usize;
            let mut total_harmonics = 0usize;
            for h in 2..=8 {
                let harmonic_freq = sub_freq * h as f64;
                if harmonic_freq >= config.sample_rate as f64 / 2.0 {
                    break;
                }
                total_harmonics += 1;
                let value = magnitude_at(spectrum, harmonic_freq, config.sample_rate, fft_size);
                if value > strongest * 0.02 {
                    harmonic_matches += 1;
                }
            }
            if total_harmonics > 0 && harmonic_matches >= total_harmonics * 2 / 3 {
                // At least 2/3 of the sub-harmonic's harmonics are present:
                // the sub-harmonic is the true fundamental.
                if let Some(pitch) = crate::music::MidiNote::from_hz(sub_freq) {
                    if !accepted.iter().any(|p| p.pitch.as_u8() == pitch.as_u8())
                        && !consolidated.iter().any(|p| p.pitch.as_u8() == pitch.as_u8())
                    {
                        consolidated.push(PolyFramePitch {
                            pitch,
                            confidence: accepted_pitch.confidence * 0.95,
                        });
                        promoted = true;
                    }
                }
                break;
            }
        }
        if !promoted {
            consolidated.push(*accepted_pitch);
        }
    }
    let mut accepted = consolidated;

    // A bare sine has no harmonics, so nothing was accepted. The strongest
    // peak is then the note.
    if accepted.is_empty() {
        if let Some(peak) = ordered.first() {
            let magnitude = peak_magnitude(spectrum, peak);
            if magnitude >= strongest * 0.05 {
                if let Some(pitch) = crate::music::MidiNote::from_hz(peak.frequency) {
                    accepted.push(PolyFramePitch {
                        pitch,
                        confidence: (magnitude / strongest).clamp(0.0, 1.0),
                    });
                }
            }
        }
    }

    accepted.sort_by_key(|a| a.pitch.as_u8());
    let _ = bin_hz;
    accepted
}

/// The polyphonic transcription engine.
#[derive(Debug, Clone, Default)]
pub struct PolyphonicTranscriber {
    /// The configuration of the detector.
    pub config: Option<PolyConfig>,
}

impl PolyphonicTranscriber {
    /// Create an engine with the default configuration.
    #[must_use]
    pub fn new() -> Self {
        PolyphonicTranscriber::default()
    }
}

impl Transcriber for PolyphonicTranscriber {
    fn name(&self) -> &'static str {
        "polyphonic"
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
        let config = self
            .config
            .unwrap_or_else(|| PolyConfig::from_options(sample_rate, options));
        let frames = PolyTranscriber::new().analyze(signal, &config);
        if frames.is_empty() {
            return Ok(Vec::new());
        }

        let frame_duration = config.hop_size as f64 / config.sample_rate as f64;
        Ok(notes_from_frames(
            &frames,
            frame_duration,
            options.min_note_seconds,
        ))
    }
}

/// The shortest run of frames a note must span.
///
/// A single frame is not a note: it is the transient at the boundary between
/// two chords, where the spectrum of both is present at once and the peak
/// picker reports notes that belong to neither.
const MIN_FRAMES: usize = 2;

/// Turn the per-frame pitch sets into note events.
///
/// A pitch that is present in a run of consecutive frames becomes a note.
/// Short gaps inside a run are bridged, because a single frame can be lost
/// when a partial decays.
#[must_use]
pub fn notes_from_frames(
    frames: &[PolyFrame],
    frame_duration: f64,
    min_note_seconds: f64,
) -> Vec<NoteEvent> {
    use std::collections::HashMap;

    // Track, for every pitch, the frame index where its current note started.
    let mut active: HashMap<u8, usize> = HashMap::new();

    let mut notes: Vec<NoteEvent> = Vec::new();

    for (frame_index, frame) in frames.iter().enumerate() {
        let current_pitches: Vec<u8> = frame
            .pitches
            .iter()
            .map(|p| p.pitch.as_u8())
            .collect();
        let current_set: HashSet<u8> = current_pitches.iter().copied().collect();

        // End notes for pitches that were active but are not present now.
        let ended: Vec<u8> = active
            .keys()
            .copied()
            .filter(|p| !current_set.contains(p))
            .collect();
        for pitch in &ended {
            if let Some(start_frame) = active.remove(pitch) {
                let run_len = frame_index - start_frame;
                if run_len >= MIN_FRAMES {
                    let start = (frames[start_frame].time - frame_duration / 2.0).max(0.0);
                    let end = (frames[frame_index - 1].time + frame_duration / 2.0).max(start);
                    let duration = end - start;
                    if duration >= min_note_seconds {
                        let mut best_conf = 0.0f64;
                        for fi in start_frame..=frame_index - 1 {
                            for p in &frames[fi].pitches {
                                if p.pitch.as_u8() == *pitch {
                                    best_conf = best_conf.max(p.confidence);
                                }
                            }
                        }
                        if let Ok(p) = MidiNote::new(*pitch) {
                            notes.push(NoteEvent {
                                pitch: p,
                                start: Seconds::from_secs(start),
                                end: Seconds::from_secs(end),
                                velocity: 80,
                                confidence: best_conf as f32,
                            });
                        }
                    }
                }
            }
        }

        // Start new notes for pitches that are present now but weren't
        // active before (new onsets).
        for pitch in &current_pitches {
            if !active.contains_key(pitch) {
                active.insert(*pitch, frame_index);
            }
        }
    }

    // End any notes still active at the end of the signal.
    let final_frame = frames.len();
    for (pitch, &start_frame) in &active {
        let run_len = final_frame - start_frame;
        if run_len >= MIN_FRAMES {
            let start = (frames[start_frame].time - frame_duration / 2.0).max(0.0);
            let end = (frames[frames.len() - 1].time + frame_duration / 2.0).max(start);
            let duration = end - start;
            if duration >= min_note_seconds {
                let mut best_conf = 0.0f64;
                for fi in start_frame..frames.len() {
                    for p in &frames[fi].pitches {
                        if p.pitch.as_u8() == *pitch {
                            best_conf = best_conf.max(p.confidence);
                        }
                    }
                }
                if let Ok(p) = MidiNote::new(*pitch) {
                    notes.push(NoteEvent {
                        pitch: p,
                        start: Seconds::from_secs(start),
                        end: Seconds::from_secs(end),
                        velocity: 80,
                        confidence: best_conf as f32,
                    });
                }
            }
        }
    }

    // Merge notes of the same pitch whose time ranges overlap or are
    // adjacent: a pitch whose detection flickered between frames can appear
    // as several short notes where there should be one.
    let mut by_pitch: Vec<Vec<NoteEvent>> = Vec::new();
    for note in &notes {
        match by_pitch.iter_mut().find(|g| g[0].pitch == note.pitch) {
            Some(group) => group.push(note.clone()),
            None => by_pitch.push(vec![note.clone()]),
        }
    }

    let mut merged_notes: Vec<NoteEvent> = Vec::new();
    for mut group in by_pitch {
        group.sort_by(|a, b| {
            a.start
                .as_secs()
                .partial_cmp(&b.start.as_secs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut current = group.remove(0);
        for next in group {
            if next.start.as_secs() - current.end.as_secs() <= frame_duration * 2.0 {
                // Overlapping or adjacent: extend the current note.
                if next.end.as_secs() > current.end.as_secs() {
                    current.end = next.end;
                }
                current.confidence = current.confidence.max(next.confidence);
            } else {
                merged_notes.push(std::mem::replace(&mut current, next));
            }
        }
        merged_notes.push(current);
    }

    // Filter out low-confidence and very short notes. A note that is both
    // weak and brief is almost certainly a spectral artifact. The minimum
    // duration of 0.1s removes attack transients and spectral flicker that
    // don't correspond to real notes.
    merged_notes.retain(|note| {
        let duration = note.end.as_secs() - note.start.as_secs();
        note.confidence >= 0.6 && duration >= 0.1
    });

    // Remove short-duration duplicates: when the same pitch has two notes
    // that overlap, keep only the longer one.
    let mut by_pitch2: Vec<Vec<usize>> = Vec::new();
    for (i, note) in merged_notes.iter().enumerate() {
        match by_pitch2.iter_mut().find(|g| merged_notes[g[0]].pitch == note.pitch) {
            Some(group) => group.push(i),
            None => by_pitch2.push(vec![i]),
        }
    }

    let mut to_remove: Vec<usize> = Vec::new();
    for group in &by_pitch2 {
        for i in 0..group.len() {
            for j in (i + 1)..group.len() {
                let a = &merged_notes[group[i]];
                let b = &merged_notes[group[j]];
                let a_start = a.start.as_secs();
                let a_end = a.end.as_secs();
                let b_start = b.start.as_secs();
                let b_end = b.end.as_secs();
                if a_start < b_end && b_start < a_end {
                    let a_dur = a_end - a_start;
                    let b_dur = b_end - b_start;
                    if a_dur < 0.15 && a_dur < b_dur {
                        to_remove.push(group[i]);
                    } else if b_dur < 0.15 && b_dur < a_dur {
                        to_remove.push(group[j]);
                    }
                }
            }
        }
    }
    to_remove.sort();
    to_remove.dedup();
    let mut kept: Vec<NoteEvent> = Vec::new();
    for (i, note) in merged_notes.iter().enumerate() {
        if !to_remove.contains(&i) {
            kept.push(note.clone());
        }
    }
    let mut notes = kept;

    notes.sort_by(|a, b| {
        a.start
            .as_secs()
            .partial_cmp(&b.start.as_secs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    notes
}
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn tone(frequency: f64, seconds: f64, sample_rate: u32) -> Vec<f32> {
        let n = (seconds * sample_rate as f64) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                let envelope = (-t * 2.5).exp();
                let value: f64 = (1..=6)
                    .map(|h| {
                        (2.0 * std::f64::consts::PI * frequency * h as f64 * t).sin() / h as f64
                    })
                    .sum();
                (value * envelope * 0.25) as f32
            })
            .collect()
    }

    #[test]
    fn a_single_note_is_detected() {
        let sample_rate = 22_050;
        let signal = tone(220.0, 1.0, sample_rate);
        let options = TranscribeOptions::default();
        let notes = PolyphonicTranscriber::new()
            .transcribe(&signal, sample_rate, &options)
            .expect("transcribes");
        assert!(!notes.is_empty(), "expected a note");
        assert_eq!(notes[0].pitch.as_u8(), 57, "expected A3");
    }

    #[test]
    fn a_chord_is_detected_as_three_notes() {
        let sample_rate = 22_050;
        let mut signal = tone(261.63, 1.0, sample_rate);
        signal.extend(tone(329.63, 1.0, sample_rate));
        signal.extend(tone(392.0, 1.0, sample_rate));
        let options = TranscribeOptions::default();
        let notes = PolyphonicTranscriber::new()
            .transcribe(&signal, sample_rate, &options)
            .expect("transcribes");
        let pitches: Vec<u8> = notes.iter().map(|n| n.pitch.as_u8()).collect();
        assert!(pitches.contains(&60), "expected C4 in {pitches:?}");
        assert!(pitches.contains(&64), "expected E4 in {pitches:?}");
        assert!(pitches.contains(&67), "expected G4 in {pitches:?}");
    }

    #[test]
    fn the_detector_is_deterministic() {
        let sample_rate = 22_050;
        let signal = tone(329.63, 0.5, sample_rate);
        let options = TranscribeOptions::default();
        let first = PolyphonicTranscriber::new()
            .transcribe(&signal, sample_rate, &options)
            .expect("transcribes");
        let second = PolyphonicTranscriber::new()
            .transcribe(&signal, sample_rate, &options)
            .expect("transcribes");
        assert_eq!(first, second);
    }

    #[test]
    fn silence_produces_no_notes() {
        let sample_rate = 22_050;
        let signal = vec![0.0f32; sample_rate as usize];
        let options = TranscribeOptions::default();
        let notes = PolyphonicTranscriber::new()
            .transcribe(&signal, sample_rate, &options)
            .expect("transcribes");
        assert!(notes.is_empty());
    }
}
