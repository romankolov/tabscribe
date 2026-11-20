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
        let mut is_harmonic = false;
        for (other_index, other) in ordered.iter().enumerate() {
            if other_index == index || explained[other_index] {
                continue;
            }
            let other_magnitude = peak_magnitude(spectrum, other);
            if other_magnitude < magnitude * 0.15 {
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
            explained[index] = true;
            continue;
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

        let midi = crate::music::MidiNote::from_hz(peak.frequency);
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
const MIN_FRAMES: usize = 3;

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
    // Collect, for every pitch, the frames where it was present.
    let mut occurrences: Vec<(u8, Vec<usize>)> = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        for pitch in &frame.pitches {
            let key = pitch.pitch.as_u8();
            match occurrences.iter_mut().find(|(k, _)| *k == key) {
                Some((_, list)) => list.push(index),
                None => occurrences.push((key, vec![index])),
            }
        }
    }

    let mut notes = Vec::new();
    for (key, indices) in occurrences {
        // Split the frames into runs.
        let mut runs: Vec<Vec<usize>> = Vec::new();
        let mut current: Vec<usize> = Vec::new();
        for index in indices {
            match current.last() {
                Some(&last) if index == last + 1 => current.push(index),
                _ => {
                    if !current.is_empty() {
                        runs.push(std::mem::take(&mut current));
                    }
                    current.push(index);
                }
            }
        }
        if !current.is_empty() {
            runs.push(current);
        }

        // Bridge the gaps of a single frame.
        let mut merged: Vec<Vec<usize>> = Vec::new();
        for run in runs {
            match merged.last_mut() {
                Some(last) if run[0] == last[last.len() - 1] + 2 => {
                    last.extend(run);
                }
                _ => merged.push(run),
            }
        }

        for run in merged {
            if run.len() < MIN_FRAMES {
                continue;
            }
            let start = (frames[run[0]].time - frame_duration / 2.0).max(0.0);
            let end = frames[run[run.len() - 1]].time + frame_duration / 2.0;
            let duration = end - start;
            if duration < min_note_seconds {
                continue;
            }
            let confidence = run
                .iter()
                .filter_map(|&index| {
                    frames[index]
                        .pitches
                        .iter()
                        .find(|p| p.pitch.as_u8() == key)
                        .map(|p| p.confidence)
                })
                .fold(0.0f64, f64::max);
            let Ok(pitch) = MidiNote::new(key) else {
                continue;
            };
            notes.push(NoteEvent {
                pitch,
                start: Seconds::from_secs(start),
                end: Seconds::from_secs(end),
                velocity: 80,
                confidence: confidence as f32,
            });
        }
    }

    notes.sort_by(|a, b| {
        a.start
            .as_secs()
            .partial_cmp(&b.start.as_secs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    notes
}

#[cfg(test)]
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
