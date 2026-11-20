//! Signal conditioning: DC removal, filtering and loudness normalisation.

/// Remove the DC offset of a signal.
///
/// A non-zero DC offset shifts the waveform asymmetrically and can fool the
/// onset detector, so it is removed before any analysis.
pub fn remove_dc_offset(samples: &mut [f32]) {
    if samples.is_empty() {
        return;
    }
    let mean = samples.iter().map(|&s| f64::from(s)).sum::<f64>() / samples.len() as f64;
    let mean = mean as f32;
    for sample in samples.iter_mut() {
        *sample -= mean;
    }
}

/// The peak amplitude of a signal.
#[must_use]
pub fn peak_amplitude(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()))
}

/// The root-mean-square amplitude of a signal.
#[must_use]
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum::<f64>();
    ((sum / samples.len() as f64).sqrt()) as f32
}

/// Scale a signal so that its peak amplitude equals `target`.
///
/// Signals whose peak is already below `target` are left alone, which avoids
/// amplifying silence and noise.
pub fn peak_normalize(samples: &mut [f32], target: f32) {
    let peak = peak_amplitude(samples);
    if peak > 1e-9 && target > 0.0 {
        let gain = target / peak;
        for sample in samples.iter_mut() {
            *sample *= gain;
        }
    }
}

/// A second-order IIR filter with Direct Form II transposed structure.
#[derive(Debug, Clone, Copy)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    /// Construct a filter from its coefficients.
    ///
    /// The denominator coefficients are normalised by `a0`, so the caller
    /// passes the raw values of the difference equation.
    #[must_use]
    pub const fn new(b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) -> Self {
        Biquad {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// A high-pass filter with the given cutoff and quality factor.
    #[must_use]
    pub fn high_pass(cutoff_hz: f64, sample_rate: u32, q: f64) -> Self {
        let w0 = 2.0 * std::f64::consts::PI * cutoff_hz / f64::from(sample_rate);
        let (sin_w0, cos_w0) = w0.sin_cos();
        let alpha = sin_w0 / (2.0 * q);
        let a0 = 1.0 + alpha;
        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) / 2.0;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Biquad::new(b0 as f32, b1 as f32, b2 as f32, a0 as f32, a1 as f32, a2 as f32)
    }

    /// A low-pass filter with the given cutoff and quality factor.
    #[must_use]
    pub fn low_pass(cutoff_hz: f64, sample_rate: u32, q: f64) -> Self {
        let w0 = 2.0 * std::f64::consts::PI * cutoff_hz / f64::from(sample_rate);
        let (sin_w0, cos_w0) = w0.sin_cos();
        let alpha = sin_w0 / (2.0 * q);
        let a0 = 1.0 + alpha;
        let b0 = (1.0 - cos_w0) / 2.0;
        let b1 = 1.0 - cos_w0;
        let b2 = (1.0 - cos_w0) / 2.0;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Biquad::new(b0 as f32, b1 as f32, b2 as f32, a0 as f32, a1 as f32, a2 as f32)
    }

    /// Filter a signal in place.
    pub fn process(&mut self, samples: &mut [f32]) {
        for sample in samples.iter_mut() {
            let input = *sample;
            let output = self.b0 * input + self.z1;
            self.z1 = self.b1 * input - self.a1 * output + self.z2;
            self.z2 = self.b2 * input - self.a2 * output;
            *sample = output;
        }
    }
}

/// Apply a high-pass filter to a signal, in place.
pub fn high_pass(samples: &mut [f32], cutoff_hz: f64, sample_rate: u32) {
    Biquad::high_pass(cutoff_hz, sample_rate, 0.7071).process(samples);
}

/// Apply a low-pass filter to a signal, in place.
pub fn low_pass(samples: &mut [f32], cutoff_hz: f64, sample_rate: u32) {
    Biquad::low_pass(cutoff_hz, sample_rate, 0.7071).process(samples);
}

/// The K-weighting filter of ITU-R BS.1770-4.
///
/// This is the two-stage filter that approximates the loudness perception of
/// the ear; it is used to measure the integrated loudness of a programme.
#[must_use]
pub fn k_weighting(sample_rate: u32) -> (Biquad, Biquad) {
    // Stage 1: the "head" filter, a high-shelf.
    let f0 = 1681.974_450_955_533;
    let gain_db = 3.999_843_853_973_347;
    let q = 0.707_175_236_955_419_6;

    let k = (std::f64::consts::PI * f0 / f64::from(sample_rate)).tan();
    let vh = 10f64.powf(gain_db / 20.0);
    let vb = vh.powf(0.499_666_774_154_541_6);
    let a0 = 1.0 + k / q + k * k;
    let b0 = vh + vb * k / q + k * k;
    let b1 = 2.0 * (k * k - vh);
    let b2 = vh - vb * k / q + k * k;
    let a1 = 2.0 * (k * k - 1.0);
    let a2 = 1.0 - k / q + k * k;
    let stage1 = Biquad::new(b0 as f32, b1 as f32, b2 as f32, a0 as f32, a1 as f32, a2 as f32);

    // Stage 2: the "RLB" filter, a high-pass.
    let f0 = 38.135_470_876_024_44;
    let q = 0.500_327_037_323_877_3;
    let k = (std::f64::consts::PI * f0 / f64::from(sample_rate)).tan();
    let a0 = 1.0 + k / q + k * k;
    let b0 = 1.0;
    let b1 = -2.0;
    let b2 = 1.0;
    let a1 = 2.0 * (k * k - 1.0);
    let a2 = (1.0 - k / q + k * k);
    let stage2 = Biquad::new(b0 as f32, b1 as f32, b2 as f32, a0 as f32, a1 as f32, a2 as f32);

    (stage1, stage2)
}

/// The gated integrated loudness of a signal, in LUFS.
///
/// Returns `None` when the signal is too quiet to measure.
#[must_use]
pub fn integrated_loudness(samples: &[f32], sample_rate: u32) -> Option<f64> {
    if samples.is_empty() || sample_rate == 0 {
        return None;
    }
    let (mut stage1, mut stage2) = k_weighting(sample_rate);
    let mut filtered = samples.to_vec();
    stage1.process(&mut filtered);
    stage2.process(&mut filtered);

    // The mean square of the filtered signal, in 400 ms blocks with a 75 %
    // overlap, gated at -70 LUFS absolute and -10 LU relative.
    let block = (0.4 * f64::from(sample_rate)) as usize;
    let hop = block / 4;
    if block == 0 || filtered.len() < block {
        return None;
    }

    // Each block contributes a mean square; the loudness of the programme is
    // the mean of the blocks that pass both gates.
    let mut block_powers: Vec<f64> = Vec::new();
    let mut start = 0usize;
    while start + block <= filtered.len() {
        let sum: f64 = filtered[start..start + block]
            .iter()
            .map(|&s| f64::from(s) * f64::from(s))
            .sum();
        block_powers.push(sum / block as f64);
        start += hop;
    }
    if block_powers.is_empty() {
        return None;
    }

    // The absolute gate is at -70 LKFS.
    let absolute_gate = 10f64.powf((-70.0 + 0.691) / 10.0);
    let gated: Vec<f64> = block_powers
        .iter()
        .copied()
        .filter(|power| *power > absolute_gate)
        .collect();
    if gated.is_empty() {
        return None;
    }

    // The relative gate sits 10 LU below the ungated mean power.
    let mean_power = gated.iter().sum::<f64>() / gated.len() as f64;
    let relative_gate = mean_power * 10f64.powf(-10.0 / 10.0);
    let final_gated: Vec<f64> = gated
        .iter()
        .copied()
        .filter(|power| *power > relative_gate)
        .collect();
    if final_gated.is_empty() {
        return None;
    }
    let mean = final_gated.iter().sum::<f64>() / final_gated.len() as f64;
    Some(-0.691 + 10.0 * mean.log10())
}

/// Scale a signal to the given integrated loudness.
///
/// Returns `false` when the loudness could not be measured, in which case the
/// signal is left untouched.
pub fn loudness_normalize(samples: &mut [f32], sample_rate: u32, target_lufs: f64) -> bool {
    let Some(loudness) = integrated_loudness(samples, sample_rate) else {
        return false;
    };
    let gain = 10f64.powf((target_lufs - loudness) / 20.0);
    if !gain.is_finite() || gain <= 0.0 {
        return false;
    }
    // Do not amplify beyond full scale.
    let peak = peak_amplitude(samples);
    let headroom = if peak > 1e-9 { 0.99 / f64::from(peak) } else { 1.0 };
    let gain = gain.min(headroom);
    for sample in samples.iter_mut() {
        *sample = (f64::from(*sample) * gain) as f32;
    }
    true
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn dc_offset_is_removed() {
        let mut samples = vec![1.1f32; 100];
        samples[50] = 0.6;
        remove_dc_offset(&mut samples);
        let mean = samples.iter().map(|&s| f64::from(s)).sum::<f64>() / 100.0;
        assert!(mean.abs() < 1e-5);
    }

    #[test]
    fn peak_normalization_reaches_the_target() {
        let mut samples = vec![0.25f32; 100];
        peak_normalize(&mut samples, 0.9);
        assert!((peak_amplitude(&samples) - 0.9).abs() < 1e-6);
    }

    #[test]
    fn peak_normalization_leaves_silence_alone() {
        let mut samples = vec![0.0f32; 10];
        peak_normalize(&mut samples, 0.9);
        assert!(samples.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn high_pass_removes_low_frequencies() {
        let sample_rate = 8_000;
        let mut samples: Vec<f32> = (0..8_000)
            .map(|i| (2.0 * std::f64::consts::PI * 20.0 * i as f64 / 8_000.0).sin() as f32)
            .collect();
        high_pass(&mut samples, 200.0, sample_rate);
        let rms_before = rms(&samples);
        assert!(rms_before < 0.1, "the 20 Hz tone should be gone, rms {rms_before}");
    }

    #[test]
    fn loudness_of_a_full_scale_tone_is_near_minus_three_lufs() {
        let sample_rate = 48_000;
        let samples: Vec<f32> = (0..48_000)
            .map(|i| {
                (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / 48_000.0).sin() as f32
            })
            .collect();
        let loudness = integrated_loudness(&samples, sample_rate).expect("measurable");
        assert!(
            (loudness + 3.0).abs() < 1.5,
            "expected about -3 LUFS, got {loudness}"
        );
    }
}

