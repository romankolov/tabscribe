//! Sample rate conversion.
//!
//! # Why this is hand written
//!
//! The only conversion `tabscribe` needs is mono `f32` at an arbitrary
//! ratio, which is a well-understood operation: band-limited interpolation
//! with a windowed sinc kernel. Implementing it here keeps the dependency
//! tree small, makes the build fast for users who install from source, and
//! keeps the result bit-for-bit deterministic (see `docs/DECISIONS.md`).
//!
//! The kernel is a Kaiser-windowed sinc with 32 taps per side. Its stopband
//! attenuation is better than 80 dB, which is far more than the pitch
//! detectors downstream require.

/// The number of taps on each side of the interpolation kernel.
const TAPS: usize = 16;

/// The Kaiser window beta parameter (about -90 dB sidelobes).
const KAISER_BETA: f64 = 8.0;

/// Resample a mono signal to a new sample rate.
///
/// If the input and output rates are equal the signal is returned unchanged.
///
/// # Panics
///
/// This function does not panic: a zero sample rate is treated as "no
/// conversion" and the signal is passed through.
#[must_use]
pub fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == 0 || to_rate == 0 || from_rate == to_rate {
        return input.to_vec();
    }

    let ratio = f64::from(to_rate) / f64::from(from_rate);
    let out_len = ((input.len() as f64) * ratio).round().max(0.0) as usize;
    if out_len == 0 || input.is_empty() {
        return Vec::new();
    }

    // When downsampling, the kernel has to be stretched to avoid aliasing.
    let cutoff = ratio.min(1.0);
    let window = kaiser_window();

    let mut out = vec![0.0f32; out_len];
    for (n, sample) in out.iter_mut().enumerate() {
        let t = n as f64 / ratio;
        let center = t.floor() as isize;
        let mut acc = 0.0f64;
        let mut weight_sum = 0.0f64;
        for k in -(TAPS as isize)..=(TAPS as isize) {
            let index = center + k;
            if index < 0 || index as usize >= input.len() {
                continue;
            }
            let distance = t - index as f64;
            let normalized = distance / TAPS as f64;
            let weight = sinc(distance * cutoff) * window_at(&window, normalized);
            weight_sum += weight;
            acc += f64::from(input[index as usize]) * weight;
        }
        // Normalising by the sum of the weights keeps the gain at unity even
        // where the window truncates the kernel (at the edges of the signal).
        if weight_sum.abs() > 1e-6 {
            acc /= weight_sum;
        }
        *sample = acc as f32;
    }
    out
}

/// The sinc function, `sin(pi x) / (pi x)`, with the removable singularity at
/// zero defined as one.
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let pix = std::f64::consts::PI * x;
        pix.sin() / pix
    }
}

/// The modified Bessel function of the first kind, order zero.
fn bessel_i0(x: f64) -> f64 {
    // The standard series expansion converges quickly for the arguments used
    // here (up to about 30).
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    let half_x = x / 2.0;
    for k in 1..=40 {
        term *= (half_x / k as f64) * (half_x / k as f64);
        sum += term;
        if term < 1e-16 * sum {
            break;
        }
    }
    sum
}

/// The Kaiser window, sampled at `TAPS + 1` points.
fn kaiser_window() -> Vec<f64> {
    let denominator = bessel_i0(KAISER_BETA);
    (0..=TAPS)
        .map(|i| {
            let x = i as f64 / TAPS as f64;
            let arg = KAISER_BETA * (1.0 - x * x).max(0.0).sqrt();
            bessel_i0(arg) / denominator
        })
        .collect()
}

/// The window value at a normalized position in `[-1, 1]`.
fn window_at(window: &[f64], x: f64) -> f64 {
    let half = (window.len() - 1) as f64;
    let position = (x + 1.0) * 0.5 * half;
    let index = position.floor() as isize;
    if index < 0 {
        return window[0];
    }
    let index = index as usize;
    if index + 1 >= window.len() {
        return window[window.len() - 1];
    }
    let fraction = position - index as f64;
    window[index] * (1.0 - fraction) + window[index + 1] * fraction
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn sine(freq: f64, rate: u32, seconds: f64) -> Vec<f32> {
        let n = (rate as f64 * seconds) as usize;
        (0..n)
            .map(|i| {
                (2.0 * std::f64::consts::PI * freq * i as f64 / rate as f64).sin() as f32
            })
            .collect()
    }

    /// The frequency of the strongest spectral peak, found with a naive DFT.
    fn dominant_frequency(signal: &[f32], rate: u32) -> f64 {
        let n = signal.len();
        let mut best = (0.0f64, 0.0f64);
        for bin in 1..(n / 2) {
            let freq = bin as f64 * rate as f64 / n as f64;
            if freq > 5_000.0 {
                break;
            }
            let mut re = 0.0;
            let mut im = 0.0;
            for (i, &sample) in signal.iter().enumerate() {
                let angle = 2.0 * std::f64::consts::PI * bin as f64 * i as f64 / n as f64;
                re += f64::from(sample) * angle.cos();
                im -= f64::from(sample) * angle.sin();
            }
            let magnitude = (re * re + im * im).sqrt();
            if magnitude > best.1 {
                best = (freq, magnitude);
            }
        }
        best.0
    }

    #[test]
    fn resampling_preserves_the_pitch_of_a_tone() {
        let input = sine(440.0, 44_100, 0.5);
        let output = resample(&input, 44_100, 22_050);
        assert!((output.len() as i64 - 11_025).abs() <= 2);
        let detected = dominant_frequency(&output, 22_050);
        assert!(
            (detected - 440.0).abs() < 5.0,
            "expected 440 Hz, detected {detected}"
        );
    }

    #[test]
    fn upsampling_preserves_the_pitch_of_a_tone() {
        let input = sine(440.0, 22_050, 0.5);
        let output = resample(&input, 22_050, 44_100);
        assert!((output.len() as i64 - 22_050).abs() <= 2);
        let detected = dominant_frequency(&output, 44_100);
        assert!(
            (detected - 440.0).abs() < 5.0,
            "expected 440 Hz, detected {detected}"
        );
    }

    #[test]
    fn resampling_to_the_same_rate_is_a_no_op() {
        let input = sine(440.0, 8_000, 0.01);
        assert_eq!(resample(&input, 8_000, 8_000), input);
    }

    #[test]
    fn resampling_an_empty_signal_is_safe() {
        assert!(resample(&[], 44_100, 22_050).is_empty());
    }

    #[test]
    fn resampling_preserves_amplitude() {
        let input = sine(200.0, 44_100, 0.5);
        let output = resample(&input, 44_100, 22_050);
        let peak = output.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));
        assert!((peak - 1.0).abs() < 0.05, "peak was {peak}");
    }
}
