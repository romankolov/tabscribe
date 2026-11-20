//! Streaming the audio in overlapping windows.
//!
//! Long files are analysed in windows so that the memory use stays bounded
//! no matter how long the recording is. The windows overlap, and the overlap
//! is what lets the analysis stages see a note that straddles a window
//! boundary.
//!
//! The stitching of the results is handled by the analysis stages themselves
//! (each stage reports the time range it covers), which keeps this module
//! free of any musical assumptions.

/// A window of a longer signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    /// The index of the first sample of the window.
    pub start: usize,
    /// The index one past the last sample of the window.
    pub end: usize,
}

impl Window {
    /// The number of samples in the window.
    #[must_use]
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Whether the window is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The parameters of the windowing.
#[derive(Debug, Clone, Copy)]
pub struct WindowConfig {
    /// The size of a window, in samples.
    pub size: usize,
    /// The overlap between consecutive windows, in samples.
    pub overlap: usize,
}

impl WindowConfig {
    /// Construct a configuration.
    ///
    /// The overlap is clamped so that it is always smaller than the window.
    #[must_use]
    pub fn new(size: usize, overlap: usize) -> Self {
        let size = size.max(1);
        let overlap = overlap.min(size - 1);
        WindowConfig { size, overlap }
    }

    /// The number of samples the window advances between windows.
    #[must_use]
    pub fn hop(&self) -> usize {
        self.size - self.overlap
    }
}

/// The windows that cover a signal of a given length.
#[must_use]
pub fn windows(total_samples: usize, config: &WindowConfig) -> Vec<Window> {
    let mut out = Vec::new();
    if total_samples == 0 {
        return out;
    }
    let hop = config.hop().max(1);
    let mut start = 0usize;
    loop {
        let end = (start + config.size).min(total_samples);
        out.push(Window { start, end });
        if end >= total_samples {
            break;
        }
        start += hop;
    }
    out
}

/// The Hann window of a given length.
#[must_use]
pub fn hann(size: usize) -> Vec<f32> {
    if size == 0 {
        return Vec::new();
    }
    if size == 1 {
        return vec![1.0];
    }
    (0..size)
        .map(|i| {
            let n = i as f64;
            let denominator = (size - 1) as f64;
            (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n / denominator).cos()) as f32
        })
        .collect()
}

/// Apply a window function to a slice of a signal.
pub fn apply_window(signal: &[f32], window: &[f32], out: &mut [f32]) {
    let n = out.len().min(signal.len()).min(window.len());
    for i in 0..n {
        out[i] = signal[i] * window[i];
    }
    for value in out.iter_mut().skip(n) {
        *value = 0.0;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn windows_cover_the_whole_signal() {
        let config = WindowConfig::new(100, 25);
        let windows = windows(250, &config);
        assert_eq!(windows.first().expect("non-empty").start, 0);
        assert_eq!(windows.last().expect("non-empty").end, 250);
        for pair in windows.windows(2) {
            assert!(pair[1].start < pair[0].end, "the windows must overlap");
        }
    }

    #[test]
    fn a_short_signal_yields_a_single_window() {
        let config = WindowConfig::new(100, 25);
        let windows = windows(50, &config);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].len(), 50);
    }

    #[test]
    fn an_empty_signal_yields_no_windows() {
        assert!(windows(0, &WindowConfig::new(100, 25)).is_empty());
    }

    #[test]
    fn hann_window_starts_and_ends_at_zero() {
        let window = hann(64);
        assert_eq!(window.len(), 64);
        assert!(window[0].abs() < 1e-6);
        assert!(window[63].abs() < 1e-6);
        assert!((window[32] - 1.0).abs() < 1e-3);
    }
}
