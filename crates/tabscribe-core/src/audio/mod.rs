//! Audio decoding, conditioning and windowing.
//!
//! The pipeline starts here: a file is decoded into samples, conditioned
//! (DC removal, filtering, loudness normalisation) and optionally resampled
//! to the rate the analysis stages expect.

pub mod decode;
pub mod normalize;
pub mod resample;
pub mod window;

pub use decode::{decode_file, decode_source, AudioBuffer};
pub use resample::resample;
pub use window::{Window, WindowConfig};
