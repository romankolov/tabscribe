//! Core error type for the `tabscribe-core` library.
//!
//! Every fallible operation in this crate returns [`Error`]. The CLI layer is
//! responsible for turning these into human-facing messages; the library keeps
//! them structured and machine-readable so that batch mode and the evaluation
//! harness can react programmatically.

use std::fmt;

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// The error type returned by all fallible `tabscribe-core` operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The input file could not be opened or read.
    #[error("could not read '{path}': {source}")]
    Io {
        /// Path that failed to open.
        path: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// The audio stream could not be decoded.
    #[error("could not decode audio: {0}")]
    Decode(String),

    /// The container/codec combination is not supported.
    #[error("unsupported audio format: {0}")]
    UnsupportedFormat(String),

    /// The file contains no decodable audio streams.
    #[error("the file contains no audio stream")]
    NoAudioStream,

    /// A tuning specification could not be parsed.
    #[error("invalid tuning '{input}': {reason}")]
    InvalidTuning {
        /// The offending user input.
        input: String,
        /// Human readable explanation.
        reason: String,
    },

    /// An instrument definition is internally inconsistent.
    #[error("invalid instrument definition: {0}")]
    InvalidInstrument(String),

    /// A pitch lies outside the playable range of the instrument.
    #[error("pitch {pitch} is outside the playable range of the instrument ({range})")]
    OutOfRange {
        /// The offending pitch.
        pitch: String,
        /// The allowed range.
        range: String,
    },

    /// The tablature optimizer could not find a feasible assignment.
    #[error("no playable tablature assignment exists: {0}")]
    NoFeasibleTab(String),

    /// The requested model is missing, corrupt, or failed verification.
    #[error("model '{name}': {reason}")]
    Model {
        /// Model identifier.
        name: String,
        /// Explanation.
        reason: String,
    },

    /// ONNX inference failed.
    #[error("neural inference failed: {0}")]
    Inference(String),

    /// A requested feature was not compiled into this build.
    #[error("the '{feature}' feature is not enabled in this build")]
    FeatureDisabled {
        /// Name of the missing cargo feature.
        feature: String,
    },

    /// The pipeline was given contradictory or impossible parameters.
    #[error("invalid configuration: {0}")]
    Config(String),

    /// A musical timing computation left the representable range.
    #[error("timing arithmetic overflow: {0}")]
    Timing(String),

    /// Serialization or deserialization failed.
    #[error("failed to (de)serialize {what}: {reason}")]
    Serde {
        /// What was being (de)serialized.
        what: String,
        /// Explanation.
        reason: String,
    },
}

impl Error {
    /// Attach a path to an [`std::io::Error`], preserving the original as source.
    pub fn io(path: impl Into<String>, source: std::io::Error) -> Self {
        Error::Io { path: path.into(), source }
    }

    /// Short, stable, machine-readable identifier for this error variant.
    ///
    /// Used by the JSON report and the evaluation harness to aggregate failures
    /// without string matching on the human-readable message.
    pub fn kind(&self) -> &'static str {
        match self {
            Error::Io { .. } => "io",
            Error::Decode(_) => "decode",
            Error::UnsupportedFormat(_) => "unsupported_format",
            Error::NoAudioStream => "no_audio_stream",
            Error::InvalidTuning { .. } => "invalid_tuning",
            Error::InvalidInstrument(_) => "invalid_instrument",
            Error::OutOfRange { .. } => "out_of_range",
            Error::NoFeasibleTab(_) => "no_feasible_tab",
            Error::Model { .. } => "model",
            Error::Inference(_) => "inference",
            Error::FeatureDisabled { .. } => "feature_disabled",
            Error::Config(_) => "config",
            Error::Timing(_) => "timing",
            Error::Serde { .. } => "serde",
        }
    }

    /// Actionable advice shown alongside the error message in the CLI.
    ///
    /// Returns `None` when no specific remediation exists.
    pub fn remedy(&self) -> Option<String> {
        Some(match self {
            Error::Io { .. } => {
                "check that the path exists and is readable".to_string()
            }
            Error::Decode(_) => {
                "the file may be corrupt or use an unsupported codec; try converting it to WAV \
                 (e.g. `ffmpeg -i input.mp3 -ar 44100 -ac 1 input.wav`)"
                    .to_string()
            }
            Error::UnsupportedFormat(_) => {
                "supported inputs are WAV, FLAC, MP3, OGG/Vorbis and AAC/M4A".to_string()
            }
            Error::NoAudioStream => "the container has no audio track".to_string(),
            Error::InvalidTuning { .. } => {
                "use a comma separated list of open strings, e.g. `-t E2,A2,D3,G3,B3,E4`, or a \
                 preset name such as `drop-d`"
                    .to_string()
            }
            Error::InvalidInstrument(_) => "use `-i guitar|bass|ukulele|custom`".to_string(),
            Error::OutOfRange { .. } => {
                "widen the range with a different tuning, or lower `--sensitivity`".to_string()
            }
            Error::NoFeasibleTab(_) => {
                "the passage may exceed the instrument's string count; try `-i custom` with a \
                 matching tuning, or reduce `--max-fret`"
                    .to_string()
            }
            Error::Model { reason, .. } if reason.contains("checksum") => {
                "re-download the model bundle; `tabscribe models verify` reports the expected \
                 SHA-256"
                    .to_string()
            }
            Error::Model { .. } => "run `tabscribe models list` to see available models".to_string(),
            Error::Inference(_) => "try `--engine dsp` to fall back to the classical engine".to_string(),
            Error::FeatureDisabled { feature } => {
                format!("rebuild with `--features {feature}`")
            }
            Error::Config(_) => "check the TOML profile against `docs/ARCHITECTURE.md`".to_string(),
            Error::Timing(_) => "this is a bug; please report it with the input file".to_string(),
            Error::Serde { .. } => "the file may be malformed".to_string(),
        })
    }
}

/// Small helper for adding context to a `Result` without losing the source error.
pub trait Context<T> {
    /// Attach a human readable context string to an error.
    fn context(self, msg: impl Into<String>) -> Result<T>;
}

impl<T, E: fmt::Display> Context<T> for std::result::Result<T, E> {
    fn context(self, msg: impl Into<String>) -> Result<T> {
        self.map_err(|e| Error::Config(format!("{}: {e}", msg.into())))
    }
}
