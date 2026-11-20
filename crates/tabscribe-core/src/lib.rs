//! `tabscribe-core` — the transcription and tablature algorithms behind
//! [`tabscribe`](https://github.com/OWNER/tabscribe).
//!
//! # Overview
//!
//! The crate is organised as a sequence of independently testable stages:
//!
//! ```text
//! audio → decode → resample → normalize → [separate] → pitch detection
//!       → note events → rhythm → quantization → harmony → cleanup
//!       → tablature assignment → technique detection → renderers
//! ```
//!
//! Each stage consumes and produces plain data types (see [`music`] and
//! [`transcription`]), and the algorithmic stages are exposed as traits so that
//! alternative implementations can be substituted.
//!
//! # Design rules
//!
//! * No I/O in this crate beyond reading bytes that the caller provides. File
//!   system access lives in the CLI.
//! * No `unsafe` code, anywhere.
//! * No `unwrap`/`expect` on paths that touch user input; errors are values.
//! * Deterministic: identical input yields identical output.
//!
//! # Example
//!
//! ```
//! use tabscribe_core::music::{Instrument, InstrumentKind, MidiNote};
//!
//! let guitar = Instrument::for_kind(InstrumentKind::Guitar)?;
//! // E4 can be played five ways on a standard guitar with 22 frets.
//! assert_eq!(guitar.positions_for(MidiNote::new(64)?).len(), 5);
//! # Ok::<(), tabscribe_core::Error>(())
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(clippy::unwrap_used, clippy::expect_used)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod audio;
pub mod error;
pub mod io;
pub mod music;
pub mod pipeline;
pub mod render;
pub mod tab;
pub mod technique;
pub mod transcribe;
pub mod transcription;

pub use error::{Error, Result};
pub use pipeline::{Pipeline, PipelineOptions, PipelineResult};
pub use transcription::{NoteEvent, Tab, TabNote, Transcription};

/// The version of this crate, as reported by `tabscribe --version`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
