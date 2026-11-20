//! Renderers: turning a tab into the formats the user asked for.

pub mod ascii;
pub mod json;
pub mod musicxml;

pub use ascii::{render_tab, AsciiTabOptions};
pub use json::{Report, ReportMeasure, ReportNote};
pub use musicxml::{MusicXmlOptions, MusicXmlOptions as XmlOptions};
