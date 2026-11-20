//! Command-line interface definition for `tabscribe`.
//!
//! The argument grammar lives here so that it can be unit tested and reused
//! by the shell-completion and man-page generators without running the binary.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

/// Top-level command-line interface.
#[derive(Debug, Parser)]
#[command(
    name = "tabscribe",
    version,
    about = "Convert audio recordings into MIDI and playable tablature",
    long_about = "\
tabscribe transcribes audio (or existing MIDI) into MIDI files and playable \
tablature. It runs fully offline: every model it needs is bundled locally and \
verified by checksum.

EXAMPLES:
  tabscribe song.mp3                     Transcribe with sensible defaults
  tabscribe transcribe song.wav -f all   Produce MIDI, tab, MusicXML and JSON
  tabscribe tabify song.mid -i bass      Turn an existing MIDI file into tab
  tabscribe analyze song.mp3             Report key, tempo, meter and range
  tabscribe batch ./songs --format txt  Transcribe a whole directory
  tabscribe doctor                       Check the installation",
    propagate_version = true
)]
pub struct Cli {
    /// The subcommand to run. When omitted, `transcribe` is assumed.
    #[command(subcommand)]
    pub command: Option<Command>,

    /// The audio file to transcribe. Shorthand for `tabscribe transcribe <INPUT>`.
    ///
    /// When neither an input file nor a subcommand is given, the help text is
    /// printed instead of an error, so that `tabscribe` on its own is not a
    /// failure.
    #[arg(value_name = "INPUT")]
    pub input: Option<PathBuf>,

    /// Global options that apply to every subcommand.
    #[command(flatten)]
    pub global: GlobalArgs,
}

/// Global options shared by all subcommands.
#[derive(Debug, Args, Clone)]
pub struct GlobalArgs {
    /// Increase log verbosity (repeat for more detail).
    #[arg(short = 'v', long = "verbose", global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Suppress all non-essential output.
    #[arg(short = 'q', long = "quiet", global = true, conflicts_with = "verbose")]
    pub quiet: bool,

    /// Do not colourise output.
    #[arg(long = "no-color", global = true)]
    pub no_color: bool,

    /// Path to a TOML configuration profile.
    #[arg(long = "config", global = true, value_name = "FILE")]
    pub config: Option<PathBuf>,
}

/// The available subcommands.
#[derive(Debug, Clone, Subcommand)]
pub enum Command {
    /// Transcribe an audio file into MIDI and tablature.
    Transcribe(TranscribeArgs),

    /// Convert an existing MIDI file into tablature, skipping audio analysis.
    Tabify(TabifyArgs),

    /// Analyse an audio file and report key, tempo, meter and pitch range.
    Analyze(AnalyzeArgs),

    /// Transcribe every audio file in a directory.
    Batch(BatchArgs),

    /// Inspect and verify the local model files.
    Models(ModelsArgs),

    /// Check the installation and report the environment.
    Doctor,

    /// Print a shell completion script.
    Completions(CompletionsArgs),

    /// Render the ASCII tablature for a MIDI file to stdout.
    Render(RenderArgs),
}

/// The instrument to assume.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum InstrumentArg {
    /// A six-string guitar in standard tuning.
    #[default]
    Guitar,
    /// A bass guitar (4, 5 or 6 strings depending on the tuning).
    Bass,
    /// A ukulele in standard GCEA tuning.
    Ukulele,
    /// A custom string layout, described with `--tuning`.
    Custom,
}

/// Which pitch-detection engine to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum EngineArg {
    /// Pick the best engine for the material.
    #[default]
    Auto,
    /// The neural polyphonic engine.
    Neural,
    /// The monophonic classical DSP engine.
    Dsp,
    /// The polyphonic spectral engine.
    Polyphonic,
}

/// How aggressively to quantize the transcribed rhythm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum QuantizeArg {
    /// Snap every note exactly to the grid.
    Strict,
    /// Snap only when the note is close enough to the grid.
    #[default]
    Musical,
    /// Keep the performed timing.
    Off,
}

/// The preferred hand position for the tablature optimizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum StyleArg {
    /// Balance movement, span and shape preferences.
    #[default]
    Balanced,
    /// Keep the hand as low on the neck as possible.
    LowPosition,
    /// Keep the hand as high on the neck as possible.
    HighPosition,
    /// Prefer open strings and simple shapes.
    OpenChords,
    /// Favour smooth, single-string lines for lead playing.
    Lead,
}

/// Which output formats to write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum FormatArg {
    /// Standard MIDI file.
    #[default]
    Midi,
    /// ASCII tablature.
    Txt,
    /// MusicXML with a tablature staff.
    MusicXml,
    /// Machine readable JSON report.
    Json,
    /// Every format above.
    All,
}

/// Options for the `transcribe` subcommand.
#[derive(Debug, Clone, Args)]
pub struct TranscribeArgs {
    /// The audio file to transcribe.
    #[arg(value_name = "INPUT")]
    pub input: PathBuf,

    /// The instrument to assume.
    #[arg(short = 'i', long = "instrument", value_enum)]
    pub instrument: Option<InstrumentArg>,

    /// The tuning: a preset name or a comma separated list of open strings.
    #[arg(short = 't', long = "tuning", value_name = "TUNING")]
    pub tuning: Option<String>,

    /// The capo position.
    #[arg(short = 'c', long = "capo", value_name = "N")]
    pub capo: Option<u8>,

    /// The directory to write output files to.
    #[arg(short = 'o', long = "out", value_name = "DIR")]
    pub out: Option<PathBuf>,

    /// The output format(s) to write.
    #[arg(
        short = 'f',
        long = "format",
        value_enum,
        value_delimiter = ',',
        default_value = "all"
    )]
    pub format: Vec<FormatArg>,

    /// The pitch detection engine.
    #[arg(long = "engine", value_enum, default_value = "auto")]
    pub engine: EngineArg,

    /// The preferred tablature position.
    #[arg(long = "style", value_enum, default_value = "balanced")]
    pub style: StyleArg,

    /// How aggressively to quantize.
    #[arg(long = "quantize", value_enum, default_value = "musical")]
    pub quantize: QuantizeArg,

    /// Only transcribe from this timestamp.
    #[arg(long = "start", value_name = "TIME")]
    pub start: Option<String>,

    /// Only transcribe until this timestamp.
    #[arg(long = "end", value_name = "TIME")]
    pub end: Option<String>,

    /// Override the detected tempo.
    #[arg(long = "tempo", value_name = "BPM")]
    pub tempo: Option<f64>,

    /// Override the detected time signature.
    #[arg(long = "time-sig", value_name = "N/D")]
    pub time_sig: Option<String>,

    /// Override the detected key.
    #[arg(long = "key", value_name = "KEY")]
    pub key: Option<String>,

    /// Tuning offset in semitones relative to standard pitch (None = auto).
    ///
    /// A positive value means the recording is sharp and will be transposed
    /// down. For example --tuning-offset 3 compensates for a recording that
    /// is three semitones sharp.
    #[arg(long = "tuning-offset", value_name = "SEMITONES")]
    pub tuning_offset: Option<f64>,

    /// Note detection sensitivity, from 0.0 (conservative) to 1.0 (permissive).
    #[arg(long = "sensitivity", value_name = "X", default_value_t = 0.5)]
    pub sensitivity: f32,

    /// The highest fret to consider.
    #[arg(long = "max-fret", value_name = "N")]
    pub max_fret: Option<u8>,

    /// The maximum fret span within a chord.
    #[arg(long = "max-span", value_name = "N")]
    pub max_span: Option<u8>,

    /// Also write a quantized MIDI file alongside the performance-accurate one.
    #[arg(long = "quantized-midi")]
    pub quantized_midi: bool,

    /// Print the machine readable report to stdout as well.
    #[arg(long = "json-report")]
    pub json_report: bool,
}

impl Default for TranscribeArgs {
    fn default() -> Self {
        TranscribeArgs {
            input: PathBuf::new(),
            instrument: None,
            tuning: None,
            capo: None,
            out: None,
            format: vec![FormatArg::All],
            engine: EngineArg::Auto,
            style: StyleArg::Balanced,
            quantize: QuantizeArg::Musical,
            start: None,
            end: None,
            tempo: None,
            time_sig: None,
            key: None,
            tuning_offset: None,
            sensitivity: 0.5,
            max_fret: None,
            max_span: None,
            quantized_midi: false,
            json_report: false,
        }
    }
}

/// Options for the `tabify` subcommand.
#[derive(Debug, Clone, Args)]
pub struct TabifyArgs {
    /// The MIDI file to convert.
    #[arg(value_name = "MIDI")]
    pub input: PathBuf,

    /// The instrument to assume.
    #[arg(short = 'i', long = "instrument", value_enum)]
    pub instrument: Option<InstrumentArg>,

    /// The tuning: a preset name or a comma separated list of open strings.
    #[arg(short = 't', long = "tuning", value_name = "TUNING")]
    pub tuning: Option<String>,

    /// The capo position.
    #[arg(short = 'c', long = "capo", value_name = "N")]
    pub capo: Option<u8>,

    /// The directory to write output files to.
    #[arg(short = 'o', long = "out", value_name = "DIR")]
    pub out: Option<PathBuf>,

    /// The output format(s) to write.
    #[arg(
        short = 'f',
        long = "format",
        value_enum,
        value_delimiter = ',',
        default_value = "all"
    )]
    pub format: Vec<FormatArg>,

    /// The preferred tablature position.
    #[arg(long = "style", value_enum, default_value = "balanced")]
    pub style: StyleArg,

    /// The highest fret to consider.
    #[arg(long = "max-fret", value_name = "N")]
    pub max_fret: Option<u8>,

    /// The maximum fret span within a chord.
    #[arg(long = "max-span", value_name = "N")]
    pub max_span: Option<u8>,

    /// Print the machine readable report to stdout as well.
    #[arg(long = "json-report")]
    pub json_report: bool,
}

impl Default for TabifyArgs {
    fn default() -> Self {
        TabifyArgs {
            input: PathBuf::new(),
            instrument: None,
            tuning: None,
            capo: None,
            out: None,
            format: vec![FormatArg::All],
            style: StyleArg::Balanced,
            max_fret: None,
            max_span: None,
            json_report: false,
        }
    }
}

/// Options for the `analyze` subcommand.
#[derive(Debug, Clone, Args)]
pub struct AnalyzeArgs {
    /// The audio file to analyse.
    #[arg(value_name = "INPUT")]
    pub input: PathBuf,

    /// Emit the report as JSON.
    #[arg(long = "json")]
    pub json: bool,
}

/// Options for the `batch` subcommand.
#[derive(Debug, Clone, Args)]
pub struct BatchArgs {
    /// The directory to scan for audio files.
    #[arg(value_name = "DIR")]
    pub input: PathBuf,

    /// The output directory.
    #[arg(short = 'o', long = "out", value_name = "DIR")]
    pub out: Option<PathBuf>,

    /// The output format(s) to write.
    #[arg(
        short = 'f',
        long = "format",
        value_enum,
        value_delimiter = ',',
        default_value = "txt"
    )]
    pub format: Vec<FormatArg>,

    /// The instrument to assume.
    #[arg(short = 'i', long = "instrument", value_enum)]
    pub instrument: Option<InstrumentArg>,

    /// The tuning: a preset name or a comma separated list of open strings.
    #[arg(short = 't', long = "tuning", value_name = "TUNING")]
    pub tuning: Option<String>,

    /// The capo position.
    #[arg(short = 'c', long = "capo", value_name = "N")]
    pub capo: Option<u8>,

    /// The pitch detection engine.
    #[arg(long = "engine", value_enum, default_value = "auto")]
    pub engine: EngineArg,

    /// The preferred tablature position.
    #[arg(long = "style", value_enum, default_value = "balanced")]
    pub style: StyleArg,

    /// How aggressively to quantize.
    #[arg(long = "quantize", value_enum, default_value = "musical")]
    pub quantize: QuantizeArg,

    /// Stop after the first failure instead of continuing.
    #[arg(long = "fail-fast")]
    pub fail_fast: bool,

    /// The number of files to process at once.
    #[arg(long = "jobs", value_name = "N")]
    pub jobs: Option<usize>,
}

/// The `models` subcommand.
#[derive(Debug, Clone, Args)]
pub struct ModelsArgs {
    /// The action to perform.
    #[command(subcommand)]
    pub action: ModelsAction,
}

/// Actions available under `tabscribe models`.
#[derive(Debug, Clone, Subcommand)]
pub enum ModelsAction {
    /// List the models `tabscribe` knows about.
    List {
        /// Show the full path of every model file.
        #[arg(long = "paths")]
        paths: bool,
    },

    /// Verify the checksum of every model file.
    Verify {
        /// Verify only this model.
        #[arg(value_name = "NAME")]
        name: Option<String>,
    },

    /// Print the model manifest as JSON.
    Manifest,
}

/// Options for the `completions` subcommand.
#[derive(Debug, Clone, Args)]
pub struct CompletionsArgs {
    /// The shell to generate completions for.
    #[arg(value_name = "SHELL")]
    pub shell: clap_complete::Shell,
}

/// Options for the `render` subcommand.
#[derive(Debug, Clone, Args)]
pub struct RenderArgs {
    /// The MIDI file to render.
    #[arg(value_name = "MIDI")]
    pub input: PathBuf,

    /// The instrument to assume.
    #[arg(short = 'i', long = "instrument", value_enum)]
    pub instrument: Option<InstrumentArg>,

    /// The tuning: a preset name or a comma separated list of open strings.
    #[arg(short = 't', long = "tuning", value_name = "TUNING")]
    pub tuning: Option<String>,

    /// The capo position.
    #[arg(short = 'c', long = "capo", value_name = "N")]
    pub capo: Option<u8>,

    /// The maximum line width in columns.
    #[arg(long = "width", value_name = "N", default_value_t = 80)]
    pub width: u16,

    /// Write to this file instead of stdout.
    #[arg(short = 'o', long = "out", value_name = "FILE")]
    pub out: Option<PathBuf>,
}
