//! The `tabscribe` command-line tool.
//!
//! # Exit codes
//!
//! | Code | Meaning                                             |
//! |------|-----------------------------------------------------|
//! | 0    | Success                                             |
//! | 1    | A general processing error                           |
//! | 2    | Invalid command-line usage (emitted by `clap`)      |
//! | 3    | The input file could not be found or read            |
//! | 4    | A model is missing or failed verification            |
//! | 5    | The audio format is not supported                    |

mod cli;
mod commands;
mod models;
mod output;
mod styles;

use std::process::ExitCode;

use clap::Parser;
use cli::{Cli, Command, TranscribeArgs};
use tracing_subscriber::EnvFilter;

/// Exit code returned when the input file cannot be found.
pub const EXIT_INPUT: u8 = 3;
/// Exit code returned when a model is missing or corrupt.
pub const EXIT_MODEL: u8 = 4;
/// Exit code returned when the audio format is not supported.
pub const EXIT_FORMAT: u8 = 5;

fn main() -> ExitCode {
    let cli = Cli::parse();

    let filter = if cli.global.quiet {
        "error"
    } else {
        match cli.global.verbose {
            0 => "warn",
            1 => "info",
            2 => "debug",
            _ => "trace",
        }
    };

    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(filter)),
        )
        .with_writer(std::io::stderr)
        .with_target(false)
        .finish();

    // `set_global_default` fails only if a subscriber was already installed,
    // which cannot happen in a single-shot CLI. Ignoring the error keeps the
    // binary free of panics without hiding anything meaningful.
    let _ = tracing::subscriber::set_global_default(subscriber);

    match dispatch(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            output::print_error(&err);
            ExitCode::from(exit_code_for(&err))
        }
    }
}

/// Dispatch to the requested subcommand.
fn dispatch(cli: &Cli) -> anyhow::Result<()> {
    let global = &cli.global;
    match &cli.command {
        // `tabscribe song.mp3` is shorthand for `tabscribe transcribe song.mp3`.
        None => {
            let Some(input) = cli.input.clone() else {
                let mut cmd = <Cli as clap::CommandFactory>::command();
                cmd.print_help()?;
                return Ok(());
            };
            let args = TranscribeArgs { input, ..TranscribeArgs::default() };
            commands::transcribe(args, global)
        }
        Some(Command::Transcribe(args)) => commands::transcribe(args.clone(), global),
        Some(Command::Tabify(args)) => commands::tabify(args.clone(), global),
        Some(Command::Analyze(args)) => commands::analyze(args.clone(), global),
        Some(Command::Batch(args)) => commands::batch(args.clone(), global),
        Some(Command::Models(args)) => commands::models(args.clone()),
        Some(Command::Doctor) => commands::doctor(),
        Some(Command::Completions(args)) => commands::completions(args.clone()),
        Some(Command::Render(args)) => commands::render(args.clone(), global),
    }
}

/// Map an error onto the documented exit code.
fn exit_code_for(err: &anyhow::Error) -> u8 {
    let mut source: Option<&(dyn std::error::Error + 'static)> = err.source();
    while let Some(current) = source {
        if let Some(core) = current.downcast_ref::<tabscribe_core::Error>() {
            return match core {
                tabscribe_core::Error::Io { .. } => EXIT_INPUT,
                tabscribe_core::Error::UnsupportedFormat(_)
                | tabscribe_core::Error::Decode(_)
                | tabscribe_core::Error::NoAudioStream => EXIT_FORMAT,
                tabscribe_core::Error::Model { .. } => EXIT_MODEL,
                _ => 1,
            };
        }
        if current.downcast_ref::<tabscribe_models::ModelError>().is_some() {
            return EXIT_MODEL;
        }
        if current.downcast_ref::<std::io::Error>().is_some() {
            return EXIT_INPUT;
        }
        source = current.source();
    }
    1
}
