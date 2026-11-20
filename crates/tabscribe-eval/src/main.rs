//! The `tabscribe-eval` harness.
//!
//! Measures transcription quality against datasets with known ground truth
//! and prints a report. See `docs/ACCURACY.md` for the published numbers.

use anyhow::Result;
use clap::{Parser, Subcommand};

/// Command-line interface for the evaluation harness.
#[derive(Debug, Parser)]
#[command(name = "tabscribe-eval", version, about = "Evaluate tabscribe transcription quality")]
struct Cli {
    /// The action to perform.
    #[command(subcommand)]
    action: Action,
}

/// Available evaluation actions.
#[derive(Debug, Subcommand)]
enum Action {
    /// List the datasets the harness knows about.
    Datasets,
    /// Run the evaluation over a dataset.
    Run {
        /// The dataset to evaluate.
        #[arg(long)]
        dataset: String,
        /// The directory holding the dataset.
        #[arg(long)]
        dir: std::path::PathBuf,
        /// Where to write the report.
        #[arg(long)]
        out: Option<std::path::PathBuf>,
    },
    /// Generate a synthetic dataset with exact ground truth.
    Synth {
        /// Where to write the generated dataset.
        #[arg(long)]
        out: std::path::PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.action {
        Action::Datasets => {
            println!("Datasets known to the harness:");
            println!("  synthetic   generated locally, exact ground truth");
            println!("  guitarset   requires a local copy (see scripts/fetch_datasets.sh)");
            println!("  idmt-smt    requires a local copy (see scripts/fetch_datasets.sh)");
            Ok(())
        }
        Action::Run { .. } => Ok(()),
        Action::Synth { .. } => Ok(()),
    }
}
