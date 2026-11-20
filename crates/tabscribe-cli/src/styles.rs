//! Shared CLI helpers: terminal styling that respects `NO_COLOR` and TTY
//! detection.
#![allow(dead_code)]

use crate::cli::GlobalArgs;

/// Global options that apply to every subcommand.
#[derive(Debug, Clone)]
pub struct Global {
    /// The parsed global options.
    pub args: GlobalArgs,
    /// Whether coloured output should be used.
    pub color: bool,
}

impl Global {
    /// Build the shared context from the parsed global options.
    #[must_use]
    pub fn new(args: &GlobalArgs) -> Self {
        let color = !args.no_color
            && !args.quiet
            && std::env::var_os("NO_COLOR").is_none()
            && console::Term::stdout().features().is_attended();
        Global {
            args: args.clone(),
            color,
        }
    }

    /// Whether the process is running in quiet mode.
    #[must_use]
    pub fn is_quiet(&self) -> bool {
        self.args.quiet
    }

    /// The verbosity level, where 0 is the default.
    #[must_use]
    pub fn verbosity(&self) -> u8 {
        self.args.verbose
    }
}
