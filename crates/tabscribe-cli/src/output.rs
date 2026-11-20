//! Terminal output helpers: error reporting and simple styling.

use console::style;

/// Print an error message, with a remediation hint when one is available.
pub fn print_error(err: &anyhow::Error) {
    eprintln!("{} {}", style("error:").red().bold(), err);

    let mut source = err.source();
    while let Some(current) = source {
        eprintln!("  caused by: {current}");
        source = current.source();
    }

    if let Some(remedy) = remedy_for(err) {
        eprintln!("{} {remedy}", style("hint:").yellow());
    }
}

/// Look for a remediation hint in the error chain.
fn remedy_for(err: &anyhow::Error) -> Option<String> {
    let mut source: Option<&(dyn std::error::Error + 'static)> = err.source();
    while let Some(current) = source {
        if let Some(core) = current.downcast_ref::<tabscribe_core::Error>() {
            return core.remedy();
        }
        source = current.source();
    }
    None
}

/// Print a warning.
pub fn print_warning(message: &str) {
    eprintln!("{} {message}", style("warning:").yellow());
}

/// Print a success line.
pub fn print_success(message: &str) {
    println!("{} {message}", style("done:").green().bold());
}
