//! The `tabscribe models` and `tabscribe doctor` commands.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tabscribe_models::{candidate_model_dirs, resolve, verify as verify_bytes, ModelManifest};

/// The manifest compiled into the binary.
pub fn builtin_manifest() -> ModelManifest {
    ModelManifest::from_json(tabscribe_models::BUILTIN_MANIFEST_JSON)
        .unwrap_or_else(|_| ModelManifest::default())
}

/// The first model directory that exists, or the default location.
#[must_use]
pub fn model_dir() -> PathBuf {
    candidate_model_dirs()
        .into_iter()
        .find(|d| d.is_dir())
        .unwrap_or_else(|| PathBuf::from("models"))
}

/// `tabscribe models list`
pub fn list(paths: bool) -> Result<()> {
    let manifest = builtin_manifest();
    let dir = model_dir();

    if manifest.models.is_empty() {
        println!("No models are registered in this build.");
        return Ok(());
    }

    println!("Model directory: {}", dir.display());
    println!();
    for entry in &manifest.models {
        let path = dir.join(&entry.file);
        let status = if path.exists() { "present" } else { "missing" };
        println!("  {:<16} {}", entry.name, status);
        println!("  {:<16} {}", "", entry.description);
        println!("  {:<16} {}", "", format_args!("license: {}", entry.license));
        println!("  {:<16} {}", "", format_args!("source: {}", entry.source));
        println!("  {:<16} {}", "", format_args!("sha256: {}", entry.sha256));
        if paths {
            println!("  {:<16} {}", "", format_args!("path: {}", path.display()));
        }
        println!();
    }
    Ok(())
}

/// `tabscribe models verify [NAME]`
pub fn verify(name: Option<&str>) -> Result<()> {
    let manifest = builtin_manifest();
    let dir = model_dir();
    let mut failures = 0usize;
    let mut checked = 0usize;

    for entry in &manifest.models {
        if let Some(filter) = name {
            if entry.name != filter {
                continue;
            }
        }
        let path = dir.join(&entry.file);
        if !path.exists() {
            println!("{}: missing ({})", entry.name, path.display());
            failures += 1;
            continue;
        }
        checked += 1;
        match std::fs::read(&path) {
            Ok(bytes) => match verify_bytes(entry, &bytes) {
                Ok(()) => println!("{}: ok ({} bytes)", entry.name, bytes.len()),
                Err(err) => {
                    println!("{}: FAILED: {err}", entry.name);
                    failures += 1;
                }
            },
            Err(err) => {
                println!("{}: could not read: {err}", entry.name);
                failures += 1;
            }
        }
    }

    if name.is_some() && checked == 0 && failures == 0 {
        println!(
            "No model named '{}' is registered.",
            name.unwrap_or_default()
        );
        return Ok(());
    }

    if failures > 0 {
        anyhow::bail!("{failures} model(s) failed verification");
    }
    Ok(())
}

/// `tabscribe models manifest`
pub fn print_manifest() -> Result<()> {
    let manifest = builtin_manifest();
    println!("{}", manifest.to_json()?);
    Ok(())
}

/// `tabscribe doctor`
pub fn doctor() -> Result<()> {
    println!("tabscribe {}", env!("CARGO_PKG_VERSION"));
    println!(
        "  os:   {} {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!(
        "  cwd:  {}",
        std::env::current_dir().unwrap_or_default().display()
    );
    println!();

    let dir = model_dir();
    println!("Model directory: {}", dir.display());
    let manifest = builtin_manifest();
    for entry in &manifest.models {
        let path = dir.join(&entry.file);
        let present = path.exists();
        println!(
            "  {:<16} {}",
            entry.name,
            if present { "present" } else { "missing" }
        );
    }
    println!();

    println!("Engines:");
    #[cfg(feature = "neural-tract")]
    println!("  neural (tract-onnx)   enabled");
    #[cfg(not(feature = "neural-tract"))]
    println!("  neural (tract-onnx)   disabled (rebuild with --features neural-tract)");
    #[cfg(feature = "neural-ort")]
    println!("  neural (onnxruntime)  enabled");
    #[cfg(not(feature = "neural-ort"))]
    println!("  neural (onnxruntime)  disabled (rebuild with --features neural-ort)");
    println!("  dsp (classical)       always available");
    println!();

    println!("Network access: none (all inference is local).");
    Ok(())
}

#[allow(dead_code)]
/// Resolve a model by name against the local model directory.
pub fn resolve_model(name: &str) -> Result<(tabscribe_models::ModelEntry, PathBuf)> {
    let manifest = builtin_manifest();
    let dir = model_dir();
    let resolved = resolve(&manifest, &dir, name, None)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .with_context(|| format!("could not resolve model '{name}'"))?;
    Ok((resolved.entry, resolved.path))
}

#[allow(dead_code)]
/// Check that `path` is a readable file.
pub fn check_readable(path: &Path) -> Result<()> {
    std::fs::metadata(path).with_context(|| format!("could not stat {}", path.display()))?;
    Ok(())
}
