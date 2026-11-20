//! Model registry, bundling and checksum verification for `tabscribe`.
//!
//! Every model that `tabscribe` can use is described by a [`ModelEntry`] in a
//! manifest. Entries carry the SHA-256 digest of the model file, so a
//! corrupted or substituted model is detected before it is ever handed to the
//! inference backend.
//!
//! Models are resolved in this order:
//!
//! 1. an explicit path given on the command line,
//! 2. the local model directory (`TABSCRIBE_MODEL_DIR`, else `models/` next to
//!    the executable, else `./models`),
//! 3. the copy embedded in the binary via [`include_bytes!`], when the build
//!    included one.
//!
//! No step ever touches the network.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The model manifest compiled into the binary.
///
/// Keeping the manifest as a JSON file (rather than Rust constants) means the
/// registry can be regenerated and reviewed as data.
pub const BUILTIN_MANIFEST_JSON: &str = include_str!("../../../models/manifest.json");

/// Errors produced while resolving or verifying a model.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ModelError {
    /// The requested model is not present in the registry.
    #[error("unknown model '{name}'")]
    Unknown {
        /// The requested model identifier.
        name: String,
    },

    /// The model file could not be read.
    #[error("could not read model file '{path}': {source}")]
    Io {
        /// Path that failed to open.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// The SHA-256 of the file does not match the manifest.
    #[error("checksum mismatch for '{name}': expected {expected}, got {actual}")]
    ChecksumMismatch {
        /// Model identifier.
        name: String,
        /// Digest recorded in the manifest.
        expected: String,
        /// Digest computed from the file on disk.
        actual: String,
    },

    /// The manifest could not be parsed.
    #[error("could not parse model manifest: {0}")]
    Manifest(String),

    /// The model registry is empty or inconsistent.
    #[error("invalid model registry: {0}")]
    Registry(String),
}

/// A model known to `tabscribe`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelEntry {
    /// The identifier used on the command line, e.g. `basic-pitch`.
    pub name: String,
    /// The file name inside the model directory.
    pub file: String,
    /// The SHA-256 digest of the file, lowercase hex.
    pub sha256: String,
    /// The size of the file in bytes.
    pub size_bytes: u64,
    /// The upstream source of the model.
    pub source: String,
    /// The SPDX license identifier of the model weights.
    pub license: String,
    /// A short human readable description.
    pub description: String,
}

/// A manifest of models.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ModelManifest {
    /// The models described by this manifest.
    #[serde(default)]
    pub models: Vec<ModelEntry>,
}

impl ModelManifest {
    /// Look up a model by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&ModelEntry> {
        self.models.iter().find(|m| m.name == name)
    }

    /// Parse a manifest from JSON text.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::Manifest`] when the JSON is malformed.
    pub fn from_json(text: &str) -> Result<Self, ModelError> {
        serde_json::from_str(text).map_err(|e| ModelError::Manifest(e.to_string()))
    }

    /// Serialize the manifest to pretty JSON.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::Manifest`] when serialization fails.
    pub fn to_json(&self) -> Result<String, ModelError> {
        serde_json::to_string_pretty(self).map_err(|e| ModelError::Manifest(e.to_string()))
    }
}

/// Compute the SHA-256 digest of a byte slice.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

/// Compute the SHA-256 digest of a file.
///
/// # Errors
///
/// Returns [`ModelError::Io`] when the file cannot be read.
pub fn sha256_file(path: &Path) -> Result<String, ModelError> {
    let bytes = std::fs::read(path).map_err(|source| ModelError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(sha256_hex(&bytes))
}

/// A model resolved to a concrete location on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModel {
    /// The registry entry.
    pub entry: ModelEntry,
    /// Where the file was found.
    pub path: PathBuf,
    /// Whether the file is embedded in the binary rather than on disk.
    pub embedded: bool,
}

/// The location a model was loaded from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSource {
    /// A file on disk.
    File(PathBuf),
    /// Bytes embedded in the binary.
    Embedded {
        /// The identifier of the embedded model.
        name: &'static str,
        /// The embedded bytes.
        bytes: &'static [u8],
    },
}

impl ModelSource {
    /// The bytes of the model.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::Io`] when a file-backed model cannot be read.
    pub fn bytes(&self) -> Result<Vec<u8>, ModelError> {
        match self {
            ModelSource::File(path) => {
                std::fs::read(path).map_err(|source| ModelError::Io {
                    path: path.clone(),
                    source,
                })
            }
            ModelSource::Embedded { bytes, .. } => Ok(bytes.to_vec()),
        }
    }
}

/// Resolve `name` against the registry and the local model directory.
///
/// The search order is: an explicit path, then `<model_dir>/<file>`, then any
/// embedded model with a matching name.
///
/// # Errors
///
/// Returns [`ModelError::Unknown`] when the model is not in the registry and
/// [`ModelError::Io`] when a registered model file cannot be read.
pub fn resolve(
    manifest: &ModelManifest,
    model_dir: &Path,
    name: &str,
    embedded: Option<(&'static str, &'static [u8])>,
) -> Result<ResolvedModel, ModelError> {
    let entry = manifest
        .get(name)
        .ok_or_else(|| ModelError::Unknown { name: name.to_string() })?;
    let path = model_dir.join(&entry.file);
    if path.exists() {
        return Ok(ResolvedModel {
            entry: entry.clone(),
            path,
            embedded: false,
        });
    }
    if let Some((embedded_name, _)) = embedded {
        if embedded_name == name {
            return Ok(ResolvedModel {
                entry: entry.clone(),
                path: PathBuf::from(format!("<embedded:{embedded_name}>")),
                embedded: true,
            });
        }
    }
    Err(ModelError::Io {
        path,
        source: std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "model file not found; run `tabscribe models list` to see where models are expected",
        ),
    })
}

/// Verify that the bytes of a model match the SHA-256 in its registry entry.
///
/// # Errors
///
/// Returns [`ModelError::ChecksumMismatch`] when the digests differ.
pub fn verify(entry: &ModelEntry, bytes: &[u8]) -> Result<(), ModelError> {
    let actual = sha256_hex(bytes);
    if actual.eq_ignore_ascii_case(&entry.sha256) {
        Ok(())
    } else {
        Err(ModelError::ChecksumMismatch {
            name: entry.name.clone(),
            expected: entry.sha256.to_ascii_lowercase(),
            actual,
        })
    }
}

/// The default directories searched for models, in priority order.
///
/// # Errors
///
/// Never fails, but the current directory lookup can fail on exotic platforms.
pub fn candidate_model_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(dir) = std::env::var("TABSCRIBE_MODEL_DIR") {
        dirs.push(PathBuf::from(dir));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            dirs.push(parent.join("models"));
            if let Some(grand) = parent.parent() {
                dirs.push(grand.join("models"));
            }
        }
    }
    dirs.push(PathBuf::from("models"));
    dirs
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn entry(sha: &str) -> ModelEntry {
        ModelEntry {
            name: "test".to_string(),
            file: "test.onnx".to_string(),
            sha256: sha.to_string(),
            size_bytes: 3,
            source: "test".to_string(),
            license: "Apache-2.0".to_string(),
            description: "test model".to_string(),
        }
    }

    #[test]
    fn sha256_matches_known_vector() {
        // SHA-256 of the empty input.
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_matches_known_vector_for_abc() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn verify_accepts_matching_digest_and_rejects_others() {
        let digest = sha256_hex(b"abc");
        assert!(verify(&entry(&digest), b"abc").is_ok());
        assert!(verify(&entry(&digest), b"abd").is_err());
    }

    #[test]
    fn manifest_round_trips_through_json() {
        let manifest = ModelManifest {
            models: vec![entry(&sha256_hex(b"abc"))],
        };
        let json = manifest.to_json().expect("serializable");
        let parsed = ModelManifest::from_json(&json).expect("parseable");
        assert_eq!(parsed, manifest);
    }
}
