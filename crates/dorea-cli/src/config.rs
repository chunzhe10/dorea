//! `dorea.toml` configuration file loading.
//!
//! Resolution order (first match wins):
//!   1. Path in `$DOREA_CONFIG` env var
//!   2. `./dorea.toml` in the current working directory
//!   3. `~/.config/dorea/config.toml`
//!   4. Built-in defaults
//!
//! CLI flags always override config file values.

use std::path::PathBuf;
use serde::Deserialize;

/// Top-level `dorea.toml` config.
#[derive(Debug, Deserialize, Default)]
pub struct DoreaConfig {
    #[serde(default)]
    pub models: ModelsConfig,
    #[serde(default)]
    pub grade: GradeDefaults,
}

/// Paths to AI model weights and the Python interpreter.
#[derive(Debug, Deserialize, Default)]
pub struct ModelsConfig {
    /// Python executable (default: `/opt/dorea-venv/bin/python`)
    pub python: Option<PathBuf>,
    /// RAUNE-Net weights `.pth`
    pub raune_weights: Option<PathBuf>,
    /// `sea_thru_poc` checkout directory (contains `models/raune_net.py`)
    pub raune_models_dir: Option<PathBuf>,
}

/// `dorea grade` defaults.
#[derive(Debug, Deserialize, Default)]
pub struct GradeDefaults {
    /// RAUNE proxy long-edge resolution for direct mode (default: 1440)
    pub raune_proxy_size: Option<usize>,
    /// Frames per RAUNE batch (default: 8)
    pub direct_batch_size: Option<usize>,
    /// Input encoding override (default: auto-detect from container/codec)
    pub input_encoding: Option<String>,
    /// Output codec override (default: "h264" for 8-bit, "prores" for 10-bit)
    pub output_codec: Option<String>,
}

impl DoreaConfig {
    /// Load config from the first location that exists.
    /// Returns `Default` if no config file is found (not an error).
    pub fn load() -> Self {
        if let Ok(path) = std::env::var("DOREA_CONFIG") {
            if let Some(cfg) = Self::try_load(&PathBuf::from(&path)) {
                log::debug!("Loaded config from $DOREA_CONFIG: {path}");
                return cfg;
            }
            log::warn!("$DOREA_CONFIG={path} does not exist or is invalid — using defaults");
        }

        if let Some(cfg) = Self::try_load(&PathBuf::from("dorea.toml")) {
            log::debug!("Loaded config from ./dorea.toml");
            return cfg;
        }

        let user_cfg = std::env::var("HOME").ok()
            .map(|h| PathBuf::from(h).join(".config").join("dorea").join("config.toml"));
        if let Some(ref path) = user_cfg {
            if let Some(cfg) = Self::try_load(path) {
                log::debug!("Loaded config from {}", path.display());
                return cfg;
            }
        }

        log::debug!("No dorea.toml found — using built-in defaults");
        Self::default()
    }

    fn try_load(path: &std::path::Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        match toml::from_str(&text) {
            Ok(cfg) => Some(cfg),
            Err(e) => {
                log::warn!("Failed to parse {}: {e}", path.display());
                None
            }
        }
    }
}
