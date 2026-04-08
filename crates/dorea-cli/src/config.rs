//! `dorea.toml` configuration file loading.
//!
//! Resolution order (first match wins):
//!   1. Path in `$DOREA_CONFIG` env var
//!   2. `./dorea.toml` in the current working directory
//!   3. `~/.config/dorea/config.toml`
//!   4. Built-in defaults (all `None` — each command fills its own fallback)
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
    pub inference: InferDefaults,
    #[serde(default)]
    pub grade: GradeDefaults,
    #[serde(default)]
    pub maxine: MaxineDefaults,
    #[serde(default)]
    pub preview: PreviewDefaults,
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
    /// Depth Anything V2 model directory or Hugging Face ID
    pub depth_model: Option<PathBuf>,
}

/// Inference engine defaults shared across commands.
#[derive(Debug, Deserialize, Default)]
pub struct InferDefaults {
    /// Proxy long-edge resolution for RAUNE + depth inference (default: 1080)
    pub proxy_size: Option<usize>,
    /// Compute device: `"cuda"` or `"cpu"` (default: auto-detect)
    pub device: Option<String>,
}

/// `dorea grade` defaults.
#[derive(Debug, Deserialize, Default)]
pub struct GradeDefaults {
    /// Warmth multiplier [0.0–2.0] (default: 1.0)
    pub warmth: Option<f32>,
    /// LUT/HSL blend strength [0.0–1.0] (default: 0.8)
    pub strength: Option<f32>,
    /// Ambiance contrast multiplier [0.0–1.0] (default: 1.0)
    pub contrast: Option<f32>,
    /// MSE threshold for keyframe detection (default: 0.005)
    pub depth_skip_threshold: Option<f32>,
    /// Maximum frames between forced keyframes (default: 12)
    pub depth_max_interval: Option<usize>,
    /// Frames per fused RAUNE+depth inference batch (default: 32)
    pub fused_batch_size: Option<usize>,
    /// Number of adaptive depth zones for calibration LUT (default: 16)
    pub depth_zones: Option<usize>,
    /// Fine zones for segment-level base LUT (default: 32)
    pub base_lut_zones: Option<usize>,
    /// Wasserstein-1 distance threshold for scene segment boundary (default: 0.15)
    pub scene_threshold: Option<f32>,
    /// Minimum keyframes per scene segment before merge (default: 5)
    pub min_segment_keyframes: Option<usize>,
    /// Zone boundary smoothing window width; 1 = no smoothing (default: 3)
    pub zone_smoothing_window: Option<usize>,
    /// Input encoding override (default: auto-detect from container/codec)
    pub input_encoding: Option<String>,
    /// Output codec (default: "h264" for 8-bit, "prores" for 10-bit)
    pub output_codec: Option<String>,
}

/// Maxine VFX SDK defaults (used when Maxine is re-enabled).
#[derive(Debug, Deserialize, Default)]
pub struct MaxineDefaults {
    /// Super-resolution upscale factor: 2, 3, or 4 (default: 2)
    pub upscale_factor: Option<u32>,
    /// Run artifact reduction before upscale (default: true)
    pub artifact_reduction: Option<bool>,
}

/// `dorea preview` defaults.
#[derive(Debug, Deserialize, Default)]
pub struct PreviewDefaults {
    /// Number of frames to sample (default: 8)
    pub frames: Option<usize>,
    /// Warmth multiplier — falls back to `[grade].warmth` if unset
    pub warmth: Option<f32>,
    /// Blend strength — falls back to `[grade].strength` if unset
    pub strength: Option<f32>,
    /// Contrast multiplier — falls back to `[grade].contrast` if unset
    pub contrast: Option<f32>,
}

impl DoreaConfig {
    /// Load config from the first location that exists, in priority order.
    /// Returns `Default` if no config file is found (not an error).
    pub fn load() -> Self {
        // 1. Explicit env var
        if let Ok(path) = std::env::var("DOREA_CONFIG") {
            if let Some(cfg) = Self::try_load(&PathBuf::from(&path)) {
                log::debug!("Loaded config from $DOREA_CONFIG: {path}");
                return cfg;
            }
            log::warn!("$DOREA_CONFIG={path} does not exist or is invalid — using defaults");
        }

        // 2. ./dorea.toml
        if let Some(cfg) = Self::try_load(&PathBuf::from("dorea.toml")) {
            log::debug!("Loaded config from ./dorea.toml");
            return cfg;
        }

        // 3. ~/.config/dorea/config.toml
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
