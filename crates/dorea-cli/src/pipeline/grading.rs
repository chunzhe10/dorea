//! Direct-mode grading: spawn Python single-process RAUNE filter.
//!
//! The Rust side is a thin wrapper that resolves video info, constructs
//! the subprocess command line, and waits for exit. All decode/encode/
//! inference runs inside `python -m dorea_inference.raune_filter`.

use anyhow::{Context, Result};
use dorea_video::ffmpeg::{OutputCodec, VideoInfo};

use crate::pipeline::PipelineConfig;

/// Run the grading stage: spawn raune_filter.py as a subprocess.
pub fn run(cfg: &PipelineConfig, info: &VideoInfo) -> Result<u64> {
    use std::process::{Command, Stdio};

    let (proxy_w, proxy_h) = dorea_video::resize::proxy_dims(
        info.width, info.height, cfg.raune_proxy_size,
    );

    // Map output codec to PyAV codec name
    let pyav_codec = match cfg.output_codec {
        OutputCodec::Hevc10 => "hevc",
        OutputCodec::H264 => "h264",
        _ => "prores_ks",
    };

    log::info!(
        "Grading: RAUNE proxy {}x{} batch={}, full-res {}x{}, codec={}",
        proxy_w, proxy_h, cfg.batch_size, info.width, info.height, pyav_codec,
    );

    let python_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().and_then(|p| p.parent())
        .map(|p| p.join("python"))
        .unwrap_or_default();

    // Fail-fast UTF-8 path validation — no silent fallbacks.
    let raune_weights_str = cfg.raune_weights.to_str()
        .ok_or_else(|| anyhow::anyhow!("RAUNE weights path is not valid UTF-8: {}", cfg.raune_weights.display()))?;
    let raune_models_dir_str = cfg.raune_models_dir.to_str()
        .ok_or_else(|| anyhow::anyhow!("RAUNE models dir path is not valid UTF-8: {}", cfg.raune_models_dir.display()))?;
    let input_str = cfg.input.to_str()
        .ok_or_else(|| anyhow::anyhow!("input path is not valid UTF-8: {}", cfg.input.display()))?;
    let output_str = cfg.output.to_str()
        .ok_or_else(|| anyhow::anyhow!("output path is not valid UTF-8: {}", cfg.output.display()))?;

    let mut raune_proc = Command::new(&cfg.python)
        .env("PYTHONPATH", &python_dir)
        .args([
            "-m", "dorea_inference.raune_filter",
            "--weights", raune_weights_str,
            "--models-dir", raune_models_dir_str,
            "--full-width", &info.width.to_string(),
            "--full-height", &info.height.to_string(),
            "--proxy-width", &proxy_w.to_string(),
            "--proxy-height", &proxy_h.to_string(),
            "--batch-size", &cfg.batch_size.to_string(),
            "--input", input_str,
            "--output", output_str,
            "--output-codec", pyav_codec,
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .context("failed to spawn raune_filter.py")?;

    let status = raune_proc.wait().context("raune_filter wait failed")?;
    if !status.success() {
        anyhow::bail!("raune_filter exited with {status}");
    }

    // The filter reports frame count via stderr; return total frames from info
    Ok(info.frame_count)
}
