// `dorea grade` — end-to-end video grading pipeline.
//
// Orchestrator: resolves config, probes video, then chains four pipeline stages:
//   KeyframeStage → FeatureStage → CalibrationStage → GradingStage

use std::path::PathBuf;
use std::time::Duration;
use clap::Args;
use anyhow::{Context, Result};

use dorea_video::ffmpeg::{self, FrameEncoder};
use dorea_video::inference::{InferenceConfig, InferenceServer};

use crate::pipeline::{self, PipelineConfig};

#[derive(Args, Debug)]
pub struct GradeArgs {
    /// Input video file (MP4/MOV/MKV)
    #[arg(long)]
    pub input: PathBuf,

    /// Output video file [default: <input-stem>_graded.mp4]
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Warmth multiplier [0.0–2.0] (config: [grade].warmth, built-in default: 1.0)
    #[arg(long)]
    pub warmth: Option<f32>,

    /// LUT/HSL blend strength [0.0–1.0] (config: [grade].strength, built-in default: 0.8)
    #[arg(long)]
    pub strength: Option<f32>,

    /// Ambiance contrast multiplier [0.0–1.0] (config: [grade].contrast, built-in default: 1.0)
    #[arg(long)]
    pub contrast: Option<f32>,

    /// Proxy resolution for inference (long edge, pixels) (config: [inference].proxy_size, built-in default: 1080)
    #[arg(long)]
    pub proxy_size: Option<usize>,

    /// MSE threshold for keyframe detection (config: [grade].depth_skip_threshold, built-in default: 0.005)
    #[arg(long)]
    pub depth_skip_threshold: Option<f32>,

    /// Maximum frames between keyframes (config: [grade].depth_max_interval, built-in default: 12)
    #[arg(long)]
    pub depth_max_interval: Option<usize>,

    /// Frames per fused RAUNE+depth batch (config: [grade].fused_batch_size, built-in default: 32)
    #[arg(long)]
    pub fused_batch_size: Option<usize>,

    /// Adaptive depth zones for calibration LUT (config: [grade].depth_zones, built-in default: 16)
    #[arg(long)]
    pub depth_zones: Option<usize>,

    /// Fine zones for segment-level base LUT (config: [grade].base_lut_zones, built-in default: 32)
    #[arg(long)]
    pub base_lut_zones: Option<usize>,

    /// Wasserstein-1 threshold for scene segment detection (config: [grade].scene_threshold, built-in default: 0.15)
    #[arg(long)]
    pub scene_threshold: Option<f32>,

    /// Disable temporal interpolation — run full pipeline on every frame
    #[arg(long)]
    pub no_depth_interp: bool,

    /// Path to RAUNE-Net weights .pth (config: [models].raune_weights)
    #[arg(long)]
    pub raune_weights: Option<PathBuf>,

    /// Path to RAUNE-Net checkout directory (config: [models].raune_models_dir)
    #[arg(long)]
    pub raune_models_dir: Option<PathBuf>,

    /// Path to Depth Anything V2 model directory (config: [models].depth_model)
    #[arg(long)]
    pub depth_model: Option<PathBuf>,

    /// Python executable (config: [models].python, built-in default: /opt/dorea-venv/bin/python)
    #[arg(long)]
    pub python: Option<PathBuf>,

    /// Force CPU-only mode (no CUDA)
    #[arg(long)]
    pub cpu_only: bool,

    /// Enable verbose logging
    #[arg(short, long)]
    pub verbose: bool,

    /// Maxine super-resolution upscale factor for RAUNE→Maxine→Depth fused batch
    /// (config: [maxine].upscale_factor, built-in default: 2)
    #[arg(long)]
    pub maxine_upscale_factor: Option<u32>,
}

pub fn run(args: GradeArgs, cfg: &crate::config::DoreaConfig) -> Result<()> {
    // Resolve config → CLI → built-in defaults
    let warmth              = args.warmth.or(cfg.grade.warmth).unwrap_or(1.0_f32);
    let strength            = args.strength.or(cfg.grade.strength).unwrap_or(0.8_f32);
    let contrast            = args.contrast.or(cfg.grade.contrast).unwrap_or(1.0_f32);
    let proxy_size          = args.proxy_size.or(cfg.inference.proxy_size).unwrap_or(1080_usize);
    let depth_skip_threshold = args.depth_skip_threshold.or(cfg.grade.depth_skip_threshold).unwrap_or(0.005_f32);
    let depth_max_interval  = args.depth_max_interval.or(cfg.grade.depth_max_interval).unwrap_or(12_usize);
    let fused_batch_size    = args.fused_batch_size.or(cfg.grade.fused_batch_size).unwrap_or(32_usize);
    let depth_zones         = args.depth_zones.or(cfg.grade.depth_zones).unwrap_or(16_usize);
    let base_lut_zones      = args.base_lut_zones.or(cfg.grade.base_lut_zones).unwrap_or(32_usize);
    let scene_threshold     = args.scene_threshold.or(cfg.grade.scene_threshold).unwrap_or(0.15_f32);
    let min_segment_kfs     = cfg.grade.min_segment_keyframes.unwrap_or(5_usize);
    let zone_smoothing_w    = cfg.grade.zone_smoothing_window.unwrap_or(3_usize);
    let maxine_upscale_factor = args.maxine_upscale_factor.or(cfg.maxine.upscale_factor).unwrap_or(2_u32);
    let python = args.python.clone()
        .or_else(|| cfg.models.python.clone())
        .unwrap_or_else(|| PathBuf::from("/opt/dorea-venv/bin/python"));
    let raune_weights    = args.raune_weights.clone().or_else(|| cfg.models.raune_weights.clone());
    let raune_models_dir = args.raune_models_dir.clone().or_else(|| cfg.models.raune_models_dir.clone());
    let depth_model      = args.depth_model.clone().or_else(|| cfg.models.depth_model.clone());
    let device = if args.cpu_only {
        Some("cpu".to_string())
    } else {
        cfg.inference.device.clone()
    };

    #[cfg(feature = "cuda")]
    if args.cpu_only {
        anyhow::bail!(
            "--cpu-only is no longer supported for dorea grade; GPU (CUDA) is required. \
             Use dorea preview for CPU-only workflows."
        );
    }
    if depth_max_interval == 0 {
        anyhow::bail!("--depth-max-interval must be >= 1");
    }

    let output = args.output.clone().unwrap_or_else(|| {
        let stem = args.input.file_stem().unwrap_or_default().to_string_lossy();
        args.input.with_file_name(format!("{stem}_graded.mp4"))
    });

    log::info!("Grading: {} → {}", args.input.display(), output.display());

    // Probe input
    let info = ffmpeg::probe(&args.input)
        .context("ffprobe failed — is ffmpeg installed?")?;
    log::info!(
        "Input: {}x{} @ {:.3}fps, {:.1}s ({} frames)",
        info.width, info.height, info.fps, info.duration_secs, info.frame_count
    );

    // Open encoder first — validate output path before doing expensive calibration.
    let audio_src = if info.has_audio { Some(args.input.as_path()) } else { None };
    let encoder = FrameEncoder::new(&output, info.width, info.height, info.fps, audio_src)
        .context("failed to spawn ffmpeg encoder")?;

    let inf_cfg = build_inference_config(
        &python, raune_weights.as_deref(), raune_models_dir.as_deref(),
        depth_model.as_deref(), device.clone(), maxine_upscale_factor,
    );

    let pipeline_cfg = PipelineConfig {
        input: args.input.clone(),
        warmth,
        strength,
        contrast,
        proxy_size,
        depth_skip_threshold,
        depth_max_interval,
        fused_batch_size,
        depth_zones,
        base_lut_zones,
        scene_threshold,
        min_segment_kfs,
        zone_smoothing_w,
        maxine_upscale_factor,
        interp_enabled: !args.no_depth_interp,
        maxine_in_fused_batch: false, // Maxine disabled — SDK not available in devcontainer
    };

    // -----------------------------------------------------------------------
    // Stage 1: Keyframe detection (proxy decode → change detect → select)
    // -----------------------------------------------------------------------
    let kf_out = pipeline::keyframe::run_keyframe_stage(&pipeline_cfg, &info)?;

    // -----------------------------------------------------------------------
    // Stage 2: Feature extraction (fused RAUNE+Maxine+depth → paged store)
    // -----------------------------------------------------------------------
    let mut inf_server = InferenceServer::spawn(&InferenceConfig {
        skip_raune: true,
        skip_depth: true,
        ..inf_cfg
    }).context("failed to spawn inference server")?;

    inf_server.load_raune(
        raune_weights.as_deref(),
        raune_models_dir.as_deref(),
    ).context("failed to load RAUNE-Net for calibration")?;
    inf_server.load_depth(
        depth_model.as_deref(),
    ).context("failed to load Depth Anything for calibration")?;

    let feat_out = pipeline::feature::run_feature_stage(&pipeline_cfg, &info, inf_server, kf_out)?;

    // -----------------------------------------------------------------------
    // Stage 3: Calibration (zones → segments → LUT build → HSL → grader init)
    // -----------------------------------------------------------------------
    let cal_out = pipeline::calibration::run_calibration_stage(&pipeline_cfg, feat_out)?;

    // -----------------------------------------------------------------------
    // Stage 4: Grading render (full-res decode → depth interp → grade → encode)
    // -----------------------------------------------------------------------
    let frame_count = pipeline::grading::run_grading_stage(&pipeline_cfg, &info, encoder, cal_out)?;

    if info.frame_count > 0 && frame_count < info.frame_count {
        log::warn!(
            "Incomplete grading: {frame_count} frames processed, {} expected \
             (decoder may have stopped early)",
            info.frame_count
        );
    }
    log::info!("Done. Graded {frame_count} frames → {}", output.display());
    Ok(())
}

fn build_inference_config(
    python: &std::path::Path,
    raune_weights: Option<&std::path::Path>,
    raune_models_dir: Option<&std::path::Path>,
    depth_model: Option<&std::path::Path>,
    device: Option<String>,
    maxine_upscale_factor: u32,
) -> InferenceConfig {
    InferenceConfig {
        python_exe: python.to_path_buf(),
        raune_weights: raune_weights.map(|p| p.to_path_buf()),
        raune_models_dir: raune_models_dir.map(|p| p.to_path_buf()),
        skip_raune: false,
        depth_model: depth_model.map(|p| p.to_path_buf()),
        skip_depth: false,
        device,
        startup_timeout: Duration::from_secs(180),
        maxine: false, // Maxine requires NVIDIA VFX SDK — disabled when not available
        maxine_upscale_factor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::grading::lerp_depth;

    #[test]
    fn lerp_depth_at_zero() {
        let a: Vec<f32> = vec![0.1, 0.2, 0.3];
        let b: Vec<f32> = vec![0.5, 0.6, 0.7];
        let result = lerp_depth(&a, &b, 0.0);
        assert_eq!(result, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn lerp_depth_at_one() {
        let a: Vec<f32> = vec![0.1, 0.2, 0.3];
        let b: Vec<f32> = vec![0.5, 0.6, 0.7];
        let result = lerp_depth(&a, &b, 1.0);
        assert_eq!(result, vec![0.5, 0.6, 0.7]);
    }

    #[test]
    fn lerp_depth_at_half() {
        let a: Vec<f32> = vec![0.0, 0.4, 0.8];
        let b: Vec<f32> = vec![1.0, 0.8, 0.6];
        let result = lerp_depth(&a, &b, 0.5);
        for (r, e) in result.iter().zip([0.5f32, 0.6, 0.7].iter()) {
            assert!((r - e).abs() < 1e-6, "expected {e}, got {r}");
        }
    }

    #[test]
    fn lerp_depth_clamps_t() {
        let a: Vec<f32> = vec![0.1, 0.2];
        let b: Vec<f32> = vec![0.9, 0.8];
        assert_eq!(lerp_depth(&a, &b, 2.0), vec![0.9, 0.8]);
        assert_eq!(lerp_depth(&a, &b, -1.0), vec![0.1, 0.2]);
    }

    #[test]
    fn build_inference_config_defaults() {
        let python = PathBuf::from("/opt/dorea-venv/bin/python");
        let cfg = build_inference_config(&python, None, None, None, None, 2);
        assert!(cfg.maxine, "Maxine should be enabled for fused batch upscaling");
        assert!(!cfg.skip_raune, "RAUNE should not be skipped in config");
        assert!(!cfg.skip_depth, "depth should not be skipped in config");
    }
}
