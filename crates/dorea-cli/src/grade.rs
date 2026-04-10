// `dorea grade` — end-to-end video grading pipeline.
//
// Orchestrator: resolves config, probes video, then chains four pipeline stages:
//   KeyframeStage → FeatureStage → CalibrationStage → GradingStage

use std::path::PathBuf;
use std::time::Duration;
use clap::Args;
use anyhow::{Context, Result};

use dorea_video::ffmpeg::{self, FrameEncoder, InputEncoding, OutputCodec};
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

    /// Input encoding: dlog-m, ilog, srgb (default: auto-detect from container)
    #[arg(long)]
    pub input_encoding: Option<String>,

    /// Output codec: prores, hevc10, h264 (default: auto for source bit depth)
    #[arg(long)]
    pub output_codec: Option<String>,

    /// Enabled grading stages (comma-separated). Default: all.
    /// Stages: hsl, ambiance, warmth, vibrance, strength, depth_mod, depth_dither, yolo_mask
    #[arg(long)]
    pub stages: Option<String>,

    /// Flat mode: RAUNE-only LUT (equivalent to --stages with no stages)
    #[arg(long)]
    pub flat: bool,

    /// Direct mode: per-frame RAUNE, no LUT pipeline (skips keyframe/calibration/CUDA stages)
    #[arg(long)]
    pub direct: bool,

    /// RAUNE proxy resolution for direct mode (long-edge pixels, default: 1920)
    #[arg(long)]
    pub raune_proxy_size: Option<usize>,
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

    // Probe input
    let info = ffmpeg::probe(&args.input)
        .context("ffprobe failed — is ffmpeg installed?")?;
    log::info!(
        "Input: {}x{} @ {:.3}fps, {:.1}s ({} frames)",
        info.width, info.height, info.fps, info.duration_secs, info.frame_count
    );

    // Resolve input encoding: CLI flag → config → auto-detect
    let input_encoding = args.input_encoding.as_deref()
        .or(cfg.grade.input_encoding.as_deref())
        .map(|s| s.parse::<InputEncoding>().map_err(|e| anyhow::anyhow!("invalid --input-encoding: {e}")))
        .transpose()?
        .unwrap_or_else(|| InputEncoding::auto_detect(&info, &args.input));

    // Resolve output codec: CLI flag → config → auto based on input encoding
    let output_codec = args.output_codec.as_deref()
        .or(cfg.grade.output_codec.as_deref())
        .map(|s| s.parse::<OutputCodec>().map_err(|e| anyhow::anyhow!("invalid --output-codec: {e}")))
        .transpose()?
        .unwrap_or_else(|| {
            if input_encoding.is_10bit() { OutputCodec::ProRes } else { OutputCodec::H264 }
        });

    log::info!("Encoding: input={input_encoding}, output={output_codec}, 10-bit={}", output_codec.is_10bit());

    let output = args.output.clone().unwrap_or_else(|| {
        let stem = args.input.file_stem().unwrap_or_default().to_string_lossy();
        let ext = if output_codec == OutputCodec::ProRes { "mov" } else { "mp4" };
        args.input.with_file_name(format!("{stem}_graded.{ext}"))
    });

    log::info!("Grading: {} → {}", args.input.display(), output.display());

    // -----------------------------------------------------------------------
    // Direct mode: per-frame RAUNE, skip LUT pipeline entirely
    // -----------------------------------------------------------------------
    let direct_mode = args.direct || cfg.grade.mode.as_deref() == Some("direct");
    if direct_mode {
        let raune_proxy_size = args.raune_proxy_size
            .or(cfg.grade.raune_proxy_size)
            .unwrap_or(1080_usize);

        let rw = raune_weights.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--direct requires RAUNE weights (set [models].raune_weights in dorea.toml)"))?;
        let rmd = raune_models_dir.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--direct requires RAUNE models dir (set [models].raune_models_dir in dorea.toml)"))?;

        let (proxy_w, proxy_h) = dorea_video::resize::proxy_dims(
            info.width, info.height, raune_proxy_size,
        );

        log::info!(
            "Direct mode: single-process OKLab transfer, RAUNE proxy {}x{} (max {raune_proxy_size}), output {}x{}",
            proxy_w, proxy_h, info.width, info.height,
        );

        let pipeline_cfg = PipelineConfig {
            input: args.input.clone(),
            warmth, strength, contrast, proxy_size,
            depth_skip_threshold, depth_max_interval, fused_batch_size,
            depth_zones, base_lut_zones, scene_threshold, min_segment_kfs,
            zone_smoothing_w, maxine_upscale_factor,
            interp_enabled: false,
            maxine_in_fused_batch: false,
            input_encoding, output_codec,
            stage_mask: 0,
        };

        let direct_cfg = pipeline::grading::DirectModeConfig {
            python: python.clone(),
            raune_weights: rw.clone(),
            raune_models_dir: rmd.clone(),
            raune_proxy_size,
            batch_size: 4,
            output: output.clone(),
        };

        let frame_count = pipeline::grading::run_grading_stage_direct(
            &pipeline_cfg, &info, &direct_cfg,
        )?;

        if info.frame_count > 0 && frame_count < info.frame_count {
            log::warn!(
                "Incomplete direct-mode grading: {frame_count} frames processed, {} expected",
                info.frame_count
            );
        }
        log::info!("Done. Direct-mode graded {frame_count} frames → {}", output.display());
        return Ok(());
    }

    // Open encoder first — validate output path before doing expensive calibration.
    let audio_src = if info.has_audio { Some(args.input.as_path()) } else { None };
    let encoder = if output_codec.is_10bit() {
        FrameEncoder::new_10bit(&output, info.width, info.height, info.fps, output_codec, audio_src)
            .context("failed to spawn 10-bit ffmpeg encoder")?
    } else {
        FrameEncoder::new(&output, info.width, info.height, info.fps, audio_src)
            .context("failed to spawn ffmpeg encoder")?
    };

    let inf_cfg = build_inference_config(
        &python, raune_weights.as_deref(), raune_models_dir.as_deref(),
        depth_model.as_deref(), device.clone(), maxine_upscale_factor,
    );

    // Resolve stage mask: --flat → 0 (no stages), --stages → parse, default → all
    let stage_mask = if args.flat {
        0u32
    } else if let Some(ref stages_str) = args.stages {
        dorea_gpu::stages::parse(stages_str)
            .map_err(|e| anyhow::anyhow!("invalid --stages: {e}"))?
    } else if let Some(ref stages_str) = cfg.grade.stages {
        dorea_gpu::stages::parse(stages_str)
            .map_err(|e| anyhow::anyhow!("invalid [grade].stages config: {e}"))?
    } else {
        dorea_gpu::stages::ALL
    };

    // In flat mode (stage_mask=0), force single zone — no depth stratification.
    let depth_zones = if stage_mask == 0 { 1 } else { depth_zones };
    let base_lut_zones = if stage_mask == 0 { 1 } else { base_lut_zones };

    log::info!("Stages: {}", dorea_gpu::stages::format(stage_mask));

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
        input_encoding,
        output_codec,
        stage_mask,
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
        assert!(!cfg.maxine, "Maxine should be disabled — SDK not available in devcontainer");
        assert!(!cfg.skip_raune, "RAUNE should not be skipped in config");
        assert!(!cfg.skip_depth, "depth should not be skipped in config");
    }

    #[test]
    fn default_encoding_for_8bit_h264() {
        use dorea_video::ffmpeg::{InputEncoding, OutputCodec, VideoInfo};
        let info = VideoInfo {
            width: 1920, height: 1080, fps: 30.0, duration_secs: 10.0,
            frame_count: 300, has_audio: true,
            codec_name: "h264".to_string(),
            pix_fmt: "yuv420p".to_string(),
            bits_per_component: 8,
        };
        let enc = InputEncoding::auto_detect(&info, std::path::Path::new("clip.mp4"));
        assert_eq!(enc, InputEncoding::Srgb);
        assert!(!enc.is_10bit());
        let codec = if enc.is_10bit() { OutputCodec::ProRes } else { OutputCodec::H264 };
        assert_eq!(codec, OutputCodec::H264);
    }

    #[test]
    fn default_encoding_for_10bit_hevc() {
        use dorea_video::ffmpeg::{InputEncoding, OutputCodec, VideoInfo};
        let info = VideoInfo {
            width: 3840, height: 2160, fps: 29.97, duration_secs: 3.0,
            frame_count: 90, has_audio: true,
            codec_name: "hevc".to_string(),
            pix_fmt: "yuv420p10le".to_string(),
            bits_per_component: 10,
        };
        let enc = InputEncoding::auto_detect(&info, std::path::Path::new("clip.mp4"));
        assert_eq!(enc, InputEncoding::DLogM);
        assert!(enc.is_10bit());
        let codec = if enc.is_10bit() { OutputCodec::ProRes } else { OutputCodec::H264 };
        assert_eq!(codec, OutputCodec::ProRes);
    }
}
