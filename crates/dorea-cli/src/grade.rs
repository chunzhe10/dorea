// dorea grade — direct-mode RAUNE + OKLab delta grading.
//
// Resolves config, probes the input video, and spawns a Python subprocess
// that decodes, runs RAUNE at proxy resolution, computes the OKLab delta,
// upscales via bilinear interpolation, applies the delta in OKLab, and
// encodes the output. The heavy lifting lives in raune_filter.py; this
// binary is a thin orchestrator.

use std::path::PathBuf;
use clap::Parser;
use anyhow::{Context, Result};

use dorea_video::ffmpeg::{self, InputEncoding, OutputCodec};

use crate::pipeline::{self, PipelineConfig};

#[derive(Parser, Debug)]
#[command(name = "dorea", about = "Underwater video direct-mode color grading")]
pub struct GradeArgs {
    /// Input video file (MP4/MOV/MKV).
    pub input: PathBuf,

    /// Output video file. Default: <input-stem>_graded.<ext>
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// RAUNE weights path (config: [models].raune_weights)
    #[arg(long)]
    pub raune_weights: Option<PathBuf>,

    /// RAUNE models directory (config: [models].raune_models_dir)
    #[arg(long)]
    pub raune_models_dir: Option<PathBuf>,

    /// Python interpreter path (config: [models].python)
    #[arg(long)]
    pub python: Option<PathBuf>,

    /// RAUNE proxy long-edge resolution (config: [grade].raune_proxy_size, default: 1440)
    #[arg(long)]
    pub raune_proxy_size: Option<usize>,

    /// Frames per RAUNE batch (config: [grade].direct_batch_size, default: 8).
    /// fp16 RAUNE halves activation memory vs fp32; batch=8 fp16 ≈ batch=4 fp32
    /// footprint — known-safe on RTX 3060 (6 GB). Values above 8 show diminishing
    /// returns; 16+ regresses throughput due to per-frame upload overhead.
    #[arg(long)]
    pub direct_batch_size: Option<usize>,

    /// Input encoding: dlog-m, ilog, srgb (config: [grade].input_encoding, default: auto)
    #[arg(long)]
    pub input_encoding: Option<String>,

    /// Output codec: prores, hevc10, h264 (config: [grade].output_codec, default: auto)
    #[arg(long)]
    pub output_codec: Option<String>,

    /// Enable verbose (debug) logging.
    #[arg(short, long)]
    pub verbose: bool,
}

pub fn run(args: GradeArgs, cfg: &crate::config::DoreaConfig) -> Result<()> {
    // Resolve config → CLI → built-in defaults
    let python = args.python
        .or_else(|| cfg.models.python.clone())
        .unwrap_or_else(|| PathBuf::from("/opt/dorea-venv/bin/python"));
    let raune_weights = args.raune_weights
        .or_else(|| cfg.models.raune_weights.clone())
        .ok_or_else(|| anyhow::anyhow!(
            "RAUNE weights required — set [models].raune_weights in dorea.toml or pass --raune-weights"
        ))?;
    let raune_models_dir = args.raune_models_dir
        .or_else(|| cfg.models.raune_models_dir.clone())
        .ok_or_else(|| anyhow::anyhow!(
            "RAUNE models dir required — set [models].raune_models_dir in dorea.toml or pass --raune-models-dir"
        ))?;

    let raune_proxy_size = args.raune_proxy_size
        .or(cfg.grade.raune_proxy_size)
        .unwrap_or(1440);

    let batch_size = args.direct_batch_size
        .or(cfg.grade.direct_batch_size)
        .unwrap_or(8);

    // Validate batch size
    if batch_size == 0 {
        anyhow::bail!("--direct-batch-size must be >= 1");
    }
    if batch_size > 32 {
        anyhow::bail!(
            "--direct-batch-size {batch_size} exceeds safe limit of 32 \
             (would risk CUDA OOM on 6GB VRAM). Use a smaller value."
        );
    }
    if batch_size > 8 {
        log::warn!(
            "--direct-batch-size={batch_size}: values above 8 may regress \
             throughput on 6GB VRAM (RTX 3060) due to per-frame upload overhead \
             in _process_batch. Measured baseline: batch=8 ~4.36 fps, batch=16 ~3.47 fps."
        );
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

    let output = args.output.unwrap_or_else(|| {
        let stem = args.input.file_stem().unwrap_or_default().to_string_lossy();
        let ext = if output_codec == OutputCodec::ProRes { "mov" } else { "mp4" };
        args.input.with_file_name(format!("{stem}_graded.{ext}"))
    });

    log::info!("Grading: {} → {}", args.input.display(), output.display());

    let (proxy_w, proxy_h) = dorea_video::resize::proxy_dims(
        info.width, info.height, raune_proxy_size,
    );

    log::info!(
        "RAUNE proxy {}x{} (max {raune_proxy_size}), batch={batch_size}, output {}x{}",
        proxy_w, proxy_h, info.width, info.height,
    );

    let pipeline_cfg = PipelineConfig {
        input: args.input,
        input_encoding,
        output_codec,
        output,
        python,
        raune_weights,
        raune_models_dir,
        raune_proxy_size,
        batch_size,
    };

    let frame_count = pipeline::grading::run(&pipeline_cfg, &info)?;

    if info.frame_count > 0 && frame_count < info.frame_count {
        log::warn!(
            "Incomplete grading: {frame_count} frames processed, {} expected",
            info.frame_count
        );
    }
    log::info!("Done. Graded {frame_count} frames → {}", pipeline_cfg.output.display());
    Ok(())
}
