// `dorea grade` — end-to-end video grading pipeline.

use std::path::PathBuf;
use std::time::Duration;
use clap::Args;
use anyhow::{Context, Result};

use dorea_cal::Calibration;
use dorea_gpu::{grade_frame, GradeParams};
use dorea_video::ffmpeg::{self, FrameEncoder};
use dorea_video::inference::{InferenceConfig, InferenceServer};
use dorea_video::scene;

#[derive(Args, Debug)]
pub struct GradeArgs {
    /// Input video file (MP4/MOV/MKV)
    #[arg(long)]
    pub input: PathBuf,

    /// Output video file [default: <input-stem>_graded.mp4]
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Pre-computed .dorea-cal calibration file.
    /// If omitted, auto-calibrates using inference subprocess.
    #[arg(long)]
    pub calibration: Option<PathBuf>,

    /// Warmth multiplier [0.0–2.0]
    #[arg(long, default_value = "1.0")]
    pub warmth: f32,

    /// LUT/HSL blend strength [0.0–1.0]
    #[arg(long, default_value = "0.8")]
    pub strength: f32,

    /// Ambiance contrast multiplier [0.0–1.0]
    #[arg(long, default_value = "1.0")]
    pub contrast: f32,

    /// Proxy resolution for inference (long edge, pixels) [default: 518]
    #[arg(long, default_value = "518")]
    pub proxy_size: usize,

    /// Frames between keyframe samples for auto-calibration
    #[arg(long, default_value = "30")]
    pub keyframe_interval: usize,

    /// Path to RAUNE-Net weights .pth (for auto-calibration)
    #[arg(long)]
    pub raune_weights: Option<PathBuf>,

    /// Path to RAUNE-Net models dir (sea_thru_poc directory)
    #[arg(long)]
    pub raune_models_dir: Option<PathBuf>,

    /// Path to Depth Anything V2 model directory
    #[arg(long)]
    pub depth_model: Option<PathBuf>,

    /// Python executable to use for the inference subprocess
    #[arg(long, default_value = "/opt/dorea-venv/bin/python")]
    pub python: PathBuf,

    /// Force CPU-only mode (no CUDA)
    #[arg(long)]
    pub cpu_only: bool,

    /// Enable verbose logging
    #[arg(short, long)]
    pub verbose: bool,
}

pub fn run(args: GradeArgs) -> Result<()> {
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

    // Determine grading parameters
    let params = GradeParams {
        warmth: args.warmth,
        strength: args.strength,
        contrast: args.contrast,
    };

    // Load or derive calibration
    let calibration = if let Some(cal_path) = &args.calibration {
        log::info!("Loading calibration from {}", cal_path.display());
        Calibration::load(cal_path).context("failed to load .dorea-cal file")?
    } else {
        log::info!("No calibration provided — auto-calibrating from keyframes");
        auto_calibrate(&args, &info)?
    };

    // Open encoder
    let audio_src = if info.has_audio { Some(args.input.as_path()) } else { None };
    let mut encoder = FrameEncoder::new(&output, info.width, info.height, info.fps, audio_src)
        .context("failed to spawn ffmpeg encoder")?;

    // Spawn inference server for per-frame depth (depth-only; RAUNE is not needed here).
    let inf_cfg = InferenceConfig {
        raune_weights: None,
        raune_models_dir: None,
        ..build_inference_config(&args)
    };
    let mut inf_server = InferenceServer::spawn(&inf_cfg)
        .context("failed to spawn inference server — check --python and --depth-model")?;

    // Decode and grade frames
    let frames = ffmpeg::decode_frames(&args.input, &info)
        .context("failed to spawn ffmpeg decoder")?;

    let mut prev_frame: Option<Vec<u8>> = None;
    let mut frame_count = 0u64;

    for frame_result in frames {
        let frame = frame_result.context("frame decode error")?;

        // Scene change detection
        let is_cut = prev_frame.as_ref()
            .map(|prev| scene::is_scene_change(prev, &frame.pixels))
            .unwrap_or(false);

        if is_cut {
            log::info!("Scene change detected at frame {}", frame.index);
            // TODO(Phase 3+): trigger re-calibration on scene cut
        }

        // Run depth inference at proxy resolution
        let (depth_proxy, dw, dh) = inf_server
            .run_depth(
                &frame.index.to_string(),
                &frame.pixels,
                frame.width,
                frame.height,
                args.proxy_size,
            )
            .unwrap_or_else(|e| {
                log::warn!("Depth inference failed for frame {}: {e} — using uniform depth", frame.index);
                let n = frame.width * frame.height;
                (vec![0.5f32; n], frame.width, frame.height)
            });

        // Upscale depth to full resolution
        let depth = if dw == frame.width && dh == frame.height {
            depth_proxy
        } else {
            InferenceServer::upscale_depth(&depth_proxy, dw, dh, frame.width, frame.height)
        };

        // Apply grading
        let graded = grade_frame(
            &frame.pixels,
            &depth,
            frame.width,
            frame.height,
            &calibration,
            &params,
        )
        .unwrap_or_else(|e| {
            log::warn!("Grading failed for frame {}: {e} — passing through", frame.index);
            frame.pixels.clone()
        });

        encoder.write_frame(&graded).context("encoder write failed")?;

        prev_frame = Some(frame.pixels);
        frame_count += 1;

        if frame_count % 100 == 0 {
            let pct = frame_count as f64 / info.frame_count.max(1) as f64 * 100.0;
            log::info!("Progress: {frame_count}/{} frames ({:.1}%)", info.frame_count, pct);
        }
    }

    // Finalize
    let _ = inf_server.shutdown();
    encoder.finish().context("ffmpeg encoder failed to finalize")?;

    log::info!("Done. Graded {frame_count} frames → {}", output.display());
    Ok(())
}

fn auto_calibrate(args: &GradeArgs, info: &ffmpeg::VideoInfo) -> Result<Calibration> {
    use crate::calibrate::{run_calibration_from_frames, CalibrationInput};

    let duration = info.duration_secs;
    let interval_secs = args.keyframe_interval as f64 / info.fps.max(1.0);
    let n_kf = ((duration / interval_secs) as usize).clamp(1, 20);

    log::info!("Auto-calibrating from {n_kf} keyframes...");

    // Extract keyframes
    let inf_cfg = build_inference_config(args);
    let mut inf_server = InferenceServer::spawn(&inf_cfg)
        .context("failed to spawn inference server for auto-calibration")?;

    let mut inputs: Vec<CalibrationInput> = Vec::new();

    for i in 0..n_kf {
        let ts = (i as f64 + 0.5) * duration / n_kf as f64;
        let pixels = ffmpeg::extract_frame_at(&args.input, ts, info.width, info.height)
            .with_context(|| format!("failed to extract keyframe at {ts:.2}s"))?;

        let id = format!("kf{i:03}");

        // Run RAUNE-Net for target
        let target = match inf_server.run_raune(&id, &pixels, info.width, info.height, 1024) {
            Ok((t, _, _)) => t,
            Err(e) => {
                log::warn!("RAUNE-Net failed for {id}: {e} — using original as target");
                pixels.clone()
            }
        };

        // Run Depth Anything for depth map
        let (depth_proxy, dw, dh) = match inf_server.run_depth(&id, &pixels, info.width, info.height, 518) {
            Ok(d) => d,
            Err(e) => {
                log::warn!("Depth failed for {id}: {e} — using uniform depth 0.5");
                let n = info.width * info.height;
                (vec![0.5f32; n], info.width, info.height)
            }
        };

        let depth = InferenceServer::upscale_depth(&depth_proxy, dw, dh, info.width, info.height);

        inputs.push(CalibrationInput { pixels, target, depth, width: info.width, height: info.height });
    }

    let _ = inf_server.shutdown();

    run_calibration_from_frames(&inputs)
        .context("calibration failed")
}

fn build_inference_config(args: &GradeArgs) -> InferenceConfig {
    InferenceConfig {
        python_exe: args.python.clone(),
        raune_weights: args.raune_weights.clone(),
        raune_models_dir: args.raune_models_dir.clone(),
        depth_model: args.depth_model.clone(),
        device: if args.cpu_only { Some("cpu".to_string()) } else { None },
        startup_timeout: Duration::from_secs(180),
    }
}
