// `dorea grade` — end-to-end video grading pipeline.

use std::path::PathBuf;
use std::time::Duration;
use clap::Args;
use anyhow::{Context, Result};

use dorea_cal::Calibration;
use dorea_gpu::{grade_frame, GradeParams};
use dorea_video::ffmpeg::{self, FrameEncoder};
use dorea_video::inference::{InferenceConfig, InferenceServer};

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

    /// MSE threshold for keyframe detection (lower = more keyframes)
    #[arg(long, default_value = "0.005")]
    pub depth_skip_threshold: f32,

    /// Maximum frames between keyframes
    #[arg(long, default_value = "12")]
    pub depth_max_interval: usize,

    /// Disable temporal interpolation — run full pipeline on every frame
    #[arg(long)]
    pub no_depth_interp: bool,

    /// Frames between keyframe samples for auto-calibration
    #[arg(long, default_value = "30")]
    pub keyframe_interval: usize,

    /// Path to RAUNE-Net weights .pth (for auto-calibration)
    #[arg(long)]
    pub raune_weights: Option<PathBuf>,

    /// Path to RAUNE-Net checkout directory (contains models/raune_net.py).
    /// Also accepts the parent sea_thru_poc dir — auto-descends to RAUNE-Net/.
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

/// Compute normalized MSE between two same-length u8 slices.
/// Returns value in [0, 1] where 0 = identical.
fn frame_mse(a: &[u8], b: &[u8]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let sum_sq: f64 = a.iter().zip(b.iter())
        .map(|(&av, &bv)| {
            let d = av as f64 - bv as f64;
            d * d
        })
        .sum();
    (sum_sq / (n * 255.0 * 255.0)) as f32
}

/// Linearly interpolate between two graded u8 frames.
fn lerp_graded(a: &[u8], b: &[u8], t: f32) -> Vec<u8> {
    let t = t.clamp(0.0, 1.0);
    a.iter().zip(b.iter())
        .map(|(&va, &vb)| {
            let v = va as f32 + (vb as f32 - va as f32) * t;
            v.round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// A buffered frame waiting to be output once the next keyframe is graded.
struct BufferedFrame {
    index: u64,
    width: usize,
    height: usize,
}

/// Write all buffered frames by interpolating between bracketing keyframe graded outputs.
fn flush_buffer_graded(
    buffer: &mut Vec<BufferedFrame>,
    graded_before: &Option<Vec<u8>>,
    graded_after: &Option<Vec<u8>>,
    encoder: &mut FrameEncoder,
    frame_count: &mut u64,
    info: &ffmpeg::VideoInfo,
) -> Result<()> {
    let Some(before) = graded_before else {
        buffer.clear();
        return Ok(());
    };

    let n_buffered = buffer.len();
    let interval = (n_buffered + 1) as f32;

    let same_keyframe = match graded_after {
        Some(after) => std::ptr::eq(before.as_ptr(), after.as_ptr()),
        None => true,
    };

    for (buf_idx, bf) in buffer.drain(..).enumerate() {
        let output = if same_keyframe {
            before.clone()
        } else {
            let t = (buf_idx + 1) as f32 / interval;
            lerp_graded(before, graded_after.as_ref().unwrap(), t)
        };

        let expected = bf.width * bf.height * 3;
        if output.len() != expected {
            return Err(anyhow::anyhow!(
                "Interpolated frame size mismatch at frame {}: got {}, expected {}",
                bf.index, output.len(), expected
            ));
        }

        encoder.write_frame(&output).context("encoder write failed")?;
        *frame_count += 1;

        if *frame_count % 100 == 0 {
            let pct = *frame_count as f64 / info.frame_count.max(1) as f64 * 100.0;
            log::info!("Progress: {frame_count}/{} frames ({:.1}%)", info.frame_count, pct);
        }
    }
    Ok(())
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

    // Open encoder first — validate output path before doing expensive calibration.
    let audio_src = if info.has_audio { Some(args.input.as_path()) } else { None };
    let mut encoder = FrameEncoder::new(&output, info.width, info.height, info.fps, audio_src)
        .context("failed to spawn ffmpeg encoder")?;

    // Load or derive calibration
    let calibration = if let Some(cal_path) = &args.calibration {
        log::info!("Loading calibration from {}", cal_path.display());
        Calibration::load(cal_path).context("failed to load .dorea-cal file")?
    } else {
        log::info!("No calibration provided — auto-calibrating from keyframes");
        auto_calibrate(&args, &info)?
    };

    // Spawn inference server for per-frame depth (depth-only; RAUNE is not needed here).
    let inf_cfg = InferenceConfig {
        skip_raune: true,
        ..build_inference_config(&args)
    };
    let mut inf_server = InferenceServer::spawn(&inf_cfg)
        .context("failed to spawn inference server — check --python and --depth-model")?;

    // Decode and grade frames
    let frames = ffmpeg::decode_frames(&args.input, &info)
        .context("failed to spawn ffmpeg decoder")?;

    let mut frame_count = 0u64;
    let interp_enabled = !args.no_depth_interp;
    let scene_cut_threshold = args.depth_skip_threshold * 10.0;

    // Temporal interpolation state
    let mut last_keyframe_proxy: Option<Vec<u8>> = None;
    let mut last_keyframe_graded: Option<Vec<u8>> = None;
    let mut frame_buffer: Vec<BufferedFrame> = Vec::new();
    let mut frames_since_keyframe = 0usize;
    let max_buffer = (args.depth_max_interval as f32 * 1.5) as usize;

    for frame_result in frames {
        let frame = frame_result.context("frame decode error")?;

        // Downscale to proxy resolution (needed for MSE check)
        let (proxy_w, proxy_h) =
            dorea_video::resize::proxy_dims(frame.width, frame.height, args.proxy_size);
        let proxy_pixels = if proxy_w != frame.width || proxy_h != frame.height {
            dorea_video::resize::resize_rgb_bilinear(
                &frame.pixels, frame.width, frame.height, proxy_w, proxy_h,
            )
        } else {
            frame.pixels.clone()
        };

        // Determine if this frame is a keyframe
        let is_keyframe = if !interp_enabled {
            true
        } else if last_keyframe_proxy.is_none() {
            true // First frame
        } else {
            let mse = frame_mse(&proxy_pixels, last_keyframe_proxy.as_ref().unwrap());
            let is_scene_cut = mse > scene_cut_threshold;
            let exceeds_interval = frames_since_keyframe >= args.depth_max_interval;
            let exceeds_threshold = mse > args.depth_skip_threshold;
            let buffer_overflow = frame_buffer.len() >= max_buffer;

            if is_scene_cut {
                log::info!("Scene cut at frame {} (MSE={:.6}) — flushing buffer", frame.index, mse);
                // Flush buffer using last keyframe graded output (no forward interp across cuts)
                flush_buffer_graded(
                    &mut frame_buffer, &last_keyframe_graded, &last_keyframe_graded,
                    &mut encoder, &mut frame_count, &info,
                )?;
            }

            is_scene_cut || exceeds_interval || exceeds_threshold || buffer_overflow
        };

        if is_keyframe {
            // Full pipeline: depth inference + grading
            let (depth_proxy, dw, dh) = inf_server
                .run_depth(
                    &frame.index.to_string(),
                    &proxy_pixels, proxy_w, proxy_h, args.proxy_size,
                )
                .unwrap_or_else(|e| {
                    log::warn!("Depth inference failed for frame {}: {e} — uniform depth", frame.index);
                    (vec![0.5f32; proxy_w * proxy_h], proxy_w, proxy_h)
                });

            let depth = if dw == frame.width && dh == frame.height {
                depth_proxy
            } else {
                InferenceServer::upscale_depth(&depth_proxy, dw, dh, frame.width, frame.height)
            };

            let graded = grade_frame(
                &frame.pixels, &depth, frame.width, frame.height, &calibration, &params,
            ).map_err(|e| anyhow::anyhow!("Grading failed for frame {}: {e}", frame.index))?;

            // Flush any buffered frames — interpolate between previous and this keyframe's graded output
            if !frame_buffer.is_empty() {
                flush_buffer_graded(
                    &mut frame_buffer, &last_keyframe_graded, &Some(graded.clone()),
                    &mut encoder, &mut frame_count, &info,
                )?;
            }

            // Write this keyframe's graded output
            encoder.write_frame(&graded).context("encoder write failed")?;
            frame_count += 1;

            if frame_count % 100 == 0 {
                let pct = frame_count as f64 / info.frame_count.max(1) as f64 * 100.0;
                log::info!("Progress: {frame_count}/{} frames ({:.1}%)", info.frame_count, pct);
            }

            // Update keyframe state
            last_keyframe_proxy = Some(proxy_pixels);
            last_keyframe_graded = Some(graded);
            frames_since_keyframe = 0;
        } else {
            // Buffer this frame — will be interpolated when next keyframe is graded
            frame_buffer.push(BufferedFrame {
                index: frame.index,
                width: frame.width,
                height: frame.height,
            });
            frames_since_keyframe += 1;
        }
    }

    // Flush any remaining buffered frames (end of video — use last keyframe, no interpolation)
    if !frame_buffer.is_empty() {
        flush_buffer_graded(
            &mut frame_buffer, &last_keyframe_graded, &last_keyframe_graded,
            &mut encoder, &mut frame_count, &info,
        )?;
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

    // Calibration runs at proxy resolution so pixels, RAUNE target, and depth all match.
    // Max 1024px on the long edge, maintaining aspect ratio.
    let cal_max = 1024usize;
    let cal_scale = (cal_max as f64 / info.width.max(info.height) as f64).min(1.0);
    let kf_w = ((info.width as f64 * cal_scale) as usize).max(1);
    let kf_h = ((info.height as f64 * cal_scale) as usize).max(1);

    let mut inputs: Vec<CalibrationInput> = Vec::new();

    for i in 0..n_kf {
        let ts = (i as f64 + 0.5) * duration / n_kf as f64;
        let pixels = ffmpeg::extract_frame_at(&args.input, ts, kf_w, kf_h)
            .with_context(|| format!("failed to extract keyframe at {ts:.2}s"))?;

        let id = format!("kf{i:03}");

        // Run RAUNE-Net for target (at proxy resolution — output matches input size)
        let target = match inf_server.run_raune(&id, &pixels, kf_w, kf_h, kf_w) {
            Ok((t, _, _)) => t,
            Err(e) => {
                log::warn!("RAUNE-Net failed for {id}: {e} — using original as target");
                pixels.clone()
            }
        };

        // Run Depth Anything for depth map, upscale to proxy resolution
        let (depth_proxy, dw, dh) = match inf_server.run_depth(&id, &pixels, kf_w, kf_h, 518) {
            Ok(d) => d,
            Err(e) => {
                log::warn!("Depth failed for {id}: {e} — using uniform depth 0.5");
                (vec![0.5f32; kf_w * kf_h], kf_w, kf_h)
            }
        };

        let depth = InferenceServer::upscale_depth(&depth_proxy, dw, dh, kf_w, kf_h);

        inputs.push(CalibrationInput { pixels, target, depth, width: kf_w, height: kf_h });
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
        skip_raune: false,
        depth_model: args.depth_model.clone(),
        device: if args.cpu_only { Some("cpu".to_string()) } else { None },
        startup_timeout: Duration::from_secs(180),
    }
}
