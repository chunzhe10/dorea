// `dorea grade` — end-to-end video grading pipeline.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use clap::Args;
use anyhow::{Context, Result};

use dorea_cal::Calibration;
use dorea_gpu::{grade_frame, GradeParams};
use dorea_video::ffmpeg::{self, FrameEncoder};
use dorea_video::inference::{DepthBatchItem, InferenceConfig, InferenceServer};

#[cfg(feature = "cuda")]
use dorea_gpu::cuda::CudaGrader;

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
    if a.is_empty() {
        return 0.0;
    }
    let n = a.len() as f64;
    let sum_sq: f64 = a.iter().zip(b.iter())
        .map(|(&av, &bv)| {
            let d = av as f64 - bv as f64;
            d * d
        })
        .sum();
    (sum_sq / (n * 255.0 * 255.0)) as f32
}

/// Linearly interpolate between two f32 depth maps.
fn lerp_depth(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    let t = t.clamp(0.0, 1.0);
    a.iter().zip(b.iter())
        .map(|(&va, &vb)| va + (vb - va) * t)
        .collect()
}

/// Maximum number of keyframes per depth inference batch.
const DEPTH_BATCH_SIZE: usize = 32;

/// A keyframe collected during the proxy-decode pass.
struct KeyframeEntry {
    frame_index: u64,
    proxy_pixels: Vec<u8>,
    /// True if this keyframe follows a scene cut (suppresses lerp across cuts).
    scene_cut_before: bool,
}

/// Grade a single frame, reusing a pre-initialized CudaGrader when available.
///
/// Falls back to `grade_frame()` (which creates its own CudaGrader per-call) when the
/// grader is `None` (e.g. CUDA init failed at startup).
#[cfg(feature = "cuda")]
fn grade_with_grader(
    grader: Option<&CudaGrader>,
    pixels: &[u8],
    depth: &[f32],
    width: usize,
    height: usize,
    calibration: &Calibration,
    params: &GradeParams,
) -> Result<Vec<u8>, dorea_gpu::GpuError> {
    if let Some(g) = grader {
        dorea_gpu::grade_frame_with_grader(g, pixels, depth, width, height)
    } else {
        grade_frame(pixels, depth, width, height, calibration, params)
    }
}

pub fn run(args: GradeArgs) -> Result<()> {
    if args.cpu_only {
        anyhow::bail!(
            "--cpu-only is no longer supported for dorea grade; GPU (CUDA) is required. \
             Use dorea preview for CPU-only workflows."
        );
    }
    if args.depth_max_interval == 0 {
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

    // Spawn inference server (depth-only; RAUNE is not needed for grading).
    let inf_cfg = InferenceConfig {
        skip_raune: true,
        ..build_inference_config(&args)
    };
    let mut inf_server = InferenceServer::spawn(&inf_cfg)
        .context("failed to spawn inference server — check --python and --depth-model")?;

    // Initialize CUDA grader (loads PTX once, builds combined LUT, reuses across all frames)
    #[cfg(feature = "cuda")]
    let cuda_grader = match CudaGrader::new(&calibration, &params) {
        Ok(g) => {
            log::info!("CUDA grader initialized (cudarc + PTX modules loaded)");
            Some(g)
        }
        Err(e) => {
            log::warn!("CUDA grader init failed: {e} — will use per-frame fallback");
            None
        }
    };

    let interp_enabled = !args.no_depth_interp;
    let scene_cut_threshold = args.depth_skip_threshold * 10.0;

    // -----------------------------------------------------------------------
    // Pass 1: proxy decode + MSE keyframe detection
    // -----------------------------------------------------------------------
    let (proxy_w, proxy_h) = dorea_video::resize::proxy_dims(info.width, info.height, args.proxy_size);
    let proxy_frames = ffmpeg::decode_frames_scaled(&args.input, &info, proxy_w, proxy_h)
        .context("failed to spawn ffmpeg proxy decoder")?;

    let mut keyframes: Vec<KeyframeEntry> = Vec::new();
    let mut last_proxy: Option<Vec<u8>> = None;
    let mut frames_since_kf = 0usize;

    for frame_result in proxy_frames {
        let frame = frame_result.context("proxy frame decode error")?;
        let mse = last_proxy.as_ref().map(|lp| frame_mse(&frame.pixels, lp));
        let scene_cut = mse.map_or(false, |m| m > scene_cut_threshold);
        let is_keyframe = !interp_enabled
            || last_proxy.is_none()
            || scene_cut
            || frames_since_kf >= args.depth_max_interval
            || mse.map_or(false, |m| m > args.depth_skip_threshold);

        if is_keyframe {
            if scene_cut {
                log::info!(
                    "Scene cut at frame {} (MSE={:.6})",
                    frame.index,
                    mse.unwrap_or(0.0),
                );
            }
            keyframes.push(KeyframeEntry {
                frame_index: frame.index,
                proxy_pixels: frame.pixels.clone(),
                scene_cut_before: scene_cut,
            });
            last_proxy = Some(frame.pixels);
            frames_since_kf = 0;
        } else {
            frames_since_kf += 1;
        }
    }
    log::info!("Pass 1 complete: {} keyframes detected", keyframes.len());

    // -----------------------------------------------------------------------
    // Batch depth inference
    // -----------------------------------------------------------------------
    let batch_items: Vec<DepthBatchItem> = keyframes.iter().enumerate().map(|(i, kf)| {
        DepthBatchItem {
            id: format!("kf_{i:06}"),
            pixels: kf.proxy_pixels.clone(),
            width: proxy_w,
            height: proxy_h,
            max_size: args.proxy_size,
        }
    }).collect();

    let mut keyframe_depths: HashMap<u64, Vec<f32>> = HashMap::new();
    for (chunk_kfs, chunk_items) in keyframes
        .chunks(DEPTH_BATCH_SIZE)
        .zip(batch_items.chunks(DEPTH_BATCH_SIZE))
    {
        let results = inf_server.run_depth_batch(chunk_items)
            .unwrap_or_else(|e| {
                log::warn!("Depth batch failed: {e} — using uniform depth 0.5");
                chunk_items.iter().map(|it| {
                    (it.id.clone(), vec![0.5f32; proxy_w * proxy_h], proxy_w, proxy_h)
                }).collect()
            });

        for (kf, (_, depth_raw, dw, dh)) in chunk_kfs.iter().zip(results.iter()) {
            let depth = if *dw == info.width && *dh == info.height {
                depth_raw.clone()
            } else {
                InferenceServer::upscale_depth(depth_raw, *dw, *dh, info.width, info.height)
            };
            keyframe_depths.insert(kf.frame_index, depth);
        }
    }
    log::info!("Batch depth inference complete ({} keyframes)", keyframes.len());

    // Ordered list of (frame_index, scene_cut_before) for pass-2 lerp lookup.
    let kf_index_list: Vec<(u64, bool)> = keyframes.iter()
        .map(|kf| (kf.frame_index, kf.scene_cut_before))
        .collect();

    // -----------------------------------------------------------------------
    // Pass 2: full-resolution decode + depth lookup + grade + encode
    // -----------------------------------------------------------------------
    let frames = ffmpeg::decode_frames(&args.input, &info)
        .context("failed to spawn ffmpeg full-res decoder")?;

    let mut kf_cursor = 0usize;
    let mut frame_count = 0u64;

    for frame_result in frames {
        let frame = frame_result.context("frame decode error")?;
        let fi = frame.index;

        // Advance cursor: kf_index_list[kf_cursor].0 is the most recent keyframe ≤ fi
        while kf_cursor + 1 < kf_index_list.len() && kf_index_list[kf_cursor + 1].0 <= fi {
            kf_cursor += 1;
        }

        let (prev_kf_idx, _) = kf_index_list[kf_cursor];
        let prev_depth = keyframe_depths
            .get(&prev_kf_idx)
            .expect("prev keyframe depth missing — logic error");

        let depth = if fi == prev_kf_idx {
            prev_depth.clone()
        } else if let Some(&(next_kf_idx, scene_cut_before_next)) = kf_index_list.get(kf_cursor + 1) {
            if scene_cut_before_next {
                prev_depth.clone() // Don't lerp across scene cut
            } else {
                let next_depth = keyframe_depths
                    .get(&next_kf_idx)
                    .expect("next keyframe depth missing — logic error");
                let t = (fi - prev_kf_idx) as f32 / (next_kf_idx - prev_kf_idx) as f32;
                lerp_depth(prev_depth, next_depth, t)
            }
        } else {
            prev_depth.clone() // Past last keyframe
        };

        #[cfg(feature = "cuda")]
        let graded = grade_with_grader(
            cuda_grader.as_ref(),
            &frame.pixels, &depth, frame.width, frame.height, &calibration, &params,
        ).map_err(|e| anyhow::anyhow!("Grading failed for frame {fi}: {e}"))?;
        #[cfg(not(feature = "cuda"))]
        let graded = grade_frame(
            &frame.pixels, &depth, frame.width, frame.height, &calibration, &params,
        ).map_err(|e| anyhow::anyhow!("Grading failed for frame {fi}: {e}"))?;

        encoder.write_frame(&graded).context("encoder write failed")?;
        frame_count += 1;

        if frame_count % 100 == 0 {
            let pct = frame_count as f64 / info.frame_count.max(1) as f64 * 100.0;
            log::info!("Progress: {frame_count}/{} frames ({:.1}%)", info.frame_count, pct);
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_mse_identical_is_zero() {
        let a: Vec<u8> = vec![100, 150, 200, 50, 75, 125];
        assert_eq!(frame_mse(&a, &a), 0.0);
    }

    #[test]
    fn frame_mse_empty_is_zero() {
        assert_eq!(frame_mse(&[], &[]), 0.0);
    }

    #[test]
    fn frame_mse_opposite_is_one() {
        let a: Vec<u8> = vec![0, 0, 0];
        let b: Vec<u8> = vec![255, 255, 255];
        let mse = frame_mse(&a, &b);
        assert!((mse - 1.0).abs() < 1e-5, "expected ~1.0, got {mse}");
    }

    #[test]
    fn frame_mse_known_value() {
        // All pixels differ by 1 → MSE = 1/(255²) ≈ 0.0000154
        let a: Vec<u8> = vec![100, 100, 100];
        let b: Vec<u8> = vec![101, 101, 101];
        let mse = frame_mse(&a, &b);
        let expected = 1.0 / (255.0 * 255.0);
        assert!((mse as f64 - expected).abs() < 1e-8, "expected {expected}, got {mse}");
    }

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
        // t > 1 should clamp to 1
        assert_eq!(lerp_depth(&a, &b, 2.0), vec![0.9, 0.8]);
        // t < 0 should clamp to 0
        assert_eq!(lerp_depth(&a, &b, -1.0), vec![0.1, 0.2]);
    }
}
