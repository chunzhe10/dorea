// `dorea grade` — end-to-end video grading pipeline.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use clap::Args;
use anyhow::{Context, Result};

use dorea_cal::Calibration;
use dorea_gpu::{grade_frame, GradeParams};
use dorea_video::ffmpeg::{self, FrameEncoder};
use dorea_video::inference::{RauneDepthBatchItem, InferenceConfig, InferenceServer};
use crate::change_detect::{ChangeDetector, MseDetector};

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

    /// Warmth multiplier [0.0–2.0]
    #[arg(long, default_value = "1.0")]
    pub warmth: f32,

    /// LUT/HSL blend strength [0.0–1.0]
    #[arg(long, default_value = "0.8")]
    pub strength: f32,

    /// Ambiance contrast multiplier [0.0–1.0]
    #[arg(long, default_value = "1.0")]
    pub contrast: f32,

    /// Proxy resolution for inference (long edge, pixels) [default: 1080]
    #[arg(long, default_value = "1080")]
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

    /// Path to RAUNE-Net weights .pth (for auto-calibration)
    #[arg(long)]
    pub raune_weights: Option<PathBuf>,

    /// Path to RAUNE-Net checkout directory (contains models/raune_net.py).
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

    /// Disable Maxine AI enhancement preprocessing (Maxine is attempted by default)
    #[arg(long)]
    pub no_maxine: bool,

    /// Disable Maxine artifact reduction before upscale [default: enabled]
    #[arg(long)]
    pub no_maxine_artifact_reduction: bool,

    /// Maxine super-resolution upscale factor [default: 2]
    #[arg(long, default_value = "2")]
    pub maxine_upscale_factor: u32,
}

/// RAII guard that deletes a temp file when dropped.
struct TempFileGuard(Option<std::path::PathBuf>);

impl TempFileGuard {
    fn new(path: std::path::PathBuf) -> Self {
        Self(Some(path))
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if let Some(ref p) = self.0 {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// Linearly interpolate between two f32 depth maps.
fn lerp_depth(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    let t = t.clamp(0.0, 1.0);
    a.iter().zip(b.iter())
        .map(|(&va, &vb)| va + (vb - va) * t)
        .collect()
}

/// Maximum frames per fused RAUNE+depth inference batch.
/// Proxy-res frames (~518px long edge); 32 frames ≈ 12–15 MB RAUNE input on GPU.
const FUSED_BATCH_SIZE: usize = 32;

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

    let use_maxine = !args.no_maxine;
    if use_maxine {
        let valid_factors = [2u32, 3, 4];
        if !valid_factors.contains(&args.maxine_upscale_factor) {
            anyhow::bail!(
                "--maxine-upscale-factor {} is not supported. Supported: {:?}",
                args.maxine_upscale_factor, valid_factors,
            );
        }
        log::info!(
            "Maxine enabled: upscale_factor={}, artifact_reduction={}",
            args.maxine_upscale_factor,
            !args.no_maxine_artifact_reduction,
        );
    }

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

    let interp_enabled = !args.no_depth_interp;

    // Spawn ONE inference server for the entire run.
    // Starts with Maxine loaded only (RAUNE+Depth loaded lazily after Pass 1).
    let maxine_start_cfg = InferenceConfig {
        skip_raune: true,
        skip_depth: true,
        ..build_inference_config(&args)
    };
    let mut inf_server = InferenceServer::spawn(&maxine_start_cfg)
        .context("failed to spawn inference server")?;

    let maxine_temp_path: Option<std::path::PathBuf> = if use_maxine {
        Some(std::env::temp_dir().join(format!("dorea_maxine_{}.mkv", std::process::id())))
    } else {
        None
    };
    let _maxine_temp_guard = maxine_temp_path.as_ref().map(|p| TempFileGuard::new(p.clone()));

    // -----------------------------------------------------------------------
    // Pass 1: decode + optional Maxine enhance + proxy downscale + keyframe detect
    // -----------------------------------------------------------------------
    let (proxy_w, proxy_h) = dorea_video::resize::proxy_dims(info.width, info.height, args.proxy_size);

    let mut keyframes: Vec<KeyframeEntry> = Vec::new();
    let mut detector: Box<dyn ChangeDetector> = Box::new(MseDetector::default());
    let mut frames_since_kf = 0usize;
    let scene_cut_threshold = args.depth_skip_threshold * 10.0;

    if use_maxine {
        // Maxine path: full-res decode → enhance → write temp → proxy downscale → keyframe detect
        use dorea_video::resize::resize_rgb_bilinear;

        let temp_path = maxine_temp_path.as_ref().unwrap();
        let mut temp_enc = FrameEncoder::new_lossless_temp(
            temp_path, info.width, info.height, info.fps,
        ).context("failed to create Maxine temp encoder")?;

        let full_frames = ffmpeg::decode_frames(&args.input, &info)
            .context("failed to spawn full-res decoder for Maxine pass")?;

        for frame_result in full_frames {
            let frame = frame_result.context("Maxine pass frame decode error")?;

            let maxine_full = inf_server.enhance(
                &frame.index.to_string(),
                &frame.pixels,
                frame.width,
                frame.height,
                !args.no_maxine_artifact_reduction,
            ).unwrap_or_else(|e| {
                log::warn!("enhance() IPC failed for frame {} — using original: {e}", frame.index);
                frame.pixels.clone()
            });

            temp_enc.write_frame(&maxine_full)
                .context("failed to write Maxine-enhanced frame to temp file")?;

            let maxine_proxy = if proxy_w == frame.width && proxy_h == frame.height {
                maxine_full.clone()
            } else {
                resize_rgb_bilinear(&maxine_full, frame.width, frame.height, proxy_w, proxy_h)
            };

            let change = detector.score(&maxine_proxy);
            let scene_cut = change < f32::MAX && change > scene_cut_threshold;
            let is_keyframe = !interp_enabled
                || keyframes.is_empty()
                || scene_cut
                || frames_since_kf >= args.depth_max_interval
                || (change < f32::MAX && change > args.depth_skip_threshold);

            if is_keyframe {
                if scene_cut {
                    log::info!("Scene cut at frame {} (change={:.6})", frame.index, change);
                    detector.reset();
                }
                keyframes.push(KeyframeEntry {
                    frame_index: frame.index,
                    proxy_pixels: maxine_proxy.clone(),
                    scene_cut_before: scene_cut,
                });
                detector.set_reference(&maxine_proxy);
                frames_since_kf = 0;
            } else {
                frames_since_kf += 1;
            }
        }

        temp_enc.finish().context("failed to finalize Maxine temp file")?;
        log::info!(
            "Maxine Pass 1 complete: {} keyframes, temp file: {}",
            keyframes.len(),
            temp_path.display(),
        );
    } else {
        // No-Maxine path: proxy decode (existing behaviour)
        let proxy_frames = ffmpeg::decode_frames_scaled(&args.input, &info, proxy_w, proxy_h)
            .context("failed to spawn ffmpeg proxy decoder")?;

        for frame_result in proxy_frames {
            let frame = frame_result.context("proxy frame decode error")?;
            let change = detector.score(&frame.pixels);
            let scene_cut = change < f32::MAX && change > scene_cut_threshold;
            let is_keyframe = !interp_enabled
                || keyframes.is_empty()
                || scene_cut
                || frames_since_kf >= args.depth_max_interval
                || (change < f32::MAX && change > args.depth_skip_threshold);

            if is_keyframe {
                if scene_cut {
                    log::info!(
                        "Scene cut at frame {} (change={:.6})",
                        frame.index,
                        change,
                    );
                    detector.reset();
                }
                keyframes.push(KeyframeEntry {
                    frame_index: frame.index,
                    proxy_pixels: frame.pixels.clone(),
                    scene_cut_before: scene_cut,
                });
                detector.set_reference(&frame.pixels);
                frames_since_kf = 0;
            } else {
                frames_since_kf += 1;
            }
        }
        log::info!("Pass 1 complete: {} keyframes detected", keyframes.len());
    }

    anyhow::ensure!(
        !keyframes.is_empty(),
        "pass 1 detected no keyframes — video may be empty or undecodable"
    );

    // -----------------------------------------------------------------------
    // Calibration + depth cache
    // -----------------------------------------------------------------------
    // Single-server lifecycle: Maxine unloaded here, then RAUNE+Depth loaded
    // for fused calibration batch. Server shut down after calibration; VRAM
    // freed for Pass 2 CUDA grading.

    // -----------------------------------------------------------------------
    // Model lifecycle transition: Maxine → RAUNE + Depth
    // -----------------------------------------------------------------------
    if use_maxine {
        inf_server.unload_maxine()
            .unwrap_or_else(|e| log::warn!("unload_maxine failed (non-fatal): {e}"));
        log::info!("Maxine unloaded — loading RAUNE+Depth for calibration");
    }
    inf_server.load_raune(
        args.raune_weights.as_deref(),
        args.raune_models_dir.as_deref(),
    ).context("failed to load RAUNE-Net for calibration")?;
    inf_server.load_depth(
        args.depth_model.as_deref(),
    ).context("failed to load Depth Anything for calibration")?;

    // -----------------------------------------------------------------------
    // Auto-calibrate: fused RAUNE+depth, dual output
    // -----------------------------------------------------------------------
    log::info!(
        "Auto-calibrating from {} keyframes (fused RAUNE+depth)",
        keyframes.len()
    );

    let fused_items: Vec<RauneDepthBatchItem> = keyframes.iter().map(|kf| {
        RauneDepthBatchItem {
            id: format!("kf_f{}", kf.frame_index),
            pixels: kf.proxy_pixels.clone(),
            width: proxy_w,
            height: proxy_h,
            raune_max_size: proxy_w.max(proxy_h),
            depth_max_size: args.proxy_size.min(1036),
        }
    }).collect();

    let mut store = PagedCalibrationStore::new()
        .context("failed to create paged calibration store")?;
    let mut kf_depths: HashMap<u64, (Vec<f32>, usize, usize)> = HashMap::new();

    for (chunk_kfs, chunk_items) in keyframes
        .chunks(FUSED_BATCH_SIZE)
        .zip(fused_items.chunks(FUSED_BATCH_SIZE))
    {
        let mut results = inf_server.run_raune_depth_batch(chunk_items)
            .unwrap_or_else(|e| {
                log::warn!(
                    "Fused RAUNE+depth batch failed: {e} — using originals + uniform depth"
                );
                chunk_items.iter().map(|item| {
                    (item.id.clone(), item.pixels.clone(),
                     item.width, item.height,
                     vec![0.5f32; item.width * item.height],
                     item.width, item.height)
                }).collect()
            });

        if results.len() < chunk_items.len() {
            log::warn!(
                "Fused batch returned {} results for {} items — padding with originals",
                results.len(), chunk_items.len()
            );
            for item in &chunk_items[results.len()..] {
                results.push((
                    item.id.clone(), item.pixels.clone(),
                    item.width, item.height,
                    vec![0.5f32; item.width * item.height],
                    item.width, item.height,
                ));
            }
        }

        for (kf, (_, enhanced, enh_w, enh_h, depth, dw, dh)) in
            chunk_kfs.iter().zip(results.into_iter())
        {
            debug_assert_eq!(enh_w, proxy_w, "RAUNE enh_w {enh_w} != proxy_w {proxy_w}");
            debug_assert_eq!(enh_h, proxy_h, "RAUNE enh_h {enh_h} != proxy_h {proxy_h}");
            let depth_for_store = if dw == proxy_w && dh == proxy_h {
                depth.clone()
            } else {
                InferenceServer::upscale_depth(&depth, dw, dh, proxy_w, proxy_h)
            };
            store.push(&kf.proxy_pixels, &enhanced, &depth_for_store, proxy_w, proxy_h)
                .context("failed to page fused result to store")?;
            kf_depths.insert(kf.frame_index, (depth, dw, dh));
        }
    }
    log::info!("Fused inference complete ({} keyframes)", keyframes.len());
    let _ = inf_server.shutdown();

    let calibration: Calibration;
    let keyframe_depths: HashMap<u64, (Vec<f32>, usize, usize)>;
    keyframe_depths = kf_depths;

    store.seal().context("failed to seal calibration store")?;

    // ---- 3-pass calibration (inline) ----
    use dorea_hsl::derive::{derive_hsl_corrections, HslCorrections, QualifierCorrection};
    use dorea_hsl::{HSL_QUALIFIERS, MIN_WEIGHT};
    use dorea_lut::apply::apply_depth_luts;
    use dorea_lut::build::{adaptive_zone_boundaries, compute_importance, StreamingLutBuilder, N_DEPTH_ZONES};

    // Pass 1: reservoir-sample depths → adaptive zone boundaries
    const RESERVOIR_CAP: usize = 1_000_000;
    let mut reservoir: Vec<f32> = Vec::with_capacity(RESERVOIR_CAP);
    let mut total_seen: u64 = 0;
    let mut rng: u64 = 0x853c49e6748fea9b_u64;

    for i in 0..store.len() {
        for d in store.depth_bytes(i).chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        {
            total_seen += 1;
            if reservoir.len() < RESERVOIR_CAP {
                reservoir.push(d);
            } else {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let j = (rng % total_seen) as usize;
                if j < RESERVOIR_CAP {
                    reservoir[j] = d;
                }
            }
        }
    }
    let zone_boundaries = adaptive_zone_boundaries(&reservoir, N_DEPTH_ZONES);
    drop(reservoir);

    // Pass 2: stream frames → build LUT
    let mut lut_builder = StreamingLutBuilder::new(zone_boundaries);
    for i in 0..store.len() {
        let (w, h) = store.dims(i);
        let (pixels_u8, target_u8) = store.pixtar_slices(i);
        let depth = store.read_depth(i);
        let original: Vec<[f32; 3]> = pixels_u8.chunks_exact(3)
            .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
            .collect();
        let target: Vec<[f32; 3]> = target_u8.chunks_exact(3)
            .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
            .collect();
        let importance = compute_importance(&depth, w, h);
        lut_builder.add_frame(&original, &target, &depth, &importance);
    }
    let depth_luts = lut_builder.finish();

    // Pass 3: stream frames → HSL corrections
    let n_quals = HSL_QUALIFIERS.len();
    let mut h_offset_acc    = vec![0.0_f64; n_quals];
    let mut s_ratio_acc     = vec![0.0_f64; n_quals];
    let mut v_offset_acc    = vec![0.0_f64; n_quals];
    let mut active_count    = vec![0_usize;  n_quals];
    let mut total_weight_acc = vec![0.0_f64; n_quals];

    for i in 0..store.len() {
        let (pixels_u8, target_u8) = store.pixtar_slices(i);
        let depth = store.read_depth(i);
        let original: Vec<[f32; 3]> = pixels_u8.chunks_exact(3)
            .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
            .collect();
        let target: Vec<[f32; 3]> = target_u8.chunks_exact(3)
            .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
            .collect();
        let lut_output = apply_depth_luts(&original, &depth, &depth_luts);
        let corrs = derive_hsl_corrections(&lut_output, &target);
        for (qi, corr) in corrs.0.iter().enumerate() {
            if corr.weight >= MIN_WEIGHT {
                let w = corr.weight as f64;
                h_offset_acc[qi]    += corr.h_offset as f64 * w;
                s_ratio_acc[qi]     += corr.s_ratio  as f64 * w;
                v_offset_acc[qi]    += corr.v_offset as f64 * w;
                active_count[qi]    += 1;
                total_weight_acc[qi] += w;
            }
        }
    }

    let mut avg_corrections: Vec<QualifierCorrection> = Vec::with_capacity(n_quals);
    for qi in 0..n_quals {
        let qual = &HSL_QUALIFIERS[qi];
        if active_count[qi] > 0 {
            let tw = total_weight_acc[qi];
            avg_corrections.push(QualifierCorrection {
                h_center: qual.h_center,
                h_width:  qual.h_width,
                h_offset: (h_offset_acc[qi] / tw) as f32,
                s_ratio:  (s_ratio_acc[qi]  / tw) as f32,
                v_offset: (v_offset_acc[qi] / tw) as f32,
                weight:   tw as f32,
            });
        } else {
            avg_corrections.push(QualifierCorrection {
                h_center: qual.h_center,
                h_width:  qual.h_width,
                h_offset: 0.0,
                s_ratio:  1.0,
                v_offset: 0.0,
                weight:   0.0,
            });
        }
    }

    calibration = Calibration::new(depth_luts, HslCorrections(avg_corrections), store.len());
    log::info!(
        "Auto-calibration complete ({} keyframes → {} depth zones)",
        store.len(), N_DEPTH_ZONES
    );

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

    // Ordered list of (frame_index, scene_cut_before) for pass-2 lerp lookup.
    let kf_index_list: Vec<(u64, bool)> = keyframes.iter()
        .map(|kf| (kf.frame_index, kf.scene_cut_before))
        .collect();

    // -----------------------------------------------------------------------
    // Pass 2: full-resolution decode + depth lookup + grade + encode
    // -----------------------------------------------------------------------
    let decode_source = maxine_temp_path.as_deref().unwrap_or(args.input.as_path());
    let frames = ffmpeg::decode_frames(decode_source, &info)
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
        let (prev_depth_proxy, dpw, dph) = keyframe_depths
            .get(&prev_kf_idx)
            .expect("prev keyframe depth missing — logic error");
        let (dpw, dph) = (*dpw, *dph);

        // Lerp at proxy resolution, then upscale once — ~50× less work than lerping full-res.
        let depth_proxy = if fi == prev_kf_idx {
            prev_depth_proxy.clone()
        } else if let Some(&(next_kf_idx, scene_cut_before_next)) = kf_index_list.get(kf_cursor + 1) {
            if scene_cut_before_next {
                prev_depth_proxy.clone() // Don't lerp across scene cut
            } else {
                let (next_depth_proxy, _, _) = keyframe_depths
                    .get(&next_kf_idx)
                    .expect("next keyframe depth missing — logic error");
                let t = (fi - prev_kf_idx) as f32 / (next_kf_idx - prev_kf_idx) as f32;
                lerp_depth(prev_depth_proxy, next_depth_proxy, t)
            }
        } else {
            prev_depth_proxy.clone() // Past last keyframe
        };

        // Upscale depth to full frame resolution for grading.
        let depth = if dpw == frame.width && dph == frame.height {
            depth_proxy
        } else {
            InferenceServer::upscale_depth(&depth_proxy, dpw, dph, frame.width, frame.height)
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

    encoder.finish().context("ffmpeg encoder failed to finalize")?;

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

/// Temporary on-disk store for calibration frames.
///
/// Two packed binary files, both memory-mapped after the inference phase:
///   depths.bin  — all depth maps concatenated (f32 LE), one region per frame
///   pixtar.bin  — all [pixels | target] pairs concatenated (u8 RGB), one pair per frame
///
/// This gives three fast sequential scans (one per calibration pass) instead of
/// 988×3 individual file open/read/close calls. Passes 2 and 3 both scan pixtar.bin
/// sequentially, so pass 3 is served entirely from the OS page cache.
struct PagedCalibrationStore {
    dir: std::path::PathBuf,
    widths: Vec<usize>,
    heights: Vec<usize>,
    /// Byte offset in depths.bin for each frame's depth region
    depth_offsets: Vec<usize>,
    /// Byte offset in pixtar.bin for each frame's pixels region (target follows immediately)
    pixtar_offsets: Vec<usize>,
    /// Write handles — None after seal()
    depth_writer: Option<std::io::BufWriter<std::fs::File>>,
    pixtar_writer: Option<std::io::BufWriter<std::fs::File>>,
    /// Read-only mmaps — Some after seal()
    depth_mmap: Option<memmap2::Mmap>,
    pixtar_mmap: Option<memmap2::Mmap>,
}

impl PagedCalibrationStore {
    fn new() -> Result<Self> {
        use std::io::BufWriter;
        let dir = std::env::temp_dir()
            .join(format!("dorea_cal_{}", std::process::id()));
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create calibration temp dir {:?}", dir))?;
        let dw = BufWriter::new(
            std::fs::File::create(dir.join("depths.bin")).context("create depths.bin")?
        );
        let pw = BufWriter::new(
            std::fs::File::create(dir.join("pixtar.bin")).context("create pixtar.bin")?
        );
        Ok(Self {
            dir,
            widths: Vec::new(),
            heights: Vec::new(),
            depth_offsets: Vec::new(),
            pixtar_offsets: Vec::new(),
            depth_writer: Some(dw),
            pixtar_writer: Some(pw),
            depth_mmap: None,
            pixtar_mmap: None,
        })
    }

    fn push(&mut self, pixels: &[u8], target: &[u8], depth: &[f32], width: usize, height: usize) -> Result<()> {
        use std::io::Write;
        let dw = self.depth_writer.as_mut().context("store already sealed")?;
        let pw = self.pixtar_writer.as_mut().context("store already sealed")?;

        let depth_off = self.depth_offsets.last().copied().unwrap_or(0)
            + self.widths.last().copied().unwrap_or(0)
            * self.heights.last().copied().unwrap_or(0) * 4;
        let pixtar_off = self.pixtar_offsets.last().copied().unwrap_or(0)
            + self.widths.last().copied().unwrap_or(0)
            * self.heights.last().copied().unwrap_or(0) * 6; // pixels + target = 3+3 bytes/px

        // Write depth as raw f32 LE
        for f in depth {
            dw.write_all(&f.to_le_bytes()).context("write depth")?;
        }
        pw.write_all(pixels).context("write pixels")?;
        pw.write_all(target).context("write target")?;

        self.depth_offsets.push(depth_off);
        self.pixtar_offsets.push(pixtar_off);
        self.widths.push(width);
        self.heights.push(height);
        Ok(())
    }

    /// Flush writers and memory-map both files for fast read access.
    fn seal(&mut self) -> Result<()> {
        use std::io::Write;
        if let Some(mut dw) = self.depth_writer.take() { dw.flush().context("flush depths")?; }
        if let Some(mut pw) = self.pixtar_writer.take() { pw.flush().context("flush pixtar")?; }
        let df = std::fs::File::open(self.dir.join("depths.bin")).context("open depths.bin")?;
        let pf = std::fs::File::open(self.dir.join("pixtar.bin")).context("open pixtar.bin")?;
        // SAFETY: files are written once, sealed, and never modified again.
        self.depth_mmap = Some(unsafe { memmap2::Mmap::map(&df) }.context("mmap depths")?);
        self.pixtar_mmap = Some(unsafe { memmap2::Mmap::map(&pf) }.context("mmap pixtar")?);
        Ok(())
    }

    fn len(&self) -> usize { self.widths.len() }

    /// Raw f32 LE bytes for frame i's depth map — zero-copy mmap slice.
    fn depth_bytes(&self, i: usize) -> &[u8] {
        let mmap = self.depth_mmap.as_ref().expect("call seal() before reading");
        let off = self.depth_offsets[i];
        let len = self.widths[i] * self.heights[i] * 4;
        &mmap[off..off + len]
    }

    /// Decode depth for frame i into a Vec<f32>.
    fn read_depth(&self, i: usize) -> Vec<f32> {
        self.depth_bytes(i).chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    /// Zero-copy mmap slices for pixels and target for frame i.
    fn pixtar_slices(&self, i: usize) -> (&[u8], &[u8]) {
        let mmap = self.pixtar_mmap.as_ref().expect("call seal() before reading");
        let off = self.pixtar_offsets[i];
        let n = self.widths[i] * self.heights[i] * 3;
        (&mmap[off..off + n], &mmap[off + n..off + n * 2])
    }

    fn dims(&self, i: usize) -> (usize, usize) { (self.widths[i], self.heights[i]) }
}

impl Drop for PagedCalibrationStore {
    fn drop(&mut self) {
        drop(self.depth_mmap.take());
        drop(self.pixtar_mmap.take());
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn build_inference_config(args: &GradeArgs) -> InferenceConfig {
    InferenceConfig {
        python_exe: args.python.clone(),
        raune_weights: args.raune_weights.clone(),
        raune_models_dir: args.raune_models_dir.clone(),
        skip_raune: false,
        depth_model: args.depth_model.clone(),
        skip_depth: false,
        device: if args.cpu_only { Some("cpu".to_string()) } else { None },
        startup_timeout: Duration::from_secs(180),
        maxine: false, // SR disabled — re-enable via !args.no_maxine when ready
        maxine_upscale_factor: args.maxine_upscale_factor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn build_inference_config_maxine_disabled() {
        let args = GradeArgs {
            input: PathBuf::from("/dev/null"),
            output: None,
            no_maxine: false,
            no_maxine_artifact_reduction: false,
            maxine_upscale_factor: 2,
            warmth: 1.0,
            strength: 0.8,
            contrast: 1.0,
            proxy_size: 1080,
            depth_skip_threshold: 0.005,
            depth_max_interval: 12,
            no_depth_interp: false,
            raune_weights: None,
            raune_models_dir: None,
            depth_model: None,
            python: PathBuf::from("/opt/dorea-venv/bin/python"),
            cpu_only: false,
            verbose: false,
        };
        let cfg = build_inference_config(&args);
        assert!(!cfg.maxine, "Maxine is disabled until SR is re-enabled");
        assert!(!cfg.skip_raune, "RAUNE should not be skipped in config");
        assert!(!cfg.skip_depth, "depth should not be skipped in config");
    }

    #[test]
    fn build_inference_config_no_maxine_flag_disables_maxine() {
        let args = GradeArgs {
            input: PathBuf::from("/dev/null"),
            output: None,
            no_maxine: true,
            no_maxine_artifact_reduction: false,
            maxine_upscale_factor: 2,
            warmth: 1.0,
            strength: 0.8,
            contrast: 1.0,
            proxy_size: 1080,
            depth_skip_threshold: 0.005,
            depth_max_interval: 12,
            no_depth_interp: false,
            raune_weights: None,
            raune_models_dir: None,
            depth_model: None,
            python: PathBuf::from("/opt/dorea-venv/bin/python"),
            cpu_only: false,
            verbose: false,
        };
        let cfg = build_inference_config(&args);
        assert!(!cfg.maxine, "no_maxine flag should disable Maxine");
    }
}
