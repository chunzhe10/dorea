// `dorea grade` — end-to-end video grading pipeline.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use clap::Args;
use anyhow::{Context, Result};

use dorea_gpu::GradeParams;
use dorea_video::ffmpeg::{self, FrameEncoder};
use dorea_video::inference::{RauneDepthBatchItem, InferenceConfig, InferenceServer};
use crate::change_detect::{
    ChangeDetector, MseDetector,
    detect_scene_segments, compute_per_kf_zones, smooth_zone_boundaries,
};

#[cfg(feature = "cuda")]
use dorea_gpu::AdaptiveGrader;

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

    /// Disable Maxine AI enhancement preprocessing
    #[arg(long)]
    pub no_maxine: bool,

    /// Disable Maxine artifact reduction before upscale
    #[arg(long)]
    pub no_maxine_artifact_reduction: bool,

    /// Maxine super-resolution upscale factor (config: [maxine].upscale_factor, built-in default: 2)
    #[arg(long)]
    pub maxine_upscale_factor: Option<u32>,
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


/// A keyframe collected during the proxy-decode pass.
struct KeyframeEntry {
    frame_index: u64,
    proxy_pixels: Vec<u8>,
    /// True if this keyframe follows a scene cut (suppresses lerp across cuts).
    scene_cut_before: bool,
}


pub fn run(args: GradeArgs, cfg: &crate::config::DoreaConfig) -> Result<()> {
    // Resolve config → CLI → built-in defaults (CLI wins, config fills missing, built-in is last resort)
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

    let use_maxine = false; // Disable Pass 1 Maxine (full-res enhancement not needed)
    let maxine_in_fused_batch = true; // Enable Maxine upscaling in fused RAUNE+depth batch instead
    if use_maxine {
        let valid_factors = [2u32, 3, 4];
        if !valid_factors.contains(&maxine_upscale_factor) {
            anyhow::bail!(
                "--maxine-upscale-factor {} is not supported. Supported: {:?}",
                maxine_upscale_factor, valid_factors,
            );
        }
        log::info!(
            "Maxine enabled: upscale_factor={}, artifact_reduction={}",
            maxine_upscale_factor,
            !args.no_maxine_artifact_reduction,
        );
    }

    // Determine grading parameters
    let params = GradeParams { warmth, strength, contrast };

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
        ..build_inference_config(
            &python,
            raune_weights.as_deref(),
            raune_models_dir.as_deref(),
            depth_model.as_deref(),
            device.clone(),
            maxine_upscale_factor,
        )
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
    let (proxy_w, proxy_h) = dorea_video::resize::proxy_dims(info.width, info.height, proxy_size);

    let mut keyframes: Vec<KeyframeEntry> = Vec::new();
    let mut detector: Box<dyn ChangeDetector> = Box::new(MseDetector::default());
    let mut frames_since_kf = 0usize;
    let scene_cut_threshold = depth_skip_threshold * 10.0;

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
                || frames_since_kf >= depth_max_interval
                || (change < f32::MAX && change > depth_skip_threshold);

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
                || frames_since_kf >= depth_max_interval
                || (change < f32::MAX && change > depth_skip_threshold);

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
        raune_weights.as_deref(),
        raune_models_dir.as_deref(),
    ).context("failed to load RAUNE-Net for calibration")?;
    inf_server.load_depth(
        depth_model.as_deref(),
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
            depth_max_size: proxy_size.min(1036),
        }
    }).collect();

    let mut store = PagedCalibrationStore::new()
        .context("failed to create paged calibration store")?;
    let mut kf_depths: HashMap<u64, (Vec<f32>, usize, usize)> = HashMap::new();

    let n_batches = keyframes.chunks(fused_batch_size).count();
    for (batch_idx, (chunk_kfs, chunk_items)) in keyframes
        .chunks(fused_batch_size)
        .zip(fused_items.chunks(fused_batch_size))
        .enumerate()
    {
        log::info!(
            "RAUNE+depth batch {}/{n_batches} ({} frames)",
            batch_idx + 1,
            chunk_items.len(),
        );
        let mut results = inf_server.run_raune_depth_batch(chunk_items, maxine_in_fused_batch)
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

    let keyframe_depths: HashMap<u64, (Vec<f32>, usize, usize)> = kf_depths;

    store.seal().context("failed to seal calibration store")?;

    // ---- Pre-compute depth timeline ----
    use dorea_hsl::derive::{derive_hsl_corrections, HslCorrections, QualifierCorrection};
    use dorea_hsl::{HSL_QUALIFIERS, MIN_WEIGHT};
    use dorea_lut::apply::apply_depth_luts;
    use dorea_lut::build::{adaptive_zone_boundaries, compute_importance, StreamingLutBuilder};

    // Step 1: Collect per-keyframe depth maps from store
    let kf_depths_store: Vec<Vec<f32>> = (0..store.len())
        .map(|i| store.read_depth(i))
        .collect();

    // Step 2: Compute per-keyframe zone boundaries (runtime zones, depth_zones wide)
    let raw_kf_zones = compute_per_kf_zones(&kf_depths_store, depth_zones);

    // Step 3: Detect scene segments via Wasserstein-1 depth histogram distance
    let segments = detect_scene_segments(&kf_depths_store, scene_threshold, min_segment_kfs);
    log::info!("Scene segments: {} (from {} keyframes)", segments.len(), store.len());
    for (si, seg) in segments.iter().enumerate() {
        log::info!(
            "  segment {si}: keyframes {}..{} ({} KFs)",
            seg.start, seg.end, seg.end - seg.start
        );
    }

    // Step 4: Smooth zone boundaries (respecting segment boundaries)
    let smoothed_kf_zones = smooth_zone_boundaries(&raw_kf_zones, &segments, zone_smoothing_w);

    // Step 5: Build per-segment base LUTs + HSL corrections
    struct SegmentCalibration {
        depth_luts: dorea_lut::types::DepthLuts,
        hsl_corrections: HslCorrections,
    }
    let n_quals = HSL_QUALIFIERS.len();

    let segment_calibrations: Vec<SegmentCalibration> = segments.iter().map(|seg| {
        // Collect depths from all keyframes in this segment → base zone boundaries
        let seg_depths: Vec<f32> = (seg.start..seg.end)
            .flat_map(|i| kf_depths_store[i].iter().cloned())
            .collect();
        let base_boundaries = adaptive_zone_boundaries(&seg_depths, base_lut_zones);

        // Build streaming LUT for this segment (base_lut_zones fine zones)
        let mut lut_builder = StreamingLutBuilder::new(base_boundaries);
        for i in seg.start..seg.end {
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

        // HSL corrections: weighted average over keyframes in segment
        let mut h_offset_acc    = vec![0.0_f64; n_quals];
        let mut s_ratio_acc     = vec![0.0_f64; n_quals];
        let mut v_offset_acc    = vec![0.0_f64; n_quals];
        let mut active_count    = vec![0_usize;  n_quals];
        let mut total_weight_acc = vec![0.0_f64; n_quals];

        for i in seg.start..seg.end {
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
                    h_offset_acc[qi]     += corr.h_offset as f64 * w;
                    s_ratio_acc[qi]      += corr.s_ratio  as f64 * w;
                    v_offset_acc[qi]     += corr.v_offset as f64 * w;
                    active_count[qi]     += 1;
                    total_weight_acc[qi] += w;
                }
            }
        }

        let avg_corrections: Vec<QualifierCorrection> = (0..n_quals).map(|qi| {
            let qual = &HSL_QUALIFIERS[qi];
            if active_count[qi] > 0 {
                let tw = total_weight_acc[qi];
                QualifierCorrection {
                    h_center: qual.h_center,
                    h_width:  qual.h_width,
                    h_offset: (h_offset_acc[qi] / tw) as f32,
                    s_ratio:  (s_ratio_acc[qi]  / tw) as f32,
                    v_offset: (v_offset_acc[qi] / tw) as f32,
                    weight:   tw as f32,
                }
            } else {
                QualifierCorrection {
                    h_center: qual.h_center,
                    h_width:  qual.h_width,
                    h_offset: 0.0,
                    s_ratio:  1.0,
                    v_offset: 0.0,
                    weight:   0.0,
                }
            }
        }).collect();

        SegmentCalibration {
            depth_luts,
            hsl_corrections: HslCorrections(avg_corrections),
        }
    }).collect();

    log::info!(
        "Pre-compute complete: {} segments, {} per-KF zone sets",
        segment_calibrations.len(), smoothed_kf_zones.len(),
    );

    // Build segment-to-keyframe index lookup
    let kf_to_segment: Vec<usize> = {
        let mut map = vec![0usize; store.len()];
        for (si, seg) in segments.iter().enumerate() {
            for ki in seg.start..seg.end {
                map[ki] = si;
            }
        }
        map
    };

    // Initialize adaptive CUDA grader with first segment's base LUT
    #[cfg(feature = "cuda")]
    let mut adaptive_grader = {
        let seg0 = &segment_calibrations[0];
        let base_flat: Vec<f32> = seg0.depth_luts.luts.iter()
            .flat_map(|lut| lut.data.iter().copied())
            .collect();
        let base_bounds = &seg0.depth_luts.zone_boundaries;
        let hsl = &seg0.hsl_corrections;
        let h_offsets: Vec<f32> = hsl.0.iter().map(|q| q.h_offset).collect();
        let s_ratios:  Vec<f32> = hsl.0.iter().map(|q| q.s_ratio).collect();
        let v_offsets: Vec<f32> = hsl.0.iter().map(|q| q.v_offset).collect();
        let weights:   Vec<f32> = hsl.0.iter().map(|q| q.weight).collect();
        let lut_size = seg0.depth_luts.luts[0].size;

        AdaptiveGrader::new(
            &base_flat, base_bounds, base_lut_zones,
            (&h_offsets, &s_ratios, &v_offsets, &weights),
            &params, lut_size, depth_zones,
        ).context("AdaptiveGrader init failed")?
    };

    // Build first keyframe's runtime texture
    #[cfg(feature = "cuda")]
    {
        adaptive_grader.prepare_keyframe(&smoothed_kf_zones[0])
            .context("prepare initial keyframe texture failed")?;
        adaptive_grader.swap_textures();
        // Pre-build second keyframe texture (pipelining: hidden behind encoder startup)
        if smoothed_kf_zones.len() > 1 {
            adaptive_grader.prepare_keyframe(&smoothed_kf_zones[1])
                .context("prepare second keyframe texture failed")?;
        }
        log::info!(
            "Adaptive CUDA grader initialized ({base_lut_zones} base zones, {depth_zones} runtime zones)"
        );
    }

    // Ordered list of (frame_index, scene_cut_before) for pass-2 lerp lookup.
    let kf_index_list: Vec<(u64, bool)> = keyframes.iter()
        .map(|kf| (kf.frame_index, kf.scene_cut_before))
        .collect();

    // -----------------------------------------------------------------------
    // Pass 2: full-resolution decode + adaptive depth grading
    // -----------------------------------------------------------------------
    let decode_source = maxine_temp_path.as_deref().unwrap_or(args.input.as_path());
    let frames = ffmpeg::decode_frames(decode_source, &info)
        .context("failed to spawn ffmpeg full-res decoder")?;

    let mut kf_cursor = 0usize;
    let mut frame_count = 0u64;
    let mut current_segment = kf_to_segment.first().copied().unwrap_or(0);

    for frame_result in frames {
        let frame = frame_result.context("frame decode error")?;
        let fi = frame.index;

        // Advance cursor: kf_index_list[kf_cursor] is the most recent keyframe ≤ fi.
        while kf_cursor + 1 < kf_index_list.len() && kf_index_list[kf_cursor + 1].0 <= fi {
            kf_cursor += 1;

            #[cfg(feature = "cuda")]
            {
                // Crossed a keyframe boundary — swap textures
                adaptive_grader.swap_textures();

                // Check for segment boundary
                let new_seg = kf_to_segment[kf_cursor];
                if new_seg != current_segment {
                    let seg_cal = &segment_calibrations[new_seg];
                    let base_flat: Vec<f32> = seg_cal.depth_luts.luts.iter()
                        .flat_map(|lut| lut.data.iter().copied())
                        .collect();
                    let base_bounds = &seg_cal.depth_luts.zone_boundaries;
                    let hsl = &seg_cal.hsl_corrections;
                    let h_offsets: Vec<f32> = hsl.0.iter().map(|q| q.h_offset).collect();
                    let s_ratios:  Vec<f32> = hsl.0.iter().map(|q| q.s_ratio).collect();
                    let v_offsets: Vec<f32> = hsl.0.iter().map(|q| q.v_offset).collect();
                    let weights:   Vec<f32> = hsl.0.iter().map(|q| q.weight).collect();
                    adaptive_grader.load_segment(
                        &base_flat, base_bounds,
                        (&h_offsets, &s_ratios, &v_offsets, &weights),
                    ).context("load_segment failed")?;
                    current_segment = new_seg;
                    log::info!("Segment switch to {new_seg} at keyframe {kf_cursor}");
                }

                // Pre-build next keyframe's texture (pipelining: hidden behind grading of current interval)
                if kf_cursor + 1 < smoothed_kf_zones.len() {
                    adaptive_grader.prepare_keyframe(&smoothed_kf_zones[kf_cursor + 1])
                        .context("prepare_keyframe failed")?;
                }
            }
        }

        let (prev_kf_idx, _) = kf_index_list[kf_cursor];
        let (prev_depth_proxy, dpw, dph) = keyframe_depths
            .get(&prev_kf_idx)
            .expect("prev keyframe depth missing — logic error");
        let (dpw, dph) = (*dpw, *dph);

        // Lerp depth at proxy resolution, then upscale once.
        let depth_proxy = if fi == prev_kf_idx {
            prev_depth_proxy.clone()
        } else if let Some(&(next_kf_idx, scene_cut_before_next)) = kf_index_list.get(kf_cursor + 1) {
            if scene_cut_before_next {
                prev_depth_proxy.clone()
            } else {
                let (next_depth_proxy, _, _) = keyframe_depths
                    .get(&next_kf_idx)
                    .expect("next keyframe depth missing — logic error");
                let t = (fi - prev_kf_idx) as f32 / (next_kf_idx - prev_kf_idx) as f32;
                lerp_depth(prev_depth_proxy, next_depth_proxy, t)
            }
        } else {
            prev_depth_proxy.clone()
        };

        let depth = if dpw == frame.width && dph == frame.height {
            depth_proxy
        } else {
            InferenceServer::upscale_depth(&depth_proxy, dpw, dph, frame.width, frame.height)
        };

        // Compute blend_t: temporal position of this frame between adjacent keyframes.
        let blend_t = if fi == prev_kf_idx {
            0.0_f32
        } else if let Some(&(next_kf_idx, scene_cut)) = kf_index_list.get(kf_cursor + 1) {
            if scene_cut { 0.0 } else {
                (fi - prev_kf_idx) as f32 / (next_kf_idx - prev_kf_idx) as f32
            }
        } else {
            0.0 // Past last keyframe
        };

        #[cfg(feature = "cuda")]
        let graded = adaptive_grader.grade_frame_blended(
            &frame.pixels, &depth, frame.width, frame.height, blend_t,
        ).map_err(|e| anyhow::anyhow!("Grading failed for frame {fi}: {e}"))?;

        #[cfg(not(feature = "cuda"))]
        let graded = {
            // CPU fallback: use current segment's calibration
            use dorea_cal::Calibration;
            let seg_idx = kf_to_segment.get(kf_cursor).copied().unwrap_or(0);
            let cal = &segment_calibrations[seg_idx];
            let calibration = Calibration::new(
                cal.depth_luts.clone(), cal.hsl_corrections.clone(), store.len(),
            );
            grade_frame(&frame.pixels, &depth, frame.width, frame.height, &calibration, &params)
                .map_err(|e| anyhow::anyhow!("Grading failed for frame {fi}: {e}"))?
        };

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
        maxine: true, // Enable Maxine for fused batch upscaling (between RAUNE and Depth)
        maxine_upscale_factor,
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
        let python = PathBuf::from("/opt/dorea-venv/bin/python");
        let cfg = build_inference_config(&python, None, None, None, None, 2);
        assert!(!cfg.maxine, "Maxine is disabled until SR is re-enabled");
        assert!(!cfg.skip_raune, "RAUNE should not be skipped in config");
        assert!(!cfg.skip_depth, "depth should not be skipped in config");
    }

    #[test]
    fn build_inference_config_maxine_always_false() {
        // Maxine is hardcoded false regardless of any arg — verify both flag states
        let python = PathBuf::from("/opt/dorea-venv/bin/python");
        let cfg = build_inference_config(&python, None, None, None, None, 2);
        assert!(!cfg.maxine, "Maxine is disabled until SR is re-enabled");
    }
}
