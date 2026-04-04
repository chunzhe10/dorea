//! `dorea calibrate` command implementation.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use dorea_cal::Calibration;
use dorea_color::dlog_m::dlog_m_to_linear;
use dorea_hsl::derive::{derive_hsl_corrections, HslCorrections, QualifierCorrection};
use dorea_hsl::qualifiers::HSL_QUALIFIERS;
use dorea_lut::apply::apply_depth_luts;
use dorea_lut::build::{build_depth_luts, compute_importance, KeyframeData};
use dorea_video::inference::{InferenceConfig, InferenceServer};
use image::ImageReader;

/// Input data for a single keyframe, as used by `run_calibration_from_frames`.
pub struct CalibrationInput {
    /// sRGB pixels f32 [0,1] in row-major order, length = width * height * 3 (as u8 RGB)
    pub pixels: Vec<u8>,
    /// RAUNE-Net enhanced target, same size, RGB u8
    pub target: Vec<u8>,
    /// Depth map f32 [0,1], length = width * height
    pub depth: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

/// Build a `Calibration` from in-memory keyframe inputs.
///
/// Used by `dorea grade` and `dorea preview` for auto-calibration.
pub fn run_calibration_from_frames(inputs: &[CalibrationInput]) -> Result<Calibration> {
    let mut keyframe_data_vec: Vec<KeyframeData> = Vec::new();

    for ci in inputs {
        let original: Vec<[f32; 3]> = ci.pixels.chunks_exact(3)
            .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
            .collect();
        let target: Vec<[f32; 3]> = ci.target.chunks_exact(3)
            .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
            .collect();
        let importance = compute_importance(&ci.depth, ci.width, ci.height);
        keyframe_data_vec.push(KeyframeData {
            original,
            target,
            depth: ci.depth.clone(),
            importance,
            width: ci.width,
            height: ci.height,
        });
    }

    let depth_luts = build_depth_luts(&keyframe_data_vec);

    let n_quals = HSL_QUALIFIERS.len();
    let mut h_offset_acc = vec![0.0_f64; n_quals];
    let mut s_ratio_acc = vec![0.0_f64; n_quals];
    let mut v_offset_acc = vec![0.0_f64; n_quals];
    let mut active_count = vec![0_usize; n_quals];
    let mut total_weight_acc = vec![0.0_f64; n_quals];

    for kd in keyframe_data_vec.iter() {
        let lut_output = apply_depth_luts(&kd.original, &kd.depth, &depth_luts);
        let corrs = derive_hsl_corrections(&lut_output, &kd.target);

        for (qi, corr) in corrs.0.iter().enumerate() {
            if corr.weight >= dorea_hsl::MIN_WEIGHT {
                let w = corr.weight as f64;
                h_offset_acc[qi] += corr.h_offset as f64 * w;
                s_ratio_acc[qi] += corr.s_ratio as f64 * w;
                v_offset_acc[qi] += corr.v_offset as f64 * w;
                active_count[qi] += 1;
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
                h_width: qual.h_width,
                h_offset: (h_offset_acc[qi] / tw) as f32,
                s_ratio: (s_ratio_acc[qi] / tw) as f32,
                v_offset: (v_offset_acc[qi] / tw) as f32,
                weight: tw as f32,
            });
        } else {
            avg_corrections.push(QualifierCorrection {
                h_center: qual.h_center,
                h_width: qual.h_width,
                h_offset: 0.0,
                s_ratio: 1.0,
                v_offset: 0.0,
                weight: 0.0,
            });
        }
    }

    let cal = Calibration::new(
        depth_luts,
        HslCorrections(avg_corrections),
        inputs.len(),
    );
    Ok(cal)
}

/// Arguments for the `calibrate` subcommand.
#[derive(clap::Args)]
pub struct CalibrateArgs {
    /// Directory of sRGB PNG keyframe images
    #[arg(long)]
    pub keyframes: PathBuf,

    /// Directory of 16-bit PNG depth maps (matched by filename stem).
    /// If omitted, Depth Anything V2 is run via the inference subprocess.
    #[arg(long)]
    pub depth: Option<PathBuf>,

    /// Directory of RAUNE-Net target PNGs (matched by filename stem).
    /// If omitted, RAUNE-Net is run via the inference subprocess.
    #[arg(long)]
    pub targets: Option<PathBuf>,

    /// Output .dorea-cal file path
    #[arg(long, default_value = "calibration.dorea-cal")]
    pub output: PathBuf,

    /// CPU-only mode (forces --device cpu in inference subprocess)
    #[arg(long)]
    pub cpu_only: bool,

    /// Verbose logging
    #[arg(short, long)]
    pub verbose: bool,

    /// Encoding of the input keyframe images.
    ///
    /// Use `srgb` (default) for standard sRGB PNGs.
    /// Use `dlog_m` if keyframes are D-Log M encoded (DJI Action 4).
    #[arg(long, default_value = "srgb", value_parser = parse_input_encoding)]
    pub input_encoding: InputEncoding,

    /// Path to RAUNE-Net weights .pth (used when --targets is omitted)
    #[arg(long)]
    pub raune_weights: Option<PathBuf>,

    /// Path to sea_thru_poc directory (contains models/raune_net.py)
    #[arg(long)]
    pub raune_models_dir: Option<PathBuf>,

    /// Path to Depth Anything V2 model directory (used when --depth is omitted)
    #[arg(long)]
    pub depth_model: Option<PathBuf>,

    /// Python executable for the inference subprocess
    #[arg(long, default_value = "/opt/dorea-venv/bin/python")]
    pub python: PathBuf,
}

/// Supported input encodings for keyframe images.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InputEncoding {
    Srgb,
    DlogM,
}

fn parse_input_encoding(s: &str) -> std::result::Result<InputEncoding, String> {
    match s {
        "srgb" => Ok(InputEncoding::Srgb),
        "dlog_m" => Ok(InputEncoding::DlogM),
        other => Err(format!("unknown input encoding '{other}'; expected 'srgb' or 'dlog_m'")),
    }
}

pub fn run(args: CalibrateArgs) -> Result<()> {
    // Logging is already initialised by main() based on the verbose flag.
    log::info!("dorea calibrate — scanning keyframes in {:?}", args.keyframes);

    // Determine whether we need the inference subprocess
    let need_inference = args.targets.is_none() || args.depth.is_none();
    let mut inf_server: Option<InferenceServer> = None;

    if need_inference {
        log::info!("Spawning inference subprocess for auto-generation of missing inputs...");
        let cfg = InferenceConfig {
            python_exe: args.python.clone(),
            raune_weights: args.raune_weights.clone(),
            raune_models_dir: args.raune_models_dir.clone(),
            skip_raune: false,
            depth_model: args.depth_model.clone(),
            device: if args.cpu_only { Some("cpu".to_string()) } else { None },
            startup_timeout: Duration::from_secs(180),
            maxine: false,
            maxine_upscale_factor: 2,
        };
        inf_server = Some(
            InferenceServer::spawn(&cfg).context("failed to spawn inference server")?,
        );
    }

    // 1. Scan keyframe directory for PNGs
    let mut keyframe_paths: Vec<PathBuf> = std::fs::read_dir(&args.keyframes)
        .with_context(|| format!("Cannot read keyframes dir: {:?}", args.keyframes))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e.eq_ignore_ascii_case("png")).unwrap_or(false))
        .collect();
    keyframe_paths.sort();

    if keyframe_paths.is_empty() {
        anyhow::bail!("No PNG files found in {:?}", args.keyframes);
    }

    log::info!("Found {} keyframe(s)", keyframe_paths.len());

    // 2. Load each keyframe + matching depth + target
    let mut keyframe_data_vec: Vec<KeyframeData> = Vec::new();

    for kf_path in &keyframe_paths {
        let stem = kf_path.file_stem().context("keyframe has no stem")?.to_string_lossy().to_string();

        log::info!("Processing keyframe: {}", kf_path.display());

        let (mut kf_pixels, kf_w, kf_h) = load_rgb_image(kf_path)
            .with_context(|| format!("Failed to load keyframe: {}", kf_path.display()))?;

        // Apply D-Log M → linear decode if requested
        if args.input_encoding == InputEncoding::DlogM {
            log::debug!("Applying D-Log M decode to keyframe {kf_path:?}");
            for px in kf_pixels.iter_mut() {
                px[0] = dlog_m_to_linear(px[0]);
                px[1] = dlog_m_to_linear(px[1]);
                px[2] = dlog_m_to_linear(px[2]);
            }
        }

        // Load or generate depth map
        let depth_pixels = if let Some(depth_dir) = &args.depth {
            let depth_path = depth_dir.join(format!("{stem}.png"));
            let (mut depth, dw, dh) = load_depth_map(&depth_path)
                .with_context(|| format!("Failed to load depth map: {}", depth_path.display()))?;
            if dw != kf_w || dh != kf_h {
                depth = resize_depth(&depth, dw, dh, kf_w, kf_h);
            }
            depth
        } else {
            // Use inference subprocess
            let srv = inf_server.as_mut().unwrap();
            let pixels_u8: Vec<u8> = kf_pixels.iter()
                .flat_map(|p| [(p[0] * 255.0) as u8, (p[1] * 255.0) as u8, (p[2] * 255.0) as u8])
                .collect();
            match srv.run_depth(&stem, &pixels_u8, kf_w, kf_h, 518) {
                Ok((depth_proxy, dw, dh)) => {
                    InferenceServer::upscale_depth(&depth_proxy, dw, dh, kf_w, kf_h)
                }
                Err(e) => {
                    log::warn!("Depth inference failed for {stem}: {e} — using uniform 0.5");
                    vec![0.5f32; kf_w * kf_h]
                }
            }
        };

        // Load or generate RAUNE-Net target
        let tgt_pixels = if let Some(targets_dir) = &args.targets {
            let target_path = targets_dir.join(format!("{stem}.png"));
            let (mut tgt, tw, th) = load_rgb_image(&target_path)
                .with_context(|| format!("Failed to load target: {}", target_path.display()))?;
            if tw != kf_w || th != kf_h {
                tgt = resize_rgb_nn(&tgt, tw, th, kf_w, kf_h);
            }
            tgt
        } else {
            // Use inference subprocess
            let srv = inf_server.as_mut().unwrap();
            let pixels_u8: Vec<u8> = kf_pixels.iter()
                .flat_map(|p| [(p[0] * 255.0) as u8, (p[1] * 255.0) as u8, (p[2] * 255.0) as u8])
                .collect();
            match srv.run_raune(&stem, &pixels_u8, kf_w, kf_h, 1024) {
                Ok((raune_u8, rw, rh)) => {
                    let raune_f32: Vec<[f32; 3]> = raune_u8.chunks_exact(3)
                        .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
                        .collect();
                    if rw != kf_w || rh != kf_h {
                        resize_rgb_nn(&raune_f32, rw, rh, kf_w, kf_h)
                    } else {
                        raune_f32
                    }
                }
                Err(e) => {
                    log::warn!("RAUNE-Net inference failed for {stem}: {e} — using original as target");
                    kf_pixels.clone()
                }
            }
        };

        // Compute importance
        let importance = compute_importance(&depth_pixels, kf_w, kf_h);

        keyframe_data_vec.push(KeyframeData {
            original: kf_pixels,
            target: tgt_pixels,
            depth: depth_pixels,
            importance,
            width: kf_w,
            height: kf_h,
        });
    }

    // Shutdown inference server
    if let Some(srv) = inf_server {
        let _ = srv.shutdown();
    }

    // 3+4. Build LUTs + HSL corrections via shared helper, then set source description
    log::info!("Building depth-stratified LUTs and HSL corrections...");
    let cal_inputs: Vec<CalibrationInput> = keyframe_data_vec.iter()
        .map(|kd| CalibrationInput {
            pixels: kd.original.iter()
                .flat_map(|p| {
                    [
                        (p[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                        (p[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                        (p[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                    ]
                })
                .collect(),
            target: kd.target.iter()
                .flat_map(|p| {
                    [
                        (p[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                        (p[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                        (p[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                    ]
                })
                .collect(),
            depth: kd.depth.clone(),
            width: kd.width,
            height: kd.height,
        })
        .collect();
    let mut cal = run_calibration_from_frames(&cal_inputs)?;
    log::info!("LUTs built ({} zones), HSL corrections derived", cal.depth_luts.n_zones());

    // 5. Set source description and save
    cal.source_description = format!(
        "{} keyframe(s) from {}",
        keyframe_data_vec.len(),
        args.keyframes.display()
    );

    cal.save(&args.output)
        .with_context(|| format!("Failed to save calibration to {:?}", args.output))?;

    log::info!("Calibration saved to {:?}", args.output);
    println!(
        "Calibration saved to {} ({} zones, {} keyframes)",
        args.output.display(),
        cal.depth_luts.n_zones(),
        cal.keyframe_count
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Image loading helpers
// ---------------------------------------------------------------------------

/// Load an sRGB PNG as Vec<[f32; 3]> in [0.0, 1.0].
fn load_rgb_image(path: &Path) -> Result<(Vec<[f32; 3]>, usize, usize)> {
    let img = ImageReader::open(path)
        .with_context(|| format!("Cannot open {}", path.display()))?
        .decode()
        .with_context(|| format!("Cannot decode {}", path.display()))?
        .into_rgb8();

    let (w, h) = (img.width() as usize, img.height() as usize);
    let pixels: Vec<[f32; 3]> = img
        .pixels()
        .map(|p| [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0])
        .collect();
    Ok((pixels, w, h))
}

/// Load a 16-bit grayscale depth map and normalize to [0.0, 1.0].
fn load_depth_map(path: &Path) -> Result<(Vec<f32>, usize, usize)> {
    let img = ImageReader::open(path)
        .with_context(|| format!("Cannot open {}", path.display()))?
        .decode()
        .with_context(|| format!("Cannot decode {}", path.display()))?;

    // Warn if source is not a grayscale format (L6).
    if !matches!(
        img.color(),
        image::ColorType::L8
            | image::ColorType::L16
            | image::ColorType::La8
            | image::ColorType::La16
    ) {
        log::warn!(
            "Depth map {:?} is not grayscale (got {:?}); converting via luminance",
            path,
            img.color()
        );
    }

    let img = img.into_luma16();

    let (w, h) = (img.width() as usize, img.height() as usize);
    let pixels: Vec<f32> = img.pixels().map(|p| p[0] as f32 / 65535.0).collect();
    Ok((pixels, w, h))
}

/// Resize a depth map (f32) using bilinear interpolation.
fn resize_depth(
    src: &[f32],
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
) -> Vec<f32> {
    let mut dst = vec![0.0_f32; dw * dh];
    for dy in 0..dh {
        for dx in 0..dw {
            let sx = dx as f32 * (sw as f32 - 1.0) / (dw as f32 - 1.0).max(1.0);
            let sy = dy as f32 * (sh as f32 - 1.0) / (dh as f32 - 1.0).max(1.0);
            let x0 = (sx.floor() as usize).min(sw - 1);
            let y0 = (sy.floor() as usize).min(sh - 1);
            let x1 = (x0 + 1).min(sw - 1);
            let y1 = (y0 + 1).min(sh - 1);
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;
            let v00 = src[y0 * sw + x0];
            let v01 = src[y0 * sw + x1];
            let v10 = src[y1 * sw + x0];
            let v11 = src[y1 * sw + x1];
            dst[dy * dw + dx] =
                v00 * (1.0 - fx) * (1.0 - fy)
                + v01 * fx * (1.0 - fy)
                + v10 * (1.0 - fx) * fy
                + v11 * fx * fy;
        }
    }
    dst
}

/// Resize an RGB image using nearest-neighbor interpolation.
fn resize_rgb_nn(
    src: &[[f32; 3]],
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
) -> Vec<[f32; 3]> {
    let mut dst = vec![[0.0_f32; 3]; dw * dh];
    for dy in 0..dh {
        for dx in 0..dw {
            let sx = ((dx as f32 * sw as f32) / dw as f32) as usize;
            let sy = ((dy as f32 * sh as f32) / dh as f32) as usize;
            let sx = sx.min(sw - 1);
            let sy = sy.min(sh - 1);
            dst[dy * dw + dx] = src[sy * sw + sx];
        }
    }
    dst
}
