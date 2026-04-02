//! `dorea calibrate` command implementation.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use dorea_cal::Calibration;
use dorea_hsl::derive::{derive_hsl_corrections, HslCorrections, QualifierCorrection};
use dorea_hsl::qualifiers::HSL_QUALIFIERS;
use dorea_lut::apply::apply_depth_luts;
use dorea_lut::build::{build_depth_luts, compute_importance, KeyframeData};
use image::ImageReader;

/// Arguments for the `calibrate` subcommand.
#[derive(clap::Args)]
pub struct CalibrateArgs {
    /// Directory of sRGB PNG keyframe images
    #[arg(long)]
    pub keyframes: PathBuf,

    /// Directory of 16-bit PNG depth maps (matched by filename stem)
    #[arg(long)]
    pub depth: PathBuf,

    /// Directory of RAUNE-Net target PNGs (matched by filename stem)
    #[arg(long)]
    pub targets: PathBuf,

    /// Output .dorea-cal file path
    #[arg(long, default_value = "calibration.dorea-cal")]
    pub output: PathBuf,

    /// CPU-only mode (placeholder for Phase 2 GPU path)
    #[arg(long)]
    pub cpu_only: bool,

    /// Verbose logging
    #[arg(short, long)]
    pub verbose: bool,
}

pub fn run(args: CalibrateArgs) -> Result<()> {
    if args.verbose {
        std::env::set_var("RUST_LOG", "debug");
    }

    log::info!("dorea calibrate — scanning keyframes in {:?}", args.keyframes);

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
    let mut pixel_counts: Vec<usize> = Vec::new();

    for kf_path in &keyframe_paths {
        let stem = kf_path.file_stem().context("keyframe has no stem")?;

        let depth_path = args.depth.join(format!("{}.png", stem.to_string_lossy()));
        let target_path = args.targets.join(format!("{}.png", stem.to_string_lossy()));

        log::info!("Processing keyframe: {}", kf_path.display());

        let (kf_pixels, kf_w, kf_h) = load_rgb_image(kf_path)
            .with_context(|| format!("Failed to load keyframe: {}", kf_path.display()))?;

        let (mut depth_pixels, depth_w, depth_h) = load_depth_map(&depth_path)
            .with_context(|| format!("Failed to load depth map: {}", depth_path.display()))?;

        let (mut tgt_pixels, tgt_w, tgt_h) = load_rgb_image(&target_path)
            .with_context(|| format!("Failed to load target: {}", target_path.display()))?;

        // Resize depth if needed (nearest-neighbor for depth, but use bilinear)
        if depth_w != kf_w || depth_h != kf_h {
            log::debug!("Resizing depth map from {depth_w}x{depth_h} to {kf_w}x{kf_h}");
            depth_pixels = resize_depth(&depth_pixels, depth_w, depth_h, kf_w, kf_h);
        }

        // Resize target if needed (nearest-neighbor)
        if tgt_w != kf_w || tgt_h != kf_h {
            log::debug!("Resizing target from {tgt_w}x{tgt_h} to {kf_w}x{kf_h}");
            tgt_pixels = resize_rgb_nn(&tgt_pixels, tgt_w, tgt_h, kf_w, kf_h);
        }

        // Compute importance
        let importance = compute_importance(&depth_pixels, kf_w, kf_h);

        pixel_counts.push(kf_w * kf_h);
        keyframe_data_vec.push(KeyframeData {
            original: kf_pixels,
            target: tgt_pixels,
            depth: depth_pixels,
            importance,
            width: kf_w,
            height: kf_h,
        });
    }

    // 3. Build depth-stratified LUTs
    log::info!("Building depth-stratified LUTs...");
    let depth_luts = build_depth_luts(&keyframe_data_vec);
    log::info!("LUTs built ({} zones)", depth_luts.n_zones());

    // 4. Derive HSL corrections per keyframe, then aggregate
    log::info!("Deriving HSL corrections...");

    // Accumulate weighted corrections across keyframes
    let n_quals = HSL_QUALIFIERS.len();
    let mut h_offset_acc = vec![0.0_f64; n_quals];
    let mut s_ratio_acc = vec![0.0_f64; n_quals];
    let mut v_offset_acc = vec![0.0_f64; n_quals];
    let mut active_count = vec![0_usize; n_quals];
    let mut total_weight_acc = vec![0.0_f64; n_quals];

    for (kd, &px_count) in keyframe_data_vec.iter().zip(pixel_counts.iter()) {
        let lut_output = apply_depth_luts(&kd.original, &kd.depth, &depth_luts);
        let corrs = derive_hsl_corrections(&lut_output, &kd.target);

        for (qi, corr) in corrs.0.iter().enumerate() {
            if corr.weight >= dorea_hsl::MIN_WEIGHT {
                let w = px_count as f64;
                h_offset_acc[qi] += corr.h_offset as f64 * w;
                s_ratio_acc[qi] += corr.s_ratio as f64 * w;
                v_offset_acc[qi] += corr.v_offset as f64 * w;
                active_count[qi] += 1;
                total_weight_acc[qi] += w;
            }
        }
    }

    // Build averaged corrections
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

    log::info!("HSL corrections derived:");
    for c in &avg_corrections {
        log::info!(
            "  {:>12}: h_offset={:+.2}° s_ratio={:.3} v_offset={:+.4} weight={:.0}",
            c.h_center,
            c.h_offset,
            c.s_ratio,
            c.v_offset,
            c.weight,
        );
    }

    let hsl_corrections = HslCorrections(avg_corrections);

    // 5. Build and save calibration
    let mut cal = Calibration::new(depth_luts, hsl_corrections, keyframe_data_vec.len());
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
        .with_context(|| format!("Cannot decode {}", path.display()))?
        .into_luma16();

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
