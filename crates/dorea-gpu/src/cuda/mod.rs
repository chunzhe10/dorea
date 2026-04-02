//! CUDA-backed grading pipeline.
//!
//! Only compiled when the `cuda` feature is enabled (detected by build.rs).
//! Provides `grade_frame_cuda` which runs LUT apply and HSL correct on GPU,
//! then finishes depth_aware_ambiance + warmth + blend on CPU.

#[cfg(feature = "cuda")]
use crate::{GradeParams, GpuError};
#[cfg(feature = "cuda")]
use dorea_cal::Calibration;
#[cfg(feature = "cuda")]
use dorea_color::lab::{srgb_to_lab, lab_to_srgb};

#[cfg(feature = "cuda")]
extern "C" {
    /// GPU launcher: depth-stratified LUT apply. Returns cudaError_t (0 = success).
    fn dorea_lut_apply_gpu(
        h_pixels_in: *const f32,
        h_depth: *const f32,
        h_luts: *const f32,
        h_zone_boundaries: *const f32,
        h_pixels_out: *mut f32,
        n_pixels: i32,
        lut_size: i32,
        n_zones: i32,
    ) -> i32;

    /// GPU launcher: HSL 6-qualifier correction. Returns cudaError_t (0 = success).
    fn dorea_hsl_correct_gpu(
        h_pixels_in: *const f32,
        h_pixels_out: *mut f32,
        h_offsets: *const f32,
        h_s_ratios: *const f32,
        h_v_offsets: *const f32,
        h_weights: *const f32,
        n_pixels: i32,
    ) -> i32;
}

/// Attempt GPU-accelerated grading.
///
/// LUT apply and HSL correct run on GPU via CUDA kernels; depth_aware_ambiance,
/// warmth, and strength blend run on CPU (LAB math not yet ported to CUDA).
/// Returns `Err` on any CUDA failure so the caller can fall back to CPU.
#[cfg(feature = "cuda")]
pub fn grade_frame_cuda(
    pixels: &[u8],
    depth: &[f32],
    width: usize,
    height: usize,
    calibration: &Calibration,
    params: &GradeParams,
) -> Result<Vec<u8>, GpuError> {
    let n = width * height;

    // --- u8 → f32 ---
    let rgb_f32: Vec<f32> = pixels.iter().map(|&p| p as f32 / 255.0).collect();

    // --- Flatten LUT data ---
    let depth_luts = &calibration.depth_luts;
    let n_zones = depth_luts.n_zones();
    let lut_size = if n_zones > 0 { depth_luts.luts[0].size } else { 33 };

    // Concatenate all zone LUT grids into one contiguous slice
    let luts_flat: Vec<f32> = depth_luts.luts.iter()
        .flat_map(|lg| lg.data.iter().copied())
        .collect();

    // --- GPU: LUT apply ---
    let mut rgb_after_lut = vec![0.0f32; n * 3];
    let status = unsafe {
        dorea_lut_apply_gpu(
            rgb_f32.as_ptr(),
            depth.as_ptr(),
            luts_flat.as_ptr(),
            depth_luts.zone_boundaries.as_ptr(),
            rgb_after_lut.as_mut_ptr(),
            n as i32,
            lut_size as i32,
            n_zones as i32,
        )
    };
    if status != 0 {
        return Err(GpuError::Cuda(format!("dorea_lut_apply_gpu returned CUDA error {status}")));
    }

    // --- Extract HSL arrays (6 qualifiers, ordered Red/Yellow/Green/Cyan/Blue/Magenta) ---
    let mut h_offsets = [0.0f32; 6];
    let mut s_ratios  = [1.0f32; 6];
    let mut v_offsets = [0.0f32; 6];
    let mut weights   = [0.0f32; 6];
    for (i, q) in calibration.hsl_corrections.0.iter().enumerate().take(6) {
        h_offsets[i] = q.h_offset;
        s_ratios[i]  = q.s_ratio;
        v_offsets[i] = q.v_offset;
        weights[i]   = q.weight;
    }

    // --- GPU: HSL correct ---
    let mut rgb_after_hsl = vec![0.0f32; n * 3];
    let status = unsafe {
        dorea_hsl_correct_gpu(
            rgb_after_lut.as_ptr(),
            rgb_after_hsl.as_mut_ptr(),
            h_offsets.as_ptr(),
            s_ratios.as_ptr(),
            v_offsets.as_ptr(),
            weights.as_ptr(),
            n as i32,
        )
    };
    if status != 0 {
        return Err(GpuError::Cuda(format!("dorea_hsl_correct_gpu returned CUDA error {status}")));
    }

    // --- CPU: depth_aware_ambiance (LAB conversions, shadow lift, S-curve, etc.) ---
    crate::cpu::depth_aware_ambiance(&mut rgb_after_hsl, depth, width, height, params.contrast);

    // --- CPU: Warmth (scale LAB a*/b*) ---
    if (params.warmth - 1.0).abs() > 1e-4 {
        let warmth_factor = 1.0 + (params.warmth - 1.0) * 0.3;
        for i in 0..n {
            let r = rgb_after_hsl[i * 3];
            let g = rgb_after_hsl[i * 3 + 1];
            let b = rgb_after_hsl[i * 3 + 2];
            let (l, a, b_ab) = srgb_to_lab(r, g, b);
            let (ro, go, bo) = lab_to_srgb(l, a * warmth_factor, b_ab * warmth_factor);
            rgb_after_hsl[i * 3]     = ro.clamp(0.0, 1.0);
            rgb_after_hsl[i * 3 + 1] = go.clamp(0.0, 1.0);
            rgb_after_hsl[i * 3 + 2] = bo.clamp(0.0, 1.0);
        }
    }

    // --- CPU: Blend with original ---
    if params.strength < 1.0 - 1e-4 {
        for i in 0..rgb_after_hsl.len() {
            let orig = pixels[i] as f32 / 255.0;
            rgb_after_hsl[i] = orig * (1.0 - params.strength) + rgb_after_hsl[i] * params.strength;
        }
    }

    // --- f32 → u8 ---
    Ok(rgb_after_hsl.iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect())
}
