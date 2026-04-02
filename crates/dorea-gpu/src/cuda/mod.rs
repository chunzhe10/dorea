//! CUDA-backed grading pipeline.
//!
//! Only compiled when the `cuda` feature is enabled (detected by build.rs).
//! Provides `grade_frame_cuda` which runs LUT apply and HSL correct on GPU,
//! returning f32 intermediate pixels. The caller (`lib.rs`) applies
//! `cpu::finish_grade` (depth_aware_ambiance, warmth, blend, u8 conversion)
//! after GPU resources are freed.

#[cfg(feature = "cuda")]
use crate::{GradeParams, GpuError};
#[cfg(feature = "cuda")]
use dorea_cal::Calibration;

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

/// Attempt GPU-accelerated grading: LUT apply + HSL correct only.
///
/// Returns the LUT+HSL-processed pixels as f32 [0,1], interleaved RGB.
/// The caller is responsible for applying `cpu::finish_grade` (depth_aware_ambiance,
/// warmth, blend, u8 conversion) after this function returns, so GPU resources
/// are freed before the CPU-heavy ambiance pass begins.
///
/// Returns `Err` on any CUDA failure so the caller can fall back to full CPU.
#[cfg(feature = "cuda")]
pub fn grade_frame_cuda(
    pixels: &[u8],
    depth: &[f32],
    width: usize,
    height: usize,
    calibration: &Calibration,
    _params: &GradeParams,
) -> Result<Vec<f32>, GpuError> {
    let n = width * height;

    // --- u8 → f32 ---
    let rgb_f32: Vec<f32> = pixels.iter().map(|&p| p as f32 / 255.0).collect();

    // --- Flatten LUT data ---
    let depth_luts = &calibration.depth_luts;
    let n_zones = depth_luts.n_zones();
    let lut_size = if n_zones > 0 { depth_luts.luts[0].size } else { 33 };

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

    // --- Extract HSL arrays (6 qualifiers) ---
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

    // GPU work done. Return f32 intermediate — caller applies CPU finish pass.
    Ok(rgb_after_hsl)
}
