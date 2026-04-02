/// CUDA-backed grading pipeline.
///
/// Only compiled when the `cuda` feature is enabled (detected by build.rs).
/// Provides `grade_frame_cuda` which launches fused CUDA kernels and falls back
/// to the CPU path on any runtime error.

#[cfg(feature = "cuda")]
use crate::GradeParams;
#[cfg(feature = "cuda")]
use dorea_cal::Calibration;
#[cfg(feature = "cuda")]
use crate::GpuError;

/// Attempt GPU-accelerated grading. Returns `Err` on any CUDA failure so the
/// caller can fall back to the CPU path.
#[cfg(feature = "cuda")]
pub fn grade_frame_cuda(
    pixels: &[u8],
    depth: &[f32],
    width: usize,
    height: usize,
    calibration: &Calibration,
    params: &GradeParams,
) -> Result<Vec<u8>, GpuError> {
    // TODO(Phase 2): Full CUDA implementation using cuLaunchKernel.
    // For now, CUDA feature flag is reserved; CPU fallback handles all work.
    // This function exists so the call site compiles when feature="cuda".
    let _ = (pixels, depth, width, height, calibration, params);
    Err(GpuError::Cuda(
        "CUDA kernel launch not yet implemented; using CPU fallback".to_string(),
    ))
}
