// dorea-gpu — GPU-accelerated (and CPU fallback) color grading pipeline.
//
// Public API:
// - `GradeParams` — user-facing grading parameters (warmth, strength, contrast)
// - `grade_frame` — applies full grading pipeline (LUT + HSL + depth_aware_ambiance)
// - `cpu::depth_aware_ambiance` — always available, pure CPU

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

use dorea_cal::Calibration;
use thiserror::Error;

/// User-visible grading parameters.
#[derive(Debug, Clone)]
pub struct GradeParams {
    /// Warmth multiplier \[0.0–2.0\]. 1.0 = neutral.
    pub warmth: f32,
    /// Blend strength between original and graded \[0.0–1.0\].
    pub strength: f32,
    /// Ambiance contrast multiplier. Typical range \[0.0–2.0\]; 1.0 = neutral default.
    pub contrast: f32,
}

impl Default for GradeParams {
    fn default() -> Self {
        Self { warmth: 1.0, strength: 0.8, contrast: 1.0 }
    }
}

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("CUDA error: {0}")]
    Cuda(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

/// Grade a single frame using the calibration and parameters.
///
/// `pixels`: interleaved sRGB u8, length = width * height * 3.
/// `depth`: f32 depth map \[0,1\], length = width * height.
///
/// Returns graded sRGB u8 pixels with the same dimensions.
///
/// Uses CUDA if compiled with the `cuda` feature and the runtime is available;
/// otherwise falls back to the CPU implementation.
pub fn grade_frame(
    pixels: &[u8],
    depth: &[f32],
    width: usize,
    height: usize,
    calibration: &Calibration,
    params: &GradeParams,
) -> Result<Vec<u8>, GpuError> {
    if pixels.len() != width * height * 3 {
        return Err(GpuError::InvalidInput(format!(
            "pixels length {} != width*height*3 {}",
            pixels.len(),
            width * height * 3
        )));
    }
    if depth.len() != width * height {
        return Err(GpuError::InvalidInput(format!(
            "depth length {} != width*height {}",
            depth.len(),
            width * height
        )));
    }

    #[cfg(feature = "cuda")]
    {
        match cuda::grade_frame_cuda(pixels, depth, width, height, calibration, params) {
            Ok(result) => return Ok(result),
            Err(e) => {
                log::warn!("CUDA grading failed ({e}), falling back to CPU");
            }
        }
    }

    cpu::grade_frame_cpu(pixels, depth, width, height, calibration, params)
        .map_err(|e| GpuError::InvalidInput(e.to_string()))
}
