// dorea-gpu — GPU-accelerated (and CPU fallback) color grading pipeline.
//
// Public API:
// - `GradeParams` — user-facing grading parameters (warmth, strength, contrast)
// - `grade_frame` — applies full grading pipeline (LUT + HSL + depth_aware_ambiance)
// - `cpu::depth_aware_ambiance` — always available, pure CPU

pub mod batcher;
pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::AdaptiveGrader;

#[cfg(feature = "cuda")]
pub mod device;

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
    #[error("CUDA OOM: {0}")]
    Oom(String),
    #[error("CUDA error: {0}")]
    CudaFail(String),
    #[error("CUDA module load failed: {0}")]
    ModuleLoad(String),
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
        match cuda::CudaGrader::new(calibration, params) {
            Ok(grader) => {
                match grader.grade_frame_cuda(pixels, depth, width, height) {
                    Ok(out) => return Ok(out),
                    Err(e) => {
                        log::warn!("CUDA grading failed: {e} — falling back to CPU");
                        return cpu::grade_frame_cpu(
                            pixels, depth, width, height, calibration, params,
                        )
                        .map_err(GpuError::CudaFail);
                    }
                }
            }
            Err(GpuError::ModuleLoad(msg)) => {
                log::error!("CUDA module load failed: {msg} — using CPU for all frames");
                return cpu::grade_frame_cpu(
                    pixels, depth, width, height, calibration, params,
                )
                .map_err(GpuError::CudaFail);
            }
            Err(e) => return Err(e),
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        cpu::grade_frame_cpu(pixels, depth, width, height, calibration, params)
            .map_err(GpuError::CudaFail)
    }
}

/// Grade multiple frames using a `CudaGrader` built with the given calibration and params.
///
/// Creates a `CudaGrader::new(calibration, params)` **once per call**, then
/// processes all frames in sequence. All frames must have the same `(width, height)`.
///
/// Returns a `Vec<Vec<u8>>` of graded sRGB u8 frames in the same order as input.
/// Returns `Err` if grader construction fails, if any frame has wrong slice lengths,
/// or if a CUDA error occurs during grading.
#[cfg(feature = "cuda")]
pub fn grade_frames_with_capacity(
    frames: &[(&[u8], &[f32])],   // (pixels, depth) per frame
    width: usize,
    height: usize,
    calibration: &Calibration,
    params: &GradeParams,
) -> Result<Vec<Vec<u8>>, GpuError> {
    if frames.is_empty() {
        return Ok(vec![]);
    }
    let grader = cuda::CudaGrader::new(calibration, params)?;
    frames
        .iter()
        .map(|(pixels, depth)| {
            grader.grade_frame_cuda(pixels, depth, width, height)
        })
        .collect()
}

/// Grade a single frame reusing an existing `CudaGrader`.
///
/// Avoids repeated PTX loading and LUT rebuilding when processing multiple frames.
/// The caller creates one `CudaGrader` and passes it to each call.
///
/// Returns graded sRGB u8 pixels with the same dimensions.
#[cfg(feature = "cuda")]
pub fn grade_frame_with_grader(
    grader: &cuda::CudaGrader,
    pixels: &[u8],
    depth: &[f32],
    width: usize,
    height: usize,
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
    grader.grade_frame_cuda(pixels, depth, width, height)
}

/// Grade a single frame using an existing `AdaptiveGrader` with dual-texture blending.
///
/// `blend_t`: 0.0 = keyframe (active set only), up to 1.0 at next keyframe.
#[cfg(feature = "cuda")]
pub fn grade_frame_with_adaptive_grader(
    grader: &cuda::AdaptiveGrader,
    pixels: &[u8],
    depth: &[f32],
    width: usize,
    height: usize,
    blend_t: f32,
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
    grader.grade_frame_blended(pixels, depth, width, height, blend_t)
}
