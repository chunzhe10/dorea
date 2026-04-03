//! CUDA-backed grading pipeline using cudarc PTX module loading.
//!
//! Only compiled when the `cuda` feature is enabled (detected by build.rs).
//!
//! `CudaGrader` holds an `Arc<CudaDevice>` and loads the three PTX modules
//! (lut_apply, hsl_correct, clarity) on construction. `grade_frame_cuda` uploads
//! host data, runs the three kernel stages, and downloads the result — all device
//! memory is RAII via `CudaSlice<T>` (drop = cudaFree).

#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::Ptx;
#[cfg(feature = "cuda")]
use crate::{GradeParams, GpuError};
#[cfg(feature = "cuda")]
use dorea_cal::Calibration;

// Embedded PTX compiled by build.rs
#[cfg(feature = "cuda")]
const LUT_APPLY_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/lut_apply.ptx"));
#[cfg(feature = "cuda")]
const HSL_CORRECT_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/hsl_correct.ptx"));
#[cfg(feature = "cuda")]
const CLARITY_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/clarity.ptx"));

/// Blur radius for the clarity box-blur passes (matches CPU path).
#[cfg(feature = "cuda")]
const BLUR_RADIUS: i32 = 30;

/// Maximum proxy long-edge for clarity downsampling (matches CPU path: 518).
#[cfg(feature = "cuda")]
const PROXY_MAX_SIZE: usize = 518;

/// CUDA grader: holds a device handle and pre-loaded PTX modules.
///
/// Create once, reuse across frames to avoid repeated PTX loading.
/// `!Send + !Sync` — CUDA contexts are thread-local.
#[cfg(feature = "cuda")]
pub struct CudaGrader {
    device: Arc<CudaDevice>,
    _not_send: std::marker::PhantomData<*const ()>,
}

#[cfg(feature = "cuda")]
impl CudaGrader {
    /// Initialise CUDA device 0 and load all PTX modules.
    pub fn new() -> Result<Self, GpuError> {
        let device = CudaDevice::new(0).map_err(|e| {
            GpuError::ModuleLoad(format!("CudaDevice::new(0) failed: {e}"))
        })?;

        // Load PTX modules — each module exposes its kernel function(s).
        device
            .load_ptx(
                Ptx::from_src(LUT_APPLY_PTX),
                "lut_apply",
                &["lut_apply_kernel"],
            )
            .map_err(|e| GpuError::ModuleLoad(format!("load lut_apply PTX: {e}")))?;

        device
            .load_ptx(
                Ptx::from_src(HSL_CORRECT_PTX),
                "hsl_correct",
                &["hsl_correct_kernel"],
            )
            .map_err(|e| GpuError::ModuleLoad(format!("load hsl_correct PTX: {e}")))?;

        device
            .load_ptx(
                Ptx::from_src(CLARITY_PTX),
                "clarity",
                &[
                    "clarity_extract_L_proxy",
                    "clarity_box_blur_rows",
                    "clarity_box_blur_cols",
                    "clarity_apply_kernel",
                ],
            )
            .map_err(|e| GpuError::ModuleLoad(format!("load clarity PTX: {e}")))?;

        Ok(Self {
            device,
            _not_send: std::marker::PhantomData,
        })
    }

    /// Run the GPU grading pipeline: LUT apply -> HSL correct -> clarity.
    ///
    /// Returns interleaved f32 RGB [0,1] with clarity applied.
    /// The caller should pass this to `cpu::finish_grade(skip_clarity=true)`.
    pub fn grade_frame_cuda(
        &self,
        pixels: &[u8],
        depth: &[f32],
        width: usize,
        height: usize,
        calibration: &Calibration,
        params: &GradeParams,
    ) -> Result<Vec<f32>, GpuError> {
        let n = width * height;
        let dev = &self.device;

        // --- u8 -> f32 ---
        let rgb_f32: Vec<f32> = pixels.iter().map(|&p| p as f32 / 255.0).collect();

        // =====================================================================
        // LUT APPLY
        // =====================================================================
        let depth_luts = &calibration.depth_luts;
        let n_zones = depth_luts.n_zones();
        let lut_size = if n_zones > 0 { depth_luts.luts[0].size } else { 33 };

        let luts_flat: Vec<f32> = depth_luts
            .luts
            .iter()
            .flat_map(|lg| lg.data.iter().copied())
            .collect();

        // Upload inputs
        let d_rgb_in = dev.htod_sync_copy(&rgb_f32).map_err(map_cudarc_error)?;
        let d_depth = dev.htod_sync_copy(depth).map_err(map_cudarc_error)?;
        let d_luts = dev.htod_sync_copy(&luts_flat).map_err(map_cudarc_error)?;
        let d_boundaries = dev
            .htod_sync_copy(&depth_luts.zone_boundaries)
            .map_err(map_cudarc_error)?;
        let d_rgb_after_lut: CudaSlice<f32> =
            dev.alloc_zeros(n * 3).map_err(map_cudarc_error)?;

        {
            let func = dev
                .get_func("lut_apply", "lut_apply_kernel")
                .ok_or_else(|| GpuError::ModuleLoad("lut_apply_kernel not found".into()))?;

            let cfg = LaunchConfig {
                grid_dim: (div_ceil(n as u32, 256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                func.launch(
                    cfg,
                    (
                        &d_rgb_in,
                        &d_depth,
                        &d_luts,
                        &d_boundaries,
                        &d_rgb_after_lut,
                        n as i32,
                        lut_size as i32,
                        n_zones as i32,
                    ),
                )
            }
            .map_err(map_cudarc_error)?;
        }

        // Free LUT-specific device memory (drop early to reduce peak VRAM)
        // Safe: cudarc uses stream-ordered cuMemFreeAsync on Ampere+ (sm_86).
        // Do not relax sm_86 target without auditing these early drops.
        drop(d_rgb_in);
        drop(d_luts);
        drop(d_boundaries);
        drop(d_depth); // depth only needed for LUT stage

        // =====================================================================
        // HSL CORRECT
        // =====================================================================
        let mut h_offsets = [0.0f32; 6];
        let mut s_ratios = [1.0f32; 6];
        let mut v_offsets = [0.0f32; 6];
        let mut weights = [0.0f32; 6];
        for (i, q) in calibration.hsl_corrections.0.iter().enumerate().take(6) {
            h_offsets[i] = q.h_offset;
            s_ratios[i] = q.s_ratio;
            v_offsets[i] = q.v_offset;
            weights[i] = q.weight;
        }

        let d_h_offsets = dev.htod_sync_copy(&h_offsets).map_err(map_cudarc_error)?;
        let d_s_ratios = dev.htod_sync_copy(&s_ratios).map_err(map_cudarc_error)?;
        let d_v_offsets = dev.htod_sync_copy(&v_offsets).map_err(map_cudarc_error)?;
        let d_weights = dev.htod_sync_copy(&weights).map_err(map_cudarc_error)?;
        let d_rgb_after_hsl: CudaSlice<f32> =
            dev.alloc_zeros(n * 3).map_err(map_cudarc_error)?;

        {
            let func = dev
                .get_func("hsl_correct", "hsl_correct_kernel")
                .ok_or_else(|| GpuError::ModuleLoad("hsl_correct_kernel not found".into()))?;

            let cfg = LaunchConfig {
                grid_dim: (div_ceil(n as u32, 256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                func.launch(
                    cfg,
                    (
                        &d_rgb_after_lut,
                        &d_rgb_after_hsl,
                        &d_h_offsets,
                        &d_s_ratios,
                        &d_v_offsets,
                        &d_weights,
                        n as i32,
                    ),
                )
            }
            .map_err(map_cudarc_error)?;
        }

        // Free HSL-specific device memory
        drop(d_rgb_after_lut);
        drop(d_h_offsets);
        drop(d_s_ratios);
        drop(d_v_offsets);
        drop(d_weights);

        // =====================================================================
        // CLARITY (proxy-resolution box blur)
        // =====================================================================
        let mean_d = depth.iter().sum::<f32>() / depth.len().max(1) as f32;
        let clarity_amount = (0.2 + 0.25 * mean_d) * params.contrast;

        let (proxy_w, proxy_h) = proxy_dims(width, height, PROXY_MAX_SIZE);
        let proxy_n = proxy_w * proxy_h;

        // Allocate proxy L buffers (ping-pong pair) and output RGB
        let d_proxy_l: CudaSlice<f32> = dev.alloc_zeros(proxy_n).map_err(map_cudarc_error)?;
        let d_blur_a: CudaSlice<f32> = dev.alloc_zeros(proxy_n).map_err(map_cudarc_error)?;
        let d_blur_b: CudaSlice<f32> = dev.alloc_zeros(proxy_n).map_err(map_cudarc_error)?;
        let d_rgb_out: CudaSlice<f32> = dev.alloc_zeros(n * 3).map_err(map_cudarc_error)?;

        // --- Sub-kernel A: extract proxy L from full-res RGB ---
        {
            let func = dev
                .get_func("clarity", "clarity_extract_L_proxy")
                .ok_or_else(|| {
                    GpuError::ModuleLoad("clarity_extract_L_proxy not found".into())
                })?;

            let cfg = LaunchConfig {
                grid_dim: (div_ceil(proxy_w as u32, 16), div_ceil(proxy_h as u32, 16), 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                func.launch(
                    cfg,
                    (
                        &d_rgb_after_hsl,
                        &d_proxy_l,
                        width as i32,
                        height as i32,
                        proxy_w as i32,
                        proxy_h as i32,
                    ),
                )
            }
            .map_err(map_cudarc_error)?;
        }

        // --- Sub-kernel B: 3-pass box blur (rows + cols each pass) ---
        // Pass 1: proxy_l -> blur_a (rows) -> blur_b (cols)
        // Pass 2: blur_b -> blur_a (rows) -> blur_b (cols) -- but we need to ping-pong
        // Actually: each pass is rows then cols. We read from src, write to dst, swap.
        //
        // Pass 1: rows(proxy_l -> blur_a), cols(blur_a -> blur_b)
        // Pass 2: rows(blur_b -> blur_a), cols(blur_a -> blur_b)
        // Pass 3: rows(blur_b -> blur_a), cols(blur_a -> blur_b)
        // Result in blur_b.

        let blur_rows_cfg = LaunchConfig {
            grid_dim: (
                div_ceil(proxy_w as u32, 32),
                div_ceil(proxy_h as u32, 8),
                1,
            ),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        };

        let blur_cols_cfg = LaunchConfig {
            grid_dim: (
                div_ceil(proxy_w as u32, 32),
                div_ceil(proxy_h as u32, 8),
                1,
            ),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        };

        for pass in 0..3 {
            // Rows: src -> blur_a
            {
                let func = dev
                    .get_func("clarity", "clarity_box_blur_rows")
                    .ok_or_else(|| {
                        GpuError::ModuleLoad("clarity_box_blur_rows not found".into())
                    })?;

                let src: &CudaSlice<f32> = if pass == 0 { &d_proxy_l } else { &d_blur_b };
                unsafe {
                    func.launch(
                        blur_rows_cfg,
                        (
                            src,
                            &d_blur_a,
                            proxy_w as i32,
                            proxy_h as i32,
                            BLUR_RADIUS,
                        ),
                    )
                }
                .map_err(map_cudarc_error)?;
            }

            // Cols: blur_a -> blur_b
            {
                let func = dev
                    .get_func("clarity", "clarity_box_blur_cols")
                    .ok_or_else(|| {
                        GpuError::ModuleLoad("clarity_box_blur_cols not found".into())
                    })?;

                unsafe {
                    func.launch(
                        blur_cols_cfg,
                        (
                            &d_blur_a,
                            &d_blur_b,
                            proxy_w as i32,
                            proxy_h as i32,
                            BLUR_RADIUS,
                        ),
                    )
                }
                .map_err(map_cudarc_error)?;
            }
        }
        // After 3 passes, blurred result is in d_blur_b.

        // --- Sub-kernel C: apply clarity at full resolution ---
        {
            let func = dev
                .get_func("clarity", "clarity_apply_kernel")
                .ok_or_else(|| {
                    GpuError::ModuleLoad("clarity_apply_kernel not found".into())
                })?;

            let cfg = LaunchConfig {
                grid_dim: (div_ceil(n as u32, 256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };

            // Match the actual kernel signature:
            // clarity_apply_kernel(rgb_in, rgb_out, blur_proxy,
            //                      clarity_amount, full_w, full_h, proxy_w, proxy_h)
            unsafe {
                func.launch(
                    cfg,
                    (
                        &d_rgb_after_hsl,
                        &d_rgb_out,
                        &d_blur_b,
                        clarity_amount,
                        width as i32,
                        height as i32,
                        proxy_w as i32,
                        proxy_h as i32,
                    ),
                )
            }
            .map_err(map_cudarc_error)?;
        }

        // Download result
        let result = dev.dtoh_sync_copy(&d_rgb_out).map_err(map_cudarc_error)?;

        // All CudaSlice values drop here, freeing device memory.
        Ok(result)
    }
}

/// Holds all device buffers sized by `(width, height)`.
///
/// Pre-allocated once and reused across frames via `htod_sync_copy_into`.
/// Reallocated only when the frame resolution changes.
#[cfg(feature = "cuda")]
struct ResolutionBuffers {
    width: usize,
    height: usize,
    proxy_w: usize,
    proxy_h: usize,
    d_rgb_in: CudaSlice<f32>,         // n * 3  — input RGB as f32
    d_depth: CudaSlice<f32>,          // n      — depth map
    d_rgb_after_lut: CudaSlice<f32>,  // n * 3  — LUT-stage output (scratch)
    d_rgb_after_hsl: CudaSlice<f32>,  // n * 3  — HSL-stage output (scratch)
    d_proxy_l: CudaSlice<f32>,        // proxy_n — clarity proxy luminance (scratch)
    d_blur_a: CudaSlice<f32>,         // proxy_n — blur ping-pong A (scratch)
    d_blur_b: CudaSlice<f32>,         // proxy_n — blur ping-pong B (scratch)
    d_rgb_out: CudaSlice<f32>,        // n * 3  — final output (scratch)
}

/// Holds all device buffers sized by `(n_zones, lut_size)`.
///
/// Pre-allocated once and reused across frames via `htod_sync_copy_into`.
/// Reallocated only when the calibration shape changes.
#[cfg(feature = "cuda")]
struct CalibrationBuffers {
    n_zones: usize,
    lut_size: usize,
    d_luts: CudaSlice<f32>,        // n_zones * lut_size^3 * 3
    d_boundaries: CudaSlice<f32>,  // n_zones + 1
    d_h_offsets: CudaSlice<f32>,   // 6
    d_s_ratios: CudaSlice<f32>,    // 6
    d_v_offsets: CudaSlice<f32>,   // 6
    d_weights: CudaSlice<f32>,     // 6
}

/// Allocate a fresh [`ResolutionBuffers`] for the given frame dimensions.
#[cfg(feature = "cuda")]
fn alloc_resolution_buffers(
    dev: &Arc<CudaDevice>,
    width: usize,
    height: usize,
) -> Result<ResolutionBuffers, GpuError> {
    let n = width * height;
    let (proxy_w, proxy_h) = proxy_dims(width, height, PROXY_MAX_SIZE);
    let proxy_n = proxy_w * proxy_h;
    Ok(ResolutionBuffers {
        width,
        height,
        proxy_w,
        proxy_h,
        d_rgb_in: dev.alloc_zeros(n * 3).map_err(map_cudarc_error)?,
        d_depth: dev.alloc_zeros(n).map_err(map_cudarc_error)?,
        d_rgb_after_lut: dev.alloc_zeros(n * 3).map_err(map_cudarc_error)?,
        d_rgb_after_hsl: dev.alloc_zeros(n * 3).map_err(map_cudarc_error)?,
        d_proxy_l: dev.alloc_zeros(proxy_n).map_err(map_cudarc_error)?,
        d_blur_a: dev.alloc_zeros(proxy_n).map_err(map_cudarc_error)?,
        d_blur_b: dev.alloc_zeros(proxy_n).map_err(map_cudarc_error)?,
        d_rgb_out: dev.alloc_zeros(n * 3).map_err(map_cudarc_error)?,
    })
}

/// Allocate a fresh [`CalibrationBuffers`] for the given calibration shape.
#[cfg(feature = "cuda")]
fn alloc_calibration_buffers(
    dev: &Arc<CudaDevice>,
    n_zones: usize,
    lut_size: usize,
) -> Result<CalibrationBuffers, GpuError> {
    Ok(CalibrationBuffers {
        n_zones,
        lut_size,
        d_luts: dev
            .alloc_zeros(n_zones * lut_size * lut_size * lut_size * 3)
            .map_err(map_cudarc_error)?,
        d_boundaries: dev.alloc_zeros(n_zones + 1).map_err(map_cudarc_error)?,
        d_h_offsets: dev.alloc_zeros(6).map_err(map_cudarc_error)?,
        d_s_ratios: dev.alloc_zeros(6).map_err(map_cudarc_error)?,
        d_v_offsets: dev.alloc_zeros(6).map_err(map_cudarc_error)?,
        d_weights: dev.alloc_zeros(6).map_err(map_cudarc_error)?,
    })
}

/// Map cudarc DriverError to GpuError, detecting OOM conditions.
#[cfg(feature = "cuda")]
fn map_cudarc_error(e: cudarc::driver::result::DriverError) -> GpuError {
    let msg = format!("{e}");
    if msg.contains("OUT_OF_MEMORY") || msg.contains("out of memory") {
        GpuError::Oom(msg)
    } else {
        GpuError::CudaFail(msg)
    }
}

/// Integer ceiling division.
#[cfg(feature = "cuda")]
fn div_ceil(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

/// Compute proxy dimensions: scale so the long edge <= max_size.
/// Inline here to avoid a dep on dorea-video just for this helper.
#[cfg(feature = "cuda")]
fn proxy_dims(src_w: usize, src_h: usize, max_size: usize) -> (usize, usize) {
    let long_edge = src_w.max(src_h);
    if long_edge <= max_size {
        return (src_w, src_h);
    }
    let scale = max_size as f64 / long_edge as f64;
    let pw = ((src_w as f64 * scale).round() as usize).max(1);
    let ph = ((src_h as f64 * scale).round() as usize).max(1);
    (pw, ph)
}

#[cfg(all(feature = "cuda", test))]
mod tests {
    use super::*;
    use crate::GradeParams;
    use dorea_cal::Calibration;
    use dorea_hsl::derive::{HslCorrections, QualifierCorrection};
    use dorea_lut::types::{DepthLuts, LutGrid};

    /// Build an identity LUT of given size: output == input for all lattice points.
    fn identity_lut(size: usize) -> LutGrid {
        let mut lut = LutGrid::new(size);
        for ri in 0..size {
            for gi in 0..size {
                for bi in 0..size {
                    let r = ri as f32 / (size - 1) as f32;
                    let g = gi as f32 / (size - 1) as f32;
                    let b = bi as f32 / (size - 1) as f32;
                    lut.set(ri, gi, bi, [r, g, b]);
                }
            }
        }
        lut
    }

    /// Build a `Calibration` with `n_zones` depth zones, each holding an identity LUT.
    fn make_calibration(n_zones: usize) -> Calibration {
        let luts: Vec<LutGrid> = (0..n_zones).map(|_| identity_lut(17)).collect();
        let boundaries: Vec<f32> = (0..=n_zones).map(|i| i as f32 / n_zones as f32).collect();
        let depth_luts = DepthLuts::new(luts, boundaries);
        let hsl = HslCorrections(vec![QualifierCorrection {
            h_center: 0.0,
            h_width: 1.0,
            h_offset: 0.0,
            s_ratio: 1.0,
            v_offset: 0.0,
            weight: 0.0, // weight=0 → qualifier inactive, HSL stage is pass-through in tests
        }]);
        Calibration::new(depth_luts, hsl, 1)
    }

    /// Build synthetic pixel and depth data for a frame of size `w * h`.
    fn make_frame(w: usize, h: usize) -> (Vec<u8>, Vec<f32>) {
        let n = w * h;
        let pixels: Vec<u8> = (0..n * 3).map(|i| ((i * 7 + 128) % 256) as u8).collect();
        let depth: Vec<f32> = (0..n).map(|i| (i as f32) / n as f32 * 0.8 + 0.1).collect();
        (pixels, depth)
    }

    /// Calling `grade_frame_cuda` twice with the same input must produce identical output.
    ///
    /// This verifies that the GPU pipeline is deterministic: no race conditions,
    /// no uninitialized memory, no stale state between calls.
    #[test]
    fn determinism() {
        let grader = CudaGrader::new().expect("CudaGrader::new() failed");
        let cal = make_calibration(5);
        let params = GradeParams::default();
        let (pixels, depth) = make_frame(320, 240);

        let out1 = grader
            .grade_frame_cuda(&pixels, &depth, 320, 240, &cal, &params)
            .expect("first grade_frame_cuda call failed");
        let out2 = grader
            .grade_frame_cuda(&pixels, &depth, 320, 240, &cal, &params)
            .expect("second grade_frame_cuda call failed");

        assert_eq!(out1.len(), out2.len(), "output length must be identical between runs");
        for (i, (a, b)) in out1.iter().zip(out2.iter()).enumerate() {
            assert!(a.is_finite(), "output[{i}] is not finite: {a}");
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "output differs at element {i}: {a} vs {b}"
            );
        }
    }

    /// Switching resolution between calls must succeed and return the correct output size.
    ///
    /// Sequence: 1280×720, then 1920×1080, then 1280×720 again.
    /// Each call must return `width * height * 3` f32 elements.
    #[test]
    fn resolution_switch_round_trip() {
        let grader = CudaGrader::new().expect("CudaGrader::new() failed");
        let cal = make_calibration(5);
        let params = GradeParams::default();

        let resolutions: &[(usize, usize)] = &[(1280, 720), (1920, 1080), (1280, 720)];

        for &(w, h) in resolutions {
            let (pixels, depth) = make_frame(w, h);
            let out = grader
                .grade_frame_cuda(&pixels, &depth, w, h, &cal, &params)
                .unwrap_or_else(|e| panic!("grade_frame_cuda failed at {w}×{h}: {e}"));

            let expected_len = w * h * 3;
            assert_eq!(
                out.len(),
                expected_len,
                "output length for {w}×{h} should be {expected_len}, got {}",
                out.len()
            );
        }
    }

    /// Switching calibration zone count between calls must succeed and return correct output size.
    ///
    /// Sequence: 5-zone calibration, then 3-zone, then 5-zone again.
    /// Output length must always equal `width * height * 3`.
    #[test]
    fn calibration_shape_switch() {
        let grader = CudaGrader::new().expect("CudaGrader::new() failed");
        let params = GradeParams::default();
        let (w, h) = (320, 240);
        let (pixels, depth) = make_frame(w, h);
        let expected_len = w * h * 3;

        let zone_counts = [5usize, 3, 5];

        for n_zones in zone_counts {
            let cal = make_calibration(n_zones);
            let out = grader
                .grade_frame_cuda(&pixels, &depth, w, h, &cal, &params)
                .unwrap_or_else(|e| {
                    panic!("grade_frame_cuda failed with {n_zones}-zone calibration: {e}")
                });

            assert_eq!(
                out.len(),
                expected_len,
                "output length for {n_zones}-zone calibration should be {expected_len}, got {}",
                out.len()
            );
        }
    }
}
