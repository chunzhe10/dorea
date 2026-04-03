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
use std::cell::RefCell;
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
    res_bufs: RefCell<Option<ResolutionBuffers>>,
    cal_bufs: RefCell<Option<CalibrationBuffers>>,
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
            res_bufs: RefCell::new(None),
            cal_bufs: RefCell::new(None),
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

        // --- Input length guards (BEFORE any borrow_mut — htod_sync_copy_into panics on mismatch) ---
        if pixels.len() != n * 3 {
            return Err(GpuError::InvalidInput(format!(
                "pixels len {} != width*height*3 {}", pixels.len(), n * 3
            )));
        }
        if depth.len() != n {
            return Err(GpuError::InvalidInput(format!(
                "depth len {} != width*height {}", depth.len(), n
            )));
        }

        // --- Calibration shape ---
        let depth_luts = &calibration.depth_luts;
        let n_zones = depth_luts.n_zones();
        if n_zones == 0 {
            return Err(GpuError::InvalidInput("n_zones must be >= 1".into()));
        }
        let lut_size = depth_luts.luts[0].size;

        // --- Ensure ResolutionBuffers for this (width, height) ---
        {
            let mut guard = self.res_bufs.borrow_mut();
            let needs_alloc = guard.as_ref()
                .map_or(true, |b| b.width != width || b.height != height);
            if needs_alloc {
                *guard = Some(alloc_resolution_buffers(dev, width, height)?);
            }
        }

        // --- Ensure CalibrationBuffers for this (n_zones, lut_size) ---
        {
            let mut guard = self.cal_bufs.borrow_mut();
            let needs_alloc = guard.as_ref()
                .map_or(true, |b| b.n_zones != n_zones || b.lut_size != lut_size);
            if needs_alloc {
                *guard = Some(alloc_calibration_buffers(dev, n_zones, lut_size)?);
            }
        }

        // --- Upload resolution-keyed data (no alloc — copy only) ---
        let rgb_f32: Vec<f32> = pixels.iter().map(|&p| p as f32 / 255.0).collect();
        {
            let mut guard = self.res_bufs.borrow_mut();
            let bufs = guard.as_mut().unwrap();
            dev.htod_sync_copy_into(&rgb_f32, &mut bufs.d_rgb_in).map_err(map_cudarc_error)?;
            dev.htod_sync_copy_into(depth, &mut bufs.d_depth).map_err(map_cudarc_error)?;
        }

        // --- Upload calibration-keyed data (no alloc — copy only) ---
        let luts_flat: Vec<f32> = depth_luts.luts.iter()
            .flat_map(|lg| lg.data.iter().copied())
            .collect();
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
        {
            let mut guard = self.cal_bufs.borrow_mut();
            let bufs = guard.as_mut().unwrap();
            dev.htod_sync_copy_into(&luts_flat,                          &mut bufs.d_luts      ).map_err(map_cudarc_error)?;
            dev.htod_sync_copy_into(&depth_luts.zone_boundaries,         &mut bufs.d_boundaries).map_err(map_cudarc_error)?;
            dev.htod_sync_copy_into(&h_offsets,                          &mut bufs.d_h_offsets ).map_err(map_cudarc_error)?;
            dev.htod_sync_copy_into(&s_ratios,                           &mut bufs.d_s_ratios  ).map_err(map_cudarc_error)?;
            dev.htod_sync_copy_into(&v_offsets,                          &mut bufs.d_v_offsets ).map_err(map_cudarc_error)?;
            dev.htod_sync_copy_into(&weights,                            &mut bufs.d_weights   ).map_err(map_cudarc_error)?;
        }

        // --- Kernel launches — immutable borrow on both buffer sets ---
        let res = self.res_bufs.borrow();
        let res_bufs = res.as_ref().unwrap();
        let cal = self.cal_bufs.borrow();
        let cal_bufs = cal.as_ref().unwrap();

        // =====================================================================
        // LUT APPLY
        // =====================================================================
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
                        &res_bufs.d_rgb_in,
                        &res_bufs.d_depth,
                        &cal_bufs.d_luts,
                        &cal_bufs.d_boundaries,
                        &res_bufs.d_rgb_after_lut,
                        n as i32,
                        lut_size as i32,
                        n_zones as i32,
                    ),
                )
            }
            .map_err(map_cudarc_error)?;
        }

        // =====================================================================
        // HSL CORRECT
        // =====================================================================
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
                        &res_bufs.d_rgb_after_lut,
                        &res_bufs.d_rgb_after_hsl,
                        &cal_bufs.d_h_offsets,
                        &cal_bufs.d_s_ratios,
                        &cal_bufs.d_v_offsets,
                        &cal_bufs.d_weights,
                        n as i32,
                    ),
                )
            }
            .map_err(map_cudarc_error)?;
        }

        // =====================================================================
        // CLARITY (proxy-resolution box blur)
        // =====================================================================
        let mean_d = depth.iter().sum::<f32>() / depth.len().max(1) as f32;
        let clarity_amount = (0.2 + 0.25 * mean_d) * params.contrast;

        let proxy_w = res_bufs.proxy_w;
        let proxy_h = res_bufs.proxy_h;

        // Sub-kernel A: extract proxy L
        {
            let func = dev
                .get_func("clarity", "clarity_extract_L_proxy")
                .ok_or_else(|| GpuError::ModuleLoad("clarity_extract_L_proxy not found".into()))?;

            let cfg = LaunchConfig {
                grid_dim: (div_ceil(proxy_w as u32, 16), div_ceil(proxy_h as u32, 16), 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                func.launch(
                    cfg,
                    (
                        &res_bufs.d_rgb_after_hsl,
                        &res_bufs.d_proxy_l,
                        width as i32,
                        height as i32,
                        proxy_w as i32,
                        proxy_h as i32,
                    ),
                )
            }
            .map_err(map_cudarc_error)?;
        }

        // Sub-kernel B: 3-pass box blur
        let blur_rows_cfg = LaunchConfig {
            grid_dim: (div_ceil(proxy_w as u32, 32), div_ceil(proxy_h as u32, 8), 1),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        };
        let blur_cols_cfg = LaunchConfig {
            grid_dim: (div_ceil(proxy_w as u32, 32), div_ceil(proxy_h as u32, 8), 1),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        };

        for pass in 0..3 {
            let src: &CudaSlice<f32> = if pass == 0 { &res_bufs.d_proxy_l } else { &res_bufs.d_blur_b };

            {
                let func = dev
                    .get_func("clarity", "clarity_box_blur_rows")
                    .ok_or_else(|| GpuError::ModuleLoad("clarity_box_blur_rows not found".into()))?;
                unsafe {
                    func.launch(blur_rows_cfg, (src, &res_bufs.d_blur_a, proxy_w as i32, proxy_h as i32, BLUR_RADIUS))
                }
                .map_err(map_cudarc_error)?;
            }

            {
                let func = dev
                    .get_func("clarity", "clarity_box_blur_cols")
                    .ok_or_else(|| GpuError::ModuleLoad("clarity_box_blur_cols not found".into()))?;
                unsafe {
                    func.launch(blur_cols_cfg, (&res_bufs.d_blur_a, &res_bufs.d_blur_b, proxy_w as i32, proxy_h as i32, BLUR_RADIUS))
                }
                .map_err(map_cudarc_error)?;
            }
        }

        // Sub-kernel C: apply clarity at full resolution
        {
            let func = dev
                .get_func("clarity", "clarity_apply_kernel")
                .ok_or_else(|| GpuError::ModuleLoad("clarity_apply_kernel not found".into()))?;

            let cfg = LaunchConfig {
                grid_dim: (div_ceil(n as u32, 256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                func.launch(
                    cfg,
                    (
                        &res_bufs.d_rgb_after_hsl,
                        &res_bufs.d_rgb_out,
                        &res_bufs.d_blur_b,
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

        // SAFETY: dtoh_sync_copy calls device.synchronize() — stream is idle after this download.
        // Buffer contents are stable on device; the next grade_frame_cuda call can safely re-upload
        // into these pre-allocated slices without data races.
        let result = dev.dtoh_sync_copy(&res_bufs.d_rgb_out).map_err(map_cudarc_error)?;

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
    d_depth: CudaSlice<f32>,          // n — depth map; held for full frame (no early drop with pre-alloc)
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
    d_luts: CudaSlice<f32>,        // n_zones * lut_size³ * 3  (lut_size cubed)
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
    let n = width.checked_mul(height).ok_or_else(|| {
        GpuError::InvalidInput("frame dimensions overflow usize".into())
    })?;
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
    if n_zones == 0 {
        return Err(GpuError::InvalidInput("n_zones must be >= 1".into()));
    }
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
