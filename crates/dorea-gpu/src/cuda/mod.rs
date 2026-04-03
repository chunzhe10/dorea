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

        // Free LUT-specific device memory (drop early)
        drop(d_rgb_in);
        drop(d_luts);
        drop(d_boundaries);

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
