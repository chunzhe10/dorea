//! CUDA-backed grading pipeline using cudarc PTX module loading.
//!
//! Only compiled when the `cuda` feature is enabled (detected by build.rs).
//!
//! `CudaGrader` holds an `Arc<CudaDevice>` and the precomputed `CombinedLut` textures.
//! Created once via `CudaGrader::new(calibration, params)`; reused across frames.

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

#[cfg(feature = "cuda")]
mod combined_lut;
#[cfg(feature = "cuda")]
pub(crate) use combined_lut::CombinedLut;

// Embedded PTX compiled by build.rs
#[cfg(feature = "cuda")]
const COMBINED_LUT_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/combined_lut.ptx"));

/// CUDA grader: holds a device handle and the precomputed combined LUT textures.
///
/// Create once via `CudaGrader::new(calibration, params)`, reuse across frames.
/// `!Send + !Sync` — CUDA contexts are thread-local.
///
/// Field order matters for drop: `combined_lut` must be declared before `device`
/// so it is dropped first (Rust drops fields in declaration order). The
/// `CombinedLut` destructor calls `cuTexObjectDestroy` / `cuArrayDestroy`, which
/// require the CUDA context to still be alive — i.e. `device` must outlive it.
#[cfg(feature = "cuda")]
pub struct CudaGrader {
    combined_lut: CombinedLut,              // dropped first — CUDA APIs require live context
    device:       Arc<CudaDevice>,          // dropped second — destroys CUDA context
    _not_send:    std::marker::PhantomData<*const ()>,
}

#[cfg(feature = "cuda")]
impl CudaGrader {
    /// Initialise CUDA device 0, load per-frame kernel PTX, and build combined LUT textures.
    pub fn new(calibration: &Calibration, params: &GradeParams) -> Result<Self, GpuError> {
        let device = CudaDevice::new(0).map_err(|e| {
            GpuError::ModuleLoad(format!("CudaDevice::new(0) failed: {e}"))
        })?;

        // Load the per-frame lookup kernel PTX
        device.load_ptx(
            Ptx::from_src(COMBINED_LUT_PTX),
            "combined_lut",
            &["combined_lut_kernel"],
        ).map_err(|e| GpuError::ModuleLoad(format!("load combined_lut PTX: {e}")))?;

        // Build combined LUT textures (GPU build kernel runs here, ~<1ms)
        let combined_lut = CombinedLut::build(&device, calibration, params)?;

        Ok(Self {
            device,
            combined_lut,
            _not_send: std::marker::PhantomData,
        })
    }

    /// Run the combined LUT kernel on one frame.
    ///
    /// Returns graded sRGB u8 pixels (interleaved RGB, same dimensions as input).
    pub fn grade_frame_cuda(
        &self,
        pixels: &[u8],
        depth: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<u8>, GpuError> {
        let n = width.checked_mul(height).ok_or_else(|| {
            GpuError::InvalidInput("frame dimensions overflow usize".into())
        })?;
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
        let dev = &self.device;

        // Upload inputs
        let d_pixels_in = dev.htod_sync_copy(pixels).map_err(map_cudarc_error)?;
        let d_depth     = dev.htod_sync_copy(depth).map_err(map_cudarc_error)?;

        // Upload texture handles (CUtexObject = u64) to device
        let d_textures: CudaSlice<u64> = dev
            .htod_sync_copy(&self.combined_lut.textures)
            .map_err(map_cudarc_error)?;

        // Upload zone boundaries
        let d_boundaries = dev
            .htod_sync_copy(&self.combined_lut.zone_boundaries)
            .map_err(map_cudarc_error)?;

        // Output buffer
        let d_pixels_out: CudaSlice<u8> = dev.alloc_zeros(n * 3).map_err(map_cudarc_error)?;

        // Launch combined_lut_kernel
        {
            let func = dev.get_func("combined_lut", "combined_lut_kernel")
                .ok_or_else(|| GpuError::ModuleLoad("combined_lut_kernel not found".into()))?;
            let cfg = LaunchConfig {
                grid_dim: (div_ceil(n as u32, 256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                func.launch(cfg, (
                    &d_pixels_in,
                    &d_depth,
                    &d_textures,
                    &d_boundaries,
                    &d_pixels_out,
                    n as i32,
                    self.combined_lut.n_zones as i32,
                    self.combined_lut.grid_size as i32,
                ))
            }.map_err(map_cudarc_error)?;
        }

        // Download result
        let result = dev.dtoh_sync_copy(&d_pixels_out).map_err(map_cudarc_error)?;
        Ok(result)
    }
}

/// Map cudarc DriverError to GpuError, detecting OOM conditions.
#[cfg(feature = "cuda")]
pub(crate) fn map_cudarc_error(e: cudarc::driver::result::DriverError) -> GpuError {
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


#[cfg(all(feature = "cuda", test))]
mod tests {
    use super::*;
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

    #[test]
    fn new_api_takes_calibration_and_params() {
        let cal = make_calibration(5);
        let params = crate::GradeParams::default();
        match CudaGrader::new(&cal, &params) {
            Ok(grader) => {
                let (pixels, depth) = make_frame(4, 4);
                let out: Vec<u8> = grader.grade_frame_cuda(&pixels, &depth, 4, 4)
                    .expect("grade_frame_cuda failed");
                assert_eq!(out.len(), 4 * 4 * 3);
            }
            Err(e) => eprintln!("SKIP: no CUDA device ({e})"),
        }
    }

    /// Calling `grade_frame_cuda` twice with the same input must produce identical output.
    ///
    /// This verifies that the GPU pipeline is deterministic: no race conditions,
    /// no uninitialized memory, no stale state between calls.
    #[test]
    fn determinism() {
        let cal = make_calibration(5);
        let params = crate::GradeParams::default();
        let (w, h) = (4, 4);
        let (pixels, depth) = make_frame(w, h);
        let grader = match CudaGrader::new(&cal, &params) {
            Ok(g) => g,
            Err(e) => { eprintln!("SKIP: {e}"); return; }
        };
        let out1 = grader.grade_frame_cuda(&pixels, &depth, w, h).expect("first call failed");
        let out2 = grader.grade_frame_cuda(&pixels, &depth, w, h).expect("second call failed");
        assert_eq!(out1, out2, "grade_frame_cuda must be deterministic");
    }

    /// Switching resolution between calls must succeed and return the correct output size.
    ///
    /// Sequence: 1280×720, then 1920×1080, then 1280×720 again.
    /// Each call must return `width * height * 3` bytes.
    #[test]
    fn resolution_switch_round_trip() {
        let cal = make_calibration(5);
        let params = crate::GradeParams::default();
        let grader = match CudaGrader::new(&cal, &params) {
            Ok(g) => g,
            Err(e) => { eprintln!("SKIP: {e}"); return; }
        };

        let resolutions: &[(usize, usize)] = &[(1280, 720), (1920, 1080), (1280, 720)];

        for &(w, h) in resolutions {
            let (pixels, depth) = make_frame(w, h);
            let out = grader.grade_frame_cuda(&pixels, &depth, w, h)
                .unwrap_or_else(|e| panic!("grade_frame_cuda failed at {w}×{h}: {e}"));
            let expected_len = w * h * 3;
            assert_eq!(out.len(), expected_len,
                "output length for {w}×{h} should be {expected_len}, got {}", out.len());
        }
    }

    #[test]
    fn combined_lut_builds_without_panic() {
        use crate::cuda::combined_lut::CombinedLut;
        let cal = make_calibration(5);
        let params = crate::GradeParams::default();
        let device = match cudarc::driver::CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => { eprintln!("SKIP: no CUDA device ({e})"); return; }
        };
        let _lut = CombinedLut::build(&device, &cal, &params).expect("CombinedLut::build failed");
    }

    /// Helper: build calibration with a gamma-shifted LUT (not identity).
    /// Uses the fixed 6-vector HSL qualifier grid that the GPU hardcodes in
    /// GP_H_CENTERS/GP_H_WIDTHS, activating the Green slot (index 2, h_center=100).
    fn make_shifted_calibration(n_zones: usize) -> Calibration {
        use dorea_lut::types::{DepthLuts, LutGrid};
        use dorea_hsl::derive::{HslCorrections, QualifierCorrection};
        use dorea_hsl::qualifiers::HSL_QUALIFIERS;
        let size = 17usize;
        let luts: Vec<LutGrid> = (0..n_zones).map(|_| {
            let mut lut = LutGrid::new(size);
            for ri in 0..size { for gi in 0..size { for bi in 0..size {
                let r = (ri as f32 / (size - 1) as f32).powf(0.9);
                let g = (gi as f32 / (size - 1) as f32).powf(0.9);
                let b = (bi as f32 / (size - 1) as f32).powf(0.9);
                lut.set(ri, gi, bi, [r, g, b]);
            }}}
            lut
        }).collect();
        let boundaries: Vec<f32> = (0..=n_zones).map(|i| i as f32 / n_zones as f32).collect();
        // Use the standard 6-vector structure (matches GP_H_CENTERS/GP_H_WIDTHS in grade_pixel.cuh).
        // Green slot (index 2, h_center=100, h_width=50) gets an active s_ratio=1.2.
        let hsl = HslCorrections(
            HSL_QUALIFIERS.iter().enumerate().map(|(i, q)| QualifierCorrection {
                h_center: q.h_center,
                h_width:  q.h_width,
                h_offset: 0.0,
                s_ratio:  if i == 2 { 1.2 } else { 1.0 },
                v_offset: 0.0,
                weight:   if i == 2 { 200.0 } else { 0.0 },
            }).collect()
        );
        Calibration::new(DepthLuts::new(luts, boundaries), hsl, 1)
    }

    #[test]
    fn combined_lut_within_2_per_255_of_cpu() {
        let cal = make_calibration(5);
        let params = crate::GradeParams::default();
        let (w, h) = (32, 32);
        let (pixels, depth) = make_frame(w, h);

        let grader = match CudaGrader::new(&cal, &params) {
            Ok(g) => g,
            Err(e) => { eprintln!("SKIP: {e}"); return; }
        };
        let gpu_out = grader.grade_frame_cuda(&pixels, &depth, w, h)
            .expect("grade_frame_cuda failed");
        let cpu_out = crate::cpu::grade_frame_cpu(&pixels, &depth, w, h, &cal, &params)
            .expect("grade_frame_cpu failed");

        assert_eq!(gpu_out.len(), cpu_out.len());
        let mut diffs: Vec<u32> = gpu_out.iter().zip(cpu_out.iter())
            .map(|(&g, &c)| (g as i32 - c as i32).unsigned_abs())
            .collect();
        diffs.sort_unstable();
        let max_diff = *diffs.last().unwrap_or(&0);
        let p99_idx  = (diffs.len() as f32 * 0.99) as usize;
        let p99_diff = diffs.get(p99_idx).copied().unwrap_or(0);
        eprintln!("GPU/CPU diff — max: {max_diff}/255, p99: {p99_diff}/255");

        assert!(max_diff <= 2,
            "GPU/CPU max diff {max_diff}/255 exceeds 2/255 tolerance.");
    }

    #[test]
    fn combined_lut_non_trivial_lut_and_hsl_within_2_per_255() {
        let cal = make_shifted_calibration(3);
        let params = GradeParams { warmth: 1.1, strength: 0.9, contrast: 1.1 };
        let (w, h) = (16, 16);
        let (pixels, depth) = make_frame(w, h);

        let grader = match CudaGrader::new(&cal, &params) {
            Ok(g) => g,
            Err(e) => { eprintln!("SKIP: {e}"); return; }
        };
        let gpu_out = grader.grade_frame_cuda(&pixels, &depth, w, h)
            .expect("grade_frame_cuda failed");
        let cpu_out = crate::cpu::grade_frame_cpu(&pixels, &depth, w, h, &cal, &params)
            .expect("grade_frame_cpu failed");

        let max_diff = gpu_out.iter().zip(cpu_out.iter())
            .map(|(&g, &c)| (g as i32 - c as i32).unsigned_abs())
            .max().unwrap_or(0);
        // 4/255 tolerance (not 2) for nonlinear LUT + active HSL: the GPU hardware
        // trilinear interpolation of the baked pipeline introduces ~4/255 quantization
        // error in regions of high curvature (gamma LUT + LAB + HSL combined).
        // Identity-LUT case (combined_lut_within_2_per_255_of_cpu) verifies 2/255.
        assert!(max_diff <= 4,
            "Non-trivial LUT: GPU/CPU max diff {max_diff}/255 exceeds 4/255");
    }

    #[test]
    fn combined_lut_edge_case_pixels() {
        let cal = make_calibration(5);
        let params = crate::GradeParams::default();
        let grader = match CudaGrader::new(&cal, &params) {
            Ok(g) => g,
            Err(e) => { eprintln!("SKIP: {e}"); return; }
        };

        let pixels: Vec<u8> = vec![
            0, 0, 0,
            255, 255, 255,
            128, 64, 32,
            128, 64, 32,
        ];
        let depth = vec![0.0f32, 1.0f32, 0.0f32, 1.0f32];

        let gpu_out = grader.grade_frame_cuda(&pixels, &depth, 4, 1)
            .expect("grade_frame_cuda failed");
        let cpu_out = crate::cpu::grade_frame_cpu(&pixels, &depth, 4, 1, &cal, &params)
            .expect("grade_frame_cpu failed");

        assert_eq!(gpu_out.len(), cpu_out.len());
        let max_diff = gpu_out.iter().zip(cpu_out.iter())
            .map(|(&g, &c)| (g as i32 - c as i32).unsigned_abs())
            .max().unwrap_or(0);
        assert!(max_diff <= 2,
            "Edge-case pixels: GPU/CPU max diff {max_diff}/255 exceeds 2/255");
    }

    #[test]
    fn combined_lut_zone_count_variants() {
        for n_zones in [1usize, 3, 8] {
            let cal = make_calibration(n_zones);
            let params = crate::GradeParams::default();
            let grader = match CudaGrader::new(&cal, &params) {
                Ok(g) => g,
                Err(e) => { eprintln!("SKIP n_zones={n_zones}: {e}"); continue; }
            };
            let (pixels, depth) = make_frame(8, 8);
            let gpu_out = grader.grade_frame_cuda(&pixels, &depth, 8, 8)
                .expect(&format!("grade_frame_cuda failed for n_zones={n_zones}"));
            assert_eq!(gpu_out.len(), 8 * 8 * 3,
                "n_zones={n_zones}: output length mismatch");
        }
    }
}
