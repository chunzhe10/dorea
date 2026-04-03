# CudaGrader Buffer Pre-Allocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate per-frame `cudaMalloc` overhead in `CudaGrader::grade_frame_cuda` by pre-allocating device buffers keyed by resolution and calibration shape, reusing them across frames.

**Architecture:** `CudaGrader` gains two `RefCell<Option<...>>` fields — `ResolutionBuffers` (8 slices, keyed by width×height) and `CalibrationBuffers` (6 slices, keyed by n_zones×lut_size). On each frame call, if the key matches the cached buffers they are reused via `htod_sync_copy_into` (copy only, no allocation); otherwise they are dropped and reallocated. Zero `cudaMalloc` calls per frame in steady state.

**Tech Stack:** Rust, cudarc 0.12.1 (`htod_sync_copy_into`, `alloc_zeros`, `dtoh_sync_copy`), `RefCell` for interior mutability, `#[cfg(feature = "cuda")]` throughout.

---

## File Structure

| File | Change |
|------|--------|
| `crates/dorea-gpu/src/cuda/mod.rs` | All changes live here. Add structs, update `CudaGrader`, refactor `grade_frame_cuda`. |

No other files change.

---

### Task 1: Add regression tests (run before refactoring to confirm baseline)

**Files:**
- Modify: `crates/dorea-gpu/src/cuda/mod.rs` (bottom `#[cfg(test)]` block)

These tests establish behavioral invariants that must hold before AND after refactoring.
They protect against the refactoring accidentally corrupting output.

- [ ] **Step 1: Open `crates/dorea-gpu/src/cuda/mod.rs` and locate the end of the file**

  The file currently ends with the closing `}` of the module at line ~424.
  Add a `#[cfg(test)]` block just before the final `}`:

```rust
#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use dorea_cal::Calibration;
    use dorea_hsl::derive::{HslCorrections, QualifierCorrection};
    use dorea_lut::types::{DepthLuts, LutGrid};

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

    fn make_calibration(n_zones: usize, lut_size: usize) -> Calibration {
        let luts: Vec<LutGrid> = (0..n_zones).map(|_| identity_lut(lut_size)).collect();
        let boundaries: Vec<f32> = (0..=n_zones).map(|i| i as f32 / n_zones as f32).collect();
        let depth_luts = DepthLuts::new(luts, boundaries);
        let hsl = HslCorrections(vec![QualifierCorrection {
            h_center: 0.0,
            h_width: 1.0,
            h_offset: 0.0,
            s_ratio: 1.0,
            v_offset: 0.0,
            weight: 0.0,
        }]);
        Calibration::new(depth_luts, hsl, 1)
    }

    fn make_pixels(w: usize, h: usize) -> Vec<u8> {
        (0..w * h * 3).map(|i| ((i * 7 + 128) % 256) as u8).collect()
    }

    fn make_depth(w: usize, h: usize) -> Vec<f32> {
        let n = w * h;
        (0..n).map(|i| i as f32 / n as f32 * 0.8 + 0.1).collect()
    }

    /// Same resolution twice → same output (determinism check).
    #[test]
    fn same_resolution_twice_is_deterministic() {
        let grader = CudaGrader::new().expect("CudaGrader::new failed — no GPU?");
        let cal = make_calibration(3, 17);
        let params = crate::GradeParams::default();
        let (w, h) = (64, 36);
        let pixels = make_pixels(w, h);
        let depth = make_depth(w, h);

        let out1 = grader
            .grade_frame_cuda(&pixels, &depth, w, h, &cal, &params)
            .expect("first call failed");
        let out2 = grader
            .grade_frame_cuda(&pixels, &depth, w, h, &cal, &params)
            .expect("second call failed");

        assert_eq!(out1.len(), w * h * 3, "output length wrong");
        assert_eq!(out1, out2, "same input twice must produce same output");
    }

    /// Resolution switch: 64×36 → 128×72 → 64×36. Output must be correct length
    /// and the third call must equal the first (buffers not corrupted by resize).
    #[test]
    fn resolution_switch_preserves_output() {
        let grader = CudaGrader::new().expect("CudaGrader::new failed — no GPU?");
        let cal = make_calibration(3, 17);
        let params = crate::GradeParams::default();

        let (w1, h1) = (64, 36);
        let pixels1 = make_pixels(w1, h1);
        let depth1 = make_depth(w1, h1);

        let (w2, h2) = (128, 72);
        let pixels2 = make_pixels(w2, h2);
        let depth2 = make_depth(w2, h2);

        let out1 = grader
            .grade_frame_cuda(&pixels1, &depth1, w1, h1, &cal, &params)
            .expect("720p call failed");
        assert_eq!(out1.len(), w1 * h1 * 3);

        let out2 = grader
            .grade_frame_cuda(&pixels2, &depth2, w2, h2, &cal, &params)
            .expect("1080p call failed");
        assert_eq!(out2.len(), w2 * h2 * 3);

        // Back to small resolution — must equal the first call exactly.
        let out3 = grader
            .grade_frame_cuda(&pixels1, &depth1, w1, h1, &cal, &params)
            .expect("return to 720p failed");
        assert_eq!(out3.len(), w1 * h1 * 3);
        assert_eq!(out1, out3, "output after resolution switch back must equal original");
    }

    /// Calibration shape change: 3 zones 17³ → 5 zones 33³. Must not error.
    #[test]
    fn calibration_shape_change_does_not_error() {
        let grader = CudaGrader::new().expect("CudaGrader::new failed — no GPU?");
        let params = crate::GradeParams::default();
        let (w, h) = (64, 36);
        let pixels = make_pixels(w, h);
        let depth = make_depth(w, h);

        let cal1 = make_calibration(3, 17);
        let out1 = grader
            .grade_frame_cuda(&pixels, &depth, w, h, &cal1, &params)
            .expect("call with 3-zone cal failed");
        assert_eq!(out1.len(), w * h * 3);

        let cal2 = make_calibration(5, 33);
        let out2 = grader
            .grade_frame_cuda(&pixels, &depth, w, h, &cal2, &params)
            .expect("call with 5-zone cal failed");
        assert_eq!(out2.len(), w * h * 3);

        // Back to original calibration — must still work.
        let out3 = grader
            .grade_frame_cuda(&pixels, &depth, w, h, &cal1, &params)
            .expect("return to 3-zone cal failed");
        assert_eq!(out1, out3, "output after calibration switch back must equal original");
    }
}
```

- [ ] **Step 2: Run the tests to confirm they pass with the current (pre-refactor) code**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-gpu --test -- cuda::tests 2>&1 | tail -20
```

Expected: all 3 tests PASS. If any fail, the current implementation has a bug — do not proceed until they pass.

- [ ] **Step 3: Commit**

```bash
git add crates/dorea-gpu/src/cuda/mod.rs
git commit -m "test(dorea-gpu): regression tests for CudaGrader resolution/calibration switching"
```

---

### Task 2: Add `ResolutionBuffers` and `CalibrationBuffers` structs with allocation helpers

**Files:**
- Modify: `crates/dorea-gpu/src/cuda/mod.rs`

- [ ] **Step 1: Add the two buffer structs after the `PROXY_MAX_SIZE` constant (around line 35)**

  Insert after:
  ```rust
  /// Maximum proxy long-edge for clarity downsampling (matches CPU path: 518).
  #[cfg(feature = "cuda")]
  const PROXY_MAX_SIZE: usize = 518;
  ```

  Add:
  ```rust
  /// Pre-allocated device buffers for a specific frame resolution.
  ///
  /// Keyed by (width, height). Dropped and reallocated when resolution changes.
  /// Holds all 8 pixel-count-scaled device slices used across LUT, HSL, and clarity stages.
  #[cfg(feature = "cuda")]
  struct ResolutionBuffers {
      width: usize,
      height: usize,
      proxy_w: usize,
      proxy_h: usize,
      d_rgb_in: CudaSlice<f32>,        // n × 3
      d_depth: CudaSlice<f32>,         // n
      d_rgb_after_lut: CudaSlice<f32>, // n × 3
      d_rgb_after_hsl: CudaSlice<f32>, // n × 3
      d_proxy_l: CudaSlice<f32>,       // proxy_n
      d_blur_a: CudaSlice<f32>,        // proxy_n
      d_blur_b: CudaSlice<f32>,        // proxy_n
      d_rgb_out: CudaSlice<f32>,       // n × 3
  }

  /// Pre-allocated device buffers for a specific calibration shape.
  ///
  /// Keyed by (n_zones, lut_size). Dropped and reallocated when calibration shape changes.
  #[cfg(feature = "cuda")]
  struct CalibrationBuffers {
      n_zones: usize,
      lut_size: usize,
      d_luts: CudaSlice<f32>,       // n_zones × lut_size³ × 3
      d_boundaries: CudaSlice<f32>, // n_zones + 1
      d_h_offsets: CudaSlice<f32>,  // [f32; 6]
      d_s_ratios: CudaSlice<f32>,   // [f32; 6]
      d_v_offsets: CudaSlice<f32>,  // [f32; 6]
      d_weights: CudaSlice<f32>,    // [f32; 6]
  }
  ```

- [ ] **Step 2: Add the two allocation helper functions after the structs**

  ```rust
  /// Allocate all 8 resolution-dependent device buffers for the given frame size.
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

  /// Allocate all 6 calibration-dependent device buffers for the given LUT shape.
  #[cfg(feature = "cuda")]
  fn alloc_calibration_buffers(
      dev: &Arc<CudaDevice>,
      n_zones: usize,
      lut_size: usize,
  ) -> Result<CalibrationBuffers, GpuError> {
      let lut_n = n_zones * lut_size * lut_size * lut_size * 3;
      let boundary_n = (n_zones + 1).max(1);
      Ok(CalibrationBuffers {
          n_zones,
          lut_size,
          d_luts: dev.alloc_zeros(lut_n).map_err(map_cudarc_error)?,
          d_boundaries: dev.alloc_zeros(boundary_n).map_err(map_cudarc_error)?,
          d_h_offsets: dev.alloc_zeros(6).map_err(map_cudarc_error)?,
          d_s_ratios: dev.alloc_zeros(6).map_err(map_cudarc_error)?,
          d_v_offsets: dev.alloc_zeros(6).map_err(map_cudarc_error)?,
          d_weights: dev.alloc_zeros(6).map_err(map_cudarc_error)?,
      })
  }
  ```

- [ ] **Step 3: Verify it compiles (structs and helpers only — `CudaGrader` not yet changed)**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo check -p dorea-gpu 2>&1 | tail -20
```

Expected: no errors. If errors appear, fix them before continuing.

- [ ] **Step 4: Commit**

```bash
git add crates/dorea-gpu/src/cuda/mod.rs
git commit -m "feat(dorea-gpu): add ResolutionBuffers and CalibrationBuffers structs with alloc helpers"
```

---

### Task 3: Add `res_bufs` and `cal_bufs` fields to `CudaGrader`

**Files:**
- Modify: `crates/dorea-gpu/src/cuda/mod.rs`

- [ ] **Step 1: Add `use std::cell::RefCell;` to the existing `cfg(feature = "cuda")` use block at the top**

  The current imports (lines 11–19) look like:
  ```rust
  #[cfg(feature = "cuda")]
  use std::sync::Arc;
  #[cfg(feature = "cuda")]
  use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
  ...
  ```

  Add after the `use std::sync::Arc;` line:
  ```rust
  #[cfg(feature = "cuda")]
  use std::cell::RefCell;
  ```

- [ ] **Step 2: Update the `CudaGrader` struct definition**

  Replace the current struct (lines ~42–45):
  ```rust
  #[cfg(feature = "cuda")]
  pub struct CudaGrader {
      device: Arc<CudaDevice>,
      _not_send: std::marker::PhantomData<*const ()>,
  }
  ```

  With:
  ```rust
  #[cfg(feature = "cuda")]
  pub struct CudaGrader {
      device: Arc<CudaDevice>,
      /// Pre-allocated device buffers for the current frame resolution.
      /// `None` until the first call; reallocated if resolution changes.
      res_bufs: RefCell<Option<ResolutionBuffers>>,
      /// Pre-allocated device buffers for the current calibration shape.
      /// `None` until the first call; reallocated if calibration shape changes.
      cal_bufs: RefCell<Option<CalibrationBuffers>>,
      _not_send: std::marker::PhantomData<*const ()>,
  }
  ```

- [ ] **Step 3: Update `CudaGrader::new()` to initialise both fields as `None`**

  In `CudaGrader::new()`, replace the `Ok(Self { ... })` at the end:

  Before:
  ```rust
  Ok(Self {
      device,
      _not_send: std::marker::PhantomData,
  })
  ```

  After:
  ```rust
  Ok(Self {
      device,
      res_bufs: RefCell::new(None),
      cal_bufs: RefCell::new(None),
      _not_send: std::marker::PhantomData,
  })
  ```

- [ ] **Step 4: Verify it compiles**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo check -p dorea-gpu 2>&1 | tail -20
```

Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add crates/dorea-gpu/src/cuda/mod.rs
git commit -m "feat(dorea-gpu): add res_bufs/cal_bufs RefCell fields to CudaGrader"
```

---

### Task 4: Refactor `grade_frame_cuda` to use cached buffers

**Files:**
- Modify: `crates/dorea-gpu/src/cuda/mod.rs` — replace `grade_frame_cuda` body

This is the core change. The method signature is unchanged. The entire body is replaced with the version below.

- [ ] **Step 1: Replace the entire `grade_frame_cuda` body**

  The current body spans lines ~103–391. Replace everything between the opening `{` and closing `}` of the method with:

```rust
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

        // --- u8 -> f32 (host side, no allocation cost) ---
        let rgb_f32: Vec<f32> = pixels.iter().map(|&p| p as f32 / 255.0).collect();

        // =====================================================================
        // ENSURE RESOLUTION BUFFERS  (reallocate only on resolution change)
        // =====================================================================
        {
            let mut slot = self.res_bufs.borrow_mut();
            let needs = slot.as_ref().map_or(true, |b| b.width != width || b.height != height);
            if needs {
                *slot = Some(alloc_resolution_buffers(dev, width, height)?);
            }
        }

        // =====================================================================
        // ENSURE CALIBRATION BUFFERS  (reallocate only on shape change)
        // =====================================================================
        let depth_luts = &calibration.depth_luts;
        let n_zones = depth_luts.n_zones();
        let lut_size = if n_zones > 0 { depth_luts.luts[0].size } else { 33 };
        {
            let mut slot = self.cal_bufs.borrow_mut();
            let needs = slot.as_ref().map_or(true, |b| b.n_zones != n_zones || b.lut_size != lut_size);
            if needs {
                *slot = Some(alloc_calibration_buffers(dev, n_zones, lut_size)?);
            }
        }

        // =====================================================================
        // FILL INPUT BUFFERS  (htod_sync_copy_into: copy only, no cudaMalloc)
        // =====================================================================
        {
            let mut res = self.res_bufs.borrow_mut();
            let res = res.as_mut().unwrap();
            dev.htod_sync_copy_into(&rgb_f32, &mut res.d_rgb_in)
                .map_err(map_cudarc_error)?;
            dev.htod_sync_copy_into(depth, &mut res.d_depth)
                .map_err(map_cudarc_error)?;
        }

        let luts_flat: Vec<f32> = depth_luts
            .luts
            .iter()
            .flat_map(|lg| lg.data.iter().copied())
            .collect();

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

        {
            let mut cal = self.cal_bufs.borrow_mut();
            let cal = cal.as_mut().unwrap();
            dev.htod_sync_copy_into(&luts_flat, &mut cal.d_luts)
                .map_err(map_cudarc_error)?;
            dev.htod_sync_copy_into(&depth_luts.zone_boundaries, &mut cal.d_boundaries)
                .map_err(map_cudarc_error)?;
            dev.htod_sync_copy_into(&h_offsets, &mut cal.d_h_offsets)
                .map_err(map_cudarc_error)?;
            dev.htod_sync_copy_into(&s_ratios, &mut cal.d_s_ratios)
                .map_err(map_cudarc_error)?;
            dev.htod_sync_copy_into(&v_offsets, &mut cal.d_v_offsets)
                .map_err(map_cudarc_error)?;
            dev.htod_sync_copy_into(&weights, &mut cal.d_weights)
                .map_err(map_cudarc_error)?;
        }

        // =====================================================================
        // KERNEL STAGES  (all reads from pre-allocated buffers)
        // =====================================================================

        // --- LUT APPLY ---
        {
            let res = self.res_bufs.borrow();
            let res = res.as_ref().unwrap();
            let cal = self.cal_bufs.borrow();
            let cal = cal.as_ref().unwrap();

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
                        &res.d_rgb_in,
                        &res.d_depth,
                        &cal.d_luts,
                        &cal.d_boundaries,
                        &res.d_rgb_after_lut,
                        n as i32,
                        lut_size as i32,
                        n_zones as i32,
                    ),
                )
            }
            .map_err(map_cudarc_error)?;
        }

        // --- HSL CORRECT ---
        {
            let res = self.res_bufs.borrow();
            let res = res.as_ref().unwrap();
            let cal = self.cal_bufs.borrow();
            let cal = cal.as_ref().unwrap();

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
                        &res.d_rgb_after_lut,
                        &res.d_rgb_after_hsl,
                        &cal.d_h_offsets,
                        &cal.d_s_ratios,
                        &cal.d_v_offsets,
                        &cal.d_weights,
                        n as i32,
                    ),
                )
            }
            .map_err(map_cudarc_error)?;
        }

        // --- CLARITY (proxy-resolution box blur) ---
        let mean_d = depth.iter().sum::<f32>() / depth.len().max(1) as f32;
        let clarity_amount = (0.2 + 0.25 * mean_d) * params.contrast;

        {
            let res = self.res_bufs.borrow();
            let res = res.as_ref().unwrap();
            let proxy_w = res.proxy_w;
            let proxy_h = res.proxy_h;

            // Sub-kernel A: extract proxy L from full-res RGB
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
                            &res.d_rgb_after_hsl,
                            &res.d_proxy_l,
                            width as i32,
                            height as i32,
                            proxy_w as i32,
                            proxy_h as i32,
                        ),
                    )
                }
                .map_err(map_cudarc_error)?;
            }

            // Sub-kernel B: 3-pass box blur (rows then cols each pass)
            // Pass 1: rows(proxy_l → blur_a), cols(blur_a → blur_b)
            // Pass 2: rows(blur_b → blur_a), cols(blur_a → blur_b)
            // Pass 3: rows(blur_b → blur_a), cols(blur_a → blur_b)
            // Result in blur_b after 3 passes.
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
                {
                    let func = dev
                        .get_func("clarity", "clarity_box_blur_rows")
                        .ok_or_else(|| {
                            GpuError::ModuleLoad("clarity_box_blur_rows not found".into())
                        })?;
                    let src: &CudaSlice<f32> = if pass == 0 { &res.d_proxy_l } else { &res.d_blur_b };
                    unsafe {
                        func.launch(
                            blur_rows_cfg,
                            (src, &res.d_blur_a, proxy_w as i32, proxy_h as i32, BLUR_RADIUS),
                        )
                    }
                    .map_err(map_cudarc_error)?;
                }
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
                                &res.d_blur_a,
                                &res.d_blur_b,
                                proxy_w as i32,
                                proxy_h as i32,
                                BLUR_RADIUS,
                            ),
                        )
                    }
                    .map_err(map_cudarc_error)?;
                }
            }

            // Sub-kernel C: apply clarity at full resolution
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

                unsafe {
                    func.launch(
                        cfg,
                        (
                            &res.d_rgb_after_hsl,
                            &res.d_rgb_out,
                            &res.d_blur_b,
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
        }

        // =====================================================================
        // DOWNLOAD RESULT
        // =====================================================================
        let res_ref = self.res_bufs.borrow();
        let result = dev
            .dtoh_sync_copy(&res_ref.as_ref().unwrap().d_rgb_out)
            .map_err(map_cudarc_error)?;
        Ok(result)
    }
```

- [ ] **Step 2: Verify it compiles**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo check -p dorea-gpu 2>&1 | tail -30
```

Expected: no errors. Common issues to watch for:
- `borrow_mut` and `borrow` overlap — all `borrow_mut` blocks must be closed (drop their `RefMut`) before any `borrow` block for the same `RefCell`.
- `htod_sync_copy_into` requires `T: DeviceRepr` — `f32` satisfies this.

- [ ] **Step 3: Run the regression tests**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-gpu 2>&1 | tail -30
```

Expected: `same_resolution_twice_is_deterministic`, `resolution_switch_preserves_output`, `calibration_shape_change_does_not_error` all PASS.
All other existing tests must also pass.

- [ ] **Step 4: Commit**

```bash
git add crates/dorea-gpu/src/cuda/mod.rs
git commit -m "perf(dorea-gpu): pre-allocate device buffers in CudaGrader, zero cudaMalloc per frame

ResolutionBuffers and CalibrationBuffers hold all device slices.
grade_frame_cuda now uses htod_sync_copy_into (copy only, no allocation)
in steady state. Buffers are reallocated only when resolution or
calibration shape changes."
```

---

### Task 5: Run the benchmark and record the improvement

**Files:**
- No code changes — benchmark run only.

- [ ] **Step 1: Run the Criterion benchmark**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo bench -p dorea-gpu 2>&1 | tee /tmp/bench_after_prealloc.txt
```

Expected: `grading/with_grader/1080p` time should drop from ~83 ms to ~5–15 ms.
The `grading/cpu/1080p` path should be unchanged (CPU path is not affected).

- [ ] **Step 2: Check Criterion comparison output**

  Criterion will print something like:
  ```
  grading/with_grader/1080p
                          time:   [6.1 ms 6.3 ms 6.5 ms]
                  change: [-92.3% -91.8% -91.1%] (p = 0.00 < 0.05)
                  Performance has improved.
  ```

- [ ] **Step 3: Record the result in corvia**

  Use the `corvia_write` MCP tool with:
  - `scope_id`: `"dorea"`
  - `source_origin`: `"repo:dorea"`
  - `content_role`: `"finding"`
  - Title: `"CudaGrader buffer pre-allocation benchmark result"`
  - Body: include the before (83 ms/frame) and after times from Criterion output.

---

## Self-Review

**Spec coverage check:**
- ✅ `ResolutionBuffers` struct → Task 2
- ✅ `CalibrationBuffers` struct → Task 2
- ✅ `RefCell<Option<...>>` fields on `CudaGrader` → Task 3
- ✅ `None` init in `CudaGrader::new()` → Task 3
- ✅ Resolution-change reallocation → Task 4
- ✅ Calibration-shape-change reallocation → Task 4
- ✅ `htod_sync_copy_into` replacing `htod_sync_copy` → Task 4
- ✅ Public API unchanged (signature unchanged) → Task 4
- ✅ Resolution-switch regression test → Task 1
- ✅ Calibration-switch regression test → Task 1
- ✅ Benchmark validation → Task 5

**Placeholder scan:** None found.

**Type consistency:** `CudaSlice<f32>` used throughout. `alloc_resolution_buffers` and `alloc_calibration_buffers` return the exact structs used in Tasks 3 and 4. `htod_sync_copy_into` signature `(&[T], &mut CudaSlice<T>)` matches usage.
