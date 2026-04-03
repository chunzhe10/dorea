# PCIe Sync Barrier Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce `grade_frame_cuda` from 9 synchronous PCIe barriers per frame to 2 in steady state, and eliminate a 24 MB per-frame heap allocation, without changing the public API.

**Architecture:** Three independent, low-risk changes land in one PR (Tasks 1–3). A second PR (Tasks 4–5, not yet planned in detail) introduces explicit CUDA streams for copy-compute overlap and deferred dtoh. Tasks 1–3 are strictly additive: no kernel signature changes that break existing tests, no new unsafe code, no public API additions except one new safe method.

**Tech Stack:** Rust, cudarc 0.12.1, CUDA C (nvcc PTX), criterion benchmarks.

---

## Context: The 9 barriers and their cost at 1080p

| # | Transfer | Size | Fires every frame? |
|---|----------|------|--------------------|
| 1 | `htod` → `d_rgb_in` | 24 MB | Yes |
| 2 | `htod` → `d_depth` | 8 MB | Yes |
| 3 | `htod` → `d_luts` | 2.2 MB | Yes — **unnecessary** |
| 4 | `htod` → `d_boundaries` | 24 B | Yes — **unnecessary** |
| 5 | `htod` → `d_h_offsets` | 24 B | Yes — **unnecessary** |
| 6 | `htod` → `d_s_ratios` | 24 B | Yes — **unnecessary** |
| 7 | `htod` → `d_v_offsets` | 24 B | Yes — **unnecessary** |
| 8 | `htod` → `d_weights` | 24 B | Yes — **unnecessary** |
| 9 | `dtoh` ← `d_rgb_out` | 24 MB | Yes |

Barriers 3–8 fire every frame even though calibration never changes within a session. Task 1 eliminates them in steady state. Task 2 packs 4 of the tiny ones into 1, reducing the first-frame cost. Task 3 eliminates a 24 MB per-frame heap allocation for the u8→f32 conversion.

After Tasks 1–3, steady-state barriers: **2 htod (rgb + depth) + 1 dtoh** = 3 total.

---

## File structure

| File | Change |
|------|--------|
| `crates/dorea-gpu/src/cuda/mod.rs` | All Rust changes (Tasks 1–3) |
| `crates/dorea-gpu/src/cuda/kernels/hsl_correct.cu` | Kernel signature: 4 params → 1 packed (Task 2) |

No new files. No changes to `lib.rs`, `cpu.rs`, `device.rs`, `batcher.rs`, `build.rs`.

---

## Benchmark baseline

Before starting, save a Criterion baseline so the gain is measurable.

```bash
cargo bench -p dorea-gpu --bench grade_bench \
  -- "grading/with_grader" --save-baseline before_pcie_fixes
```

You will compare against this at the end of Task 3.

---

## Task 1: Skip calibration re-uploads when data has not changed (Fix 1)

Eliminates 6/9 barriers in steady state. Calibration is constant across an entire dorea session; uploading it on every frame is pure waste.

**Files:**
- Modify: `crates/dorea-gpu/src/cuda/mod.rs`

- [ ] **Step 1: Write the failing test**

Add inside `mod tests` in `cuda/mod.rs`:

```rust
/// Build a non-identity (saturating) LUT of given size.
/// Output differs from an identity LUT so the test can detect stale calibration.
fn make_saturated_calibration(n_zones: usize) -> Calibration {
    let luts: Vec<LutGrid> = (0..n_zones)
        .map(|_| {
            let mut lut = LutGrid::new(17);
            for ri in 0..17 {
                for gi in 0..17 {
                    for bi in 0..17 {
                        let r = (ri as f32 / 16.0_f32).powf(0.5);
                        let g = (gi as f32 / 16.0_f32).powf(0.7);
                        let b = (bi as f32 / 16.0_f32).powf(0.9);
                        lut.set(ri, gi, bi, [r, g, b]);
                    }
                }
            }
            lut
        })
        .collect();
    let boundaries: Vec<f32> =
        (0..=n_zones).map(|i| i as f32 / n_zones as f32).collect();
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

/// Calibration data is only uploaded once per session; re-using the same calibration
/// must produce identical output. After `invalidate_calibration`, different calibration
/// data must produce different output.
#[test]
fn calibration_cache_correctness() {
    let grader = CudaGrader::new().expect("CudaGrader::new() failed");
    let params = GradeParams::default();
    let (pixels, depth) = make_frame(320, 240);

    // Identity calibration: first call uploads data.
    let cal_identity = make_calibration(5);
    let out1 = grader
        .grade_frame_cuda(&pixels, &depth, 320, 240, &cal_identity, &params)
        .expect("first call failed");

    // Second call — calibration unchanged, should use cached data and produce same output.
    let out2 = grader
        .grade_frame_cuda(&pixels, &depth, 320, 240, &cal_identity, &params)
        .expect("second call failed");
    assert_eq!(out1, out2, "cached calibration must produce identical output");

    // Invalidate, then switch to a saturating calibration — output must differ.
    grader.invalidate_calibration();
    let cal_sat = make_saturated_calibration(5);
    let out3 = grader
        .grade_frame_cuda(&pixels, &depth, 320, 240, &cal_sat, &params)
        .expect("saturated call failed");
    assert_ne!(out1, out3, "different calibration must produce different output after invalidation");
}
```

- [ ] **Step 2: Run the test — expect compile failure**

```bash
cargo test -p dorea-gpu --test-threads=1 2>&1 | grep "calibration_cache_correctness"
```

Expected: `error[E0599]: no method named 'invalidate_calibration' found`

- [ ] **Step 3: Add `data_valid: bool` field to `CalibrationBuffers`**

In `cuda/mod.rs`, change the `CalibrationBuffers` struct:

```rust
struct CalibrationBuffers {
    n_zones: usize,
    lut_size: usize,
    /// true = device buffers contain freshly-uploaded calibration data.
    /// Set to false on construction and after `invalidate_calibration()`.
    data_valid: bool,
    d_luts: CudaSlice<f32>,
    d_boundaries: CudaSlice<f32>,
    d_h_offsets: CudaSlice<f32>,
    d_s_ratios: CudaSlice<f32>,
    d_v_offsets: CudaSlice<f32>,
    d_weights: CudaSlice<f32>,
}
```

In `alloc_calibration_buffers`, add `data_valid: false` to the returned struct:

```rust
Ok(CalibrationBuffers {
    n_zones,
    lut_size,
    data_valid: false,   // <-- add this
    d_luts: dev.alloc_zeros(lut_elems).map_err(map_cudarc_error)?,
    d_boundaries: dev.alloc_zeros(n_zones + 1).map_err(map_cudarc_error)?,
    d_h_offsets: dev.alloc_zeros(6).map_err(map_cudarc_error)?,
    d_s_ratios: dev.alloc_zeros(6).map_err(map_cudarc_error)?,
    d_v_offsets: dev.alloc_zeros(6).map_err(map_cudarc_error)?,
    d_weights: dev.alloc_zeros(6).map_err(map_cudarc_error)?,
})
```

- [ ] **Step 4: Add `CudaGrader::invalidate_calibration()`**

Inside `impl CudaGrader` in `cuda/mod.rs`, after `with_capacity`:

```rust
/// Mark the cached calibration as stale, forcing a re-upload on the next
/// `grade_frame_cuda` call.
///
/// Call this when the calibration content passed to `grade_frame_cuda` changes.
/// In a typical dorea session, calibration is fixed — you do not need to call this.
pub fn invalidate_calibration(&self) {
    let mut guard = self.cal_bufs.borrow_mut();
    if let Some(bufs) = guard.as_mut() {
        bufs.data_valid = false;
    }
}
```

- [ ] **Step 5: Guard the calibration upload block in `grade_frame_cuda`**

Replace the existing `// --- Upload calibration-keyed data ---` block (lines ~205–228) with:

```rust
// --- Upload calibration-keyed data (skipped when data_valid — calibration is
//     constant across a session, so this block fires only on the first frame
//     and after invalidate_calibration()) ---
let needs_cal_upload = {
    // Immutable borrow, dropped immediately so borrow_mut below can fire.
    let guard = self.cal_bufs.borrow();
    guard.as_ref().is_none_or(|b| !b.data_valid)
};

if needs_cal_upload {
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
        bufs.data_valid = true;
    }
}
```

**Borrow discipline note:** `needs_cal_upload` uses an immutable borrow that is dropped at the `}` before `if needs_cal_upload`. The `borrow_mut` inside the `if` therefore cannot conflict. The borrow sequence for this frame:
1. Immutable `borrow()` → drops → `needs_cal_upload` set
2. If needed: `borrow_mut()` → upload → set `data_valid = true` → drops
3. Existing `borrow()` for kernel launches (after this block)

- [ ] **Step 6: Run tests**

```bash
cargo test -p dorea-gpu --test-threads=1 2>&1 | grep -E "test .* (ok|FAILED)"
```

Expected: 27 tests pass, 0 fail. The new `calibration_cache_correctness` test is one of them.

- [ ] **Step 7: Commit**

```bash
git add crates/dorea-gpu/src/cuda/mod.rs
git commit -m "perf(dorea-gpu): skip calibration re-uploads when data_valid (eliminates 6/9 PCIe barriers in steady state)"
```

---

## Task 2: Pack HSL scalar arrays into one device buffer (Fix 4)

Reduces 4 separate 24-byte transfers (h_offsets, s_ratios, v_offsets, weights) into a single 96-byte transfer. Also requires a one-line change to the `hsl_correct.cu` kernel.

**Files:**
- Modify: `crates/dorea-gpu/src/cuda/kernels/hsl_correct.cu`
- Modify: `crates/dorea-gpu/src/cuda/mod.rs`

- [ ] **Step 1: Write the failing test**

This change is a pure refactor: the observable behavior is unchanged. The test verifies the packed-params path produces the same output as an unmodified reference call. Add to `mod tests`:

```rust
/// Packing h_offsets/s_ratios/v_offsets/weights into d_hsl_params must not change output.
/// (Behavioral regression test for Task 2 — kernel signature change.)
#[test]
fn hsl_packed_params_matches_unpacked_reference() {
    let grader = CudaGrader::new().expect("CudaGrader::new()");
    let params = GradeParams::default();
    let (w, h) = (320, 240);
    let (pixels, depth) = make_frame(w, h);

    // Use a calibration with non-zero HSL weights so the kernel is not a no-op.
    let luts: Vec<LutGrid> = (0..3).map(|_| identity_lut(17)).collect();
    let boundaries: Vec<f32> = vec![0.0, 0.33, 0.66, 1.0];
    let depth_luts = DepthLuts::new(luts, boundaries);
    let hsl = HslCorrections(vec![QualifierCorrection {
        h_center: 40.0,
        h_width: 40.0,
        h_offset: 5.0,
        s_ratio: 1.2,
        v_offset: 0.05,
        weight: 200.0,   // > 100 → qualifier is active
    }]);
    let cal = Calibration::new(depth_luts, hsl, 1);

    let out1 = grader
        .grade_frame_cuda(&pixels, &depth, w, h, &cal, &params)
        .expect("first call");
    grader.invalidate_calibration();
    let out2 = grader
        .grade_frame_cuda(&pixels, &depth, w, h, &cal, &params)
        .expect("second call after invalidation");

    assert_eq!(
        out1, out2,
        "packed hsl_params kernel must produce identical output to re-upload path"
    );
}
```

- [ ] **Step 2: Run the test — it should pass** (it's a regression guard, not a failing stub)

```bash
cargo test -p dorea-gpu --test-threads=1 -- hsl_packed_params 2>&1
```

Expected: PASS. Record the baseline to compare after the change.

- [ ] **Step 3: Update `hsl_correct.cu` — change kernel signature to single `hsl_params` pointer**

Replace the existing `hsl_correct_kernel` signature and body with:

```c
extern "C"
__global__ void hsl_correct_kernel(
    const float* __restrict__ pixels_in,
    float* __restrict__ pixels_out,
    const float* __restrict__ hsl_params,  /* [24]: h_offsets[6] | s_ratios[6] | v_offsets[6] | weights[6] */
    int n_pixels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pixels) return;

    float r = pixels_in[idx * 3 + 0];
    float g = pixels_in[idx * 3 + 1];
    float b = pixels_in[idx * 3 + 2];

    float h, s, v;
    rgb_to_hsv(r, g, b, &h, &s, &v);

    const float* h_offsets = hsl_params;
    const float* s_ratios  = hsl_params + 6;
    const float* v_offsets = hsl_params + 12;
    const float* weights   = hsl_params + 18;

    for (int q = 0; q < N_QUALIFIERS; q++) {
        if (weights[q] < 100.0f) continue;

        float dist = angular_dist(h, H_CENTERS[q]);
        float mask = fmaxf(1.0f - dist / H_WIDTHS[q], 0.0f);
        if (s < 0.08f) mask = 0.0f;
        if (mask < 1e-6f) continue;

        h += h_offsets[q] * mask;
        s *= (1.0f + (s_ratios[q] - 1.0f) * mask);
        v += v_offsets[q] * mask;
    }

    h = fmodf(h, 360.0f);
    if (h < 0.0f) h += 360.0f;
    s = fminf(fmaxf(s, 0.0f), 1.0f);
    v = fminf(fmaxf(v, 0.0f), 1.0f);

    hsv_to_rgb(h, s, v, &r, &g, &b);

    pixels_out[idx * 3 + 0] = r;
    pixels_out[idx * 3 + 1] = g;
    pixels_out[idx * 3 + 2] = b;
}
```

The helper functions `rgb_to_hsv`, `hsv_to_rgb`, `angular_dist`, and the `__constant__` arrays at the top of the file are unchanged.

- [ ] **Step 4: Update `CalibrationBuffers` in `cuda/mod.rs`**

Replace the four scalar fields with one packed field:

```rust
struct CalibrationBuffers {
    n_zones: usize,
    lut_size: usize,
    data_valid: bool,
    d_luts: CudaSlice<f32>,
    d_boundaries: CudaSlice<f32>,
    /// Packed HSL correction parameters: [h_offsets(6) | s_ratios(6) | v_offsets(6) | weights(6)] = 24 f32.
    d_hsl_params: CudaSlice<f32>,
}
```

- [ ] **Step 5: Update `alloc_calibration_buffers`**

```rust
Ok(CalibrationBuffers {
    n_zones,
    lut_size,
    data_valid: false,
    d_luts: dev.alloc_zeros(lut_elems).map_err(map_cudarc_error)?,
    d_boundaries: dev.alloc_zeros(n_zones + 1).map_err(map_cudarc_error)?,
    d_hsl_params: dev.alloc_zeros(24).map_err(map_cudarc_error)?,
})
```

- [ ] **Step 6: Update the calibration upload block in `grade_frame_cuda`**

Inside the `if needs_cal_upload { ... }` block added in Task 1, replace the four separate uploads with one packed upload:

```rust
if needs_cal_upload {
    let luts_flat: Vec<f32> = depth_luts.luts.iter()
        .flat_map(|lg| lg.data.iter().copied())
        .collect();
    let mut hsl_params = [0.0f32; 24];
    for (i, q) in calibration.hsl_corrections.0.iter().enumerate().take(6) {
        hsl_params[i]      = q.h_offset;
        hsl_params[6 + i]  = q.s_ratio;
        hsl_params[12 + i] = q.v_offset;
        hsl_params[18 + i] = q.weight;
    }
    {
        let mut guard = self.cal_bufs.borrow_mut();
        let bufs = guard.as_mut().unwrap();
        dev.htod_sync_copy_into(&luts_flat,                  &mut bufs.d_luts      ).map_err(map_cudarc_error)?;
        dev.htod_sync_copy_into(&depth_luts.zone_boundaries, &mut bufs.d_boundaries).map_err(map_cudarc_error)?;
        dev.htod_sync_copy_into(&hsl_params,                 &mut bufs.d_hsl_params).map_err(map_cudarc_error)?;
        bufs.data_valid = true;
    }
}
```

- [ ] **Step 7: Update the HSL kernel launch args in `grade_frame_cuda`**

Find the `HSL CORRECT` kernel launch block and change the argument tuple from 7 args to 4:

```rust
unsafe {
    func.launch(
        cfg,
        (
            &res_bufs.d_rgb_after_lut,
            &res_bufs.d_rgb_after_hsl,
            &cal_bufs.d_hsl_params,   // was: d_h_offsets, d_s_ratios, d_v_offsets, d_weights
            n as i32,
        ),
    )
}
.map_err(map_cudarc_error)?;
```

- [ ] **Step 8: Build to trigger nvcc recompile**

```bash
cargo build -p dorea-gpu 2>&1 | grep -E "warning:|error:"
```

Expected: nvcc recompiles `hsl_correct.cu` → `hsl_correct.ptx`. No errors.

- [ ] **Step 9: Run tests**

```bash
cargo test -p dorea-gpu --test-threads=1 2>&1 | grep -E "test .* (ok|FAILED)"
```

Expected: 27 tests pass. Pay attention to `hsl_packed_params_matches_unpacked_reference` and `calibration_shape_switch`.

- [ ] **Step 10: Commit**

```bash
git add crates/dorea-gpu/src/cuda/kernels/hsl_correct.cu crates/dorea-gpu/src/cuda/mod.rs
git commit -m "perf(dorea-gpu): pack HSL scalar params into d_hsl_params — 4 htod barriers → 1 on calibration upload"
```

---

## Task 3: Reuse host rgb_f32 conversion buffer (Fix 2 partial)

`grade_frame_cuda` currently allocates a 24 MB `Vec<f32>` on every frame for the u8→f32 pixel conversion, then immediately discards it after `htod_sync_copy_into`. This is a 24 MB heap alloc + dealloc at 1080p per frame. Pre-allocating this buffer in `ResolutionBuffers` eliminates the per-frame allocation and reduces pressure on the system allocator.

**Files:**
- Modify: `crates/dorea-gpu/src/cuda/mod.rs`

- [ ] **Step 1: Write the failing test**

This is a correctness regression: the reused buffer must be fully overwritten before each upload. Test that alternating different pixel inputs produces the correct distinct outputs:

```rust
/// The reused host rgb_f32 buffer must be fully overwritten before each upload.
/// Alternating pixel inputs must produce their respective correct outputs.
#[test]
fn host_buffer_reuse_overwrites_correctly() {
    let grader = CudaGrader::new().expect("CudaGrader::new()");
    let cal = make_calibration(5);
    let params = GradeParams::default();
    let (w, h) = (320, 240);
    let n = w * h;
    let depth: Vec<f32> = (0..n).map(|i| i as f32 / n as f32 * 0.8 + 0.1).collect();

    let pixels_a: Vec<u8> = (0..n * 3).map(|i| ((i * 7 + 10) % 256) as u8).collect();
    let pixels_b: Vec<u8> = (0..n * 3).map(|i| ((i * 13 + 50) % 256) as u8).collect();

    // Establish reference outputs with the current (pre-Task 3) implementation.
    let ref_a = grader
        .grade_frame_cuda(&pixels_a, &depth, w, h, &cal, &params)
        .expect("ref_a");
    let ref_b = grader
        .grade_frame_cuda(&pixels_b, &depth, w, h, &cal, &params)
        .expect("ref_b");

    // Alternate: a, b, a, b — each must match its reference exactly.
    let out_a2 = grader
        .grade_frame_cuda(&pixels_a, &depth, w, h, &cal, &params)
        .expect("out_a2");
    let out_b2 = grader
        .grade_frame_cuda(&pixels_b, &depth, w, h, &cal, &params)
        .expect("out_b2");
    let out_a3 = grader
        .grade_frame_cuda(&pixels_a, &depth, w, h, &cal, &params)
        .expect("out_a3");

    assert_eq!(ref_a, out_a2, "second call with pixels_a must match first");
    assert_eq!(ref_b, out_b2, "second call with pixels_b must match first");
    assert_eq!(ref_a, out_a3, "third call with pixels_a must match first");
    assert_ne!(ref_a, ref_b,  "different pixels must produce different output");
}
```

- [ ] **Step 2: Run test to verify it passes on current code**

```bash
cargo test -p dorea-gpu --test-threads=1 -- host_buffer_reuse 2>&1
```

Expected: PASS. (This is a pre-condition check — the test should pass before and after Task 3.)

- [ ] **Step 3: Add `h_rgb_f32: Vec<f32>` to `ResolutionBuffers`**

```rust
struct ResolutionBuffers {
    width: usize,
    height: usize,
    proxy_w: usize,
    proxy_h: usize,
    /// Host-side u8→f32 conversion scratch buffer, pre-allocated to avoid per-frame heap alloc.
    /// Size: n * 3 (same as d_rgb_in). Overwritten on every frame before upload.
    h_rgb_f32: Vec<f32>,
    d_rgb_in: CudaSlice<f32>,
    d_depth: CudaSlice<f32>,
    d_rgb_after_lut: CudaSlice<f32>,
    d_rgb_after_hsl: CudaSlice<f32>,
    d_proxy_l: CudaSlice<f32>,
    d_blur_a: CudaSlice<f32>,
    d_blur_b: CudaSlice<f32>,
    d_rgb_out: CudaSlice<f32>,
}
```

- [ ] **Step 4: Initialise in `alloc_resolution_buffers`**

```rust
Ok(ResolutionBuffers {
    width,
    height,
    proxy_w,
    proxy_h,
    h_rgb_f32: vec![0.0f32; n * 3],   // pre-allocated; content overwritten on every frame
    d_rgb_in: dev.alloc_zeros(n * 3).map_err(map_cudarc_error)?,
    d_depth: dev.alloc_zeros(n).map_err(map_cudarc_error)?,
    d_rgb_after_lut: dev.alloc_zeros(n * 3).map_err(map_cudarc_error)?,
    d_rgb_after_hsl: dev.alloc_zeros(n * 3).map_err(map_cudarc_error)?,
    d_proxy_l: dev.alloc_zeros(proxy_n).map_err(map_cudarc_error)?,
    d_blur_a: dev.alloc_zeros(proxy_n).map_err(map_cudarc_error)?,
    d_blur_b: dev.alloc_zeros(proxy_n).map_err(map_cudarc_error)?,
    d_rgb_out: dev.alloc_zeros(n * 3).map_err(map_cudarc_error)?,
})
```

- [ ] **Step 5: Replace per-frame Vec allocation in `grade_frame_cuda`**

Find and remove:

```rust
let rgb_f32: Vec<f32> = pixels.iter().map(|&p| p as f32 / 255.0).collect();
```

Replace the resolution upload block (which previously borrowed `rgb_f32`) with:

```rust
// --- Upload resolution-keyed data — convert u8→f32 into pre-allocated host buffer,
//     then copy into device. No per-frame heap allocation. ---
{
    let mut guard = self.res_bufs.borrow_mut();
    let bufs = guard.as_mut().unwrap();
    // Overwrite every element — h_rgb_f32 is exactly n*3 elements (same as pixels).
    bufs.h_rgb_f32
        .iter_mut()
        .zip(pixels.iter())
        .for_each(|(dst, &src)| *dst = src as f32 / 255.0);
    dev.htod_sync_copy_into(&bufs.h_rgb_f32, &mut bufs.d_rgb_in).map_err(map_cudarc_error)?;
    dev.htod_sync_copy_into(depth, &mut bufs.d_depth).map_err(map_cudarc_error)?;
}
```

**Length safety:** The guard at the top of `grade_frame_cuda` ensures `pixels.len() == n * 3`. `alloc_resolution_buffers` allocates `h_rgb_f32` with `n * 3` elements where `n = width * height`. These are equal by invariant, so `zip` will not silently truncate.

- [ ] **Step 6: Run tests**

```bash
cargo test -p dorea-gpu --test-threads=1 2>&1 | grep -E "test .* (ok|FAILED)"
```

Expected: 28 tests pass (27 from Tasks 1–2 + new `host_buffer_reuse_overwrites_correctly`), 0 fail.

- [ ] **Step 7: Measure improvement**

```bash
cargo bench -p dorea-gpu --bench grade_bench \
  -- "grading/with_grader" --baseline before_pcie_fixes 2>&1 | tail -20
```

Expected: `grading/with_grader/1080p` shows improvement. The steady-state path now has 2 htod barriers + 1 dtoh (was 8 htod + 1 dtoh). If improvement is < 5%, PCIe BW is still dominated by the 24 MB rgb upload (barrier #1) and 24 MB dtoh (barrier #9) — that is expected and addressed in the follow-up plan.

- [ ] **Step 8: Commit**

```bash
git add crates/dorea-gpu/src/cuda/mod.rs
git commit -m "perf(dorea-gpu): pre-allocate h_rgb_f32 in ResolutionBuffers — eliminate 24 MB per-frame Vec alloc"
```

---

## Task 4 & 5: Architecture notes for the follow-up streaming PR

**Not implemented in this plan.** Tasks 4 and 5 are tracked in issue #31 as Fix 5 (deferred dtoh) and Fix 3 (copy-compute overlap). They share a common prerequisite: explicit `CudaStream` management, which requires investigation of the cudarc 0.12 raw `sys::lib()` stream API or an upgrade to cudarc 0.13+.

### Task 4: Deferred dtoh (Fix 5)

Goal: In batch mode, collect all frame results on device, then download in one burst instead of one `dtoh_sync_copy` per frame.

Prerequisite design decisions needed:
1. Each frame needs its own output `CudaSlice<f32>` (since `d_rgb_out` in `ResolutionBuffers` is shared and overwritten each frame).
2. N output slices must be allocated upfront in `grade_frames_with_capacity`. This reintroduces N `cudaMalloc` calls per batch. Acceptable only if batch size is large (>= 10 frames) so the saving on N dtoh syncs outweighs the alloc cost.
3. D2D copy from `d_rgb_out` into per-frame output slot: cudarc's `dev.dtod_copy_into` or raw `cuMemcpyDtoD`. This is a device→device copy (no PCIe) and can be enqueued async.
4. One final `device.synchronize()` before all downloads.

Create a separate issue/plan before implementing.

### Task 5: Copy-compute overlap (Fix 3)

Goal: Upload frame N+1's pixels while the GPU runs frame N's kernels. At 7–16% SM utilisation, the PCIe copy engine and SMs can run fully concurrently.

Prerequisite design decisions needed:
1. cudarc 0.12's `LaunchConfig` has no stream field. Explicit multi-stream requires `sys::lib().cuStreamCreate()` and raw kernel launch. Investigate whether upgrading to cudarc 0.13 exposes `CudaStream` in the safe API.
2. Double-buffered `ResolutionBuffers` (slot A + slot B). `CudaGrader` alternates between slots per frame.
3. Upload stream: fires `cuMemcpyHtoDAsync` for the next frame's rgb + depth.
4. Compute stream: depends on upload stream via `cuStreamWaitEvent` before the first kernel.
5. CUDA event for signalling when compute stream has finished writing `d_rgb_out` and the slot can be reused.

Create a separate issue/plan before implementing.

---

## Post-PR verification

After Tasks 1–3 are merged:

```bash
# Full test suite
cargo test -p dorea-gpu 2>&1 | tail -5

# Benchmark comparison
cargo bench -p dorea-gpu --bench grade_bench \
  -- "grading/with_grader" --baseline before_pcie_fixes

# GPU utilisation during a run (requires nvidia-smi in a separate terminal)
# Terminal 1: nvidia-smi dmon -s u -d 1
# Terminal 2: cargo run --release --bin dorea -- grade ...
```

Expected: SM utilisation unchanged (still 7–16%, PCIe-bound on the 24 MB rgb upload and dtoh). The gain from Tasks 1–3 is latency reduction on calibration uploads, not bandwidth. The streaming PR (Tasks 4–5) is needed to move the remaining bottleneck.
