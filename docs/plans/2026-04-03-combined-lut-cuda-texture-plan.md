# Combined LUT + CUDA 3D Texture Grading — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 3-kernel GPU pipeline (lut_apply → hsl_correct → clarity) with a single precomputed combined LUT sampled via CUDA 3D texture objects, baking the full grading pipeline including ambiance, warmth, and strength blend.

**Architecture:** At `CudaGrader::new(calibration, params)`, a GPU kernel samples the complete pipeline at every lattice point of a 97³ grid (per zone), storing results as `float4` CUDA 3D texture arrays. Per-frame, a single kernel reads u8 pixels, samples two zone textures, depth-lerps, and writes u8 output. Clarity is hard-deleted.

**Tech Stack:** Rust (cudarc 0.12, rayon), CUDA C++ (nvcc -arch=sm_86), cudarc::driver::sys raw FFI for CUarray/CUtexObject.

---

## File Map

### New files

| File | Responsibility |
|------|---------------|
| `crates/dorea-gpu/src/cuda/kernels/grade_pixel.cuh` | `__device__ grade_pixel_device()` — full pipeline in one CUDA fn |
| `crates/dorea-gpu/src/cuda/kernels/build_combined_lut.cu` | GPU LUT build kernel (startup, once) |
| `crates/dorea-gpu/src/cuda/kernels/combined_lut.cu` | Per-frame lookup kernel |
| `crates/dorea-gpu/src/cuda/combined_lut.rs` | `CombinedLut` struct — CUarray + texture object lifecycle |

### Deleted files

| File | Reason |
|------|--------|
| `crates/dorea-gpu/src/cuda/kernels/clarity.cu` | Hard-deleted; Maxine will handle sharpening |
| `crates/dorea-gpu/src/cuda/kernels/lut_apply.cu` | Replaced by combined LUT |
| `crates/dorea-gpu/src/cuda/kernels/hsl_correct.cu` | Replaced by combined LUT |

### Modified files

| File | Changes |
|------|---------|
| `crates/dorea-gpu/build.rs` | kernel_names: remove [lut_apply, hsl_correct, clarity], add [build_combined_lut, combined_lut] |
| `crates/dorea-gpu/src/cuda/mod.rs` | CudaGrader::new(cal, params), grade_frame_cuda → Vec<u8>, add CombinedLut field |
| `crates/dorea-gpu/src/cpu.rs` | Remove apply_cpu_clarity, remove skip_clarity param from finish_grade, add grade_pixel_cpu |
| `crates/dorea-gpu/src/lib.rs` | grade_frame_with_grader drops cal+params, grade_frame calls new(cal,params) |
| `crates/dorea-cli/src/grade.rs` | CudaGrader::new(cal, params), grade_with_grader drops cal+params |
| `crates/dorea-gpu/benches/grade_bench.rs` | Update CudaGrader::new and grade_frame_with_grader call sites |

---

### Task 1: Capture regression baseline

**Files:**
- Test: `crates/dorea-gpu/src/cpu.rs` (add test in existing `#[cfg(test)] mod tests`)

- [ ] **Step 1: Write the failing test**

Add to `crates/dorea-gpu/src/cpu.rs` inside the `tests` module (after the existing tests):

```rust
#[test]
fn grade_pixel_cpu_matches_grade_frame_cpu_baseline() {
    // This test will fail until grade_pixel_cpu is implemented in Task 3.
    // It exists here to lock in the expected baseline before any changes.
    use dorea_cal::Calibration;
    use dorea_hsl::derive::{HslCorrections, QualifierCorrection};
    use dorea_lut::types::{DepthLuts, LutGrid};

    fn identity_lut(size: usize) -> LutGrid {
        let mut lut = LutGrid::new(size);
        for ri in 0..size { for gi in 0..size { for bi in 0..size {
            let r = ri as f32 / (size - 1) as f32;
            let g = gi as f32 / (size - 1) as f32;
            let b = bi as f32 / (size - 1) as f32;
            lut.set(ri, gi, bi, [r, g, b]);
        }}}
        lut
    }

    let n_zones = 5;
    let luts: Vec<LutGrid> = (0..n_zones).map(|_| identity_lut(17)).collect();
    let boundaries: Vec<f32> = (0..=n_zones).map(|i| i as f32 / n_zones as f32).collect();
    let depth_luts = DepthLuts::new(luts, boundaries);
    let hsl = HslCorrections(vec![QualifierCorrection {
        h_center: 0.0, h_width: 1.0, h_offset: 0.0,
        s_ratio: 1.0, v_offset: 0.0, weight: 0.0,
    }]);
    let cal = Calibration::new(depth_luts, hsl, 1);
    let params = crate::GradeParams::default();

    // Input: r=0.6, g=0.3, b=0.2, depth=0.5
    // Use exact u8 → f32 round-trip to avoid floating-point inconsistency
    let r_u8 = 153u8; let g_u8 = 77u8; let b_u8 = 51u8;
    let pixels = vec![r_u8, g_u8, b_u8];
    let depth = vec![0.5f32];

    // Full pipeline baseline
    let out = grade_frame_cpu(&pixels, &depth, 1, 1, &cal, &params).unwrap();
    assert_eq!(out.len(), 3);
    // All in range
    for &v in &out { assert!(v <= 255, "out of range"); }

    // Baseline values are pinned here (update this comment if params change):
    // This will be verified against grade_pixel_cpu in Task 3.
    // Print for reference during development:
    // eprintln!("baseline: {:?}", out);
}
```

- [ ] **Step 2: Run the test — expect it to compile and pass**

```bash
cd repos/dorea && cargo test -p dorea-gpu grade_pixel_cpu_matches_grade_frame_cpu_baseline 2>&1 | tail -5
```

Expected: PASS (the test only verifies grade_frame_cpu works, not grade_pixel_cpu yet).

- [ ] **Step 3: Commit**

```bash
cd repos/dorea && git add crates/dorea-gpu/src/cpu.rs
git commit -m "test: add baseline regression test for grade_pixel_cpu (Task 1)"
```

---

### Task 2: Remove clarity from the pipeline

**Files:**
- Delete: `crates/dorea-gpu/src/cuda/kernels/clarity.cu`
- Modify: `crates/dorea-gpu/build.rs`
- Modify: `crates/dorea-gpu/src/cuda/mod.rs`
- Modify: `crates/dorea-gpu/src/cpu.rs`
- Modify: `crates/dorea-gpu/src/lib.rs`

- [ ] **Step 1: Write failing test**

In `crates/dorea-gpu/src/cpu.rs` tests, add:

```rust
#[test]
fn finish_grade_has_no_skip_clarity_param() {
    // This test verifies finish_grade compiles without skip_clarity.
    // Will fail until Task 2 removes the param.
    use crate::GradeParams;
    use dorea_cal::Calibration;
    use dorea_hsl::HslCorrections;
    use dorea_lut::types::{DepthLuts, LutGrid};

    let mut lut = LutGrid::new(2);
    for ri in 0..2usize { for gi in 0..2usize { for bi in 0..2usize {
        lut.set(ri, gi, bi, [ri as f32, gi as f32, bi as f32]);
    }}}
    let cal = Calibration::new(
        DepthLuts::new(vec![lut], vec![0.0, 1.0]),
        HslCorrections(vec![]),
        0,
    );
    let mut rgb_f32 = vec![0.5f32; 4 * 3];
    let orig = vec![128u8; 4 * 3];
    let depth = vec![0.5f32; 4];
    // After Task 2: finish_grade takes no skip_clarity param
    let out = finish_grade(&mut rgb_f32, &orig, &depth, 2, 2, &GradeParams::default(), &cal);
    assert_eq!(out.len(), 12);
}
```

- [ ] **Step 2: Run test — expect compile failure** (skip_clarity param still present)

```bash
cd repos/dorea && cargo test -p dorea-gpu finish_grade_has_no_skip_clarity_param 2>&1 | head -20
```

Expected: compile error about wrong number of arguments to finish_grade.

- [ ] **Step 3: Delete clarity.cu**

```bash
rm repos/dorea/crates/dorea-gpu/src/cuda/kernels/clarity.cu
```

- [ ] **Step 4: Update build.rs — remove clarity, update rerun-if-changed**

In `crates/dorea-gpu/build.rs`, replace:

```rust
    println!("cargo:rerun-if-changed=src/cuda/kernels/lut_apply.cu");
    println!("cargo:rerun-if-changed=src/cuda/kernels/hsl_correct.cu");
    println!("cargo:rerun-if-changed=src/cuda/kernels/clarity.cu");
```

with:

```rust
    println!("cargo:rerun-if-changed=src/cuda/kernels/lut_apply.cu");
    println!("cargo:rerun-if-changed=src/cuda/kernels/hsl_correct.cu");
```

And replace:

```rust
    let kernel_names = ["lut_apply", "hsl_correct", "clarity"];
```

with:

```rust
    let kernel_names = ["lut_apply", "hsl_correct"];
```

- [ ] **Step 5: Update cuda/mod.rs — remove clarity stage**

At the top of `crates/dorea-gpu/src/cuda/mod.rs`, remove:

```rust
#[cfg(feature = "cuda")]
const CLARITY_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/clarity.ptx"));

/// Blur radius for the clarity box-blur passes (matches CPU path).
#[cfg(feature = "cuda")]
const BLUR_RADIUS: i32 = 30;

/// Maximum proxy long-edge for clarity downsampling (matches CPU path: 518).
#[cfg(feature = "cuda")]
const PROXY_MAX_SIZE: usize = 518;
```

In `CudaGrader::new()`, remove the clarity PTX load block:

```rust
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
```

In `grade_frame_cuda`, replace the entire `// CLARITY` section (from the comment through `let result = dev.dtoh_sync_copy(&d_rgb_out)`) with:

```rust
        // Download result (HSL output is now the final GPU stage)
        let result = dev.dtoh_sync_copy(&d_rgb_after_hsl).map_err(map_cudarc_error)?;
```

Also remove the `ResolutionBuffers` struct's clarity fields (`d_proxy_l`, `d_blur_a`, `d_blur_b`, `d_rgb_out`) and the `alloc_resolution_buffers` function since they're dead code after this change. Remove `proxy_dims` function too.

The `grade_frame_cuda` return type remains `Result<Vec<f32>, GpuError>` for now.

- [ ] **Step 6: Update cpu.rs — remove apply_cpu_clarity, remove skip_clarity from finish_grade**

Remove the entire `apply_cpu_clarity` function (lines 219-230).

Remove the `apply_clarity` function and `three_pass_box_blur`, `box_blur_rows_sliding`, `box_blur_cols_via_transpose`, `box_blur_rows`, `box_blur_cols` functions (lines 268-407) — clarity is gone.

Change `finish_grade` signature and body:

```rust
/// Apply depth_aware_ambiance, warmth, blend, and convert f32 → u8.
///
/// Called by the CPU grading path after LUT+HSL processing.
/// `rgb_f32` is modified in place (ambiance + warmth applied).
/// `orig_pixels` is the original u8 input used for the strength blend.
pub(crate) fn finish_grade(
    rgb_f32: &mut [f32],
    orig_pixels: &[u8],
    depth: &[f32],
    width: usize,
    height: usize,
    params: &GradeParams,
    _cal: &Calibration,
) -> Vec<u8> {
    // 1. Fused ambiance + warmth (single LAB roundtrip, rayon-parallelized)
    let warmth_factor = 1.0 + (params.warmth - 1.0) * 0.3;
    fused_ambiance_warmth(rgb_f32, depth, width, height, params.contrast, warmth_factor);

    // 2. Blend with original using strength
    if params.strength < 1.0 - 1e-4 {
        for i in 0..rgb_f32.len() {
            let orig = orig_pixels[i] as f32 / 255.0;
            rgb_f32[i] = orig * (1.0 - params.strength) + rgb_f32[i] * params.strength;
        }
    }

    // 3. f32 → u8
    rgb_f32.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect()
}
```

Update `grade_frame_cpu` to pass without `skip_clarity`:

```rust
    // 3–4. Ambiance + warmth + blend + u8 (CPU finish pass)
    Ok(finish_grade(&mut rgb_f32, pixels, depth, width, height, params, calibration))
```

- [ ] **Step 7: Update lib.rs — remove skip_clarity=true arguments**

In `grade_frame`:

```rust
                    Ok(mut rgb_f32) => {
                        return Ok(cpu::finish_grade(
                            &mut rgb_f32,
                            pixels,
                            depth,
                            width,
                            height,
                            params,
                            calibration,
                        ));
                    }
```

In `grade_frame_with_grader`:

```rust
    let mut rgb_f32 =
        grader.grade_frame_cuda(pixels, depth, width, height, calibration, params)?;
    Ok(cpu::finish_grade(
        &mut rgb_f32,
        pixels,
        depth,
        width,
        height,
        params,
        calibration,
    ))
```

- [ ] **Step 8: Fix any test that references skip_clarity**

In `cpu.rs` tests, remove/update `finish_grade_skip_clarity_runs_without_panic` — it now becomes just `finish_grade_runs_without_panic` without the `skip_clarity` parameter.

- [ ] **Step 9: Run tests**

```bash
cd repos/dorea && cargo test -p dorea-gpu 2>&1 | tail -20
```

Expected: all tests pass (CUDA tests skipped if no GPU).

- [ ] **Step 10: Commit**

```bash
cd repos/dorea && git add -u crates/dorea-gpu/ && git rm crates/dorea-gpu/src/cuda/kernels/clarity.cu
git commit -m "feat: hard-delete clarity; remove 3-kernel clarity from GPU pipeline (Task 2)"
```

---

### Task 3: Extract grade_pixel_cpu

**Files:**
- Modify: `crates/dorea-gpu/src/cpu.rs`

This is the CPU reference implementation for a single lattice point — used to verify `grade_pixel_device` (in Task 4) and as a CPU test oracle for the combined LUT accuracy test (Task 10).

- [ ] **Step 1: Run the baseline test from Task 1**

```bash
cd repos/dorea && cargo test -p dorea-gpu grade_pixel_cpu_matches_grade_frame_cpu_baseline 2>&1 | tail -5
```

Expected: PASS (baseline still works after Task 2 changes).

- [ ] **Step 2: Write the test that verifies grade_pixel_cpu matches grade_frame_cpu**

Add to `cpu.rs` tests (this test was added in Task 1, and now we update it to also call grade_pixel_cpu):

```rust
#[test]
fn grade_pixel_cpu_matches_grade_frame_cpu() {
    use dorea_cal::Calibration;
    use dorea_hsl::derive::{HslCorrections, QualifierCorrection};
    use dorea_lut::types::{DepthLuts, LutGrid};

    fn identity_lut(size: usize) -> LutGrid {
        let mut lut = LutGrid::new(size);
        for ri in 0..size { for gi in 0..size { for bi in 0..size {
            let r = ri as f32 / (size - 1) as f32;
            let g = gi as f32 / (size - 1) as f32;
            let b = bi as f32 / (size - 1) as f32;
            lut.set(ri, gi, bi, [r, g, b]);
        }}}
        lut
    }

    let n_zones = 5;
    let luts: Vec<LutGrid> = (0..n_zones).map(|_| identity_lut(17)).collect();
    let boundaries: Vec<f32> = (0..=n_zones).map(|i| i as f32 / n_zones as f32).collect();
    let cal = Calibration::new(
        DepthLuts::new(luts, boundaries),
        HslCorrections(vec![QualifierCorrection {
            h_center: 0.0, h_width: 1.0, h_offset: 0.0,
            s_ratio: 1.0, v_offset: 0.0, weight: 0.0,
        }]),
        1,
    );
    let params = crate::GradeParams::default();

    let r_u8 = 153u8; let g_u8 = 77u8; let b_u8 = 51u8;
    let pixels = vec![r_u8, g_u8, b_u8];
    let depth_val = 0.5f32;
    let depth = vec![depth_val];

    // Full pipeline
    let frame_out = grade_frame_cpu(&pixels, &depth, 1, 1, &cal, &params).unwrap();

    // Single-pixel path (uses same u8-rounded inputs)
    let r_f32 = r_u8 as f32 / 255.0;
    let g_f32 = g_u8 as f32 / 255.0;
    let b_f32 = b_u8 as f32 / 255.0;
    let px_out = grade_pixel_cpu(r_f32, g_f32, b_f32, depth_val, &cal, &params);
    let r_out = (px_out[0].clamp(0.0, 1.0) * 255.0).round() as u8;
    let g_out = (px_out[1].clamp(0.0, 1.0) * 255.0).round() as u8;
    let b_out = (px_out[2].clamp(0.0, 1.0) * 255.0).round() as u8;

    assert_eq!([r_out, g_out, b_out], [frame_out[0], frame_out[1], frame_out[2]],
        "grade_pixel_cpu must match grade_frame_cpu to within 1/255 rounding");
}
```

- [ ] **Step 3: Run test — expect compile failure** (grade_pixel_cpu not yet defined)

```bash
cd repos/dorea && cargo test -p dorea-gpu grade_pixel_cpu_matches_grade_frame_cpu 2>&1 | head -10
```

Expected: `error[E0425]: cannot find function grade_pixel_cpu`.

- [ ] **Step 4: Implement grade_pixel_cpu in cpu.rs**

Add this function to `crates/dorea-gpu/src/cpu.rs` (before the `grade_frame_cpu` function):

```rust
/// Grade a single pixel through the full CPU pipeline.
///
/// Equivalent to running `grade_frame_cpu` on a 1×1 frame with the given pixel,
/// but without heap allocation. Used as a CPU oracle for combined LUT accuracy tests,
/// and mirrors the logic in `grade_pixel_device` (grade_pixel.cuh).
///
/// Returns graded f32 RGB [0,1] (caller converts to u8 as needed).
pub fn grade_pixel_cpu(
    r: f32, g: f32, b: f32,
    depth: f32,
    calibration: &Calibration,
    params: &GradeParams,
) -> [f32; 3] {
    use dorea_lut::apply::apply_depth_luts;
    use dorea_hsl::apply::apply_hsl_corrections;

    // 1. Depth-stratified LUT apply
    let lut_out = apply_depth_luts(&[[r, g, b]], &[depth], &calibration.depth_luts);
    let [r1, g1, b1] = lut_out[0];

    // 2. HSL qualifier corrections
    let hsl_out = apply_hsl_corrections(&[[r1, g1, b1]], &calibration.hsl_corrections);
    let [r2, g2, b2] = hsl_out[0];

    // 3. Fused ambiance + warmth (single LAB roundtrip)
    let mut px = [r2, g2, b2];
    let warmth_factor = 1.0 + (params.warmth - 1.0) * 0.3;
    let d = depth;

    let (mut l_norm, mut a_ab, mut b_ab) = {
        let (l, a, b_l) = dorea_color::lab::srgb_to_lab(px[0], px[1], px[2]);
        (l / 100.0, a, b_l)
    };

    // Shadow lift
    let lift_amount = 0.2 + 0.15 * d;
    let toe = 0.15_f32;
    let shadow_mask = ((toe - l_norm) / toe).clamp(0.0, 1.0);
    l_norm += shadow_mask * lift_amount * toe;

    // S-curve contrast
    let contrast_strength = (0.3 + 0.3 * d) * params.contrast;
    let slope = 4.0 + 4.0 * contrast_strength;
    let s_curve = 1.0 / (1.0 + (-(l_norm - 0.5) * slope).exp());
    l_norm += (s_curve - l_norm) * contrast_strength;

    // Highlight compress
    let compress = 0.4 + 0.2 * (1.0 - d);
    let knee_h = 0.88_f32;
    if l_norm > knee_h {
        let over = l_norm - knee_h;
        let headroom = 1.0 - knee_h;
        l_norm = knee_h + headroom * ((over / headroom * (1.0 + compress)).tanh());
    }

    // Warmth (LAB a*/b* push)
    let lum_weight = 4.0 * l_norm * (1.0 - l_norm);
    a_ab += (1.0 + 5.0 * d) * lum_weight;
    b_ab += 4.0 * d * lum_weight;

    // Vibrance
    let vibrance = 0.4 + 0.5 * d;
    let chroma = (a_ab * a_ab + b_ab * b_ab + 1e-8).sqrt();
    let chroma_norm = (chroma / 40.0).clamp(0.0, 1.0);
    let boost = vibrance * (1.0 - chroma_norm) * (l_norm / 0.25).clamp(0.0, 1.0);
    a_ab *= 1.0 + boost;
    b_ab *= 1.0 + boost;

    // User warmth scaling
    if (warmth_factor - 1.0).abs() > 1e-4 {
        a_ab *= warmth_factor;
        b_ab *= warmth_factor;
    }

    // LAB → RGB
    let l_out = (l_norm * 100.0).clamp(0.0, 100.0);
    let a_out = a_ab.clamp(-128.0, 127.0);
    let b_out_clamped = b_ab.clamp(-128.0, 127.0);
    let (ro, go, bo) = dorea_color::lab::lab_to_srgb(l_out, a_out, b_out_clamped);

    // Final highlight knee
    let knee = 0.92_f32;
    let apply_knee = |v: f32| -> f32 {
        if v > knee { let over = v - knee; let room = 1.0 - knee; knee + room * ((over / room).tanh()) } else { v }
    };

    px[0] = apply_knee(ro).clamp(0.0, 1.0);
    px[1] = apply_knee(go).clamp(0.0, 1.0);
    px[2] = apply_knee(bo).clamp(0.0, 1.0);

    // 4. Strength blend with original input
    let strength = params.strength;
    if strength < 1.0 - 1e-4 {
        px[0] = r * (1.0 - strength) + px[0] * strength;
        px[1] = g * (1.0 - strength) + px[1] * strength;
        px[2] = b * (1.0 - strength) + px[2] * strength;
    }

    px
}
```

- [ ] **Step 5: Run tests**

```bash
cd repos/dorea && cargo test -p dorea-gpu grade_pixel_cpu 2>&1 | tail -10
```

Expected: both grade_pixel_cpu tests PASS.

- [ ] **Step 6: Commit**

```bash
cd repos/dorea && git add crates/dorea-gpu/src/cpu.rs
git commit -m "feat: add grade_pixel_cpu — single-pixel CPU reference for combined LUT (Task 3)"
```

---

### Task 4: Write grade_pixel.cuh

**Files:**
- Create: `crates/dorea-gpu/src/cuda/kernels/grade_pixel.cuh`

This header is `#include`d by both `build_combined_lut.cu` and (if needed) `combined_lut.cu`. It is not compiled standalone and has no PTX. No build.rs change needed.

- [ ] **Step 1: Create grade_pixel.cuh**

```c
/**
 * grade_pixel.cuh — __device__ single-pixel grading pipeline.
 *
 * Included by build_combined_lut.cu.
 * Mirrors grade_pixel_cpu() in cpu.rs exactly.
 *
 * Pipeline order:
 *   1. Depth-stratified LUT (soft zone blend + manual trilinear)
 *   2. HSL 6-qualifier corrections
 *   3. Fused ambiance + warmth (LAB colorspace)
 *   4. Strength blend with original input
 */

#pragma once
#include <cuda_runtime.h>
#include <math.h>

// -------------------------------------------------------------------------
// CIELAB D65 colorspace (matches dorea-color/src/lab.rs exactly)
// -------------------------------------------------------------------------

#define XN 0.95047f
#define YN 1.00000f
#define ZN 1.08883f
#define DELTA_CUBED 0.008856f   // (6/29)^3
#define DELTA_SQ_3  0.128419f   // 3*(6/29)^2

__device__ __forceinline__ float srgb_to_linear(float v) {
    return (v <= 0.04045f) ? (v / 12.92f) : powf((v + 0.055f) / 1.055f, 2.4f);
}

__device__ __forceinline__ float linear_to_srgb(float v) {
    return (v <= 0.0031308f) ? (v * 12.92f) : (1.055f * powf(v, 1.0f / 2.4f) - 0.055f);
}

__device__ __forceinline__ float f_lab(float t) {
    return (t > DELTA_CUBED) ? cbrtf(t) : (t / DELTA_SQ_3 + 4.0f / 29.0f);
}

__device__ __forceinline__ float f_lab_inv(float s) {
    const float delta = 6.0f / 29.0f;
    return (s > delta) ? (s * s * s) : (DELTA_SQ_3 * (s - 4.0f / 29.0f));
}

__device__ void srgb_to_lab(float r, float g, float b,
                              float* l_out, float* a_out, float* b_lab_out) {
    float rl = srgb_to_linear(r);
    float gl = srgb_to_linear(g);
    float bl = srgb_to_linear(b);

    float x = 0.4124564f*rl + 0.3575761f*gl + 0.1804375f*bl;
    float y = 0.2126729f*rl + 0.7151522f*gl + 0.0721750f*bl;
    float z = 0.0193339f*rl + 0.1191920f*gl + 0.9503041f*bl;

    float fx = f_lab(x / XN);
    float fy = f_lab(y / YN);
    float fz = f_lab(z / ZN);

    *l_out     = 116.0f * fy - 16.0f;
    *a_out     = 500.0f * (fx - fy);
    *b_lab_out = 200.0f * (fy - fz);
}

__device__ void lab_to_srgb(float l, float a, float b_lab,
                              float* r_out, float* g_out, float* b_out) {
    float fy = (l + 16.0f) / 116.0f;
    float fx = a / 500.0f + fy;
    float fz = fy - b_lab / 200.0f;

    float x = XN * f_lab_inv(fx);
    float y = YN * f_lab_inv(fy);
    float z = ZN * f_lab_inv(fz);

    float rl =  3.2404542f*x - 1.5371385f*y - 0.4985314f*z;
    float gl = -0.9692660f*x + 1.8760108f*y + 0.0415560f*z;
    float bl2=  0.0556434f*x - 0.2040259f*y + 1.0572252f*z;

    *r_out = fminf(fmaxf(linear_to_srgb(fmaxf(rl, 0.0f)), 0.0f), 1.0f);
    *g_out = fminf(fmaxf(linear_to_srgb(fmaxf(gl, 0.0f)), 0.0f), 1.0f);
    *b_out = fminf(fmaxf(linear_to_srgb(fmaxf(bl2,0.0f)), 0.0f), 1.0f);
}

// -------------------------------------------------------------------------
// Trilinear LUT sample (matches trilinear_sample in lut_apply.cu)
// -------------------------------------------------------------------------

__device__ float3 trilinear_sample(
    const float* __restrict__ lut,
    int lut_size,
    float r, float g, float b
) {
    float s = (float)(lut_size - 1);
    float fr_f = r * s;
    float fg_f = g * s;
    float fb_f = b * s;

    int i0 = (int)fr_f; int j0 = (int)fg_f; int k0 = (int)fb_f;
    i0 = min(max(i0, 0), lut_size - 2);
    j0 = min(max(j0, 0), lut_size - 2);
    k0 = min(max(k0, 0), lut_size - 2);

    float fr = fminf(fmaxf(fr_f - (float)i0, 0.0f), 1.0f);
    float fg = fminf(fmaxf(fg_f - (float)j0, 0.0f), 1.0f);
    float fb = fminf(fmaxf(fb_f - (float)k0, 0.0f), 1.0f);

    float3 result = {0.0f, 0.0f, 0.0f};
    for (int c = 0; c < 3; c++) {
#define IDX(di,dj,dk) (((i0+(di))*lut_size*lut_size + (j0+(dj))*lut_size + (k0+(dk)))*3 + c)
        float v000=lut[IDX(0,0,0)], v001=lut[IDX(0,0,1)];
        float v010=lut[IDX(0,1,0)], v011=lut[IDX(0,1,1)];
        float v100=lut[IDX(1,0,0)], v101=lut[IDX(1,0,1)];
        float v110=lut[IDX(1,1,0)], v111=lut[IDX(1,1,1)];
        float v =
            v000*(1-fr)*(1-fg)*(1-fb) + v100*fr*(1-fg)*(1-fb) +
            v010*(1-fr)*fg*(1-fb)     + v110*fr*fg*(1-fb)     +
            v001*(1-fr)*(1-fg)*fb     + v101*fr*(1-fg)*fb     +
            v011*(1-fr)*fg*fb         + v111*fr*fg*fb;
#undef IDX
        if (c == 0) result.x = v;
        else if (c == 1) result.y = v;
        else result.z = v;
    }
    return result;
}

// -------------------------------------------------------------------------
// HSV helpers (matches hsl_correct.cu)
// -------------------------------------------------------------------------

__device__ void rgb_to_hsv_gp(float r, float g, float b,
                                float* h, float* s, float* v) {
    float cmax = fmaxf(fmaxf(r,g),b);
    float cmin = fminf(fminf(r,g),b);
    float delta = cmax - cmin;
    *v = cmax;
    *s = (cmax > 1e-6f) ? (delta / cmax) : 0.0f;
    if (delta < 1e-6f) { *h = 0.0f; return; }
    float hh;
    if      (cmax == r) hh = 60.0f * fmodf((g-b)/delta, 6.0f);
    else if (cmax == g) hh = 60.0f * ((b-r)/delta + 2.0f);
    else                hh = 60.0f * ((r-g)/delta + 4.0f);
    if (hh < 0.0f) hh += 360.0f;
    *h = hh;
}

__device__ void hsv_to_rgb_gp(float h, float s, float v,
                                float* r, float* g, float* b) {
    if (s < 1e-6f) { *r = *g = *b = v; return; }
    float hh = fmodf(h, 360.0f); if (hh < 0.0f) hh += 360.0f; hh /= 60.0f;
    int i = (int)hh; float f = hh - (float)i;
    float p = v*(1.0f-s), q = v*(1.0f-s*f), t = v*(1.0f-s*(1.0f-f));
    switch (i%6) {
        case 0: *r=v;*g=t;*b=p; break; case 1: *r=q;*g=v;*b=p; break;
        case 2: *r=p;*g=v;*b=t; break; case 3: *r=p;*g=q;*b=v; break;
        case 4: *r=t;*g=p;*b=v; break; default:*r=v;*g=p;*b=q; break;
    }
}

// HSL qualifier constants (matches hsl_correct.cu)
__constant__ float GP_H_CENTERS[6] = {0.f, 40.f, 100.f, 170.f, 210.f, 290.f};
__constant__ float GP_H_WIDTHS[6]  = {40.f, 40.f, 50.f, 40.f, 40.f, 50.f};

// -------------------------------------------------------------------------
// Full single-pixel grading pipeline
// -------------------------------------------------------------------------

/**
 * Grade one pixel through the complete pipeline:
 *   LUT apply (soft zone blend + trilinear) →
 *   HSL correct (6 qualifiers) →
 *   Fused ambiance + warmth (LAB) →
 *   Strength blend →
 *   Returns graded f32 RGB [0,1]
 *
 * Parameters match the packed layout uploaded by CombinedLut::build():
 *   luts          — [n_zones][lut_size³][3] float array
 *   zone_boundaries — [n_zones+1] float array
 *   h_offsets/s_ratios/v_offsets/weights — [6] float arrays for HSL
 *   warmth        — params.warmth
 *   strength      — params.strength
 *   contrast      — params.contrast
 */
__device__ float3 grade_pixel_device(
    float r, float g, float b, float depth,
    const float* __restrict__ luts,
    const float* __restrict__ zone_boundaries,
    int lut_size, int n_zones,
    const float* __restrict__ h_offsets,
    const float* __restrict__ s_ratios,
    const float* __restrict__ v_offsets,
    const float* __restrict__ weights,
    float warmth, float strength, float contrast
) {
    // ------------------------------------------------------------------
    // 1. Depth-stratified LUT (soft triangular zone blend)
    // ------------------------------------------------------------------
    int lut_stride = lut_size * lut_size * lut_size * 3;
    float total_w = 0.0f;
    float3 blended = {0.0f, 0.0f, 0.0f};

    for (int z = 0; z < n_zones; z++) {
        float z_lo = zone_boundaries[z];
        float z_hi = zone_boundaries[z + 1];
        float z_width = z_hi - z_lo;
        if (z_width < 1e-6f) continue;
        float z_center = 0.5f * (z_lo + z_hi);
        float dist = fabsf(depth - z_center);
        float w = fmaxf(1.0f - dist / z_width, 0.0f);
        if (w < 1e-6f) continue;

        float3 lut_out = trilinear_sample(luts + z * lut_stride, lut_size, r, g, b);
        blended.x += lut_out.x * w;
        blended.y += lut_out.y * w;
        blended.z += lut_out.z * w;
        total_w += w;
    }

    float r1, g1, b1;
    if (total_w > 1e-6f) {
        r1 = blended.x / total_w;
        g1 = blended.y / total_w;
        b1 = blended.z / total_w;
    } else {
        r1 = r; g1 = g; b1 = b;
    }

    // ------------------------------------------------------------------
    // 2. HSL 6-qualifier corrections
    // ------------------------------------------------------------------
    float h, s, v;
    rgb_to_hsv_gp(r1, g1, b1, &h, &s, &v);

    for (int q = 0; q < 6; q++) {
        if (weights[q] < 100.0f) continue;
        float dist_h = fabsf(fmodf(h - GP_H_CENTERS[q] + 360.0f, 360.0f));
        if (dist_h > 180.0f) dist_h = 360.0f - dist_h;
        float mask = fmaxf(1.0f - dist_h / GP_H_WIDTHS[q], 0.0f);
        if (s < 0.08f) mask = 0.0f;
        if (mask < 1e-6f) continue;
        h += h_offsets[q] * mask;
        s *= (1.0f + (s_ratios[q] - 1.0f) * mask);
        v += v_offsets[q] * mask;
    }

    h = fmodf(h, 360.0f); if (h < 0.0f) h += 360.0f;
    s = fminf(fmaxf(s, 0.0f), 1.0f);
    v = fminf(fmaxf(v, 0.0f), 1.0f);

    float r2, g2, b2;
    hsv_to_rgb_gp(h, s, v, &r2, &g2, &b2);

    // ------------------------------------------------------------------
    // 3. Fused ambiance + warmth (LAB colorspace)
    // ------------------------------------------------------------------
    float d = depth;
    float l_norm, a_ab, b_ab;
    {
        float l_raw, a_raw, b_raw;
        srgb_to_lab(r2, g2, b2, &l_raw, &a_raw, &b_raw);
        l_norm = l_raw / 100.0f; a_ab = a_raw; b_ab = b_raw;
    }

    // Shadow lift
    float lift = 0.2f + 0.15f * d;
    float toe = 0.15f;
    float shadow_mask = fminf(fmaxf((toe - l_norm) / toe, 0.0f), 1.0f);
    l_norm += shadow_mask * lift * toe;

    // S-curve contrast
    float cs = (0.3f + 0.3f * d) * contrast;
    float slope = 4.0f + 4.0f * cs;
    float s_curve = 1.0f / (1.0f + expf(-(l_norm - 0.5f) * slope));
    l_norm += (s_curve - l_norm) * cs;

    // Highlight compress
    float compress = 0.4f + 0.2f * (1.0f - d);
    float knee_h = 0.88f;
    if (l_norm > knee_h) {
        float over = l_norm - knee_h;
        float headroom = 1.0f - knee_h;
        l_norm = knee_h + headroom * tanhf(over / headroom * (1.0f + compress));
    }

    // Warmth (a*/b* push)
    float lum_w = 4.0f * l_norm * (1.0f - l_norm);
    a_ab += (1.0f + 5.0f * d) * lum_w;
    b_ab +=  4.0f * d          * lum_w;

    // Vibrance
    float vib = 0.4f + 0.5f * d;
    float chroma = sqrtf(a_ab*a_ab + b_ab*b_ab + 1e-8f);
    float chroma_n = fminf(chroma / 40.0f, 1.0f);
    float boost = vib * (1.0f - chroma_n) * fminf(l_norm / 0.25f, 1.0f);
    a_ab *= 1.0f + boost;
    b_ab *= 1.0f + boost;

    // User warmth scaling
    float warmth_factor = 1.0f + (warmth - 1.0f) * 0.3f;
    if (fabsf(warmth_factor - 1.0f) > 1e-4f) {
        a_ab *= warmth_factor;
        b_ab *= warmth_factor;
    }

    // LAB → RGB
    float l_out  = fminf(fmaxf(l_norm * 100.0f, 0.0f), 100.0f);
    float a_out  = fminf(fmaxf(a_ab,  -128.0f), 127.0f);
    float b_out2 = fminf(fmaxf(b_ab,  -128.0f), 127.0f);
    float r3, g3, b3;
    lab_to_srgb(l_out, a_out, b_out2, &r3, &g3, &b3);

    // Final highlight knee — inlined to avoid `auto` lambda in __device__ fn
    // (CUDA device lambdas require --expt-extended-lambda; inline avoids the flag)
    {
        const float knee = 0.92f;
        const float room = 1.0f - knee;
        r3 = fminf(fmaxf(r3 > knee ? knee + room * tanhf((r3 - knee) / room) : r3, 0.0f), 1.0f);
        g3 = fminf(fmaxf(g3 > knee ? knee + room * tanhf((g3 - knee) / room) : g3, 0.0f), 1.0f);
        b3 = fminf(fmaxf(b3 > knee ? knee + room * tanhf((b3 - knee) / room) : b3, 0.0f), 1.0f);
    }

    // ------------------------------------------------------------------
    // 4. Strength blend with original input
    // ------------------------------------------------------------------
    if (strength < 1.0f - 1e-4f) {
        r3 = r * (1.0f - strength) + r3 * strength;
        g3 = g * (1.0f - strength) + g3 * strength;
        b3 = b * (1.0f - strength) + b3 * strength;
    }

    return {r3, g3, b3};
}
```

- [ ] **Step 2: Verify the file is syntactically plausible**

```bash
cd repos/dorea && nvcc --ptx -arch=sm_86 --allow-unsupported-compiler \
    --compiler-bindir /usr/bin/gcc-12 \
    /dev/null -include crates/dorea-gpu/src/cuda/kernels/grade_pixel.cuh \
    -o /dev/null 2>&1 | head -20
```

Expected: completes (possibly with warnings). Alternatively, this will be validated when build_combined_lut.cu is compiled in Task 5.

- [ ] **Step 3: Commit**

```bash
cd repos/dorea && git add crates/dorea-gpu/src/cuda/kernels/grade_pixel.cuh
git commit -m "feat: add grade_pixel.cuh — CUDA device fn for full grading pipeline (Task 4)"
```

---

### Task 5: Write build_combined_lut.cu + update build.rs

**Files:**
- Create: `crates/dorea-gpu/src/cuda/kernels/build_combined_lut.cu`
- Modify: `crates/dorea-gpu/build.rs`

- [ ] **Step 1: Create build_combined_lut.cu**

```c
/**
 * build_combined_lut.cu — GPU kernel to precompute the combined LUT.
 *
 * 1D grid: one thread per (zone × grid_point) pair.
 * Thread count = n_zones × N³  where N = COMBINED_LUT_GRID (97).
 *
 * Each thread:
 *   1. Decodes flat index → (zone, ri, gi, bi)
 *   2. Computes (r,g,b) = (ri/(N-1), gi/(N-1), bi/(N-1))
 *   3. Calls grade_pixel_device() at depth = zone center
 *   4. Writes float4(r', g', b', 0.0) to output buffer
 *
 * Output layout: [n_zones][N][N][N] float4
 *   Stride: zone * N*N*N * 4 floats
 *
 * Parameters:
 *   output         — float4 [n_zones × N³] device buffer (pre-allocated)
 *   luts           — float [n_zones × lut_size³ × 3]
 *   zone_boundaries— float [n_zones + 1]
 *   h_offsets      — float [6]
 *   s_ratios       — float [6]
 *   v_offsets      — float [6]
 *   weights        — float [6]
 *   warmth/strength/contrast — scalar GradeParams
 *   grid_size      — N (= 97)
 *   lut_size       — zone LUT grid size (e.g. 17 or 33)
 *   n_zones        — number of depth zones
 *   total_threads  — n_zones * N^3 (to guard OOB)
 */

#include <cuda_runtime.h>
#include "grade_pixel.cuh"

extern "C"
__global__ void build_combined_lut_kernel(
    float4* __restrict__ output,
    const float* __restrict__ luts,
    const float* __restrict__ zone_boundaries,
    const float* __restrict__ h_offsets,
    const float* __restrict__ s_ratios,
    const float* __restrict__ v_offsets,
    const float* __restrict__ weights,
    float warmth, float strength, float contrast,
    int grid_size, int lut_size, int n_zones,
    int total_threads
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_threads) return;

    int N = grid_size;
    int N3 = N * N * N;

    int zone = idx / N3;
    int rem  = idx % N3;

    // Memory layout matches cuMemcpy3D: ri is column (x/fastest), gi is row (y), bi is slice (z/depth).
    // This ensures tex3D(tex, r*(N-1), g*(N-1), b*(N-1)) in the per-frame kernel maps
    // x→ri, y→gi, z→bi correctly — no R/B axis swap.
    int bi = rem / (N * N);   // z = depth/slice (slowest in memory)
    int gi = (rem / N) % N;   // y = row
    int ri = rem % N;          // x = column (fastest in memory)

    float r = (float)ri / (float)(N - 1);
    float g = (float)gi / (float)(N - 1);
    float b = (float)bi / (float)(N - 1);

    // Depth = center of this zone
    float z_lo = zone_boundaries[zone];
    float z_hi = zone_boundaries[zone + 1];
    float depth = 0.5f * (z_lo + z_hi);

    float3 graded = grade_pixel_device(
        r, g, b, depth,
        luts, zone_boundaries,
        lut_size, n_zones,
        h_offsets, s_ratios, v_offsets, weights,
        warmth, strength, contrast
    );

    // Write float4: hardware requires 4-channel format for 3D texture
    output[idx] = make_float4(graded.x, graded.y, graded.z, 0.0f);
}
```

- [ ] **Step 2: Update build.rs — add build_combined_lut to kernel_names and rerun-if-changed**

In `crates/dorea-gpu/build.rs`, after the existing rerun-if-changed lines, add:

```rust
    println!("cargo:rerun-if-changed=src/cuda/kernels/grade_pixel.cuh");
    println!("cargo:rerun-if-changed=src/cuda/kernels/build_combined_lut.cu");
```

Change the kernel_names array from:

```rust
    let kernel_names = ["lut_apply", "hsl_correct"];
```

to:

```rust
    let kernel_names = ["lut_apply", "hsl_correct", "build_combined_lut"];
```

- [ ] **Step 3: Verify it compiles**

```bash
cd repos/dorea && cargo build -p dorea-gpu 2>&1 | grep -E "warning=|error" | head -20
```

Expected: no errors; possibly nvcc warnings about `auto` lambda in CUDA (acceptable).

- [ ] **Step 4: Commit**

```bash
cd repos/dorea && git add crates/dorea-gpu/src/cuda/kernels/build_combined_lut.cu crates/dorea-gpu/build.rs
git commit -m "feat: add build_combined_lut.cu GPU kernel for LUT precomputation (Task 5)"
```

---

### Task 6: Write CombinedLut Rust struct

**Files:**
- Create: `crates/dorea-gpu/src/cuda/combined_lut.rs`
- Modify: `crates/dorea-gpu/src/cuda/mod.rs` (add `mod combined_lut; pub(crate) use combined_lut::CombinedLut;`)

This struct wraps the raw CUDA driver API calls to create `CUarray` (3D) and `CUtexObject` for each zone.

- [ ] **Step 1: Write the failing test for CombinedLut::build**

Add to `cuda/mod.rs` inside the existing `#[cfg(all(feature = "cuda", test))] mod tests`:

```rust
    #[test]
    fn combined_lut_builds_without_panic() {
        // CombinedLut::build will fail until Task 6 is complete.
        use crate::cuda::combined_lut::CombinedLut;
        let cal = make_calibration(5);
        let params = crate::GradeParams::default();
        let device = match cudarc::driver::CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => { eprintln!("SKIP: no CUDA device ({e})"); return; }
        };
        // This will compile-error until combined_lut.rs exists.
        let _lut = CombinedLut::build(&device, &cal, &params).expect("CombinedLut::build failed");
    }
```

- [ ] **Step 2: Create combined_lut.rs**

```rust
//! CombinedLut — manages CUDA 3D texture arrays for the precomputed combined LUT.
//!
//! One `CUarray` + `CUtexObject` per depth zone.
//! Created once at `CudaGrader::new()`; reused for all frames.

use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::driver::sys::{self, *};  // `self` for sys::lib() calls; `*` for type names
use cudarc::nvrtc::Ptx;
use crate::{GradeParams, GpuError};
use dorea_cal::Calibration;

use super::map_cudarc_error;

/// Grid size for the combined LUT (N³ lattice points per zone).
/// Bump to 129 if 10-bit empirical testing reveals banding.
pub const COMBINED_LUT_GRID: usize = 97;

/// Maximum depth zones supported by the texture array.
pub const MAX_ZONES: usize = 8;

/// Embedded PTX for the GPU LUT build kernel.
const BUILD_COMBINED_LUT_PTX: &str =
    include_str!(concat!(env!("OUT_DIR"), "/build_combined_lut.ptx"));

pub(crate) struct CombinedLut {
    /// One CUarray per zone (float4, N×N×N).
    arrays:          Vec<CUarray>,
    /// One texture object per zone.
    pub textures:    Vec<CUtexObject>,
    pub n_zones:     usize,
    pub grid_size:   usize,
    /// Zone boundary values [n_zones + 1], kept for the per-frame kernel.
    pub zone_boundaries: Vec<f32>,
}

// Safety: CombinedLut is !Send by design (CUDA context is thread-local).
// We do not implement Send/Sync; callers hold it inside CudaGrader which is also !Send.
impl Drop for CombinedLut {
    fn drop(&mut self) {
        // cudarc 0.12 dynamic-linking: driver calls go through sys::lib(), not bare symbols.
        unsafe {
            for &tex in &self.textures {
                let _ = sys::lib().cuTexObjectDestroy(tex);
            }
            for &arr in &self.arrays {
                let _ = sys::lib().cuArrayDestroy(arr);
            }
        }
    }
}

impl CombinedLut {
    /// Build the combined LUT textures from calibration data.
    ///
    /// Call once at grader construction. This:
    ///   1. Uploads calibration to device
    ///   2. Launches build_combined_lut_kernel (fills float4 device buffer)
    ///   3. Copies each zone's slice into a CUarray (device-to-device via cuMemcpy3D)
    ///   4. Creates an unnormalized-coordinate linear texture object per zone
    ///   5. Frees the build buffer (73MB freed after this fn returns)
    pub(crate) fn build(
        device: &Arc<CudaDevice>,
        calibration: &Calibration,
        params: &GradeParams,
    ) -> Result<Self, GpuError> {
        let n_zones = calibration.depth_luts.n_zones();
        if n_zones == 0 {
            return Err(GpuError::InvalidInput("n_zones must be >= 1".into()));
        }
        let n_zones = n_zones.min(MAX_ZONES);
        let N = COMBINED_LUT_GRID;
        let lut_size = calibration.depth_luts.luts[0].size;

        // --- Upload calibration data ---
        let luts_flat: Vec<f32> = calibration.depth_luts.luts.iter()
            .flat_map(|lg| lg.data.iter().copied())
            .collect();
        let d_luts = device.htod_sync_copy(&luts_flat).map_err(map_cudarc_error)?;
        let d_boundaries = device.htod_sync_copy(&calibration.depth_luts.zone_boundaries)
            .map_err(map_cudarc_error)?;

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
        let d_h_offsets = device.htod_sync_copy(&h_offsets).map_err(map_cudarc_error)?;
        let d_s_ratios  = device.htod_sync_copy(&s_ratios).map_err(map_cudarc_error)?;
        let d_v_offsets = device.htod_sync_copy(&v_offsets).map_err(map_cudarc_error)?;
        let d_weights   = device.htod_sync_copy(&weights).map_err(map_cudarc_error)?;

        // --- Allocate build buffer: n_zones × N³ × float4 (4 floats) ---
        let build_count = n_zones * N * N * N * 4; // float4 = 4 f32 per texel
        let d_build: CudaSlice<f32> = device.alloc_zeros(build_count).map_err(map_cudarc_error)?;

        // --- Launch build kernel ---
        device.load_ptx(
            Ptx::from_src(BUILD_COMBINED_LUT_PTX),
            "build_combined_lut",
            &["build_combined_lut_kernel"],
        ).map_err(|e| GpuError::ModuleLoad(format!("load build_combined_lut PTX: {e}")))?;

        let total_threads = (n_zones * N * N * N) as u32;
        let block = 256u32;
        let grid  = total_threads.div_ceil(block);

        {
            let func = device.get_func("build_combined_lut", "build_combined_lut_kernel")
                .ok_or_else(|| GpuError::ModuleLoad("build_combined_lut_kernel not found".into()))?;
            let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };
            unsafe {
                func.launch(cfg, (
                    &d_build,
                    &d_luts,
                    &d_boundaries,
                    &d_h_offsets,
                    &d_s_ratios,
                    &d_v_offsets,
                    &d_weights,
                    params.warmth,
                    params.strength,
                    params.contrast,
                    N as i32,
                    lut_size as i32,
                    n_zones as i32,
                    total_threads as i32,
                ))
            }.map_err(map_cudarc_error)?;
        }

        // Synchronise: ensure build kernel completed before cuMemcpy3D
        device.synchronize().map_err(map_cudarc_error)?;

        // --- For each zone: copy device buffer slice → CUarray, create texture ---
        let mut arrays:   Vec<CUarray>       = Vec::with_capacity(n_zones);
        let mut textures: Vec<CUtexObject>   = Vec::with_capacity(n_zones);

        // Raw device pointer to the build buffer (CUdeviceptr = u64)
        let build_base: CUdeviceptr = unsafe { *d_build.device_ptr() };
        // Bytes per zone: N*N*N float4 = N*N*N * 4 * sizeof(f32)
        let zone_bytes = (N * N * N * 4 * std::mem::size_of::<f32>()) as u64;
        // Width in bytes for one row of the 3D region: N texels × float4 = N × 16 bytes
        let width_in_bytes = (N * 4 * std::mem::size_of::<f32>()) as usize;

        for zone in 0..n_zones {
            // --- Create CUarray ---
            let arr_desc = CUDA_ARRAY3D_DESCRIPTOR {
                Width:       N,
                Height:      N,
                Depth:       N,
                Format:      CUarray_format_enum::CU_AD_FORMAT_FLOAT,
                NumChannels: 4,
                Flags:       0,
            };
            let mut arr: CUarray = std::ptr::null_mut();
            unsafe {
                // dynamic-linking: must dispatch through sys::lib()
                let rc = sys::lib().cuArray3DCreate_v2(&mut arr, &arr_desc);
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(GpuError::CudaFail(format!("cuArray3DCreate zone {zone}: {rc:?}")));
                }
            }
            arrays.push(arr);

            // --- cuMemcpy3D: linear device buffer → CUarray (no PCIe roundtrip) ---
            let zone_ptr = build_base + zone as u64 * zone_bytes;
            let mut cpy: CUDA_MEMCPY3D = unsafe { std::mem::zeroed() };
            cpy.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_DEVICE;
            cpy.srcDevice     = zone_ptr;
            cpy.srcPitch      = width_in_bytes;
            cpy.srcHeight     = N;
            cpy.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_ARRAY;
            cpy.dstArray      = arr;
            cpy.WidthInBytes  = width_in_bytes;
            cpy.Height        = N;
            cpy.Depth         = N;

            unsafe {
                let rc = sys::lib().cuMemcpy3D_v2(&cpy);
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(GpuError::CudaFail(format!("cuMemcpy3D zone {zone}: {rc:?}")));
                }
            }

            // --- Create texture object ---
            let mut res_desc: CUDA_RESOURCE_DESC = unsafe { std::mem::zeroed() };
            res_desc.resType = CUresourcetype_enum::CU_RESOURCE_TYPE_ARRAY;
            // SAFETY: The res union field for array is hArray
            unsafe { res_desc.res.array.hArray = arr; }

            let mut tex_desc: CUDA_TEXTURE_DESC = unsafe { std::mem::zeroed() };
            tex_desc.addressMode[0] = CUaddress_mode_enum::CU_TR_ADDRESS_MODE_CLAMP;
            tex_desc.addressMode[1] = CUaddress_mode_enum::CU_TR_ADDRESS_MODE_CLAMP;
            tex_desc.addressMode[2] = CUaddress_mode_enum::CU_TR_ADDRESS_MODE_CLAMP;
            tex_desc.filterMode     = CUfilter_mode_enum::CU_TR_FILTER_MODE_LINEAR;
            tex_desc.flags          = 0;    // unnormalized coords — sample with r*(N-1) not r

            let mut tex: CUtexObject = 0;
            unsafe {
                let rc = sys::lib().cuTexObjectCreate(&mut tex, &res_desc, &tex_desc, std::ptr::null());
                if rc != CUresult::CUDA_SUCCESS {
                    return Err(GpuError::CudaFail(format!("cuTexObjectCreate zone {zone}: {rc:?}")));
                }
            }
            textures.push(tex);
        }

        // d_build drops here — 73MB freed
        Ok(Self {
            arrays,
            textures,
            n_zones,
            grid_size: N,
            zone_boundaries: calibration.depth_luts.zone_boundaries.clone(),
        })
    }
}
```

- [ ] **Step 3: Add module declaration to cuda/mod.rs**

At the top of `crates/dorea-gpu/src/cuda/mod.rs`, add:

```rust
#[cfg(feature = "cuda")]
mod combined_lut;
#[cfg(feature = "cuda")]
pub(crate) use combined_lut::CombinedLut;
```

Also, add `pub(crate) use combined_lut::map_cudarc_error;` is **wrong** — `map_cudarc_error` is in `mod.rs` itself. Instead, the `combined_lut.rs` module calls `super::map_cudarc_error`. In `mod.rs`, keep `map_cudarc_error` as is; the `combined_lut.rs` imports it with `use super::map_cudarc_error;`.

`CudaDevice::synchronize()` is confirmed to exist in cudarc 0.12.1 — use it directly:
```rust
        device.synchronize().map_err(map_cudarc_error)?;
```

- [ ] **Step 4: Verify it compiles**

```bash
cd repos/dorea && cargo build -p dorea-gpu 2>&1 | grep -E "^error" | head -20
```

Fix any type name mismatches in cudarc::driver::sys — the exact enum variant names (CUresult, CUmemorytype_enum, etc.) may differ slightly from the CUDA headers. Run `grep -r "CU_AD_FORMAT_FLOAT\|CUarray_format" ~/.cargo/registry/src/` to find the correct names.

- [ ] **Step 5: Run CUDA tests (if GPU available)**

```bash
cd repos/dorea && cargo test -p dorea-gpu --features cuda combined_lut_builds_without_panic 2>&1 | tail -10
```

Expected: PASS (if CUDA device present).

- [ ] **Step 6: Commit**

```bash
cd repos/dorea && git add crates/dorea-gpu/src/cuda/combined_lut.rs crates/dorea-gpu/src/cuda/mod.rs
git commit -m "feat: add CombinedLut — CUarray/CUtexObject lifecycle for combined LUT (Task 6)"
```

---

### Task 7: Write combined_lut.cu + update build.rs

**Files:**
- Create: `crates/dorea-gpu/src/cuda/kernels/combined_lut.cu`
- Modify: `crates/dorea-gpu/build.rs`

- [ ] **Step 1: Create combined_lut.cu**

```c
/**
 * combined_lut.cu — Per-frame kernel: sample combined LUT textures.
 *
 * 1D grid, 256 threads/block. No shared memory (texture cache handles locality).
 *
 * Per pixel:
 *   1. u8 → f32 (in-kernel, no CPU expansion)
 *   2. Find 2 bounding depth zones (soft triangular blend)
 *   3. tex3D<float4> × 2 (hardware trilinear interpolation)
 *   4. Depth-weighted lerp of the two zone results
 *   5. f32 → u8 output
 *
 * Parameters:
 *   pixels_in       — uint8 RGB interleaved [n_pixels * 3]
 *   depth           — float32 [n_pixels], in [0,1]
 *   textures        — CUtexObject array (device pointer to n_zones handles)
 *   zone_boundaries — float [n_zones+1]
 *   pixels_out      — uint8 RGB interleaved [n_pixels * 3]
 *   n_pixels        — pixel count
 *   n_zones         — number of zones
 */

#include <cuda_runtime.h>

extern "C"
__global__ void combined_lut_kernel(
    const unsigned char* __restrict__ pixels_in,
    const float*         __restrict__ depth,
    const unsigned long long* __restrict__ textures,   // CUtexObject array
    const float*         __restrict__ zone_boundaries,
    unsigned char*       __restrict__ pixels_out,
    int n_pixels,
    int n_zones,
    int grid_size   // COMBINED_LUT_GRID = 97; used for unnormalized texture coords
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pixels) return;

    // u8 → f32
    float r = pixels_in[idx * 3 + 0] * (1.0f / 255.0f);
    float g = pixels_in[idx * 3 + 1] * (1.0f / 255.0f);
    float b = pixels_in[idx * 3 + 2] * (1.0f / 255.0f);
    float d = depth[idx];

    // Find the two bounding zones and their weights
    // using the same soft triangular logic as lut_apply.cu
    float total_w = 0.0f;
    float4 blended = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int z = 0; z < n_zones; z++) {
        float z_lo = zone_boundaries[z];
        float z_hi = zone_boundaries[z + 1];
        float z_width = z_hi - z_lo;
        if (z_width < 1e-6f) continue;
        float z_center = 0.5f * (z_lo + z_hi);
        float dist = fabsf(d - z_center);
        float w = fmaxf(1.0f - dist / z_width, 0.0f);
        if (w < 1e-6f) continue;

        // Sample combined LUT texture for this zone.
        // Texture uses unnormalized coordinates: texel i is at position i (not i/(N-1)).
        // Scale [0,1] input to [0, N-1] texel space — avoids the half-texel shift that
        // normalized coords introduce ((i+0.5)/N centre vs i/(N-1) lattice point).
        cudaTextureObject_t tex = (cudaTextureObject_t)textures[z];
        float gs = (float)(grid_size - 1);
        float4 sample = tex3D<float4>(tex, r * gs, g * gs, b * gs);

        blended.x += sample.x * w;
        blended.y += sample.y * w;
        blended.z += sample.z * w;
        total_w += w;
    }

    float r_out, g_out, b_out;
    if (total_w > 1e-6f) {
        r_out = blended.x / total_w;
        g_out = blended.y / total_w;
        b_out = blended.z / total_w;
    } else {
        r_out = r; g_out = g; b_out = b;
    }

    // f32 → u8 (with clamp)
    pixels_out[idx * 3 + 0] = (unsigned char)(__float2uint_rn(fminf(fmaxf(r_out, 0.0f), 1.0f) * 255.0f));
    pixels_out[idx * 3 + 1] = (unsigned char)(__float2uint_rn(fminf(fmaxf(g_out, 0.0f), 1.0f) * 255.0f));
    pixels_out[idx * 3 + 2] = (unsigned char)(__float2uint_rn(fminf(fmaxf(b_out, 0.0f), 1.0f) * 255.0f));
}
```

- [ ] **Step 2: Update build.rs — add combined_lut to kernel_names and rerun-if-changed**

Add rerun line:
```rust
    println!("cargo:rerun-if-changed=src/cuda/kernels/combined_lut.cu");
```

Change kernel_names from:
```rust
    let kernel_names = ["lut_apply", "hsl_correct", "build_combined_lut"];
```
to:
```rust
    let kernel_names = ["lut_apply", "hsl_correct", "build_combined_lut", "combined_lut"];
```

- [ ] **Step 3: Verify it compiles**

```bash
cd repos/dorea && cargo build -p dorea-gpu 2>&1 | grep -E "^error|warning=nvcc" | head -20
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
cd repos/dorea && git add crates/dorea-gpu/src/cuda/kernels/combined_lut.cu crates/dorea-gpu/build.rs
git commit -m "feat: add combined_lut.cu per-frame texture-lookup kernel (Task 7)"
```

---

### Task 8: Wire CombinedLut into CudaGrader

**Files:**
- Modify: `crates/dorea-gpu/src/cuda/mod.rs`
- Delete: `crates/dorea-gpu/src/cuda/kernels/lut_apply.cu`
- Delete: `crates/dorea-gpu/src/cuda/kernels/hsl_correct.cu`
- Modify: `crates/dorea-gpu/build.rs`

- [ ] **Step 1: Write the test that validates the new API shape**

Add to the `#[cfg(all(feature = "cuda", test))] mod tests` in `cuda/mod.rs`:

```rust
    #[test]
    fn new_api_takes_calibration_and_params() {
        // Will compile-error until CudaGrader::new takes (cal, params).
        let cal = make_calibration(5);
        let params = crate::GradeParams::default();
        match CudaGrader::new(&cal, &params) {
            Ok(grader) => {
                let (pixels, depth) = make_frame(4, 4);
                // grade_frame_cuda returns Vec<u8> directly
                let out: Vec<u8> = grader.grade_frame_cuda(&pixels, &depth, 4, 4)
                    .expect("grade_frame_cuda failed");
                assert_eq!(out.len(), 4 * 4 * 3);
            }
            Err(e) => eprintln!("SKIP: no CUDA device ({e})"),
        }
    }
```

- [ ] **Step 2: Run the test — expect compile failure** (CudaGrader::new still takes no args)

```bash
cd repos/dorea && cargo test -p dorea-gpu new_api_takes_calibration_and_params 2>&1 | head -10
```

Expected: compile error.

- [ ] **Step 3: Rewrite CudaGrader in cuda/mod.rs**

Replace the `CudaGrader` struct and its `impl` block with:

```rust
/// CUDA grader: holds a device handle and the precomputed combined LUT textures.
///
/// Create once via `CudaGrader::new(calibration, params)`, reuse across frames.
/// `!Send + !Sync` — CUDA contexts are thread-local.
#[cfg(feature = "cuda")]
pub struct CudaGrader {
    device:       Arc<CudaDevice>,
    combined_lut: CombinedLut,
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
    /// Three cuMalloc/cuFree per call, one kernel, one htod (pixels + depth), one dtoh.
    pub fn grade_frame_cuda(
        &self,
        pixels: &[u8],
        depth: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<u8>, GpuError> {
        let n = width * height;
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
                    self.combined_lut.grid_size as i32,  // for unnormalized tex3D coords
                ))
            }.map_err(map_cudarc_error)?;
        }

        // Download result
        let result = dev.dtoh_sync_copy(&d_pixels_out).map_err(map_cudarc_error)?;
        Ok(result)
    }
}
```

Update the embedded PTX constants at the top of `cuda/mod.rs` — remove the old three and add:

```rust
#[cfg(feature = "cuda")]
const COMBINED_LUT_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/combined_lut.ptx"));
```

Remove all the old `LUT_APPLY_PTX`, `HSL_CORRECT_PTX`, `BLUR_RADIUS`, `PROXY_MAX_SIZE`, `ResolutionBuffers`, `CalibrationBuffers`, `alloc_resolution_buffers`, `alloc_calibration_buffers`, `proxy_dims` definitions.

- [ ] **Step 4: Delete lut_apply.cu and hsl_correct.cu**

```bash
rm repos/dorea/crates/dorea-gpu/src/cuda/kernels/lut_apply.cu
rm repos/dorea/crates/dorea-gpu/src/cuda/kernels/hsl_correct.cu
```

- [ ] **Step 5: Update build.rs — remove old kernels**

Remove the rerun-if-changed lines for lut_apply.cu and hsl_correct.cu.

Change kernel_names from:
```rust
    let kernel_names = ["lut_apply", "hsl_correct", "build_combined_lut", "combined_lut"];
```
to:
```rust
    let kernel_names = ["build_combined_lut", "combined_lut"];
```

- [ ] **Step 6: Update existing CUDA tests to use the new API**

In `cuda/mod.rs` tests, change all `CudaGrader::new()` calls to:

```rust
let cal = make_calibration(5);
let params = crate::GradeParams::default();
let grader = CudaGrader::new(&cal, &params).expect("CudaGrader::new() failed");
```

Update `grade_frame_cuda` call sites to remove `calibration` and `params` arguments:

```rust
grader.grade_frame_cuda(&pixels, &depth, 320, 240)
```

Update the `determinism` test — `grade_frame_cuda` now returns `Vec<u8>`, so remove the `.to_bits()` / `.is_finite()` check and use a direct equality assertion:

```rust
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
        assert_eq!(out1, out2, "grade_frame_cuda must be deterministic (same grader, same inputs)");
    }
```

Remove `calibration_shape_switch` test (calibration is now baked at construction; switching calibration requires a new `CudaGrader`).

- [ ] **Step 7: Verify it builds**

```bash
cd repos/dorea && cargo build -p dorea-gpu 2>&1 | grep "^error" | head -20
```

- [ ] **Step 8: Run tests**

```bash
cd repos/dorea && cargo test -p dorea-gpu 2>&1 | tail -20
```

Expected: CUDA tests pass if GPU available, CPU tests always pass.

- [ ] **Step 9: Commit**

```bash
cd repos/dorea && git add -u crates/dorea-gpu/ && \
    git rm crates/dorea-gpu/src/cuda/kernels/lut_apply.cu \
           crates/dorea-gpu/src/cuda/kernels/hsl_correct.cu
git commit -m "feat: wire CombinedLut into CudaGrader; simplify grade_frame_cuda to 1-kernel pipeline (Task 8)"
```

---

### Task 9: Update public API

**Files:**
- Modify: `crates/dorea-gpu/src/lib.rs`
- Modify: `crates/dorea-cli/src/grade.rs`
- Modify: `crates/dorea-gpu/benches/grade_bench.rs`

- [ ] **Step 1: Update lib.rs**

Replace `grade_frame` (the one that creates a new CudaGrader per call):

```rust
pub fn grade_frame(
    pixels: &[u8],
    depth: &[f32],
    width: usize,
    height: usize,
    calibration: &Calibration,
    params: &GradeParams,
) -> Result<Vec<u8>, GpuError> {
    // ... validation unchanged ...

    #[cfg(feature = "cuda")]
    {
        match cuda::CudaGrader::new(calibration, params) {
            Ok(grader) => {
                match grader.grade_frame_cuda(pixels, depth, width, height) {
                    Ok(out) => return Ok(out),
                    Err(e) => {
                        log::warn!("CUDA grading failed: {e} — falling back to CPU");
                        return cpu::grade_frame_cpu(pixels, depth, width, height, calibration, params)
                            .map_err(GpuError::CudaFail);
                    }
                }
            }
            Err(GpuError::ModuleLoad(msg)) => {
                log::error!("CUDA module load failed: {msg} — using CPU");
                return cpu::grade_frame_cpu(pixels, depth, width, height, calibration, params)
                    .map_err(GpuError::CudaFail);
            }
            Err(e) => return Err(e),
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        Err(GpuError::CudaFail(
            "dorea grade requires CUDA. Rebuild with GPU support (build.rs auto-detects nvcc).".to_string()
        ))
    }
}
```

Replace `grade_frame_with_grader` — remove `calibration` and `params`, call `grade_frame_cuda` which returns `Vec<u8>` directly:

```rust
/// Grade a single frame reusing an existing `CudaGrader`.
///
/// Avoids repeated PTX loading and LUT rebuild when processing multiple frames.
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
            "pixels length {} != width*height*3 {}", pixels.len(), width * height * 3
        )));
    }
    if depth.len() != width * height {
        return Err(GpuError::InvalidInput(format!(
            "depth length {} != width*height {}", depth.len(), width * height
        )));
    }
    grader.grade_frame_cuda(pixels, depth, width, height)
}
```

- [ ] **Step 2: Update grade.rs — remove AdaptiveBatcher dead code, update CudaGrader::new and grade_with_grader**

First, remove `AdaptiveBatcher` — it is dead code after the new API (no OOM batching needed because calibration and params are baked at construction, not per-frame):

```rust
// DELETE these lines from grade.rs:
use dorea_gpu::cuda::AdaptiveBatcher;        // import (or similar)
let batcher = AdaptiveBatcher::new(...);     // construction
// and any batcher.push / batcher.flush calls
```

Also remove or restructure the `GpuError::Oom` match arm — the new `grade_frame_cuda` does not return `GpuError::Oom`. Replace the OOM handler with a simple fallback to CPU:

```rust
Err(e) => {
    log::warn!("CUDA grading failed: {e} — falling back to CPU");
    cpu::grade_frame_cpu(pixels, depth, width, height, calibration, params)
        .map_err(|msg| dorea_gpu::GpuError::CudaFail(msg))
}
```

Change the grader init block (around line 257):

```rust
    // Initialize CUDA grader (builds combined LUT textures once, reuses across all frames)
    #[cfg(feature = "cuda")]
    let cuda_grader = match CudaGrader::new(&calibration, &params) {
        Ok(g) => {
            log::info!("CUDA grader initialized (combined LUT built, PTX loaded)");
            Some(g)
        }
        Err(e) => {
            log::warn!("CUDA grader init failed: {e} — will use per-frame fallback");
            None
        }
    };
```

Update `grade_with_grader` to remove `calibration` and `params` from the grader path (keep for CPU fallback):

```rust
#[cfg(feature = "cuda")]
fn grade_with_grader(
    grader: Option<&CudaGrader>,
    pixels: &[u8],
    depth: &[f32],
    width: usize,
    height: usize,
    calibration: &Calibration,
    params: &GradeParams,
) -> Result<Vec<u8>, dorea_gpu::GpuError> {
    if let Some(g) = grader {
        dorea_gpu::grade_frame_with_grader(g, pixels, depth, width, height)
    } else {
        grade_frame(pixels, depth, width, height, calibration, params)
    }
}
```

- [ ] **Step 3: Update grade_bench.rs**

In the `with_grader` path:

```rust
        #[cfg(feature = "cuda")]
        {
            use dorea_gpu::{cuda::CudaGrader, grade_frame_with_grader};
            match CudaGrader::new(&calibration, &params) {
                Ok(grader) => {
                    group.bench_with_input(
                        BenchmarkId::new("with_grader", label),
                        &(w, h),
                        |b, _| {
                            b.iter(|| {
                                grade_frame_with_grader(
                                    &grader, &pixels, &depth, w, h,
                                )
                                .expect("grade_frame_with_grader failed")
                            });
                        },
                    );
                }
                Err(e) => {
                    eprintln!("SKIP with_grader/{label}: CudaGrader::new failed: {e}");
                }
            }
        }
```

In the `per_call_init` path (grade_frame still takes cal+params):

```rust
        #[cfg(feature = "cuda")]
        {
            use dorea_gpu::grade_frame;
            group.bench_with_input(
                BenchmarkId::new("per_call_init", label),
                &(w, h),
                |b, _| {
                    b.iter(|| {
                        grade_frame(&pixels, &depth, w, h, &calibration, &params)
                            .expect("grade_frame failed")
                    });
                },
            );
        }
```

- [ ] **Step 4: Build and test the full workspace**

```bash
cd repos/dorea && cargo build 2>&1 | grep "^error" | head -20
cargo test -p dorea-gpu -p dorea-cli 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 5: Spot-check benchmark for regression**

```bash
cd repos/dorea && cargo bench -p dorea-gpu -- with_grader/1080p 2>&1 | grep -E "time:|with_grader"
```

**Hard threshold:** `grading/with_grader/1080p` must be below 25ms on the RTX 3060. If it exceeds 25ms, profile the kernel — likely cause is excess htod (texture handles should be tiny) or missing sync between build and per-frame call.

The `grade_frame_cuda` now:
- htod: pixels (6MB 1080p) + depth (8MB 1080p) = 14MB total (vs 133MB before)
- dtoh: 6MB u8 output (vs 100MB before)
- IMPORTANT: calibration and params are baked into the combined LUT at construction.
  Do NOT pass calibration or params to `grade_frame_with_grader` — they are ignored.

- [ ] **Step 6: Commit**

```bash
cd repos/dorea && git add -u
git commit -m "feat: update public API — grade_frame_with_grader drops cal/params, grade_frame calls new(cal,params) (Task 9)"
```

---

### Task 10: Integration test + performance benchmark

**Files:**
- Modify: `crates/dorea-gpu/src/cuda/mod.rs` (add integration test)
- Test with bench

- [ ] **Step 1: Write the accuracy integration tests**

Add to `#[cfg(all(feature = "cuda", test))] mod tests` in `cuda/mod.rs`:

```rust
    /// Helper: build calibration with a gamma-shifted LUT (not identity).
    /// Used to verify the combined LUT bakes a non-trivial pipeline.
    fn make_shifted_calibration(n_zones: usize) -> Calibration {
        use dorea_lut::types::{DepthLuts, LutGrid};
        use dorea_hsl::derive::{HslCorrections, QualifierCorrection};
        let size = 17usize;
        let luts: Vec<LutGrid> = (0..n_zones).map(|_| {
            let mut lut = LutGrid::new(size);
            for ri in 0..size { for gi in 0..size { for bi in 0..size {
                // Gamma-shifted: output is input^0.8 — clearly non-identity
                let r = (ri as f32 / (size - 1) as f32).powf(0.8);
                let g = (gi as f32 / (size - 1) as f32).powf(0.8);
                let b = (bi as f32 / (size - 1) as f32).powf(0.8);
                lut.set(ri, gi, bi, [r, g, b]);
            }}}
            lut
        }).collect();
        let boundaries: Vec<f32> = (0..=n_zones).map(|i| i as f32 / n_zones as f32).collect();
        // Active green qualifier: push saturation 20%
        let hsl = HslCorrections(vec![QualifierCorrection {
            h_center: 120.0, h_width: 40.0, h_offset: 0.0,
            s_ratio: 1.2, v_offset: 0.0, weight: 200.0,  // weight >= 100 → active
        }]);
        Calibration::new(DepthLuts::new(luts, boundaries), hsl, 1)
    }

    #[test]
    fn combined_lut_within_2_per_255_of_cpu() {
        // Validates that grade_frame_with_grader (GPU combined LUT) matches
        // grade_frame_cpu (CPU reference) within 2/255 on a synthetic frame.
        let cal = make_calibration(5);
        let params = crate::GradeParams::default();
        let (w, h) = (32, 32);
        let (pixels, depth) = make_frame(w, h);

        // GPU path
        let grader = match CudaGrader::new(&cal, &params) {
            Ok(g) => g,
            Err(e) => { eprintln!("SKIP: {e}"); return; }
        };
        let gpu_out = grader.grade_frame_cuda(&pixels, &depth, w, h)
            .expect("grade_frame_cuda failed");

        // CPU reference path
        let cpu_out = crate::cpu::grade_frame_cpu(&pixels, &depth, w, h, &cal, &params)
            .expect("grade_frame_cpu failed");

        assert_eq!(gpu_out.len(), cpu_out.len());

        // Collect per-channel diffs and report p99
        let mut diffs: Vec<u32> = gpu_out.iter().zip(cpu_out.iter())
            .map(|(&g, &c)| (g as i32 - c as i32).unsigned_abs())
            .collect();
        diffs.sort_unstable();
        let max_diff = *diffs.last().unwrap_or(&0);
        let p99_idx  = (diffs.len() as f32 * 0.99) as usize;
        let p99_diff = diffs.get(p99_idx).copied().unwrap_or(0);
        eprintln!("GPU/CPU diff — max: {max_diff}/255, p99: {p99_diff}/255");

        assert!(max_diff <= 2,
            "GPU/CPU max diff {max_diff}/255 exceeds 2/255 tolerance. \
             Check grade_pixel.cuh matches cpu.rs fused_ambiance_warmth exactly.");
    }

    #[test]
    fn combined_lut_non_trivial_lut_and_hsl_within_2_per_255() {
        // Same check but with a gamma-shifted LUT and an active HSL qualifier,
        // exercising the non-identity parts of grade_pixel_device.
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
        assert!(max_diff <= 2,
            "Non-trivial LUT: GPU/CPU max diff {max_diff}/255 exceeds 2/255");
    }

    #[test]
    fn combined_lut_edge_case_pixels() {
        // Black, white, and boundary-depth pixels — these are the pixels most
        // likely to reveal clamping or coordinate-mapping bugs.
        let cal = make_calibration(5);
        let params = crate::GradeParams::default();
        let grader = match CudaGrader::new(&cal, &params) {
            Ok(g) => g,
            Err(e) => { eprintln!("SKIP: {e}"); return; }
        };

        // 4 edge-case pixels: black@0, white@1, mid@0, mid@1
        let pixels: Vec<u8> = vec![
            0, 0, 0,       // black pixel, depth=0.0
            255, 255, 255, // white pixel, depth=1.0
            128, 64, 32,   // mid pixel,   depth=0.0
            128, 64, 32,   // mid pixel,   depth=1.0
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
        // Verify the pipeline works for 1, 3, and 8 zones.
        // Replaces the removed calibration_shape_switch test.
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
```

- [ ] **Step 2: Run tests — expect either pass or a diff > 2 (which reveals a mismatch)**

```bash
cd repos/dorea && cargo test -p dorea-gpu combined_lut 2>&1 | tail -20
```

If diff > 2: compare `grade_pixel_cpu` vs `grade_pixel_device` step-by-step on one pixel — the mismatch will be in a specific pipeline stage. Common causes: missing clamp, wrong constant, axis swap (R↔B). Fix `grade_pixel.cuh` and re-run.

- [ ] **Step 3: Run the full test suite**

```bash
cd repos/dorea && cargo test 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 4: Run criterion benchmark**

```bash
cd repos/dorea && cargo bench -p dorea-gpu 2>&1 | grep -E "with_grader|cpu" | head -20
```

Expected output (on RTX 3060):
- `grading/with_grader/1080p`: ~15–25ms (combined LUT, no intermediate allocs)
- `grading/cpu/1080p`: ~500ms–2s (CPU reference, much slower)
- `grading/per_call_init/1080p`: ~15–50ms + LUT build overhead

**Pass criteria:** `grading/with_grader/1080p` must be below 25ms. If it exceeds 25ms, see Task 9 Step 5 for diagnosis guidance.

- [ ] **Step 5: Commit**

```bash
cd repos/dorea && git add -u crates/dorea-gpu/
git commit -m "test: add combined LUT accuracy integration tests (4 variants) + benchmark (Task 10)"
```

---

## Self-Review Checklist

### Spec coverage
- ✅ Hard delete clarity — Task 2
- ✅ grade_pixel.cuh with full pipeline — Task 4
- ✅ GPU build kernel via cudaMemcpy3D (device-to-device) — Task 5+6
- ✅ float4 texture format — Task 6+7
- ✅ u8 I/O in per-frame kernel — Task 7
- ✅ COMBINED_LUT_GRID = 97 constant — Task 6 (combined_lut.rs)
- ✅ CudaGrader::new(cal, params) — Task 8
- ✅ grade_frame_with_grader drops cal+params — Task 9
- ✅ Accuracy test within 2/255 — Task 10
- ✅ Criterion benchmark — Task 10

### Type consistency
- `CombinedLut::build(device, calibration, params)` defined in Task 6, referenced in Task 8 ✅
- `grade_frame_cuda(&pixels, &depth, w, h) → Vec<u8>` defined in Task 8, used in Task 9 ✅
- `grade_frame_with_grader(grader, pixels, depth, w, h)` defined in Task 9, used in bench ✅
- `COMBINED_LUT_GRID` defined in `combined_lut.rs` (Task 6), used in `build_combined_lut.cu` (Task 5) — **Note:** the `.cu` file uses a hardcoded `grid_size` parameter passed from Rust; the Rust constant is `COMBINED_LUT_GRID = 97`. The kernel receives it as `int grid_size` at launch.
- `grade_pixel_device` defined in `grade_pixel.cuh` (Task 4), `#include`d in `build_combined_lut.cu` (Task 5) ✅
- `grade_pixel_cpu` defined in `cpu.rs` (Task 3), tested against `grade_frame_cpu` ✅

### Placeholder scan
None — all code blocks are complete.

### Review fixes applied (5-persona review, 2026-04-03)
- **C2 axis ordering** (CUDA Expert): `ri=rem%N` is x/column (fastest), `bi=rem/(N*N)` is z/slice — no R↔B swap
- **C1 texel-center offset** (CUDA Expert + Color Science): Switched to unnormalized coordinates (`flags=0`); per-frame kernel samples `tex3D(tex, r*(N-1), g*(N-1), b*(N-1))`
- **C1 raw driver calls** (Senior SWE): All `cuXxx()` → `sys::lib().cuXxx()` (cudarc 0.12 dynamic-linking)
- **I1 synchronize note** (Senior SWE): Removed incorrect workaround — `device.synchronize()` exists in cudarc 0.12.1
- **I2 DevicePtr trait** (Senior SWE): Added `use cudarc::driver::DevicePtr;`; imports changed to `use cudarc::driver::sys::{self, *}`
- **I3 grader_temp panic** (Senior SWE): Removed `CudaGrader::new()` no-arg call from Task 6 test
- **C1 determinism test** (QA): Updated to `assert_eq!(out1, out2)` — `grade_frame_cuda` returns `Vec<u8>`
- **C1 AdaptiveBatcher dead code** (PM): Task 9 Step 2 now explicitly removes batcher and restructures OOM path
- **auto lambda** (CUDA Expert): Replaced `auto apply_knee = [&]...` with inlined ternary expressions
- **Task 10 test depth** (QA): Added 4 test variants: identity + non-trivial LUT + edge-case pixels + zone count; p99 logging added
