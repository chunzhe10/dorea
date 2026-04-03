# CudaGrader Buffer Pre-Allocation Design

**Date:** 2026-04-03
**Status:** Approved

## Problem

`grade_frame_cuda` performs 14 `cudaMalloc`-backed operations per frame:
- 8 × `htod_sync_copy` (alloc + async copy + stream sync)
- 6 × `alloc_zeros` (explicit `cudaMalloc`)
- 1 × `dtoh_sync_copy` (result download)

Each `cudaMalloc` costs ~3–8 ms on the RTX 3060 Laptop GPU. The actual kernel
compute (LUT trilinear + HSL correct + 3-pass clarity blur) is <1 ms total at
1080p. The result: GPU SM utilisation is 10–20% with 1–3% memory bandwidth
during grading — the GPU idles through allocations rather than computing.

Measured throughput before this change: ~83 ms/frame at 1080p (benchmark result
2026-04-03). Expected after: ~6–10 ms/frame. `cudaMalloc` overhead (~60–70 ms)
is eliminated; 8 `htod_sync_copy_into` calls each still call `stream.synchronize()`
contributing ~3–6 ms total.

## Decision

Pre-allocate all device buffers in `CudaGrader` and reuse them across frames.
Two cached buffer sets, each invalidated independently:

- **`ResolutionBuffers`** — keyed by `(width, height)`. Holds all pixel-count-
  scaled device slices (8 total). Reallocated when resolution changes.
- **`CalibrationBuffers`** — keyed by `(n_zones, lut_size)`. Holds the LUT and
  HSL parameter device slices (6 total). Reallocated when calibration shape
  changes.

Both caches stored as `RefCell<Option<...>>` fields on `CudaGrader` for interior
mutability. Safe because `CudaGrader` is `!Send + !Sync`.

## Rejected Alternatives

**Two-slot resolution cache** — Keeps buffers for the two most-recent resolutions
simultaneously. Eliminates realloc even when alternating between 1080p and 4K.
Rejected: holds ~440 MB of VRAM for a case the user confirmed is rare. Not worth
the complexity.

**Max-resolution static pre-allocation** — Allocate once for the largest known
resolution. Rejected: requires knowing max resolution at construction time and
permanently wastes ~250 MB when processing 1080p-only sessions.

## Architecture

### New structs in `crates/dorea-gpu/src/cuda/mod.rs`

```
ResolutionBuffers {
    width, height, proxy_w, proxy_h: usize,
    d_rgb_in:        CudaSlice<f32>,  // n × 3  (input pixels, f32)
    d_depth:         CudaSlice<f32>,  // n      (depth map)
    d_rgb_after_lut: CudaSlice<f32>,  // n × 3  (LUT → HSL stage boundary)
    d_rgb_after_hsl: CudaSlice<f32>,  // n × 3  (HSL → clarity stage boundary)
    d_proxy_l:       CudaSlice<f32>,  // proxy_n (clarity: downsampled L)
    d_blur_a:        CudaSlice<f32>,  // proxy_n (clarity: ping buffer)
    d_blur_b:        CudaSlice<f32>,  // proxy_n (clarity: pong buffer)
    d_rgb_out:       CudaSlice<f32>,  // n × 3  (final output)
}

CalibrationBuffers {
    n_zones, lut_size: usize,
    d_luts:       CudaSlice<f32>,  // n_zones × lut_size³ × 3
    d_boundaries: CudaSlice<f32>,  // n_zones + 1
    d_h_offsets:  CudaSlice<f32>,  // [f32; 6]
    d_s_ratios:   CudaSlice<f32>,  // [f32; 6]
    d_v_offsets:  CudaSlice<f32>,  // [f32; 6]
    d_weights:    CudaSlice<f32>,  // [f32; 6]
}
```

### `CudaGrader` struct additions

```rust
res_bufs: RefCell<Option<ResolutionBuffers>>,
cal_bufs: RefCell<Option<CalibrationBuffers>>,
```

### Per-frame flow in `grade_frame_cuda`

1. Borrow `res_bufs` mutably; if `(w, h)` differs from cached, drop and
   reallocate all 8 resolution slices.
2. Borrow `cal_bufs` mutably; if `(n_zones, lut_size)` differs, drop and
   reallocate all 6 calibration slices.
3. Copy input data into pre-allocated slices using `htod_sync_copy_into` (no
   allocation; copies into existing device slice then calls `stream.synchronize()`).
   8 sync barriers remain per frame — these take ~3–6 ms total, not zero.
4. Launch LUT → HSL → clarity kernels (unchanged).
5. `dtoh_sync_copy` from `d_rgb_out` to return the result.

**Steady state: 0 `cudaMalloc` calls per frame.**

### Public API

No changes. `grade_frame_cuda(&self, ...)` signature is unchanged.
`grade_frame_with_grader` and `grade_frame` in `lib.rs` are unchanged.

## Testing

- **Resolution-switch test**: call `grade_frame_cuda` at 720p, then 1080p, then
  720p again. Assert no error and pixel output is not all-zero at each step.
- **Calibration-switch test**: call twice with calibrations of different
  `(n_zones, lut_size)`. Assert correct output on both calls.
- **Benchmark regression**: `cargo bench -p dorea-gpu -- grading/cpu` must not
  regress (CPU path is unaffected). `grading/with_grader` throughput must
  improve vs. the 83 ms/frame baseline measured 2026-04-03.

## VRAM Impact

| Resolution | ResolutionBuffers | CalibrationBuffers | Total held |
|------------|-------------------|--------------------|------------|
| 720p       | ~38 MB            | ~2 MB              | ~40 MB     |
| 1080p      | ~87 MB            | ~2 MB              | ~89 MB     |
| 4K         | ~345 MB           | ~2 MB              | ~347 MB    |

RTX 3060 Laptop has 6 144 MB VRAM. Even at 4K, buffer overhead is ~6% of total
VRAM. No impact on the 6 GB constraint.
