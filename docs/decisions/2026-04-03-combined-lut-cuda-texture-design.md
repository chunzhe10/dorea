# Combined LUT + CUDA 3D Texture Grading — Design (2026-04-03)

## Summary

Replace the 3-kernel GPU grading pipeline (lut_apply → hsl_correct → clarity) with a
single precomputed combined LUT sampled via CUDA 3D texture objects. Clarity is hard-deleted.
Expected frame time at 4K: ~15ms vs 332ms current (~22× speedup on grading).

## Motivation

- Clarity is being permanently removed (Maxine will handle sharpening when implemented)
- With clarity gone, the remaining pipeline (LUT + HSL + ambiance) is purely per-pixel with
  no spatial dependencies — it can be precomputed as a 3D color lookup table
- CUDA 3D textures provide hardware trilinear interpolation and dedicated L1 texture cache,
  making the combined LUT faster than the multi-kernel pipeline even at full 4K resolution
- Eliminating intermediate buffers drops per-frame allocations from 14 to 3

## Grid Size: 97³

Empirically anchored on issue #25 validation (65³ measured: p99=0.81/255, max=2.61/255).
Scaling by (64/96)² factor:

| Grid | VRAM (5 zones) | p99 err/255 | p99 10-bit LSBs | Verdict |
|------|---------------|-------------|-----------------|---------|
| 33³  | 2MB           | 3.24        | 13.0            | FAILS 10-bit |
| 65³  | 16MB          | 0.81        | 3.2             | Risky 10-bit |
| **97³**  | **55MB**  | **0.36**    | **1.4**         | **Borderline-safe 10-bit** |
| 129³ | 129MB         | 0.20        | 0.8             | Safe 10-bit |

97³ chosen: safe for current 8-bit footage, forward-compatible with planned 10-bit switch.
Grid size exposed as named constant `COMBINED_LUT_GRID = 97` — bump to 129 if empirical
10-bit testing reveals banding. 257³ excluded: 1GB VRAM conflicts with Maxine budget.

## Depth Bands: Calibration Zones (5 adaptive)

Reuse existing `calibration.depth_luts` zone boundaries (adaptive, fitted per clip).
Same soft triangular blend logic as current `lut_apply.cu`. Two zone lookups + lerp per
pixel at runtime. No resampling step needed.

## Architecture

### New files

```
crates/dorea-gpu/src/cuda/kernels/grade_pixel.cuh      — __device__ grade_pixel_device()
crates/dorea-gpu/src/cuda/kernels/build_combined_lut.cu — build kernel (startup, once)
crates/dorea-gpu/src/cuda/kernels/combined_lut.cu       — per-frame lookup kernel
crates/dorea-gpu/src/cuda/combined_lut.rs               — CombinedLut struct + build()
```

### Deleted files

```
crates/dorea-gpu/src/cuda/kernels/lut_apply.cu
crates/dorea-gpu/src/cuda/kernels/hsl_correct.cu
crates/dorea-gpu/src/cuda/kernels/clarity.cu
```

### Modified files

```
crates/dorea-gpu/build.rs              — remove clarity PTX compilation
crates/dorea-gpu/src/cuda/mod.rs       — replace 3-kernel pipeline with combined_lut call
crates/dorea-gpu/src/cpu.rs            — remove skip_clarity, add grade_pixel_cpu (test baseline)
crates/dorea-gpu/src/lib.rs            — update public API signatures
crates/dorea-cli/src/grade.rs          — update CudaGrader::new() call site
```

## Component Design

### `grade_pixel.cuh` — shared device function

```c
__device__ float3 grade_pixel_device(
    float r, float g, float b, float depth,
    const float* luts, const float* zone_boundaries,
    int lut_size, int n_zones,
    const float* h_offsets, const float* s_ratios,
    const float* v_offsets, const float* weights,
    float warmth, float strength, float contrast
);
```

Runs in order: LUT apply (soft zone blend + trilinear) → HSL correct (6 qualifiers) →
depth-aware ambiance → warmth → strength blend. Mirrors existing kernel logic.

### `build_combined_lut.cu` — GPU LUT build (startup)

- 1D grid: one thread per `(grid_point_index, zone)` pair, ~4.5M total
- Thread computes `(r,g,b) = (i/(N-1), j/(N-1), k/(N-1))` from flat index
- Calls `grade_pixel_device` at `depth = zone_center[zone]`
- Writes `float4(r', g', b', 0.0)` to output device buffer [n_zones × N³]
- Build time: <1ms GPU (vs ~34ms CPU rayon alternative)
- Output stays on device — copied to `cudaArray3D` via `cudaMemcpy3D` without CPU roundtrip

### `CombinedLut` (combined_lut.rs)

```rust
pub(crate) struct CombinedLut {
    arrays:          [cudaArray_t; MAX_ZONES],
    textures:        [cudaTextureObject_t; MAX_ZONES],
    n_zones:         usize,
    grid_size:       usize,           // COMBINED_LUT_GRID = 97
    zone_boundaries: Vec<f32>,
}
impl Drop for CombinedLut { /* cudaDestroyTextureObject + cudaFreeArray */ }
```

Build sequence in `CombinedLut::build(device, calibration, params)`:
1. Upload calibration data to device (depth luts + HSL params, ~2.1MB)
2. Launch `build_combined_lut_kernel` → 55MB device buffer
3. Per zone: `cudaMemcpy3D` device→`cudaArray3D` (device-to-device, no PCIe)
4. Create `cudaTextureObject_t`: `filterMode=Linear`, `addressMode=Clamp`,
   `normalizedCoords=true`, channel format `float4`
5. Free 55MB device build buffer

### `combined_lut.cu` — per-frame kernel

```c
__global__ void combined_lut_kernel(
    const uint8_t* pixels_in,    // u8 RGB interleaved [n * 3]
    const float*   depth,         // f32 [n]
    cudaTextureObject_t tex[MAX_ZONES],
    const float* zone_boundaries,
    uint8_t* pixels_out,          // u8 RGB interleaved [n * 3]
    int n_pixels, int n_zones
)
// Per pixel:
//   u8→f32 in-kernel → find 2 bounding zones → tex3D<float4> × 2 → lerp → f32→u8
```

1D grid, 256 threads/block. No shared memory (texture cache handles locality).
`float4` texture: all 3 output channels from a single hardware-interpolated fetch.

### `CudaGrader` API changes

```rust
// BEFORE
pub fn new() -> Result<Self, GpuError>
pub fn grade_frame_cuda(&self, pixels, depth, w, h, calibration, params) -> Result<Vec<u8>>

// AFTER
pub fn new(calibration: &Calibration, params: &GradeParams) -> Result<Self, GpuError>
pub fn grade_frame_cuda(&self, pixels: &[u8], depth: &[f32], w: usize, h: usize)
    -> Result<Vec<u8>, GpuError>
```

Public `grade_frame_with_grader` in `lib.rs` mirrors this — calibration + params removed
from per-call signature, baked into textures at construction.

### Per-frame data movement at 4K

| Step | Before | After |
|------|--------|-------|
| CPU u8→f32 expand | ~10ms | 0ms (in-kernel) |
| htod (pixels + depth) | 133MB / ~9ms | 58MB / ~4ms |
| GPU kernels | 3 kernels + 5 intermediate allocs | 1 kernel, 0 intermediate |
| dtoh | 100MB / ~7ms | 25MB / ~2ms |
| Per-frame allocs | 14 cuMalloc/cuFree | 3 cuMalloc/cuFree |
| **Estimated total** | **~332ms** | **~15ms** |

## Testing

| Test | Validates |
|------|-----------|
| `grade_pixel_cpu` matches `grade_frame_cpu` on 1×1 synthetic frame | CPU extraction correct |
| GPU build vs CPU build within 1/255 on 1000 random samples | `grade_pixel_device` matches CPU |
| `grade_frame_with_grader` vs `grade_frame_cpu` within 2/255 | Combined LUT accuracy |
| Existing `frame_mse_*`, `lerp_depth_*` tests | No regression in grade.rs helpers |
| Criterion bench `with_grader/1080p` and `with_grader/4K` | Performance validates ~15ms |

## Decisions

- **Hard delete clarity**: no flag, no fallback. Maxine will handle sharpening when implemented.
- **97³ grid**: forward-compatible with 10-bit; `COMBINED_LUT_GRID` constant for easy bump to 129.
- **GPU LUT build**: device→device via `cudaMemcpy3D`; avoids 55MB PCIe roundtrip.
- **u8 I/O in kernel**: eliminates CPU f32 expansion and 75MB htod overhead per frame.
- **float4 texture format**: single `tex3D<float4>` fetch returns all 3 channels.
- **Calibration zones reused**: no resampling, no additional approximation error at zone boundaries.
- **Calibration + params baked at construction**: `CudaGrader::new(cal, params)` — not per-call.
