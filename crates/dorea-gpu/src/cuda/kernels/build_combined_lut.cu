/**
 * build_combined_lut.cu — GPU kernel to precompute the combined LUT.
 *
 * 1D grid: one thread per (zone × grid_point) pair.
 * Thread count = runtime_n_zones × N³  where N = COMBINED_LUT_GRID (97).
 *
 * Each thread:
 *   1. Decodes flat index → (zone, ri, gi, bi) using RUNTIME zones for output indexing
 *   2. Computes (r,g,b) = (ri/(N-1), gi/(N-1), bi/(N-1))
 *   3. Calls grade_pixel_device() at depth = runtime zone center, using BASE LUT for lookup
 *   4. Writes float4(r', g', b', 0.0) to output buffer
 *
 * Output layout: [runtime_n_zones][N][N][N] float4
 *   Stride: zone * N*N*N * 4 floats
 *
 * Parameters:
 *   output                 — float4 [runtime_n_zones × N³] device buffer (pre-allocated)
 *   base_luts              — float [base_n_zones × lut_size³ × 3]  (32-zone calibration LUT)
 *   base_zone_boundaries   — float [base_n_zones + 1]               (32-zone boundaries)
 *   base_n_zones           — int                                     (32)
 *   runtime_zone_boundaries— float [runtime_n_zones + 1]            (8 per-keyframe zones)
 *   runtime_n_zones        — int                                     (8)
 *   h_offsets              — float [6]
 *   s_ratios               — float [6]
 *   v_offsets              — float [6]
 *   weights                — float [6]
 *   warmth/strength/contrast — scalar GradeParams
 *   grid_size              — N (= 97)
 *   lut_size               — base LUT grid size (e.g. 33)
 *   total_threads          — runtime_n_zones * N^3 (to guard OOB)
 */

#include <cuda_runtime.h>
#include "grade_pixel.cuh"

extern "C"
__global__ void build_combined_lut_kernel(
    float4* __restrict__ output,
    const float* __restrict__ base_luts,
    const float* __restrict__ base_zone_boundaries,
    int base_n_zones,
    const float* __restrict__ runtime_zone_boundaries,
    int runtime_n_zones,
    const float* __restrict__ h_offsets,
    const float* __restrict__ s_ratios,
    const float* __restrict__ v_offsets,
    const float* __restrict__ weights,
    float warmth, float strength, float contrast,
    int grid_size, int lut_size,
    int total_threads
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_threads) return;

    int N = grid_size;
    int N3 = N * N * N;

    int zone = idx / N3;
    if (zone >= runtime_n_zones) return; // belt-and-suspenders: total_threads should prevent this
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

    // Depth = center of this RUNTIME zone (8 adaptive per-keyframe zones)
    float z_lo = runtime_zone_boundaries[zone];
    float z_hi = runtime_zone_boundaries[zone + 1];
    float depth = 0.5f * (z_lo + z_hi);

    // grade_pixel_device reads from the 32-zone BASE LUT for color-science lookup
    float3 graded = grade_pixel_device(
        r, g, b, depth,
        base_luts, base_zone_boundaries,
        lut_size, base_n_zones,
        h_offsets, s_ratios, v_offsets, weights,
        warmth, strength, contrast
    );

    // Write float4: hardware requires 4-channel format for 3D texture
    output[idx] = make_float4(graded.x, graded.y, graded.z, 0.0f);
}
