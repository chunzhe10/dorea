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
