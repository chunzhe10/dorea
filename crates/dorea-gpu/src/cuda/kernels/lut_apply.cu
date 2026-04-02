/**
 * lut_apply.cu — Fused depth-stratified LUT application kernel.
 *
 * Each thread processes one pixel. Performs:
 * 1. Soft zone weight computation from depth value
 * 2. Trilinear interpolation within each zone's LUT
 * 3. Weighted blend across zones
 *
 * Grid: 1D, blockDim.x = 256, gridDim.x = ceil(n_pixels / 256)
 *
 * Parameters:
 *   pixels_in  — float3 array [n_pixels], RGB in [0,1]
 *   depth      — float array [n_pixels], depth in [0,1]
 *   luts       — float array [n_zones * lut_size^3 * 3], zone LUTs row-major RGBO
 *   zone_boundaries — float array [n_zones+1], adaptive zone edges
 *   pixels_out — float3 array [n_pixels], output RGB in [0,1]
 *   n_pixels   — number of pixels
 *   lut_size   — LUT grid size (33)
 *   n_zones    — number of depth zones (5)
 */

#include <cuda_runtime.h>
#include <math.h>

__device__ float3 trilinear_sample(
    const float* __restrict__ lut,
    int lut_size,
    float r, float g, float b
) {
    float s = (float)(lut_size - 1);
    float fr_f = r * s;
    float fg_f = g * s;
    float fb_f = b * s;

    int i0 = (int)fr_f;
    int j0 = (int)fg_f;
    int k0 = (int)fb_f;

    // Clamp to valid range
    i0 = min(max(i0, 0), lut_size - 2);
    j0 = min(max(j0, 0), lut_size - 2);
    k0 = min(max(k0, 0), lut_size - 2);

    float fr = fr_f - (float)i0;
    float fg = fg_f - (float)j0;
    float fb = fb_f - (float)k0;

    // Clamp fractions
    fr = min(max(fr, 0.0f), 1.0f);
    fg = min(max(fg, 0.0f), 1.0f);
    fb = min(max(fb, 0.0f), 1.0f);

    // 8-corner trilinear interpolation
    auto idx = [&](int di, int dj, int dk) -> int {
        return ((i0+di) * lut_size * lut_size + (j0+dj) * lut_size + (k0+dk)) * 3;
    };

    float3 result;
    for (int c = 0; c < 3; c++) {
        float v000 = lut[idx(0,0,0)+c];
        float v001 = lut[idx(0,0,1)+c];
        float v010 = lut[idx(0,1,0)+c];
        float v011 = lut[idx(0,1,1)+c];
        float v100 = lut[idx(1,0,0)+c];
        float v101 = lut[idx(1,0,1)+c];
        float v110 = lut[idx(1,1,0)+c];
        float v111 = lut[idx(1,1,1)+c];

        float v = v000*(1-fr)*(1-fg)*(1-fb) +
                  v100*fr*(1-fg)*(1-fb) +
                  v010*(1-fr)*fg*(1-fb) +
                  v110*fr*fg*(1-fb) +
                  v001*(1-fr)*(1-fg)*fb +
                  v101*fr*(1-fg)*fb +
                  v011*(1-fr)*fg*fb +
                  v111*fr*fg*fb;

        if (c == 0) result.x = v;
        else if (c == 1) result.y = v;
        else result.z = v;
    }
    return result;
}

extern "C"
__global__ void lut_apply_kernel(
    const float* __restrict__ pixels_in,
    const float* __restrict__ depth,
    const float* __restrict__ luts,
    const float* __restrict__ zone_boundaries,
    float* __restrict__ pixels_out,
    int n_pixels,
    int lut_size,
    int n_zones
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pixels) return;

    float r = pixels_in[idx * 3 + 0];
    float g = pixels_in[idx * 3 + 1];
    float b = pixels_in[idx * 3 + 2];
    float d = depth[idx];

    // Compute soft zone weights: triangular kernel, width = zone_width
    float total_w = 0.0f;
    float3 blended = {0.0f, 0.0f, 0.0f};

    int lut_stride = lut_size * lut_size * lut_size * 3;

    for (int z = 0; z < n_zones; z++) {
        float z_lo = zone_boundaries[z];
        float z_hi = zone_boundaries[z + 1];
        float z_center = 0.5f * (z_lo + z_hi);
        float z_width = z_hi - z_lo;
        if (z_width < 1e-6f) continue;

        float dist = fabsf(d - z_center);
        float w = fmaxf(1.0f - dist / z_width, 0.0f);
        if (w < 1e-6f) continue;

        const float* zone_lut = luts + z * lut_stride;
        float3 lut_out = trilinear_sample(zone_lut, lut_size, r, g, b);

        blended.x += lut_out.x * w;
        blended.y += lut_out.y * w;
        blended.z += lut_out.z * w;
        total_w += w;
    }

    if (total_w > 1e-6f) {
        pixels_out[idx * 3 + 0] = blended.x / total_w;
        pixels_out[idx * 3 + 1] = blended.y / total_w;
        pixels_out[idx * 3 + 2] = blended.z / total_w;
    } else {
        // No zone matched — pass through
        pixels_out[idx * 3 + 0] = r;
        pixels_out[idx * 3 + 1] = g;
        pixels_out[idx * 3 + 2] = b;
    }
}
