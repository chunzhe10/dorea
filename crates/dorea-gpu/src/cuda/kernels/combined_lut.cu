/**
 * combined_lut.cu — Per-frame kernel: sample combined LUT textures.
 *
 * 1D grid, 256 threads/block. No shared memory (texture cache handles locality).
 *
 * Per pixel:
 *   1. u8 -> f32 (in-kernel, no CPU expansion)
 *   2. Find bounding depth zones (soft triangular blend)
 *   3. tex3D<float4> per zone (hardware trilinear interpolation)
 *   4. Depth-weighted blend of zone results
 *   5. f32 -> u8 output
 *
 * Parameters:
 *   pixels_in       — uint8 RGB interleaved [n_pixels * 3]
 *   depth           — float32 [n_pixels], in [0,1]
 *   textures        — CUtexObject array (device pointer to n_zones handles)
 *   zone_boundaries — float [n_zones+1]
 *   pixels_out      — uint8 RGB interleaved [n_pixels * 3]
 *   n_pixels        — pixel count
 *   n_zones         — number of zones
 *   grid_size       — COMBINED_LUT_GRID = 97; used for unnormalized texture coords
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

    // u8 -> f32
    float r = pixels_in[idx * 3 + 0] * (1.0f / 255.0f);
    float g = pixels_in[idx * 3 + 1] * (1.0f / 255.0f);
    float b = pixels_in[idx * 3 + 2] * (1.0f / 255.0f);

    float d = depth[idx];

    // Find the bounding zones and their weights
    // using the same soft triangular logic as the build kernel
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
        // Scale [0,1] input to [0, N-1] texel space.
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

    // f32 -> u8 (with clamp)
    pixels_out[idx * 3 + 0] = (unsigned char)(__float2uint_rn(fminf(fmaxf(r_out, 0.0f), 1.0f) * 255.0f));
    pixels_out[idx * 3 + 1] = (unsigned char)(__float2uint_rn(fminf(fmaxf(g_out, 0.0f), 1.0f) * 255.0f));
    pixels_out[idx * 3 + 2] = (unsigned char)(__float2uint_rn(fminf(fmaxf(b_out, 0.0f), 1.0f) * 255.0f));
}
