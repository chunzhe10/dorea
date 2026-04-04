/**
 * combined_lut.cu — Per-frame kernel: sample combined LUT textures with dual-texture blending.
 *
 * 1D grid, 256 threads/block.
 *
 * Per pixel:
 *   1. u8 -> f32
 *   2. Soft triangular zone blend from texture set A (active keyframe)
 *   3. If blend_t > 0: soft triangular zone blend from texture set B (next keyframe)
 *   4. Temporal lerp: output = A * (1 - blend_t) + B * blend_t
 *   5. f32 -> u8 output
 *
 * blend_t = 0.0 on keyframes (skips set B entirely), 0.0–1.0 between keyframes.
 */

#include <cuda_runtime.h>

extern "C"
__global__ void combined_lut_kernel(
    const unsigned char* __restrict__ pixels_in,
    const float*         __restrict__ depth,
    // Texture set A (active / "before" keyframe)
    const unsigned long long* __restrict__ textures_a,
    const float*              __restrict__ zone_boundaries_a,
    // Texture set B ("after" keyframe)
    const unsigned long long* __restrict__ textures_b,
    const float*              __restrict__ zone_boundaries_b,
    // Blend factor: 0.0 = all A, 1.0 = all B
    float blend_t,
    unsigned char*       __restrict__ pixels_out,
    int n_pixels,
    int n_zones,
    int grid_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pixels) return;

    float r = pixels_in[idx * 3 + 0] * (1.0f / 255.0f);
    float g = pixels_in[idx * 3 + 1] * (1.0f / 255.0f);
    float b = pixels_in[idx * 3 + 2] * (1.0f / 255.0f);
    float d = depth[idx];
    float gs = (float)(grid_size - 1);

    // --- Sample texture set A ---
    float total_w_a = 0.0f;
    float4 blended_a = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int z = 0; z < n_zones; z++) {
        float z_lo = zone_boundaries_a[z];
        float z_hi = zone_boundaries_a[z + 1];
        float z_width = z_hi - z_lo;
        if (z_width < 1e-6f) continue;
        float z_center = 0.5f * (z_lo + z_hi);
        float dist = fabsf(d - z_center);
        float w = fmaxf(1.0f - dist / z_width, 0.0f);
        if (w < 1e-6f) continue;
        cudaTextureObject_t tex = (cudaTextureObject_t)textures_a[z];
        float4 s = tex3D<float4>(tex, r * gs, g * gs, b * gs);
        blended_a.x += s.x * w;
        blended_a.y += s.y * w;
        blended_a.z += s.z * w;
        total_w_a += w;
    }

    float r_a, g_a, b_a;
    if (total_w_a > 1e-6f) {
        r_a = blended_a.x / total_w_a;
        g_a = blended_a.y / total_w_a;
        b_a = blended_a.z / total_w_a;
    } else {
        r_a = r; g_a = g; b_a = b;
    }

    // --- Early out: skip set B when blend_t ≈ 0 (on keyframes) ---
    float r_out, g_out, b_out;
    if (blend_t < 1e-4f) {
        r_out = r_a; g_out = g_a; b_out = b_a;
    } else {
        // --- Sample texture set B ---
        float total_w_b = 0.0f;
        float4 blended_b = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int z = 0; z < n_zones; z++) {
            float z_lo = zone_boundaries_b[z];
            float z_hi = zone_boundaries_b[z + 1];
            float z_width = z_hi - z_lo;
            if (z_width < 1e-6f) continue;
            float z_center = 0.5f * (z_lo + z_hi);
            float dist = fabsf(d - z_center);
            float w = fmaxf(1.0f - dist / z_width, 0.0f);
            if (w < 1e-6f) continue;
            cudaTextureObject_t tex = (cudaTextureObject_t)textures_b[z];
            float4 s = tex3D<float4>(tex, r * gs, g * gs, b * gs);
            blended_b.x += s.x * w;
            blended_b.y += s.y * w;
            blended_b.z += s.z * w;
            total_w_b += w;
        }

        float r_b, g_b, b_b;
        if (total_w_b > 1e-6f) {
            r_b = blended_b.x / total_w_b;
            g_b = blended_b.y / total_w_b;
            b_b = blended_b.z / total_w_b;
        } else {
            r_b = r; g_b = g; b_b = b;
        }

        // Temporal blend
        float inv_t = 1.0f - blend_t;
        r_out = r_a * inv_t + r_b * blend_t;
        g_out = g_a * inv_t + g_b * blend_t;
        b_out = b_a * inv_t + b_b * blend_t;
    }

    pixels_out[idx * 3 + 0] = (unsigned char)(__float2uint_rn(fminf(fmaxf(r_out, 0.0f), 1.0f) * 255.0f));
    pixels_out[idx * 3 + 1] = (unsigned char)(__float2uint_rn(fminf(fmaxf(g_out, 0.0f), 1.0f) * 255.0f));
    pixels_out[idx * 3 + 2] = (unsigned char)(__float2uint_rn(fminf(fmaxf(b_out, 0.0f), 1.0f) * 255.0f));
}
