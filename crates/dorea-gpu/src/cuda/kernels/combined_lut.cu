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
    int grid_size,
    // Depth map dimensions (may differ from frame — kernel does bilinear sampling)
    int frame_w,
    int frame_h,
    int depth_w,
    int depth_h,
    // Class mask (0=water, 1=diver) at mask resolution. NULL if no YOLO-seg.
    const unsigned char* __restrict__ class_mask,
    int mask_w,
    int mask_h,
    // Median diver depth — all diver pixels use this value for uniform correction.
    float diver_depth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pixels) return;

    float r = pixels_in[idx * 3 + 0] * (1.0f / 255.0f);
    float g = pixels_in[idx * 3 + 1] * (1.0f / 255.0f);
    float b = pixels_in[idx * 3 + 2] * (1.0f / 255.0f);

    // Bilinear sample depth at proxy resolution — avoids blocky upscale artifacts.
    float d;
    if (depth_w == frame_w && depth_h == frame_h) {
        d = depth[idx];
    } else {
        int px = idx % frame_w;
        int py = idx / frame_w;
        float sx = (float)px * (float)(depth_w - 1) / (float)(frame_w - 1);
        float sy = (float)py * (float)(depth_h - 1) / (float)(frame_h - 1);
        int x0 = (int)sx;
        int y0 = (int)sy;
        int x1 = min(x0 + 1, depth_w - 1);
        int y1 = min(y0 + 1, depth_h - 1);
        float fx = sx - (float)x0;
        float fy = sy - (float)y0;
        d = depth[y0 * depth_w + x0] * (1.0f - fx) * (1.0f - fy)
          + depth[y0 * depth_w + x1] * fx * (1.0f - fy)
          + depth[y1 * depth_w + x0] * (1.0f - fx) * fy
          + depth[y1 * depth_w + x1] * fx * fy;
    }

    // Depth dithering: add sub-zone noise to break up banding at zone transitions.
    // Uses a simple hash of pixel position for spatially-stable noise (no temporal flicker).
    {
        unsigned int hash = (unsigned int)(idx * 2654435761u);  // Knuth multiplicative hash
        float noise = ((float)(hash & 0xFFFF) / 65535.0f - 0.5f);  // [-0.5, +0.5]
        // Scale noise to half a zone width — enough to smooth transitions but not shift zones
        float avg_zone_width = 1.0f / (float)n_zones;
        d += noise * avg_zone_width * 0.5f;
        d = fminf(fmaxf(d, 0.0f), 1.0f);
    }

    // If YOLO-seg mask available and this pixel is a diver, override depth with
    // the uniform diver_depth. This gives all diver pixels the same color correction,
    // eliminating banding and temporal flicker on the subject.
    if (class_mask != 0) {
        int px = idx % frame_w;
        int py = idx / frame_w;
        float mx = (float)px * (float)(mask_w - 1) / (float)(frame_w - 1);
        float my = (float)py * (float)(mask_h - 1) / (float)(frame_h - 1);
        int mi = min((int)my, mask_h - 1) * mask_w + min((int)mx, mask_w - 1);
        if (class_mask[mi] > 0) {
            d = diver_depth;
        }
    }

    float gs = (float)(grid_size - 1);

    // --- Sample texture set A ---
    // Zone blending: each zone's influence extends 2x its width (overlaps neighbors).
    // This prevents hard color discontinuities at zone boundaries.
    float total_w_a = 0.0f;
    float4 blended_a = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int z = 0; z < n_zones; z++) {
        float z_lo = zone_boundaries_a[z];
        float z_hi = zone_boundaries_a[z + 1];
        float z_width = z_hi - z_lo;
        if (z_width < 1e-6f) continue;
        float z_center = 0.5f * (z_lo + z_hi);
        float dist = fabsf(d - z_center);
        float blend_radius = z_width * 2.0f;  // 2x overlap for smooth transitions
        float w = fmaxf(1.0f - dist / blend_radius, 0.0f);
        w = w * w;  // squared falloff for smoother transitions than linear
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
            float blend_radius = z_width * 2.0f;
            float w = fmaxf(1.0f - dist / blend_radius, 0.0f);
            w = w * w;
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

    // Depth-dependent strength: full correction on subjects (mid/near depth),
    // reduced only in the very far background (d < 0.15) where SNR is lowest.
    float depth_strength = fminf(d / 0.15f, 1.0f);  // 0→0 at d=0, ramps to 1.0 at d=0.15+

    // Clamp per-channel correction to prevent extreme noise amplification on very dark pixels.
    float max_shift = 0.35f;  // max ±0.35 per channel (±89/255) — strong but bounded
    r_out = r + fminf(fmaxf(r_out - r, -max_shift), max_shift) * depth_strength;
    g_out = g + fminf(fmaxf(g_out - g, -max_shift), max_shift) * depth_strength;
    b_out = b + fminf(fmaxf(b_out - b, -max_shift), max_shift) * depth_strength;

    // Output dithering: add ±0.5/255 triangular-PDF noise per channel to break up
    // LUT quantization banding. Uses spatially-stable hash (no temporal flicker).
    // Triangular PDF: sum of two uniform → concentrates near zero, less visible than uniform.
    {
        unsigned int h1 = (unsigned int)(idx * 2654435761u);
        unsigned int h2 = (unsigned int)(idx * 340573321u + 1013904223u);
        float u1 = (float)(h1 & 0xFFFF) / 65536.0f;  // [0, 1)
        float u2 = (float)(h2 & 0xFFFF) / 65536.0f;
        float tri = (u1 + u2 - 1.0f);  // triangular PDF in [-1, +1]
        float dither = tri / 255.0f;    // ±1/255 max
        r_out += dither;
        // Different hash per channel to avoid correlated dither
        unsigned int h3 = h1 ^ (h2 << 13);
        unsigned int h4 = h2 ^ (h1 >> 7);
        float u3 = (float)(h3 & 0xFFFF) / 65536.0f;
        float u4 = (float)(h4 & 0xFFFF) / 65536.0f;
        g_out += (u3 + u4 - 1.0f) / 255.0f;
        unsigned int h5 = h1 ^ (h2 >> 5) ^ 0xDEADBEEF;
        unsigned int h6 = h2 ^ (h1 << 11) ^ 0xCAFEBABE;
        float u5 = (float)(h5 & 0xFFFF) / 65536.0f;
        float u6 = (float)(h6 & 0xFFFF) / 65536.0f;
        b_out += (u5 + u6 - 1.0f) / 255.0f;
    }

    pixels_out[idx * 3 + 0] = (unsigned char)(__float2uint_rn(fminf(fmaxf(r_out, 0.0f), 1.0f) * 255.0f));
    pixels_out[idx * 3 + 1] = (unsigned char)(__float2uint_rn(fminf(fmaxf(g_out, 0.0f), 1.0f) * 255.0f));
    pixels_out[idx * 3 + 2] = (unsigned char)(__float2uint_rn(fminf(fmaxf(b_out, 0.0f), 1.0f) * 255.0f));
}
