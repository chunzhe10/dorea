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

// Stage bitmask constants — must match dorea_gpu::stages in Rust.
#define STAGE_HSL          (1 << 0)
#define STAGE_AMBIANCE     (1 << 1)
#define STAGE_WARMTH       (1 << 2)
#define STAGE_VIBRANCE     (1 << 3)
#define STAGE_STRENGTH     (1 << 4)
#define STAGE_DEPTH_MOD    (1 << 5)
#define STAGE_DEPTH_DITHER (1 << 6)
#define STAGE_YOLO_MASK    (1 << 7)

// -------------------------------------------------------------------------
// OKLab colorspace, rescaled to CIELab-compatible ranges.
// Matches dorea-color/src/lab.rs exactly.
//
// OKLab (Björn Ottosson, 2020) with output rescaled:
//   L: OKLab [0,1] × 100 → [0,100]
//   a: OKLab [-0.4,0.4] × 300 → [-120,120]
//   b: OKLab [-0.4,0.4] × 300 → [-120,120]
// -------------------------------------------------------------------------

#define L_SCALE  100.0f
#define AB_SCALE 300.0f

__device__ __forceinline__ float srgb_to_linear(float v) {
    return (v <= 0.04045f) ? (v / 12.92f) : __powf((v + 0.055f) / 1.055f, 2.4f);
}

__device__ __forceinline__ float linear_to_srgb(float v) {
    return (v <= 0.0031308f) ? (v * 12.92f) : (1.055f * __powf(v, 1.0f / 2.4f) - 0.055f);
}

__device__ void srgb_to_lab(float r, float g, float b,
                              float* l_out, float* a_out, float* b_lab_out) {
    float rl = srgb_to_linear(r);
    float gl = srgb_to_linear(g);
    float bl = srgb_to_linear(b);

    // Linear RGB → LMS
    float l = 0.4122214708f*rl + 0.5363325363f*gl + 0.0514459929f*bl;
    float m = 0.2119034982f*rl + 0.6806995451f*gl + 0.1073969566f*bl;
    float s = 0.0883024619f*rl + 0.2817188376f*gl + 0.6299787005f*bl;

    // Cube root
    float l_ = cbrtf(fmaxf(l, 0.0f));
    float m_ = cbrtf(fmaxf(m, 0.0f));
    float s_ = cbrtf(fmaxf(s, 0.0f));

    // LMS' → OKLab, rescaled to CIELab ranges
    *l_out     = (0.2104542553f*l_ + 0.7936177850f*m_ - 0.0040720468f*s_) * L_SCALE;
    *a_out     = (1.9779984951f*l_ - 2.4285922050f*m_ + 0.4505937099f*s_) * AB_SCALE;
    *b_lab_out = (0.0259040371f*l_ + 0.7827717662f*m_ - 0.8086757660f*s_) * AB_SCALE;
}

__device__ void lab_to_srgb(float l, float a, float b_lab,
                              float* r_out, float* g_out, float* b_out) {
    // Unscale from CIELab-compatible ranges to native OKLab
    float ok_l = l / L_SCALE;
    float ok_a = a / AB_SCALE;
    float ok_b = b_lab / AB_SCALE;

    // OKLab → LMS'
    float l_ = ok_l + 0.3963377774f*ok_a + 0.2158037573f*ok_b;
    float m_ = ok_l - 0.1055613458f*ok_a - 0.0638541728f*ok_b;
    float s_ = ok_l - 0.0894841775f*ok_a - 1.2914855480f*ok_b;

    // Cube (inverse of cube root)
    float lc = l_ * l_ * l_;
    float mc = m_ * m_ * m_;
    float sc = s_ * s_ * s_;

    // LMS → linear RGB (corrected inverse of forward RGB→LMS matrix)
    float rl =  4.0767416613f*lc - 3.3077115904f*mc + 0.2309699287f*sc;
    float gl = -1.2684380041f*lc + 2.6097574007f*mc - 0.3413193963f*sc;
    float bl =  -0.0041960863f*lc - 0.7034186145f*mc + 1.7076147010f*sc;

    *r_out = fminf(fmaxf(linear_to_srgb(fmaxf(rl, 0.0f)), 0.0f), 1.0f);
    *g_out = fminf(fmaxf(linear_to_srgb(fmaxf(gl, 0.0f)), 0.0f), 1.0f);
    *b_out = fminf(fmaxf(linear_to_srgb(fmaxf(bl, 0.0f)), 0.0f), 1.0f);
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
 *   LUT apply (soft zone blend + trilinear) ->
 *   HSL correct (6 qualifiers) ->
 *   Fused ambiance + warmth (LAB) ->
 *   Strength blend ->
 *   Returns graded f32 RGB [0,1]
 *
 * Parameters match the packed layout uploaded by CombinedLut::build():
 *   luts          - [n_zones][lut_size^3][3] float array
 *   zone_boundaries - [n_zones+1] float array
 *   h_offsets/s_ratios/v_offsets/weights - [6] float arrays for HSL
 *   warmth        - params.warmth
 *   strength      - params.strength
 *   contrast      - params.contrast
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
    float warmth, float strength, float contrast,
    int stage_mask
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
    // 2. HSL 6-qualifier corrections (STAGE_HSL)
    // ------------------------------------------------------------------
    float r2 = r1, g2 = g1, b2 = b1;
    if (stage_mask & STAGE_HSL) {
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

        hsv_to_rgb_gp(h, s, v, &r2, &g2, &b2);
    }

    // ------------------------------------------------------------------
    // 3. Fused ambiance + warmth (LAB colorspace)
    // ------------------------------------------------------------------
    float r3 = r2, g3 = g2, b3 = b2;
    if (stage_mask & (STAGE_AMBIANCE | STAGE_WARMTH | STAGE_VIBRANCE)) {
        float d = depth;
        float l_norm, a_ab, b_ab;
        {
            float l_raw, a_raw, b_raw;
            srgb_to_lab(r2, g2, b2, &l_raw, &a_raw, &b_raw);
            l_norm = l_raw / 100.0f; a_ab = a_raw; b_ab = b_raw;
        }

        if (stage_mask & STAGE_AMBIANCE) {
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
        }

        if (stage_mask & STAGE_WARMTH) {
            // Warmth (a*/b* push)
            float lum_w = 4.0f * l_norm * (1.0f - l_norm);
            a_ab += (1.0f + 5.0f * d) * lum_w;
            b_ab +=  4.0f * d          * lum_w;

            // User warmth scaling
            float warmth_factor = 1.0f + (warmth - 1.0f) * 0.3f;
            if (fabsf(warmth_factor - 1.0f) > 1e-4f) {
                a_ab *= warmth_factor;
                b_ab *= warmth_factor;
            }
        }

        if (stage_mask & STAGE_VIBRANCE) {
            // Vibrance
            float vib = 0.4f + 0.5f * d;
            float chroma = sqrtf(a_ab*a_ab + b_ab*b_ab + 1e-8f);
            float chroma_n = fminf(chroma / 40.0f, 1.0f);
            float boost = vib * (1.0f - chroma_n) * fminf(l_norm / 0.25f, 1.0f);
            a_ab *= 1.0f + boost;
            b_ab *= 1.0f + boost;
        }

        // LAB -> RGB
        float l_out  = fminf(fmaxf(l_norm * 100.0f, 0.0f), 100.0f);
        float a_out  = fminf(fmaxf(a_ab,  -128.0f), 127.0f);
        float b_out2 = fminf(fmaxf(b_ab,  -128.0f), 127.0f);
        lab_to_srgb(l_out, a_out, b_out2, &r3, &g3, &b3);

        // Final highlight knee (part of AMBIANCE)
        if (stage_mask & STAGE_AMBIANCE) {
            const float knee = 0.92f;
            const float room = 1.0f - knee;
            r3 = fminf(fmaxf(r3 > knee ? knee + room * tanhf((r3 - knee) / room) : r3, 0.0f), 1.0f);
            g3 = fminf(fmaxf(g3 > knee ? knee + room * tanhf((g3 - knee) / room) : g3, 0.0f), 1.0f);
            b3 = fminf(fmaxf(b3 > knee ? knee + room * tanhf((b3 - knee) / room) : b3, 0.0f), 1.0f);
        }
    }

    // ------------------------------------------------------------------
    // 4. Strength blend with original input (STAGE_STRENGTH)
    // ------------------------------------------------------------------
    if ((stage_mask & STAGE_STRENGTH) && strength < 1.0f - 1e-4f) {
        r3 = r * (1.0f - strength) + r3 * strength;
        g3 = g * (1.0f - strength) + g3 * strength;
        b3 = b * (1.0f - strength) + b3 * strength;
    }

    return make_float3(r3, g3, b3);
}
