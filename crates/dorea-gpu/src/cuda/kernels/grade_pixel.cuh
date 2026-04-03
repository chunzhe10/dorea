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

    // LAB -> RGB
    float l_out  = fminf(fmaxf(l_norm * 100.0f, 0.0f), 100.0f);
    float a_out  = fminf(fmaxf(a_ab,  -128.0f), 127.0f);
    float b_out2 = fminf(fmaxf(b_ab,  -128.0f), 127.0f);
    float r3, g3, b3;
    lab_to_srgb(l_out, a_out, b_out2, &r3, &g3, &b3);

    // Final highlight knee -- inlined to avoid `auto` lambda in __device__ fn
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

    return make_float3(r3, g3, b3);
}
