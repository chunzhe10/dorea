/**
 * hsl_correct.cu — HSL 6-qualifier correction kernel.
 *
 * Each thread processes one pixel. Applies per-qualifier H/S/V corrections
 * with soft qualifier masks.
 *
 * Grid: 1D, blockDim.x = 256
 *
 * Qualifiers (matching dorea-hsl/src/qualifiers.rs):
 *   0 Red/Skin:  h_center=0,   h_width=40
 *   1 Yellow:    h_center=40,  h_width=40
 *   2 Green:     h_center=100, h_width=50
 *   3 Cyan:      h_center=170, h_width=40
 *   4 Blue:      h_center=210, h_width=40
 *   5 Magenta:   h_center=290, h_width=50
 *
 * Parameters:
 *   pixels_in   — float3 [n_pixels] RGB [0,1]
 *   pixels_out  — float3 [n_pixels] RGB [0,1]
 *   h_offsets   — float [6] hue offsets in degrees
 *   s_ratios    — float [6] saturation ratio multipliers
 *   v_offsets   — float [6] value offsets in [0,1]
 *   weights     — float [6] total qualifier weights (skip if < 100)
 *   n_pixels    — total pixel count
 */

#include <cuda_runtime.h>
#include <math.h>

#define N_QUALIFIERS 6

static __constant__ float H_CENTERS[N_QUALIFIERS] = {0.f, 40.f, 100.f, 170.f, 210.f, 290.f};
static __constant__ float H_WIDTHS[N_QUALIFIERS]  = {40.f, 40.f, 50.f, 40.f, 40.f, 50.f};

__device__ void rgb_to_hsv(float r, float g, float b,
                            float* h, float* s, float* v)
{
    float cmax = fmaxf(fmaxf(r, g), b);
    float cmin = fminf(fminf(r, g), b);
    float delta = cmax - cmin;

    *v = cmax;
    *s = (cmax > 1e-6f) ? (delta / cmax) : 0.0f;

    if (delta < 1e-6f) {
        *h = 0.0f;
        return;
    }

    float hh;
    if (cmax == r) {
        hh = 60.0f * fmodf((g - b) / delta, 6.0f);
    } else if (cmax == g) {
        hh = 60.0f * ((b - r) / delta + 2.0f);
    } else {
        hh = 60.0f * ((r - g) / delta + 4.0f);
    }

    if (hh < 0.0f) hh += 360.0f;
    *h = hh;
}

__device__ void hsv_to_rgb(float h, float s, float v,
                            float* r, float* g, float* b)
{
    if (s < 1e-6f) {
        *r = *g = *b = v;
        return;
    }
    float hh = fmodf(h, 360.0f);
    if (hh < 0.0f) hh += 360.0f;
    hh /= 60.0f;
    int i = (int)hh;
    float f = hh - (float)i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));

    switch (i % 6) {
        case 0: *r=v; *g=t; *b=p; break;
        case 1: *r=q; *g=v; *b=p; break;
        case 2: *r=p; *g=v; *b=t; break;
        case 3: *r=p; *g=q; *b=v; break;
        case 4: *r=t; *g=p; *b=v; break;
        case 5: *r=v; *g=p; *b=q; break;
    }
}

__device__ float angular_dist(float a, float b) {
    float d = fabsf(fmodf(a - b + 360.0f, 360.0f));
    if (d > 180.0f) d = 360.0f - d;
    return d;
}

extern "C"
__global__ void hsl_correct_kernel(
    const float* __restrict__ pixels_in,
    float* __restrict__ pixels_out,
    const float* __restrict__ h_offsets,
    const float* __restrict__ s_ratios,
    const float* __restrict__ v_offsets,
    const float* __restrict__ weights,
    int n_pixels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pixels) return;

    float r = pixels_in[idx * 3 + 0];
    float g = pixels_in[idx * 3 + 1];
    float b = pixels_in[idx * 3 + 2];

    float h, s, v;
    rgb_to_hsv(r, g, b, &h, &s, &v);

    for (int q = 0; q < N_QUALIFIERS; q++) {
        if (weights[q] < 100.0f) continue;

        float dist = angular_dist(h, H_CENTERS[q]);
        float mask = fmaxf(1.0f - dist / H_WIDTHS[q], 0.0f);
        // Saturation gate: skip near-grey pixels
        if (s < 0.08f) mask = 0.0f;
        if (mask < 1e-6f) continue;

        h += h_offsets[q] * mask;
        s *= (1.0f + (s_ratios[q] - 1.0f) * mask);
        v += v_offsets[q] * mask;
    }

    // Wrap/clamp
    h = fmodf(h, 360.0f);
    if (h < 0.0f) h += 360.0f;
    s = fminf(fmaxf(s, 0.0f), 1.0f);
    v = fminf(fmaxf(v, 0.0f), 1.0f);

    hsv_to_rgb(h, s, v, &r, &g, &b);

    pixels_out[idx * 3 + 0] = r;
    pixels_out[idx * 3 + 1] = g;
    pixels_out[idx * 3 + 2] = b;
}
