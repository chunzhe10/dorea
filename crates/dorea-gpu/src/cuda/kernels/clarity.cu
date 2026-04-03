/**
 * clarity.cu — GPU clarity enhancement kernel.
 *
 * Clarity = high-frequency luminance detail boost, computed at proxy resolution
 * to avoid the O(N×radius) CPU box blur at 4K.
 *
 * Algorithm (3 passes):
 *   Pass A: Downsample full-res f32 RGB → proxy-res L channel (bilinear, sRGB→LAB)
 *   Pass B: 3-pass separable box blur on proxy L (naive O(N×r) — fast at proxy size)
 *   Pass C: For each full-res pixel: upsample blurred L, compute detail, apply to RGB
 *
 * Host entry point: dorea_clarity_gpu(h_rgb_in, h_rgb_out, full_w, full_h,
 *                                      proxy_w, proxy_h, blur_radius, clarity_amount)
 * Returns cudaError_t as int; 0 = success.
 */

#include <cuda_runtime.h>
#include <math.h>

// ---------------------------------------------------------------------------
// sRGB ↔ CIE LAB device helpers (D65 illuminant)
// ---------------------------------------------------------------------------

__device__ static float srgb_to_linear_d(float c) {
    return (c <= 0.04045f)
        ? (c / 12.92f)
        : powf((c + 0.055f) / 1.055f, 2.4f);
}

__device__ static float linear_to_srgb_d(float c) {
    c = fmaxf(0.0f, fminf(1.0f, c));
    return (c <= 0.0031308f)
        ? (12.92f * c)
        : (1.055f * powf(c, 1.0f / 2.4f) - 0.055f);
}

__device__ static float lab_f_d(float t) {
    return (t > 0.008856f) ? cbrtf(t) : (7.787f * t + 16.0f / 116.0f);
}

__device__ static float lab_f_inv_d(float t) {
    float t3 = t * t * t;
    return (t3 > 0.008856f) ? t3 : ((t - 16.0f / 116.0f) / 7.787f);
}

#define D65_XN 0.95047f
#define D65_YN 1.00000f
#define D65_ZN 1.08883f

__device__ static void srgb_to_lab_d(float r, float g, float b,
                                       float* L, float* A, float* B) {
    float rl = srgb_to_linear_d(r);
    float gl = srgb_to_linear_d(g);
    float bl = srgb_to_linear_d(b);

    float x = 0.4124564f * rl + 0.3575761f * gl + 0.1804375f * bl;
    float y = 0.2126729f * rl + 0.7151522f * gl + 0.0721750f * bl;
    float z = 0.0193339f * rl + 0.1191920f * gl + 0.9503041f * bl;

    float fx = lab_f_d(x / D65_XN);
    float fy = lab_f_d(y / D65_YN);
    float fz = lab_f_d(z / D65_ZN);

    *L = 116.0f * fy - 16.0f;
    *A = 500.0f * (fx - fy);
    *B = 200.0f * (fy - fz);
}

__device__ static void lab_to_srgb_d(float L, float A, float B,
                                       float* r, float* g, float* b) {
    float fy = (L + 16.0f) / 116.0f;
    float fx = A / 500.0f + fy;
    float fz = fy - B / 200.0f;

    float x = lab_f_inv_d(fx) * D65_XN;
    float y = lab_f_inv_d(fy) * D65_YN;
    float z = lab_f_inv_d(fz) * D65_ZN;

    float rl =  3.2404542f * x - 1.5371385f * y - 0.4985314f * z;
    float gl = -0.9692660f * x + 1.8760108f * y + 0.0415560f * z;
    float bl_out =  0.0556434f * x - 0.2040259f * y + 1.0572252f * z;

    *r = linear_to_srgb_d(rl);
    *g = linear_to_srgb_d(gl);
    *b = linear_to_srgb_d(bl_out);
}

// ---------------------------------------------------------------------------
// Kernel A: downsample full-res f32 RGB → proxy-res L channel (bilinear)
// ---------------------------------------------------------------------------
extern "C"
__global__ void clarity_extract_L_proxy(
    const float* __restrict__ rgb_full,   // [full_w * full_h * 3]
    float* __restrict__       l_proxy,    // [proxy_w * proxy_h]
    int full_w, int full_h,
    int proxy_w, int proxy_h
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= proxy_w || py >= proxy_h) return;

    // Map proxy pixel → full-res bilinear coordinates
    float sx = (proxy_w > 1)
        ? ((float)px / (float)(proxy_w - 1) * (float)(full_w - 1))
        : 0.0f;
    float sy = (proxy_h > 1)
        ? ((float)py / (float)(proxy_h - 1) * (float)(full_h - 1))
        : 0.0f;

    int x0 = (int)sx;  int x1 = min(x0 + 1, full_w - 1);
    int y0 = (int)sy;  int y1 = min(y0 + 1, full_h - 1);
    float fx = sx - (float)x0;
    float fy = sy - (float)y0;

    // Extract L from each corner via sRGB→LAB
    float L00, L10, L01, L11, A_unused, B_unused;
    int i00 = (y0 * full_w + x0) * 3;
    int i10 = (y0 * full_w + x1) * 3;
    int i01 = (y1 * full_w + x0) * 3;
    int i11 = (y1 * full_w + x1) * 3;
    srgb_to_lab_d(rgb_full[i00], rgb_full[i00+1], rgb_full[i00+2], &L00, &A_unused, &B_unused);
    srgb_to_lab_d(rgb_full[i10], rgb_full[i10+1], rgb_full[i10+2], &L10, &A_unused, &B_unused);
    srgb_to_lab_d(rgb_full[i01], rgb_full[i01+1], rgb_full[i01+2], &L01, &A_unused, &B_unused);
    srgb_to_lab_d(rgb_full[i11], rgb_full[i11+1], rgb_full[i11+2], &L11, &A_unused, &B_unused);

    float L = L00*(1-fx)*(1-fy) + L10*fx*(1-fy)
            + L01*(1-fx)*fy     + L11*fx*fy;
    l_proxy[py * proxy_w + px] = L;
}

// ---------------------------------------------------------------------------
// Kernel B-row: box blur along rows of the proxy L channel (one thread per pixel)
// ---------------------------------------------------------------------------
extern "C"
__global__ void clarity_box_blur_rows(
    const float* __restrict__ in,
    float* __restrict__       out,
    int width, int height, int radius
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;

    int lo = max(0, col - radius);
    int hi = min(width - 1, col + radius);
    float s = 0.0f;
    int base = row * width;
    for (int k = lo; k <= hi; k++) s += in[base + k];
    out[base + col] = s / (float)(hi - lo + 1);
}

// ---------------------------------------------------------------------------
// Kernel B-col: box blur along columns of the proxy L channel (one thread per pixel)
// ---------------------------------------------------------------------------
extern "C"
__global__ void clarity_box_blur_cols(
    const float* __restrict__ in,
    float* __restrict__       out,
    int width, int height, int radius
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;

    int lo = max(0, row - radius);
    int hi = min(height - 1, row + radius);
    float s = 0.0f;
    for (int k = lo; k <= hi; k++) s += in[k * width + col];
    out[row * width + col] = s / (float)(hi - lo + 1);
}

// ---------------------------------------------------------------------------
// Kernel C: apply clarity — upsample blurred L to full res, compute detail,
//           reconstruct RGB with modified L only.
// ---------------------------------------------------------------------------
extern "C"
__global__ void clarity_apply_kernel(
    const float* __restrict__ rgb_in,      // [full_w * full_h * 3]
    float* __restrict__       rgb_out,     // [full_w * full_h * 3]
    const float* __restrict__ blur_proxy,  // [proxy_w * proxy_h], L in [0..100]
    float clarity_amount,
    int full_w, int full_h,
    int proxy_w, int proxy_h
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= full_w * full_h) return;

    int fy = idx / full_w;
    int fx = idx % full_w;

    float r = rgb_in[idx * 3 + 0];
    float g = rgb_in[idx * 3 + 1];
    float b = rgb_in[idx * 3 + 2];

    // Convert pixel to LAB
    float L_full, A, B;
    srgb_to_lab_d(r, g, b, &L_full, &A, &B);
    float L_norm = L_full / 100.0f;  // normalise to [0,1]

    // Bilinear sample blurred proxy L at this full-res pixel's proxy coords
    float sx = (proxy_w > 1)
        ? ((float)fx / (float)(full_w - 1) * (float)(proxy_w - 1))
        : 0.0f;
    float sy = (proxy_h > 1)
        ? ((float)fy / (float)(full_h - 1) * (float)(proxy_h - 1))
        : 0.0f;
    int px0 = (int)sx;  int px1 = min(px0 + 1, proxy_w - 1);
    int py0 = (int)sy;  int py1 = min(py0 + 1, proxy_h - 1);
    float bfx = sx - (float)px0;
    float bfy = sy - (float)py0;

    float blur_sampled =
        blur_proxy[py0 * proxy_w + px0] * (1-bfx) * (1-bfy) +
        blur_proxy[py0 * proxy_w + px1] *  bfx    * (1-bfy) +
        blur_proxy[py1 * proxy_w + px0] * (1-bfx) *  bfy    +
        blur_proxy[py1 * proxy_w + px1] *  bfx    *  bfy;

    float blur_norm = blur_sampled / 100.0f;  // same normalisation as L_norm

    // Detail boost: tanh((L - blur)*3)/3, then add scaled detail to L
    float detail = tanhf((L_norm - blur_norm) * 3.0f) / 3.0f;
    float L_new = fminf(fmaxf(L_norm + detail * clarity_amount, 0.0f), 1.0f);

    // Reconstruct RGB: only L changed, A and B unchanged
    float ro, go, bo;
    lab_to_srgb_d(L_new * 100.0f, A, B, &ro, &go, &bo);

    rgb_out[idx * 3 + 0] = ro;
    rgb_out[idx * 3 + 1] = go;
    rgb_out[idx * 3 + 2] = bo;
}

