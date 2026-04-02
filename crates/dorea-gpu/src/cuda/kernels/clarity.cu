/**
 * clarity.cu — Separable Gaussian blur + clarity detail boost at proxy resolution.
 *
 * Two-pass separable Gaussian:
 *   pass 1: blur rows  (L channel, proxy resolution)
 *   pass 2: blur cols  (L channel, proxy resolution)
 *   pass 3: add detail = tanh((L - blur) * 3) / 3 * clarity_amount
 *
 * Each thread handles one pixel.
 * Grid: 1D, blockDim.x = 256
 *
 * Parameters (all at proxy resolution W x H):
 *   L_in      — float [W*H] luminance channel [0,1]
 *   L_out     — float [W*H] output luminance
 *   tmp       — float [W*H] temp buffer for intermediate blur
 *   width, height — proxy resolution
 *   sigma     — Gaussian sigma in pixels (typically 30.0 at proxy)
 *   clarity   — scalar clarity amount [0,1]
 *   radius    — kernel radius = ceil(3 * sigma)
 */

#include <cuda_runtime.h>
#include <math.h>

extern "C"
__global__ void gaussian_blur_rows(
    const float* __restrict__ L_in,
    float* __restrict__ L_out,
    int width, int height,
    int radius,
    const float* __restrict__ kernel  // pre-computed Gaussian kernel [radius+1]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;

    int row = idx / width;
    int col = idx % width;

    float acc = 0.0f;
    float total_w = 0.0f;

    for (int k = -radius; k <= radius; k++) {
        int c = col + k;
        if (c < 0 || c >= width) continue;
        float w = kernel[abs(k)];
        acc += L_in[row * width + c] * w;
        total_w += w;
    }

    L_out[idx] = (total_w > 0.0f) ? (acc / total_w) : L_in[idx];
}

extern "C"
__global__ void gaussian_blur_cols(
    const float* __restrict__ L_in,
    float* __restrict__ L_out,
    int width, int height,
    int radius,
    const float* __restrict__ kernel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;

    int row = idx / width;
    int col = idx % width;

    float acc = 0.0f;
    float total_w = 0.0f;

    for (int k = -radius; k <= radius; k++) {
        int r = row + k;
        if (r < 0 || r >= height) continue;
        float w = kernel[abs(k)];
        acc += L_in[r * width + col] * w;
        total_w += w;
    }

    L_out[idx] = (total_w > 0.0f) ? (acc / total_w) : L_in[idx];
}

extern "C"
__global__ void clarity_apply(
    const float* __restrict__ L_original,
    const float* __restrict__ L_blurred,
    float* __restrict__ L_out,
    int n_pixels,
    float clarity_amount
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pixels) return;

    float L = L_original[idx];
    float blur = L_blurred[idx];
    float detail = tanhf((L - blur) * 3.0f) / 3.0f;
    float result = L + detail * clarity_amount;
    L_out[idx] = fminf(fmaxf(result, 0.0f), 1.0f);
}
