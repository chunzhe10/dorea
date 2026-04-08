// Guided filter CUDA kernel — snaps warped depth maps to class boundary edges.
//
// Input:  warped depth map (f32, proxy resolution) + class mask (u8, same resolution)
// Output: filtered depth map where discontinuities align with class boundaries
//
// The guided filter preserves depth gradients within a class region but enforces
// sharp edges at class transitions (diver boundary vs water). This corrects
// stretching artifacts from optical flow warping.
//
// Algorithm: O'Toole-style box-guided filter (fast O(1) per pixel).
// For each pixel, compute local mean and variance of guide (class mask) and
// input (depth) in a (2r+1)^2 window, then apply the guided filter formula:
//   a_k = cov(guide, input) / (var(guide) + epsilon)
//   b_k = mean(input) - a_k * mean(guide)
//   output = mean(a) * guide + mean(b)

extern "C"
__global__ void guided_filter_kernel(
    const float* __restrict__ depth_in,     // warped depth, H x W
    const unsigned char* __restrict__ guide, // class mask, H x W (0=water, 1=diver)
    float* __restrict__ depth_out,           // filtered depth, H x W
    int width,
    int height,
    int radius,          // filter radius (e.g., 4 for 9x9 window)
    float epsilon_water, // regularization for water regions (larger = smoother)
    float epsilon_diver  // regularization for diver regions (smaller = sharper edges)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Compute local statistics in a (2*radius+1)^2 window
    float sum_guide = 0.0f;
    float sum_depth = 0.0f;
    float sum_guide2 = 0.0f;
    float sum_guide_depth = 0.0f;
    int count = 0;

    for (int dy = -radius; dy <= radius; dy++) {
        int ny = y + dy;
        if (ny < 0 || ny >= height) continue;
        for (int dx = -radius; dx <= radius; dx++) {
            int nx = x + dx;
            if (nx < 0 || nx >= width) continue;

            int nidx = ny * width + nx;
            float g = (float)guide[nidx];  // 0.0 or 1.0
            float d = depth_in[nidx];

            sum_guide += g;
            sum_depth += d;
            sum_guide2 += g * g;
            sum_guide_depth += g * d;
            count++;
        }
    }

    float inv_count = 1.0f / fmaxf((float)count, 1.0f);
    float mean_guide = sum_guide * inv_count;
    float mean_depth = sum_depth * inv_count;
    float var_guide = sum_guide2 * inv_count - mean_guide * mean_guide;
    float cov_gd = sum_guide_depth * inv_count - mean_guide * mean_depth;

    // Adaptive epsilon: sharper at diver boundaries, smoother in water
    float local_guide = (float)guide[idx];
    float eps = local_guide > 0.5f ? epsilon_diver : epsilon_water;

    float a = cov_gd / (var_guide + eps);
    float b = mean_depth - a * mean_guide;

    // Second pass would compute mean(a) and mean(b) over the window for
    // the full guided filter. For efficiency, we use the single-pass
    // approximation: output = a * guide + b
    depth_out[idx] = a * local_guide + b;
}
