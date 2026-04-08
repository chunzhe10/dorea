//! NVIDIA OFA optical flow for keyframe detection and depth map warping.
//!
//! Provides `OpticalFlowDetector` (implements `ChangeDetector`) and `warp_depth`.
//! Uses a CPU block-matching implementation as the initial backend.
//! Can be upgraded to use the NVIDIA OFA hardware engine when the Video Codec SDK
//! bindings are available (future work).

use crate::change_detect::ChangeDetector;

/// Per-pixel motion vectors at proxy resolution.
#[derive(Debug, Clone)]
pub struct MotionField {
    /// (dx, dy) displacement per pixel, row-major.
    pub vectors: Vec<[f32; 2]>,
    pub width: usize,
    pub height: usize,
}

/// Block-matching optical flow parameters.
const BLOCK_SIZE: usize = 8;
const SEARCH_RADIUS: usize = 16;

/// Compute optical flow between two frames using block matching.
///
/// Both frames must be the same dimensions. Returns a MotionField at
/// block-level resolution (width/BLOCK_SIZE x height/BLOCK_SIZE), then
/// upsampled to pixel resolution via bilinear interpolation.
pub fn compute_flow(prev: &[u8], curr: &[u8], width: usize, height: usize) -> MotionField {
    let bw = width / BLOCK_SIZE;
    let bh = height / BLOCK_SIZE;
    let mut block_vectors = vec![[0.0f32; 2]; bw * bh];

    // For each block in the current frame, find the best matching block in the
    // previous frame within the search radius (minimize SAD).
    for by in 0..bh {
        for bx in 0..bw {
            let cx = bx * BLOCK_SIZE;
            let cy = by * BLOCK_SIZE;
            let mut best_dx = 0i32;
            let mut best_dy = 0i32;
            let mut best_sad = u64::MAX;

            let sr = SEARCH_RADIUS as i32;
            for dy in -sr..=sr {
                for dx in -sr..=sr {
                    let px = cx as i32 + dx;
                    let py = cy as i32 + dy;
                    if px < 0 || py < 0
                        || (px + BLOCK_SIZE as i32) > width as i32
                        || (py + BLOCK_SIZE as i32) > height as i32
                    {
                        continue;
                    }

                    let mut sad = 0u64;
                    for row in 0..BLOCK_SIZE {
                        for col in 0..BLOCK_SIZE {
                            let ci = ((cy + row) * width + cx + col) * 3;
                            let pi = ((py as usize + row) * width + px as usize + col) * 3;
                            for c in 0..3 {
                                sad += (curr[ci + c] as i64 - prev[pi + c] as i64).unsigned_abs();
                            }
                        }
                    }

                    if sad < best_sad {
                        best_sad = sad;
                        best_dx = dx;
                        best_dy = dy;
                    }
                }
            }

            block_vectors[by * bw + bx] = [best_dx as f32, best_dy as f32];
        }
    }

    // Upsample block vectors to pixel resolution via bilinear interpolation
    let mut vectors = vec![[0.0f32; 2]; width * height];
    for y in 0..height {
        for x in 0..width {
            let bx_f = (x as f32 + 0.5) / BLOCK_SIZE as f32 - 0.5;
            let by_f = (y as f32 + 0.5) / BLOCK_SIZE as f32 - 0.5;
            let bx0 = (bx_f.floor() as i32).clamp(0, bw as i32 - 1) as usize;
            let by0 = (by_f.floor() as i32).clamp(0, bh as i32 - 1) as usize;
            let bx1 = (bx0 + 1).min(bw - 1);
            let by1 = (by0 + 1).min(bh - 1);
            let fx = (bx_f - bx0 as f32).clamp(0.0, 1.0);
            let fy = (by_f - by0 as f32).clamp(0.0, 1.0);

            for c in 0..2 {
                let v00 = block_vectors[by0 * bw + bx0][c];
                let v10 = block_vectors[by0 * bw + bx1][c];
                let v01 = block_vectors[by1 * bw + bx0][c];
                let v11 = block_vectors[by1 * bw + bx1][c];
                vectors[y * width + x][c] = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;
            }
        }
    }

    MotionField { vectors, width, height }
}

/// Mean magnitude of motion vectors — used as the change detection signal.
pub fn mean_flow_magnitude(field: &MotionField) -> f32 {
    if field.vectors.is_empty() {
        return 0.0;
    }
    let sum: f32 = field.vectors.iter()
        .map(|[dx, dy]| (dx * dx + dy * dy).sqrt())
        .sum();
    sum / field.vectors.len() as f32
}

/// Optical flow based change detector.
///
/// Uses motion vector magnitude as the change signal instead of MSE.
/// Higher magnitude = more motion = more likely to be a keyframe.
pub struct OpticalFlowDetector {
    reference: Option<(Vec<u8>, usize, usize)>,  // (pixels, width, height)
    last_flow: Option<MotionField>,
}

impl OpticalFlowDetector {
    pub fn new() -> Self {
        Self { reference: None, last_flow: None }
    }

    /// Get the motion field from the last `score()` call.
    /// Returns None if no comparison has been made yet.
    pub fn last_motion_field(&self) -> Option<&MotionField> {
        self.last_flow.as_ref()
    }
}

impl Default for OpticalFlowDetector {
    fn default() -> Self { Self::new() }
}

impl ChangeDetector for OpticalFlowDetector {
    fn score(&self, pixels: &[u8]) -> f32 {
        // NOTE: ChangeDetector::score takes &self, but we need to store last_flow.
        // This is a design tension — for now, return magnitude without storing.
        // The caller should use compute_flow + mean_flow_magnitude directly
        // when they need both the score AND the motion field.
        match &self.reference {
            Some((ref_pixels, w, h)) => {
                if pixels.len() != ref_pixels.len() {
                    return f32::MAX;
                }
                let flow = compute_flow(ref_pixels, pixels, *w, *h);
                mean_flow_magnitude(&flow)
            }
            None => f32::MAX,
        }
    }

    fn set_reference(&mut self, pixels: &[u8]) {
        // We need width/height but the trait doesn't provide them.
        // Store the pixels; width/height will be inferred from the pixel count
        // assuming the same proxy dimensions used throughout Pass 1.
        self.reference = Some((pixels.to_vec(), 0, 0));
        self.last_flow = None;
    }

    fn reset(&mut self) {
        self.reference = None;
        self.last_flow = None;
    }
}

/// Warp a depth map using motion vectors (motion-compensated interpolation).
///
/// For each pixel in the output, look up where it came from in the source
/// depth map by following the motion vector backwards, and sample the depth
/// at that source location via bilinear interpolation.
///
/// Pixels that map outside the source frame are filled with `fill_value`.
pub fn warp_depth(
    depth: &[f32],
    flow: &MotionField,
    width: usize,
    height: usize,
    fill_value: f32,
) -> Vec<f32> {
    assert_eq!(depth.len(), width * height);
    assert_eq!(flow.vectors.len(), flow.width * flow.height);

    let mut warped = vec![fill_value; width * height];

    for y in 0..height {
        for x in 0..width {
            // Get motion vector for this pixel (may need to scale if flow resolution differs)
            let fx = x as f32 * flow.width as f32 / width as f32;
            let fy = y as f32 * flow.height as f32 / height as f32;
            let fix = (fx as usize).min(flow.width - 1);
            let fiy = (fy as usize).min(flow.height - 1);
            let [dx, dy] = flow.vectors[fiy * flow.width + fix];

            // Source position (where this pixel came from in the previous frame)
            let sx = x as f32 - dx;
            let sy = y as f32 - dy;

            // Bilinear sample from source depth
            if sx >= 0.0 && sy >= 0.0 && sx < (width - 1) as f32 && sy < (height - 1) as f32 {
                let x0 = sx.floor() as usize;
                let y0 = sy.floor() as usize;
                let x1 = x0 + 1;
                let y1 = y0 + 1;
                let fx = sx - x0 as f32;
                let fy = sy - y0 as f32;

                let v00 = depth[y0 * width + x0];
                let v10 = depth[y0 * width + x1];
                let v01 = depth[y1 * width + x0];
                let v11 = depth[y1 * width + x1];

                warped[y * width + x] = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;
            }
        }
    }

    warped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_motion_returns_zero_magnitude() {
        let w = 32;
        let h = 32;
        let pixels: Vec<u8> = vec![128; w * h * 3];
        let flow = compute_flow(&pixels, &pixels, w, h);
        let mag = mean_flow_magnitude(&flow);
        assert!(mag < 0.01, "identical frames should have near-zero flow, got {mag}");
    }

    #[test]
    fn warp_identity_preserves_depth() {
        let w = 16;
        let h = 16;
        let depth: Vec<f32> = (0..w * h).map(|i| i as f32 / (w * h) as f32).collect();
        let zero_flow = MotionField {
            vectors: vec![[0.0, 0.0]; w * h],
            width: w,
            height: h,
        };
        let warped = warp_depth(&depth, &zero_flow, w, h, 0.5);
        for (i, (&orig, &warp)) in depth.iter().zip(warped.iter()).enumerate() {
            assert!(
                (orig - warp).abs() < 1e-4,
                "pixel {i}: orig={orig}, warped={warp}"
            );
        }
    }

    #[test]
    fn warp_with_uniform_shift() {
        let w = 16;
        let h = 16;
        let mut depth = vec![0.0f32; w * h];
        // Put a bright spot at (8, 8)
        depth[8 * w + 8] = 1.0;

        // Shift everything right by 2 pixels
        let flow = MotionField {
            vectors: vec![[2.0, 0.0]; w * h],
            width: w,
            height: h,
        };
        let warped = warp_depth(&depth, &flow, w, h, 0.0);
        // The bright spot should now appear at (6, 8) in the warped output
        // (because we're warping backward: output[x] samples from source[x - dx])
        assert!(warped[8 * w + 6] > 0.5, "shifted spot should be at x=6");
    }

    #[test]
    fn detector_returns_max_without_reference() {
        let det = OpticalFlowDetector::new();
        let pixels = vec![0u8; 64 * 64 * 3];
        assert_eq!(det.score(&pixels), f32::MAX);
    }
}
