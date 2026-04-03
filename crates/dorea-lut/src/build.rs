//! Build depth-stratified LUTs from keyframe data.
//!
//! Ported from `run_fixed_hsl_lut_poc.py::build_fixed_depth_luts`.

use crate::types::{DepthLuts, LutGrid};
use rayon::prelude::*;

pub const LUT_SIZE: usize = 33;
pub const N_DEPTH_ZONES: usize = 5;
pub const EDGE_SCALE: f32 = 0.3;
pub const CONTRAST_SCALE: f32 = 0.3;

/// Input data for one calibration keyframe.
pub struct KeyframeData {
    /// Original sRGB image, pixel values in [0.0, 1.0], length = width * height
    pub original: Vec<[f32; 3]>,
    /// RAUNE-Net target sRGB, same shape
    pub target: Vec<[f32; 3]>,
    /// Depth map, values in [0.0, 1.0], same shape
    pub depth: Vec<f32>,
    /// Per-pixel importance weights (pre-computed)
    pub importance: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

/// Compute per-pixel importance weights from depth map.
///
/// importance = depth × (1 + EDGE_SCALE × edge_norm) × (1 + CONTRAST_SCALE × contrast_norm)
/// floored at 0.1
pub fn compute_importance(depth: &[f32], width: usize, height: usize) -> Vec<f32> {
    let n = width * height;
    assert_eq!(depth.len(), n);
    assert_eq!(n, width * height, "n ({n}) must equal width ({width}) × height ({height})");

    // --- Sobel edge detection (row-parallel) ---
    let mut edge_mag = vec![0.0_f32; n];
    edge_mag
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, out) in row.iter_mut().enumerate() {
                let mut dx = 0.0_f32;
                let mut dy = 0.0_f32;
                for ky in -1_i32..=1 {
                    for kx in -1_i32..=1 {
                        let ny = (y as i32 + ky).clamp(0, height as i32 - 1) as usize;
                        let nx = (x as i32 + kx).clamp(0, width as i32 - 1) as usize;
                        let v = depth[ny * width + nx];
                        let kx_w = kx as f32;
                        let ky_w = ky as f32;
                        let wx = kx_w * (2.0 - ky_w.abs());
                        let wy = ky_w * (2.0 - kx_w.abs());
                        dx += v * wx;
                        dy += v * wy;
                    }
                }
                *out = (dx * dx + dy * dy).sqrt();
            }
        });

    // --- Gaussian blur σ=8 on edge magnitude (separated 1D) ---
    let sigma = 8.0_f32;
    let radius = (3.0 * sigma).ceil() as usize; // = 24
    // Build 1D kernel
    let kernel_len = 2 * radius + 1;
    let mut kernel = vec![0.0_f32; kernel_len];
    let mut kernel_sum = 0.0_f32;
    for (i, slot) in kernel.iter_mut().enumerate() {
        let offset = i as f32 - radius as f32;
        let v = (-offset * offset / (2.0 * sigma * sigma)).exp();
        *slot = v;
        kernel_sum += v;
    }
    for k in kernel.iter_mut() {
        *k /= kernel_sum;
    }

    // Horizontal pass (row-parallel)
    let mut temp = vec![0.0_f32; n];
    temp.par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, out) in row.iter_mut().enumerate() {
                let mut acc = 0.0_f32;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let nx = (x as i32 + ki as i32 - radius as i32)
                        .clamp(0, width as i32 - 1) as usize;
                    acc += edge_mag[y * width + nx] * kv;
                }
                *out = acc;
            }
        });
    // Vertical pass (row-parallel).
    // IMPORTANT: Must stay sequential with the H-pass above — `temp` is fully written
    // by H-pass `for_each` (which blocks) before this begins. Do NOT run H+V concurrently
    // (e.g. via rayon::join) — that would race on `temp`.
    let mut edge_dilated = vec![0.0_f32; n];
    edge_dilated
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..width {
                let mut acc = 0.0_f32;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let ny = (y as i32 + ki as i32 - radius as i32)
                        .clamp(0, height as i32 - 1) as usize;
                    acc += temp[ny * width + x] * kv;
                }
                row[x] = acc;
            }
        });

    let edge_max = edge_dilated.iter().cloned().fold(0.0_f32, f32::max).max(1e-6);
    let edge_norm: Vec<f32> = edge_dilated.iter().map(|&v| v / edge_max).collect();

    // --- Local variance via box filter (31×31 window) ---
    // Use integral image for efficient box filtering
    let box_r = 15_usize; // half-size of 31×31 window
    let local_var = local_variance_box(depth, width, height, box_r);

    let std_max = local_var.iter().cloned().map(|v| v.sqrt()).fold(0.0_f32, f32::max).max(1e-6);
    let contrast_norm: Vec<f32> = local_var.iter().map(|&v| v.sqrt() / std_max).collect();

    // --- Combine ---
    let mut importance = vec![0.0_f32; n];
    for i in 0..n {
        let imp = depth[i]
            * (1.0 + EDGE_SCALE * edge_norm[i])
            * (1.0 + CONTRAST_SCALE * contrast_norm[i]);
        importance[i] = imp.max(0.1);
    }
    importance
}

/// Compute local variance in a (2*radius+1)² window using integral images.
fn local_variance_box(data: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let n = width * height;
    // Build integral images for sum and sum-of-squares
    let mut isum = vec![0.0_f64; (width + 1) * (height + 1)];
    let mut isumsq = vec![0.0_f64; (width + 1) * (height + 1)];
    let w1 = width + 1;

    for y in 0..height {
        for x in 0..width {
            let v = data[y * width + x] as f64;
            isum[(y + 1) * w1 + (x + 1)] = v
                + isum[y * w1 + (x + 1)]
                + isum[(y + 1) * w1 + x]
                - isum[y * w1 + x];
            isumsq[(y + 1) * w1 + (x + 1)] = v * v
                + isumsq[y * w1 + (x + 1)]
                + isumsq[(y + 1) * w1 + x]
                - isumsq[y * w1 + x];
        }
    }

    let mut var_out = vec![0.0_f32; n];
    for y in 0..height {
        for x in 0..width {
            let y0 = y.saturating_sub(radius);
            let x0 = x.saturating_sub(radius);
            let y1 = (y + radius + 1).min(height);
            let x1 = (x + radius + 1).min(width);
            let count = ((y1 - y0) * (x1 - x0)) as f64;

            let s = isum[y1 * w1 + x1] - isum[y0 * w1 + x1]
                - isum[y1 * w1 + x0] + isum[y0 * w1 + x0];
            let sq = isumsq[y1 * w1 + x1] - isumsq[y0 * w1 + x1]
                - isumsq[y1 * w1 + x0] + isumsq[y0 * w1 + x0];

            let mean = s / count;
            let var = (sq / count - mean * mean).max(0.0);
            var_out[y * width + x] = var as f32;
        }
    }
    var_out
}

/// Index coordinates of a single LUT cell (r, g, b) in the [0, LUT_SIZE) grid.
type CellIdx = [usize; 3];

/// Find the nearest populated LUT cell to `empty_cell` by brute-force L2 in index space.
///
/// `populated` must be non-empty. Returns (empty_cell, best_populated_cell).
fn find_nearest(empty_cell: &CellIdx, populated: &[CellIdx]) -> (CellIdx, CellIdx) {
    let mut best_dist = u64::MAX;
    let mut best_pop = populated[0];
    for &pc in populated {
        let dr = (empty_cell[0] as i64 - pc[0] as i64).pow(2) as u64;
        let dg = (empty_cell[1] as i64 - pc[1] as i64).pow(2) as u64;
        let db = (empty_cell[2] as i64 - pc[2] as i64).pow(2) as u64;
        let dist = dr + dg + db;
        if dist < best_dist {
            best_dist = dist;
            best_pop = pc;
        }
    }
    (*empty_cell, best_pop)
}

/// Build adaptive zone boundaries from depth distribution.
///
/// Returns n_zones+1 boundary values ensuring each zone covers ~equal fraction of pixels.
pub fn adaptive_zone_boundaries(all_depths: &[f32], n_zones: usize) -> Vec<f32> {
    // C2: guard against empty input — return uniform linspace.
    if all_depths.is_empty() {
        return (0..=n_zones)
            .map(|i| i as f32 / n_zones as f32)
            .collect();
    }

    let mut sorted = all_depths.to_vec();
    // I2: use total_cmp to handle NaN values without panic.
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();

    let mut boundaries = Vec::with_capacity(n_zones + 1);
    boundaries.push(0.0_f32);
    for z in 1..n_zones {
        let idx = (z * n) / n_zones;
        // C2: use saturating_sub to avoid underflow if n == 0 (already guarded above).
        let idx = idx.min(n.saturating_sub(1));
        boundaries.push(sorted[idx]);
    }
    boundaries.push(1.0_f32);
    boundaries
}

/// Streaming LUT builder — accumulates keyframe contributions one frame at a time.
///
/// Accepts pre-computed `zone_boundaries` (e.g. from reservoir-sampled depths).
/// Call `add_frame` for each keyframe, then `finish()` to produce the `DepthLuts`.
/// Peak RAM = O(LUT_SIZE³ × n_zones) ≈ a few MB, regardless of frame count.
pub struct StreamingLutBuilder {
    zone_boundaries: Vec<f32>,
    n_zones: usize,
    /// Per-zone weighted RGB sums: lut_wsums[z][cell * 3 + c]
    lut_wsums: Vec<Vec<f64>>,
    /// Per-zone weighted counts: lut_wcounts[z][cell]
    lut_wcounts: Vec<Vec<f64>>,
}

impl StreamingLutBuilder {
    pub fn new(zone_boundaries: Vec<f32>) -> Self {
        let n_zones = zone_boundaries.len().saturating_sub(1);
        let n_cells = LUT_SIZE * LUT_SIZE * LUT_SIZE;
        Self {
            zone_boundaries,
            n_zones,
            lut_wsums: vec![vec![0.0_f64; n_cells * 3]; n_zones],
            lut_wcounts: vec![vec![0.0_f64; n_cells]; n_zones],
        }
    }

    /// Accumulate one keyframe's pixels into the LUT accumulators.
    pub fn add_frame(
        &mut self,
        original: &[[f32; 3]],
        target: &[[f32; 3]],
        depth: &[f32],
        importance: &[f32],
    ) {
        let n_px = original.len();
        for z in 0..self.n_zones {
            let d_lo = self.zone_boundaries[z];
            let d_hi = self.zone_boundaries[z + 1];
            let is_last = z == self.n_zones - 1;
            let wsum = &mut self.lut_wsums[z];
            let wcount = &mut self.lut_wcounts[z];

            for i in 0..n_px {
                let d = depth[i];
                let in_zone = if is_last { d >= d_lo } else { d >= d_lo && d < d_hi };
                if !in_zone {
                    continue;
                }
                let orig = original[i];
                let tgt = target[i];
                let w = importance[i] as f64;

                let ri = ((orig[0] * (LUT_SIZE as f32 - 1.0)).round() as usize).min(LUT_SIZE - 1);
                let gi = ((orig[1] * (LUT_SIZE as f32 - 1.0)).round() as usize).min(LUT_SIZE - 1);
                let bi = ((orig[2] * (LUT_SIZE as f32 - 1.0)).round() as usize).min(LUT_SIZE - 1);

                let cell = ri * LUT_SIZE * LUT_SIZE + gi * LUT_SIZE + bi;
                wsum[cell * 3]     += tgt[0] as f64 * w;
                wsum[cell * 3 + 1] += tgt[1] as f64 * w;
                wsum[cell * 3 + 2] += tgt[2] as f64 * w;
                wcount[cell] += w;
            }
        }
    }

    /// Normalise, apply NN fill, and return the completed `DepthLuts`.
    pub fn finish(self) -> DepthLuts {
        let mut luts = Vec::with_capacity(self.n_zones);

        for z in 0..self.n_zones {
            let wsum = &self.lut_wsums[z];
            let wcount = &self.lut_wcounts[z];

            let mut lut = LutGrid::new(LUT_SIZE);
            let mut populated: Vec<[usize; 3]> = Vec::new();
            let mut empty: Vec<[usize; 3]> = Vec::new();

            for ri in 0..LUT_SIZE {
                for gi in 0..LUT_SIZE {
                    for bi in 0..LUT_SIZE {
                        let cell = ri * LUT_SIZE * LUT_SIZE + gi * LUT_SIZE + bi;
                        let wc = wcount[cell];
                        if wc > 0.0 {
                            lut.set(ri, gi, bi, [
                                (wsum[cell * 3]     / wc) as f32,
                                (wsum[cell * 3 + 1] / wc) as f32,
                                (wsum[cell * 3 + 2] / wc) as f32,
                            ]);
                            populated.push([ri, gi, bi]);
                        } else {
                            empty.push([ri, gi, bi]);
                        }
                    }
                }
            }

            if !populated.is_empty() && !empty.is_empty() {
                let filled: Vec<([usize; 3], [usize; 3])> = empty
                    .par_iter()
                    .map(|ec| find_nearest(ec, &populated))
                    .collect();
                for (ec, best_pop) in filled {
                    let val = lut.get(best_pop[0], best_pop[1], best_pop[2]);
                    lut.set(ec[0], ec[1], ec[2], val);
                }
            } else if populated.is_empty() {
                for ri in 0..LUT_SIZE {
                    for gi in 0..LUT_SIZE {
                        for bi in 0..LUT_SIZE {
                            let r = ri as f32 / (LUT_SIZE - 1) as f32;
                            let g = gi as f32 / (LUT_SIZE - 1) as f32;
                            let b = bi as f32 / (LUT_SIZE - 1) as f32;
                            lut.set(ri, gi, bi, [r, g, b]);
                        }
                    }
                }
            }

            luts.push(lut);
        }

        DepthLuts::new(luts, self.zone_boundaries)
    }
}

/// Build depth-stratified LUTs from keyframe data.
///
/// Returns `DepthLuts` with `N_DEPTH_ZONES` zones, σ=0 (no Gaussian smoothing), NN fill.
pub fn build_depth_luts(keyframes: &[KeyframeData]) -> DepthLuts {
    // I7: validate that all keyframe slices have consistent lengths.
    for (i, kd) in keyframes.iter().enumerate() {
        assert_eq!(kd.original.len(), kd.target.len(),     "keyframe {i}: original and target length mismatch");
        assert_eq!(kd.original.len(), kd.depth.len(),      "keyframe {i}: original and depth length mismatch");
        assert_eq!(kd.original.len(), kd.importance.len(), "keyframe {i}: original and importance length mismatch");
    }

    let all_depths: Vec<f32> = keyframes.iter().flat_map(|kd| kd.depth.iter().cloned()).collect();
    let zone_boundaries = adaptive_zone_boundaries(&all_depths, N_DEPTH_ZONES);

    let mut builder = StreamingLutBuilder::new(zone_boundaries);
    for kd in keyframes {
        builder.add_frame(&kd.original, &kd.target, &kd.depth, &kd.importance);
    }
    builder.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nn_fill_no_identity() {
        // Build a simple keyframe where all pixels have red channel boosted.
        // Target always has r = orig.r + 0.2 (clamped), same g/b.
        // Verify no cell has identity mapping for the red channel in the populated region.
        let size = 4;
        let mut original = Vec::new();
        let mut target = Vec::new();
        let mut depth = Vec::new();
        let mut importance = Vec::new();

        // Use a grid of RGB values that uniformly samples the input space
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let rf = r as f32 / (size - 1) as f32;
                    let gf = g as f32 / (size - 1) as f32;
                    let bf = b as f32 / (size - 1) as f32;
                    original.push([rf, gf, bf]);
                    target.push([(rf + 0.2).min(1.0), gf, bf]);
                    depth.push(0.5); // all in middle zone
                    importance.push(1.0);
                }
            }
        }

        let kf = KeyframeData {
            original,
            target,
            depth,
            importance,
            width: size * size,
            height: size,
        };

        let depth_luts = build_depth_luts(&[kf]);

        // All pixels have depth=0.5, so with adaptive boundaries all land in the same zone.
        // Find the zone that actually has data (populated cells) by checking which has non-zero
        // values distinct from the identity mapping.
        // With adaptive quantiles and all depths equal, zone 4 (last) captures all pixels.
        // Search all zones for the populated one.
        let populated_zone = (0..N_DEPTH_ZONES)
            .find(|&z| {
                // A populated zone will have r ≈ 0.867 for cell [21,0,0] (boosted from 2/3).
                // An identity-filled zone will have cell[21,0,0][0] ≈ 21/32 ≈ 0.656.
                let v = depth_luts.luts[z].get(21, 0, 0)[0];
                (v - 0.656).abs() > 0.1
            })
            .expect("No populated zone found — all zones appear to be identity");

        let lut = &depth_luts.luts[populated_zone];

        // Check populated cell at index 21 (input r ≈ 2/3 = 0.6667, g=0, b=0)
        // Target r = min(0.6667 + 0.2, 1.0) = 0.8667
        let cell_21 = lut.get(21, 0, 0);
        let input_r_21 = 2.0_f32 / 3.0;
        let expected_r_21 = (input_r_21 + 0.2).min(1.0); // ≈ 0.867
        assert!(
            (cell_21[0] - expected_r_21).abs() < 0.05,
            "Zone {populated_zone} cell[21,0,0] output_r: expected ≈ {expected_r_21:.3}, got {:.3}",
            cell_21[0]
        );
        // Verify it's not identity
        assert!(
            (cell_21[0] - input_r_21).abs() > 0.1,
            "Zone {populated_zone} cell[21,0,0] appears to be identity: input {input_r_21:.3}, output {:.3}",
            cell_21[0]
        );

        // Check pure-red cell (index 32): input r=1.0, target r=min(1.2, 1.0)=1.0
        let cell_32 = lut.get(LUT_SIZE - 1, 0, 0);
        assert!(
            (cell_32[0] - 1.0_f32).abs() < 0.05,
            "Zone {populated_zone} cell[32,0,0] output_r: expected ≈ 1.0, got {:.3}",
            cell_32[0]
        );
        // Green/blue channels should be ~0 for pure-red variants
        assert!(
            cell_32[1].abs() < 0.05,
            "Green should be ~0 for all-red input, got {:.3}",
            cell_32[1]
        );
    }
}
