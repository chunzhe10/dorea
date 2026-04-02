//! Apply depth-stratified LUTs via trilinear interpolation + depth blending.
//!
//! Ported from `run_fixed_hsl_lut_poc.py::apply_depth_luts`.

use crate::types::{DepthLuts, LutGrid};
use rayon::prelude::*;

/// Trilinear interpolation through a single LutGrid for one pixel.
fn trilinear(lut: &LutGrid, rgb: [f32; 3]) -> [f32; 3] {
    // L8: LUT must have at least 2 entries per axis for interpolation.
    debug_assert!(lut.size >= 2, "LutGrid must have size >= 2 for trilinear interpolation");
    let size = lut.size;
    let scale = (size - 1) as f32;

    let sr = (rgb[0] * scale).clamp(0.0, scale);
    let sg = (rgb[1] * scale).clamp(0.0, scale);
    let sb = (rgb[2] * scale).clamp(0.0, scale);

    let i0 = (sr.floor() as usize).min(size - 2);
    let j0 = (sg.floor() as usize).min(size - 2);
    let k0 = (sb.floor() as usize).min(size - 2);

    let fr = (sr - i0 as f32).clamp(0.0, 1.0);
    let fg = (sg - j0 as f32).clamp(0.0, 1.0);
    let fb = (sb - k0 as f32).clamp(0.0, 1.0);

    let i1 = i0 + 1;
    let j1 = j0 + 1;
    let k1 = k0 + 1;

    // 8 corners
    let c000 = lut.get(i0, j0, k0);
    let c001 = lut.get(i0, j0, k1);
    let c010 = lut.get(i0, j1, k0);
    let c011 = lut.get(i0, j1, k1);
    let c100 = lut.get(i1, j0, k0);
    let c101 = lut.get(i1, j0, k1);
    let c110 = lut.get(i1, j1, k0);
    let c111 = lut.get(i1, j1, k1);

    // Trilinear blend
    let mut out = [0.0_f32; 3];
    for c in 0..3 {
        let v00 = c000[c] * (1.0 - fb) + c001[c] * fb;
        let v01 = c010[c] * (1.0 - fb) + c011[c] * fb;
        let v10 = c100[c] * (1.0 - fb) + c101[c] * fb;
        let v11 = c110[c] * (1.0 - fb) + c111[c] * fb;

        let v0 = v00 * (1.0 - fg) + v01 * fg;
        let v1 = v10 * (1.0 - fg) + v11 * fg;

        out[c] = v0 * (1.0 - fr) + v1 * fr;
    }
    out
}

/// Apply depth-stratified LUTs to an image using trilinear interpolation + depth blending.
///
/// For each pixel:
/// 1. Compute soft weight for each zone: `w_z = max(1 - |d - center_z| / zone_width_z, 0)`
///    where `zone_width_z` is the actual adaptive width of zone z from `luts.zone_boundaries`.
/// 2. For each zone with w_z > 0, apply LUT via trilinear interpolation
/// 3. Blend: `result = Σ(lut_result_z × w_z) / Σ(w_z)`
pub fn apply_depth_luts(
    pixels: &[[f32; 3]],
    depth: &[f32],
    luts: &DepthLuts,
) -> Vec<[f32; 3]> {
    // C3: pixels and depth must have the same length.
    assert_eq!(
        pixels.len(),
        depth.len(),
        "pixels ({}) and depth ({}) must have same length",
        pixels.len(),
        depth.len()
    );

    let n_zones = luts.n_zones();

    let mut result = vec![[0.0_f32; 3]; pixels.len()];

    // `luts` is a shared `&DepthLuts` across threads. DepthLuts contains only
    // Vec<LutGrid> + Vec<f32>, which are Sync — safe to share across rayon threads.
    result
        .par_iter_mut()
        .zip(pixels.par_iter())
        .zip(depth.par_iter())
        .for_each(|((out, &px), &d)| {
            let mut acc = [0.0_f32; 3];
            let mut total_w = 0.0_f32;

            for z in 0..n_zones {
                let zone_width = luts.zone_boundaries[z + 1] - luts.zone_boundaries[z];
                let dist = (d - luts.zone_centers[z]).abs();
                let w = (1.0 - dist / zone_width.max(1e-6)).max(0.0);
                if w < 1e-7 {
                    continue;
                }
                let lut_out = trilinear(&luts.luts[z], px);
                for (c, lo) in lut_out.iter().enumerate() {
                    acc[c] += lo * w;
                }
                total_w += w;
            }

            if total_w > 1e-6 {
                for c in 0..3 {
                    out[c] = (acc[c] / total_w).clamp(0.0, 1.0);
                }
            } else {
                *out = px;
            }
        });

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::{adaptive_zone_boundaries, N_DEPTH_ZONES};

    // ---------------------------------------------------------------------------
    // I5a: test_apply_identity_lut
    // ---------------------------------------------------------------------------

    /// Build an identity LutGrid of given size (every cell maps input to itself).
    fn identity_lut(size: usize) -> LutGrid {
        let mut lut = LutGrid::new(size);
        for ri in 0..size {
            for gi in 0..size {
                for bi in 0..size {
                    let r = ri as f32 / (size - 1) as f32;
                    let g = gi as f32 / (size - 1) as f32;
                    let b = bi as f32 / (size - 1) as f32;
                    lut.set(ri, gi, bi, [r, g, b]);
                }
            }
        }
        lut
    }

    #[test]
    fn test_apply_identity_lut() {
        // Build DepthLuts with 5 identity zones.
        let n_zones = 5;
        let luts: Vec<LutGrid> = (0..n_zones).map(|_| identity_lut(17)).collect();
        let boundaries: Vec<f32> = (0..=n_zones).map(|i| i as f32 / n_zones as f32).collect();
        let depth_luts = DepthLuts::new(luts, boundaries);

        let pixels: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.3, 0.8],
            [0.25, 0.75, 0.1],
            [1.0, 1.0, 1.0],
        ];
        // Place each pixel at the center of its zone so blending is deterministic.
        let depth: Vec<f32> = pixels.iter().enumerate().map(|(i, _)| {
            let z = i % n_zones;
            (depth_luts.zone_boundaries[z] + depth_luts.zone_boundaries[z + 1]) / 2.0
        }).collect();

        let result = apply_depth_luts(&pixels, &depth, &depth_luts);

        for (i, (orig, out)) in pixels.iter().zip(result.iter()).enumerate() {
            for c in 0..3 {
                assert!(
                    (orig[c] - out[c]).abs() < 1e-4,
                    "pixel {i} channel {c}: expected {:.4} got {:.4}",
                    orig[c],
                    out[c]
                );
            }
        }
    }

    // ---------------------------------------------------------------------------
    // I5b: test_adaptive_zone_boundaries_edge_cases
    // ---------------------------------------------------------------------------

    #[test]
    fn test_adaptive_zone_boundaries_empty_input() {
        let bounds = adaptive_zone_boundaries(&[], N_DEPTH_ZONES);
        assert_eq!(bounds.len(), N_DEPTH_ZONES + 1);
        // Should be uniform linspace 0, 0.2, 0.4, 0.6, 0.8, 1.0
        for (i, &b) in bounds.iter().enumerate() {
            let expected = i as f32 / N_DEPTH_ZONES as f32;
            assert!(
                (b - expected).abs() < 1e-6,
                "bounds[{i}] expected {expected}, got {b}"
            );
        }
    }

    #[test]
    fn test_adaptive_zone_boundaries_all_equal() {
        // All depths equal — quantile boundaries degenerate to the same value.
        let depths = vec![0.5_f32; 100];
        let bounds = adaptive_zone_boundaries(&depths, N_DEPTH_ZONES);
        assert_eq!(bounds.len(), N_DEPTH_ZONES + 1);
        // First boundary must be 0.0 and last must be 1.0.
        assert!((bounds[0] - 0.0).abs() < 1e-6);
        assert!((bounds[N_DEPTH_ZONES] - 1.0).abs() < 1e-6);
        // Interior boundaries all equal the single depth value.
        for &b in &bounds[1..N_DEPTH_ZONES] {
            assert!(
                (b - 0.5).abs() < 1e-6,
                "interior boundary should be 0.5 for uniform depths, got {b}"
            );
        }
    }

    #[test]
    fn test_adaptive_zone_boundaries_single_value() {
        let bounds = adaptive_zone_boundaries(&[0.3_f32], N_DEPTH_ZONES);
        assert_eq!(bounds.len(), N_DEPTH_ZONES + 1);
        assert!((bounds[0] - 0.0).abs() < 1e-6);
        assert!((bounds[N_DEPTH_ZONES] - 1.0).abs() < 1e-6);
    }
}
