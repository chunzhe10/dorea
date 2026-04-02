//! Apply depth-stratified LUTs via trilinear interpolation + depth blending.
//!
//! Ported from `run_fixed_hsl_lut_poc.py::apply_depth_luts`.

use crate::types::{DepthLuts, LutGrid};

/// Trilinear interpolation through a single LutGrid for one pixel.
fn trilinear(lut: &LutGrid, rgb: [f32; 3]) -> [f32; 3] {
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
/// 1. Compute soft weight for each zone: `w_z = max(1 - |d - center_z| / zone_width, 0)`
/// 2. For each zone with w_z > 0, apply LUT via trilinear interpolation
/// 3. Blend: `result = Σ(lut_result_z × w_z) / Σ(w_z)`
pub fn apply_depth_luts(
    pixels: &[[f32; 3]],
    depth: &[f32],
    luts: &DepthLuts,
) -> Vec<[f32; 3]> {
    let n_zones = luts.n_zones();
    let zone_width = 1.0_f32 / n_zones as f32;

    let mut result = vec![[0.0_f32; 3]; pixels.len()];

    for (idx, (&px, &d)) in pixels.iter().zip(depth.iter()).enumerate() {
        let mut acc = [0.0_f32; 3];
        let mut total_w = 0.0_f32;

        for z in 0..n_zones {
            let dist = (d - luts.zone_centers[z]).abs();
            let w = (1.0 - dist / zone_width).max(0.0);
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
            for (c, val) in acc.iter().enumerate() {
                result[idx][c] = (val / total_w).clamp(0.0, 1.0);
            }
        } else {
            result[idx] = px;
        }
    }

    result
}
