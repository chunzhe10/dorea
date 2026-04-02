//! Apply HSL qualifier corrections to image pixels.
//!
//! Ported from `run_fixed_hsl_lut_poc.py::apply_hsl_corrections`.

use dorea_color::hsv::{hsv_to_rgb, rgb_to_hsv};

use crate::derive::HslCorrections;
use crate::qualifiers::MIN_SATURATION;

/// Apply HSL qualifier corrections to image pixels.
///
/// Each active correction (weight ≥ MIN_WEIGHT) applies a soft-masked
/// hue offset, saturation multiplier, and value offset.
pub fn apply_hsl_corrections(
    pixels: &[[f32; 3]],
    corrections: &HslCorrections,
) -> Vec<[f32; 3]> {
    pixels
        .iter()
        .map(|&[r, g, b]| {
            let (mut h, mut s, mut v) = rgb_to_hsv(r, g, b);

            for corr in &corrections.0 {
                if corr.weight < crate::qualifiers::MIN_WEIGHT {
                    continue;
                }

                let hc = corr.h_center;
                let hw = corr.h_width;

                let h_dist = {
                    let diff = (h - hc).abs();
                    diff.min(360.0 - diff)
                };
                let soft_mask = (1.0 - h_dist / hw).max(0.0)
                    * if s > MIN_SATURATION { 1.0 } else { 0.0 };

                h += corr.h_offset * soft_mask;
                s = (s * (1.0 + (corr.s_ratio - 1.0) * soft_mask)).clamp(0.0, 1.0);
                v = (v + corr.v_offset * soft_mask).clamp(0.0, 1.0);
            }

            // Wrap hue to [0, 360)
            h = h.rem_euclid(360.0);

            let (ro, go, bo) = hsv_to_rgb(h, s, v);
            [ro, go, bo]
        })
        .collect()
}
