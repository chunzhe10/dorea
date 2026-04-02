//! Derive per-qualifier HSL corrections from (lut_output, raune_target) pair.
//!
//! Ported from `run_fixed_hsl_lut_poc.py::derive_hsl_corrections`.

use dorea_color::hsv::rgb_to_hsv;

use crate::qualifiers::{HSL_QUALIFIERS, MIN_SATURATION, MIN_WEIGHT};

/// Per-qualifier HSL correction parameters.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QualifierCorrection {
    pub h_center: f32,
    pub h_width: f32,
    /// Degrees to add to hue
    pub h_offset: f32,
    /// Saturation multiplier
    pub s_ratio: f32,
    /// Value offset in [0, 1]
    pub v_offset: f32,
    /// Total weight; 0 = inactive qualifier
    pub weight: f32,
}

/// Collection of per-qualifier corrections.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HslCorrections(pub Vec<QualifierCorrection>);

/// Derive per-qualifier HSL corrections from (lut_output, raune_target) pixel pairs.
///
/// Both slices must be the same length. Values in [0.0, 1.0].
pub fn derive_hsl_corrections(
    lut_output: &[[f32; 3]],
    target: &[[f32; 3]],
) -> HslCorrections {
    assert_eq!(
        lut_output.len(),
        target.len(),
        "lut_output ({}) and target ({}) must have the same length",
        lut_output.len(),
        target.len()
    );

    // Convert to HSV
    let lut_hsv: Vec<(f32, f32, f32)> = lut_output
        .iter()
        .map(|&[r, g, b]| rgb_to_hsv(r, g, b))
        .collect();
    let tgt_hsv: Vec<(f32, f32, f32)> = target
        .iter()
        .map(|&[r, g, b]| rgb_to_hsv(r, g, b))
        .collect();

    let mut corrections = Vec::with_capacity(HSL_QUALIFIERS.len());

    for qual in HSL_QUALIFIERS {
        let hc = qual.h_center;
        let hw = qual.h_width;

        // Build soft qualifier mask
        let mut total_weight = 0.0_f64;
        let mut weights: Vec<f32> = Vec::with_capacity(lut_hsv.len());

        for &(lh, ls, _lv) in &lut_hsv {
            let h_dist = angular_dist(lh, hc);
            let mask = (1.0 - h_dist / hw).max(0.0) * if ls > MIN_SATURATION { 1.0 } else { 0.0 };
            weights.push(mask);
            total_weight += mask as f64;
        }

        if total_weight < MIN_WEIGHT as f64 {
            corrections.push(QualifierCorrection {
                h_center: hc,
                h_width: hw,
                h_offset: 0.0,
                s_ratio: 1.0,
                v_offset: 0.0,
                weight: 0.0,
            });
            continue;
        }

        // Weighted hue difference (circular, wrapped to [-180, 180])
        let mut h_offset_sum = 0.0_f64;
        let mut lut_s_sum = 0.0_f64;
        let mut tgt_s_sum = 0.0_f64;
        let mut v_diff_sum = 0.0_f64;

        for (i, &w) in weights.iter().enumerate() {
            if w < 1e-7 {
                continue;
            }
            let w = w as f64;
            let (lh, ls, lv) = lut_hsv[i];
            let (th, ts, tv) = tgt_hsv[i];

            let h_diff = wrap_hue_diff(th - lh);
            h_offset_sum += h_diff as f64 * w;
            lut_s_sum += ls as f64 * w;
            tgt_s_sum += ts as f64 * w;
            v_diff_sum += (tv - lv) as f64 * w;
        }

        let h_offset = (h_offset_sum / total_weight) as f32;
        let lut_s_mean = (lut_s_sum / total_weight) as f32;
        let tgt_s_mean = (tgt_s_sum / total_weight) as f32;
        let s_ratio = tgt_s_mean / lut_s_mean.max(1e-6);
        let v_offset = (v_diff_sum / total_weight) as f32;

        corrections.push(QualifierCorrection {
            h_center: hc,
            h_width: hw,
            h_offset,
            s_ratio,
            v_offset,
            weight: total_weight as f32,
        });
    }

    HslCorrections(corrections)
}

/// Angular distance between two hue values (degrees), wrapped to [0, 180].
#[allow(dead_code)]
#[inline]
fn angular_dist(h: f32, center: f32) -> f32 {
    let diff = (h - center).abs();
    diff.min(360.0 - diff)
}

/// Wrap a hue difference to [-180, 180].
#[inline]
fn wrap_hue_diff(diff: f32) -> f32 {
    (diff + 180.0).rem_euclid(360.0) - 180.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use dorea_color::hsv::hsv_to_rgb;

    /// I5c: When lut_output == target, all corrections should be zero/identity.
    #[test]
    fn test_derive_same_image() {
        // Build a diverse set of saturated pixels that will activate qualifiers.
        let mut pixels: Vec<[f32; 3]> = Vec::new();
        // Generate pixels in different hue ranges with sufficient saturation.
        for &(h, s, v) in &[
            (0.0_f32, 0.8, 0.6),    // Red
            (10.0, 0.9, 0.7),
            (40.0, 0.85, 0.6),      // Yellow
            (50.0, 0.9, 0.7),
            (100.0, 0.8, 0.6),      // Green
            (115.0, 0.9, 0.7),
            (170.0, 0.85, 0.65),    // Cyan
            (180.0, 0.9, 0.6),
            (210.0, 0.8, 0.7),      // Blue
            (220.0, 0.9, 0.65),
            (290.0, 0.85, 0.6),     // Magenta
            (300.0, 0.9, 0.7),
        ] {
            // Repeat each pixel many times to exceed MIN_WEIGHT.
            for _ in 0..20 {
                let (r, g, b) = hsv_to_rgb(h, s, v);
                pixels.push([r, g, b]);
            }
        }

        // When lut_output == target, corrections must be identity.
        let corrs = derive_hsl_corrections(&pixels, &pixels);
        for (i, c) in corrs.0.iter().enumerate() {
            if c.weight < crate::qualifiers::MIN_WEIGHT {
                continue;
            }
            assert!(
                c.h_offset.abs() < 0.5,
                "qualifier {i}: h_offset should be ~0, got {}",
                c.h_offset
            );
            assert!(
                (c.s_ratio - 1.0).abs() < 0.01,
                "qualifier {i}: s_ratio should be ~1.0, got {}",
                c.s_ratio
            );
            assert!(
                c.v_offset.abs() < 0.01,
                "qualifier {i}: v_offset should be ~0, got {}",
                c.v_offset
            );
        }
    }
}
