//! OKLab colorspace conversion, rescaled to CIELab-compatible ranges.
//!
//! Uses Björn Ottosson's OKLab (2020) internally, with output rescaled to match
//! CIELab numeric ranges so downstream consumers (ambiance, warmth, vibrance)
//! need no constant changes:
//!   L: OKLab [0,1] × 100 → [0,100]
//!   a: OKLab [-0.4,0.4] × 300 → [-120,120]
//!   b: OKLab [-0.4,0.4] × 300 → [-120,120]
//!
//! sRGB linearization uses IEC 61966-2-1 piecewise gamma (unchanged).

// OKLab rescale factors — map OKLab ranges to CIELab-compatible ranges.
const L_SCALE: f32 = 100.0;
const AB_SCALE: f32 = 300.0;

/// sRGB component → linear light (IEC 61966-2-1).
#[inline]
fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear light → sRGB component (IEC 61966-2-1).
#[inline]
fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// Convert sRGB [0,1] to OKLab, rescaled to CIELab-compatible ranges.
///
/// Returns (L, a, b) where L ∈ [0,100], a/b ∈ approx [-120, 120].
pub fn srgb_to_lab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // sRGB → linear
    let rl = srgb_to_linear(r);
    let gl = srgb_to_linear(g);
    let bl = srgb_to_linear(b);

    // Linear RGB → LMS
    let l = 0.4122214708 * rl + 0.5363325363 * gl + 0.0514459929 * bl;
    let m = 0.2119034982 * rl + 0.6806995451 * gl + 0.1073969566 * bl;
    let s = 0.0883024619 * rl + 0.2817188376 * gl + 0.6299787005 * bl;

    // Cube root
    let l_ = l.max(0.0).cbrt();
    let m_ = m.max(0.0).cbrt();
    let s_ = s.max(0.0).cbrt();

    // LMS' → OKLab
    let ok_l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
    let ok_a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
    let ok_b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;

    // Rescale to CIELab-compatible ranges
    (ok_l * L_SCALE, ok_a * AB_SCALE, ok_b * AB_SCALE)
}

/// Convert OKLab (rescaled) to sRGB [0,1] (clamped).
pub fn lab_to_srgb(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    // Unscale from CIELab-compatible ranges to native OKLab
    let ok_l = l / L_SCALE;
    let ok_a = a / AB_SCALE;
    let ok_b = b / AB_SCALE;

    // OKLab → LMS'
    let l_ = ok_l + 0.3963377774 * ok_a + 0.2158037573 * ok_b;
    let m_ = ok_l - 0.1055613458 * ok_a - 0.0638541728 * ok_b;
    let s_ = ok_l - 0.0894841775 * ok_a - 1.2914855480 * ok_b;

    // Cube (inverse of cube root)
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    // LMS → linear RGB (proper inverse of forward RGB→LMS matrix)
    let rl =  4.0767416613 * l - 3.3077115904 * m + 0.2309699287 * s;
    let gl = -1.2684380041 * l + 2.6097574007 * m - 0.3413193963 * s;
    let bl = -0.0041960863 * l - 0.7034186145 * m + 1.7076147010 * s;

    // Linear → sRGB, clamped
    let r = linear_to_srgb(rl.max(0.0));
    let g = linear_to_srgb(gl.max(0.0));
    let b_out = linear_to_srgb(bl.max(0.0));

    (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b_out.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(r: f32, g: f32, b: f32) {
        let (l, a, lab_b) = srgb_to_lab(r, g, b);
        let (r2, g2, b2) = lab_to_srgb(l, a, lab_b);
        assert!(
            (r2 - r).abs() < 1e-3 && (g2 - g).abs() < 1e-3 && (b2 - b).abs() < 1e-3,
            "Round-trip ({r},{g},{b}) -> Lab({l:.2},{a:.2},{lab_b:.2}) -> ({r2:.5},{g2:.5},{b2:.5})"
        );
    }

    #[test]
    fn test_lab_round_trip() {
        round_trip(1.0, 1.0, 1.0); // white
        round_trip(0.0, 0.0, 0.0); // black
        round_trip(0.5, 0.5, 0.5); // mid grey
        round_trip(1.0, 0.0, 0.0); // red
        round_trip(0.0, 1.0, 0.0); // green
        round_trip(0.0, 0.0, 1.0); // blue
    }

    #[test]
    fn test_white_point() {
        let (l, a, b) = srgb_to_lab(1.0, 1.0, 1.0);
        // OKLab white = (1,0,0), rescaled L = 100, a = 0, b = 0
        assert!((l - 100.0).abs() < 1e-2, "White L: expected 100, got {l}");
        assert!(a.abs() < 1e-2, "White a: expected 0, got {a}");
        assert!(b.abs() < 1e-2, "White b: expected 0, got {b}");
    }

    #[test]
    fn test_black_point() {
        let (l, a, b) = srgb_to_lab(0.0, 0.0, 0.0);
        assert!(l.abs() < 1e-3, "Black L: expected 0, got {l}");
        assert!(a.abs() < 1e-3, "Black a: expected 0, got {a}");
        assert!(b.abs() < 1e-3, "Black b: expected 0, got {b}");
    }

    #[test]
    fn test_l_range() {
        // L should span [0, 100] for sRGB gamut
        let (l_black, _, _) = srgb_to_lab(0.0, 0.0, 0.0);
        let (l_white, _, _) = srgb_to_lab(1.0, 1.0, 1.0);
        assert!(l_black < 1.0, "Black L should be near 0, got {l_black}");
        assert!(l_white > 99.0, "White L should be near 100, got {l_white}");
    }
}
