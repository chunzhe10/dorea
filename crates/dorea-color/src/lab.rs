//! Standard sRGB ↔ CIELAB D65 conversion.
//!
//! IEC 61966-2-1 piecewise linearisation + D65 matrix + CIE Lab formula.

// D65 whitepoint
const XN: f32 = 0.95047;
const YN: f32 = 1.0;
const ZN: f32 = 1.08883;

// Threshold for f(t) piecewise: (6/29)^3
const DELTA_CUBED: f32 = (6.0 / 29.0) * (6.0 / 29.0) * (6.0 / 29.0); // ≈ 0.008856
const DELTA_SQ_3: f32 = 3.0 * (6.0 / 29.0) * (6.0 / 29.0); // 3*(6/29)^2 ≈ 0.12842

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

/// CIE Lab f(t) function.
#[inline]
fn f_lab(t: f32) -> f32 {
    if t > DELTA_CUBED {
        t.cbrt()
    } else {
        t / DELTA_SQ_3 + 4.0 / 29.0
    }
}

/// Inverse of f(t): f⁻¹(s) = s³ if s > 6/29, else 3*(6/29)²*(s - 4/29)
#[inline]
fn f_lab_inv(s: f32) -> f32 {
    let delta = 6.0_f32 / 29.0;
    if s > delta {
        s * s * s
    } else {
        DELTA_SQ_3 * (s - 4.0 / 29.0)
    }
}

/// Convert sRGB [0,1] to CIELAB (D65).
///
/// Returns (L, a, b) where L ∈ [0,100], a/b ∈ approx [-128, 127].
pub fn srgb_to_lab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // sRGB → linear
    let rl = srgb_to_linear(r);
    let gl = srgb_to_linear(g);
    let bl = srgb_to_linear(b);

    // Linear RGB → XYZ (D65, Rec. 709)
    let x = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl;
    let y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl;
    let z = 0.0193339 * rl + 0.119_192 * gl + 0.9503041 * bl;

    // XYZ → Lab
    let fx = f_lab(x / XN);
    let fy = f_lab(y / YN);
    let fz = f_lab(z / ZN);

    let l = 116.0 * fy - 16.0;
    let a_lab = 500.0 * (fx - fy);
    let b_lab = 200.0 * (fy - fz);

    (l, a_lab, b_lab)
}

/// Convert CIELAB (D65) to sRGB [0,1] (clamped).
pub fn lab_to_srgb(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;

    let x = XN * f_lab_inv(fx);
    let y = YN * f_lab_inv(fy);
    let z = ZN * f_lab_inv(fz);

    // XYZ → linear RGB (D65, Rec. 709) — inverse matrix
    let rl = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let gl = -0.969_266 * x + 1.8760108 * y + 0.0415560 * z;
    let bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;

    // Linear → sRGB, clamped
    let r = linear_to_srgb(rl.clamp(0.0, 1.0));
    let g = linear_to_srgb(gl.clamp(0.0, 1.0));
    let b_out = linear_to_srgb(bl.clamp(0.0, 1.0));

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
        assert!((l - 100.0).abs() < 1e-3, "White L: expected 100, got {l}");
        assert!(a.abs() < 1e-3, "White a: expected 0, got {a}");
        assert!(b.abs() < 1e-3, "White b: expected 0, got {b}");
    }
}
