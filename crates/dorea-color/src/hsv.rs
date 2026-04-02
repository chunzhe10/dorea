//! Standard RGB ↔ HSV conversion.
//!
//! H ∈ [0, 360), S ∈ [0, 1], V ∈ [0, 1]. Input/output RGB ∈ [0, 1].

/// Convert RGB to HSV.
///
/// Returns `(H, S, V)` where H ∈ [0, 360), S ∈ [0, 1], V ∈ [0, 1].
pub fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let cmax = r.max(g).max(b);
    let cmin = r.min(g).min(b);
    let delta = cmax - cmin;

    let v = cmax;

    let s = if cmax < 1e-7 { 0.0 } else { delta / cmax };

    let h = if delta < 1e-7 {
        0.0
    } else if (cmax - r).abs() < 1e-7 {
        60.0 * (((g - b) / delta) % 6.0)
    } else if (cmax - g).abs() < 1e-7 {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };

    let h = if h < 0.0 { h + 360.0 } else { h };

    (h, s, v)
}

/// Convert HSV to RGB.
///
/// H ∈ [0, 360), S ∈ [0, 1], V ∈ [0, 1]. Returns `(R, G, B)` ∈ [0, 1].
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    if s < 1e-7 {
        return (v, v, v);
    }

    let h = h % 360.0;
    let sector = h / 60.0;
    let i = sector.floor() as i32;
    let f = sector - i as f32;

    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(r: f32, g: f32, b: f32) {
        let (h, s, v) = rgb_to_hsv(r, g, b);
        let (r2, g2, b2) = hsv_to_rgb(h, s, v);
        assert!(
            (r2 - r).abs() < 1e-5 && (g2 - g).abs() < 1e-5 && (b2 - b).abs() < 1e-5,
            "Round-trip ({r},{g},{b}) -> HSV({h:.2},{s:.4},{v:.4}) -> ({r2:.6},{g2:.6},{b2:.6})"
        );
    }

    #[test]
    fn test_hsv_round_trip() {
        round_trip(1.0, 0.0, 0.0); // red
        round_trip(0.0, 1.0, 0.0); // green
        round_trip(0.0, 0.0, 1.0); // blue
        round_trip(1.0, 1.0, 1.0); // white
        round_trip(0.0, 0.0, 0.0); // black
        round_trip(0.5, 0.5, 0.5); // grey
        round_trip(0.2, 0.6, 0.8); // arbitrary
        round_trip(1.0, 0.5, 0.0); // orange
    }

    #[test]
    fn test_pure_red() {
        let (h, s, v) = rgb_to_hsv(1.0, 0.0, 0.0);
        assert!((h - 0.0).abs() < 1e-5, "Red hue: expected 0, got {h}");
        assert!((s - 1.0).abs() < 1e-5, "Red sat: expected 1, got {s}");
        assert!((v - 1.0).abs() < 1e-5, "Red val: expected 1, got {v}");
    }
}
