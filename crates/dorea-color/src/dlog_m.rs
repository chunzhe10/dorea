//! D-Log M transfer function (DJI Action 4).
//!
//! Ported from `pipeline_utils.py::dlog_m_to_linear` and `linear_to_dlog_m`.

// D-Log M curve parameters (DJI published curve)
const A: f64 = 0.9892;
const B: f64 = 0.0108;
const C: f64 = 0.256663;
const D: f64 = 0.584555;
const CUT_ENCODED: f64 = 0.14;

fn cut_linear() -> f64 {
    (10_f64.powf((CUT_ENCODED - D) / C) - B) / A
}

fn slope() -> f64 {
    let cl = cut_linear();
    C * A / ((A * cl + B) * 10_f64.ln())
}

fn intercept() -> f64 {
    CUT_ENCODED - slope() * cut_linear()
}

/// Convert D-Log M encoded values to scene-linear light.
///
/// Middle grey (~0.39) decodes to ~0.18.
pub fn dlog_m_to_linear(x: f32) -> f32 {
    let x = x as f64;
    let sl = slope();
    let ic = intercept();

    let linear = if x <= CUT_ENCODED {
        (x - ic) / sl
    } else {
        (10_f64.powf((x - D) / C) - B) / A
    };

    linear.max(0.0) as f32
}

/// Convert scene-linear light values to D-Log M encoding.
///
/// Inverse of `dlog_m_to_linear`.
pub fn linear_to_dlog_m(x: f32) -> f32 {
    let x = x as f64;
    let cl = cut_linear();
    let sl = slope();
    let ic = intercept();

    let encoded = if x <= cl {
        sl * x + ic
    } else {
        C * (A * x + B).log10() + D
    };

    encoded.clamp(0.0, 1.0) as f32
}

use crate::TransferFunction;

/// D-Log M transfer function (DJI Action 4).
pub struct DLogM;

impl TransferFunction for DLogM {
    fn to_linear(&self, encoded: f32) -> f32 {
        dlog_m_to_linear(encoded)
    }

    fn from_linear(&self, linear: f32) -> f32 {
        linear_to_dlog_m(linear)
    }

    fn shoulder(&self) -> f32 {
        0.85
    }

    fn name(&self) -> &str {
        "D-Log M"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dlog_m_round_trip() {
        let values = [0.0_f32, 0.1, 0.2, 0.39, 0.5, 0.7, 0.9, 1.0];
        for &v in &values {
            let encoded = linear_to_dlog_m(v);
            let decoded = dlog_m_to_linear(encoded);
            assert!(
                (decoded - v).abs() < 1e-5,
                "Round-trip failed for {v}: got {decoded}"
            );
        }
    }

    #[test]
    fn test_dlog_m_middle_grey() {
        // Middle grey ~0.39 in D-Log M should decode to ~0.18 linear
        let linear = dlog_m_to_linear(0.39);
        assert!(
            (linear - 0.18).abs() < 0.02,
            "Middle grey decode: expected ~0.18, got {linear}"
        );
    }

    #[test]
    fn test_dlog_m_as_trait_object() {
        let tf: Box<dyn crate::TransferFunction> = Box::new(super::DLogM);
        let encoded = tf.from_linear(0.18);
        let decoded = tf.to_linear(encoded);
        assert!((decoded - 0.18).abs() < 1e-4, "Trait round-trip: expected ~0.18, got {decoded}");
        assert_eq!(tf.shoulder(), 0.85);
        assert_eq!(tf.name(), "D-Log M");
    }
}
