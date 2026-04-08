//! I-Log transfer function (Insta360 X5).
//!
//! Uses S-Log3 as a proxy curve until empirical extraction is available.
//! Replace with `LutBased` once a 1D .cube file is produced from a
//! grayscale chart shot on the X5.

use crate::TransferFunction;

// Sony S-Log3 parameters (normalised, i.e. code value / 1023)
// Log segment: encoded = (420 + 261.5 * log10((x + 0.01) / 0.19)) / 1023
//              => A * log10(x + B) + D,  where B = 0.01, D = 420/1023, A = 261.5/1023
// Linear segment: encoded = (x * slope_cv + 95) / 1023
//              => slope_cv = (171.2102946929 - 95) / 0.01125
const A: f64 = 261.5 / 1023.0;
const B: f64 = 0.01;
const D: f64 = 420.0 / 1023.0;
// Linear-segment slope (CV/linear unit, then normalised)
const LIN_SLOPE: f64 = (171.2102946929 - 95.0) / 0.01125 / 1023.0;
const LIN_OFFSET: f64 = 95.0 / 1023.0;
// Linear cut point: below this, encoding is linear
const LIN_CUT: f64 = 0.01125;
// Encoded cut point: S-Log3 value at LIN_CUT (≈ 171.2/1023)
const ENC_CUT: f64 = 171.2102946929 / 1023.0;

/// Convert I-Log (S-Log3 proxy) encoded value to scene-linear light.
pub fn ilog_to_linear(x: f32) -> f32 {
    let x = x as f64;
    let linear = if x >= ENC_CUT {
        10_f64.powf((x - D) / A) * 0.19 - B
    } else {
        (x - LIN_OFFSET) / LIN_SLOPE
    };
    linear.max(0.0) as f32
}

/// Convert scene-linear light to I-Log (S-Log3 proxy) encoding.
pub fn linear_to_ilog(x: f32) -> f32 {
    let x = x as f64;
    let encoded = if x >= LIN_CUT {
        A * ((x + B) / 0.19).log10() + D
    } else {
        x * LIN_SLOPE + LIN_OFFSET
    };
    encoded.clamp(0.0, 1.0) as f32
}

/// I-Log transfer function (Insta360 X5, S-Log3 proxy).
pub struct ILog;

impl TransferFunction for ILog {
    fn to_linear(&self, encoded: f32) -> f32 {
        ilog_to_linear(encoded)
    }

    fn from_linear(&self, linear: f32) -> f32 {
        linear_to_ilog(linear)
    }

    fn shoulder(&self) -> f32 {
        0.88
    }

    fn name(&self) -> &str {
        "I-Log"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ilog_round_trip() {
        let values = [0.0_f32, 0.01, 0.05, 0.18, 0.5, 0.9, 1.0];
        for &v in &values {
            let encoded = linear_to_ilog(v);
            let decoded = ilog_to_linear(encoded);
            assert!(
                (decoded - v).abs() < 1e-4,
                "Round-trip failed for {v}: encoded={encoded}, decoded={decoded}"
            );
        }
    }

    #[test]
    fn test_ilog_middle_grey() {
        // S-Log3 middle grey: 18% linear ≈ 0.41 encoded
        let encoded = linear_to_ilog(0.18);
        assert!(
            (0.35..0.50).contains(&encoded),
            "Middle grey encode: expected 0.35–0.50, got {encoded}"
        );
    }

    #[test]
    fn test_ilog_monotonic() {
        let mut prev = ilog_to_linear(0.0);
        for i in 1..=100 {
            let x = i as f32 / 100.0;
            let lin = ilog_to_linear(x);
            assert!(lin >= prev, "Non-monotonic at x={x}: {lin} < {prev}");
            prev = lin;
        }
    }

    #[test]
    fn test_ilog_trait() {
        let tf: Box<dyn crate::TransferFunction> = Box::new(ILog);
        assert_eq!(tf.shoulder(), 0.88);
        assert_eq!(tf.name(), "I-Log");
    }
}
