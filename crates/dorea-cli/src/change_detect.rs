/// Abstracted change detection between video frames.
///
/// Implementations hold a reference frame and compute a scalar change score
/// against it. The caller decides when to update the reference (i.e. on keyframe
/// detection) — this is intentional: we compare each frame to the *last keyframe*,
/// not to the immediately preceding frame.
pub trait ChangeDetector: Send {
    /// Change score from `pixels` vs the stored reference frame.
    /// Returns `f32::MAX` when no reference has been set yet.
    fn score(&self, pixels: &[u8]) -> f32;

    /// Accept `pixels` as the new reference for future `score()` calls.
    fn set_reference(&mut self, pixels: &[u8]);

    /// Clear the reference — next `score()` will return `f32::MAX`.
    fn reset(&mut self);
}

/// Mean-squared-error change detector.
#[derive(Default)]
pub struct MseDetector {
    reference: Option<Vec<u8>>,
}

/// Normalized MSE between two equal-length u8 slices.
/// Returns value in [0, 1] where 0 = identical.
pub fn frame_mse(a: &[u8], b: &[u8]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    let n = a.len() as f64;
    let sum_sq: f64 = a.iter().zip(b.iter())
        .map(|(&av, &bv)| {
            let d = av as f64 - bv as f64;
            d * d
        })
        .sum();
    (sum_sq / (n * 255.0 * 255.0)) as f32
}

impl ChangeDetector for MseDetector {
    fn score(&self, pixels: &[u8]) -> f32 {
        match self.reference.as_ref() {
            Some(r) => frame_mse(pixels, r),
            None => f32::MAX,
        }
    }

    fn set_reference(&mut self, pixels: &[u8]) {
        self.reference = Some(pixels.to_vec());
    }

    fn reset(&mut self) {
        self.reference = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_mse_identical_is_zero() {
        let a: Vec<u8> = vec![100, 150, 200, 50, 75, 125];
        assert_eq!(frame_mse(&a, &a), 0.0);
    }

    #[test]
    fn frame_mse_empty_is_zero() {
        assert_eq!(frame_mse(&[], &[]), 0.0);
    }

    #[test]
    fn frame_mse_opposite_is_one() {
        let a: Vec<u8> = vec![0, 0, 0];
        let b: Vec<u8> = vec![255, 255, 255];
        let mse = frame_mse(&a, &b);
        assert!((mse - 1.0).abs() < 1e-5, "expected ~1.0, got {mse}");
    }

    #[test]
    fn frame_mse_known_value() {
        let a: Vec<u8> = vec![100, 100, 100];
        let b: Vec<u8> = vec![101, 101, 101];
        let mse = frame_mse(&a, &b);
        let expected = 1.0 / (255.0 * 255.0);
        assert!((mse as f64 - expected).abs() < 1e-8, "expected {expected}, got {mse}");
    }

    #[test]
    fn mse_detector_no_reference_returns_max() {
        let det = MseDetector::default();
        let pixels = vec![128u8; 100];
        assert_eq!(det.score(&pixels), f32::MAX);
    }

    #[test]
    fn mse_detector_same_as_reference_returns_zero() {
        let mut det = MseDetector::default();
        let pixels = vec![200u8; 300];
        det.set_reference(&pixels);
        assert_eq!(det.score(&pixels), 0.0);
    }

    #[test]
    fn mse_detector_set_reference_compares_to_set_frame_not_latest() {
        let mut det = MseDetector::default();
        let ref_frame = vec![0u8; 3];
        let other_frame = vec![255u8; 3];
        det.set_reference(&ref_frame);

        // score returns MSE vs ref_frame — not updated by calling score itself
        let s1 = det.score(&other_frame);
        let s2 = det.score(&other_frame);
        assert_eq!(s1, s2, "score should be deterministic without set_reference");
        assert!((s1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn mse_detector_reset_clears_reference() {
        let mut det = MseDetector::default();
        det.set_reference(&vec![0u8; 3]);
        det.reset();
        assert_eq!(det.score(&vec![0u8; 3]), f32::MAX);
    }
}
