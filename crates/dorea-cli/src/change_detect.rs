use dorea_lut::build::adaptive_zone_boundaries;

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

/// Wasserstein-1 (earth mover's) distance between two depth distributions.
///
/// Quantizes depths into `n_bins` histogram bins over [0, 1], computes CDFs,
/// and returns the area between them (normalized to [0, 1]).
/// Returns 0.0 if either input is empty.
pub fn depth_distribution_distance(a: &[f32], b: &[f32], n_bins: usize) -> f32 {
    if a.is_empty() || b.is_empty() || n_bins == 0 {
        return 0.0;
    }

    let mut hist_a = vec![0u32; n_bins];
    let mut hist_b = vec![0u32; n_bins];

    for &d in a {
        let bin = ((d.clamp(0.0, 1.0) * n_bins as f32) as usize).min(n_bins - 1);
        hist_a[bin] += 1;
    }
    for &d in b {
        let bin = ((d.clamp(0.0, 1.0) * n_bins as f32) as usize).min(n_bins - 1);
        hist_b[bin] += 1;
    }

    let na = a.len() as f64;
    let nb = b.len() as f64;
    let mut cdf_a = 0.0_f64;
    let mut cdf_b = 0.0_f64;
    let mut distance = 0.0_f64;

    for i in 0..n_bins {
        cdf_a += hist_a[i] as f64 / na;
        cdf_b += hist_b[i] as f64 / nb;
        distance += (cdf_a - cdf_b).abs();
    }

    (distance / n_bins as f64) as f32
}

/// A contiguous range of keyframe indices forming one scene segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentRange {
    /// Inclusive start keyframe index (into the keyframe Vec, not frame index).
    pub start: usize,
    /// Exclusive end keyframe index.
    pub end: usize,
}

/// Detect scene segments from per-keyframe depth maps.
///
/// Scans consecutive keyframe depth distributions using Wasserstein-1 distance.
/// When the distance exceeds `threshold`, a segment boundary is placed.
/// Segments shorter than `min_keyframes` are merged into the previous segment.
/// The first segment is never merged — if it is shorter than `min_keyframes`, it is kept as-is.
pub fn detect_scene_segments(
    keyframe_depths: &[Vec<f32>],
    threshold: f32,
    min_keyframes: usize,
) -> Vec<SegmentRange> {
    let n = keyframe_depths.len();
    if n == 0 {
        return vec![];
    }

    const N_BINS: usize = 64;
    let mut boundaries: Vec<usize> = vec![0];

    for i in 1..n {
        let dist = depth_distribution_distance(
            &keyframe_depths[i - 1],
            &keyframe_depths[i],
            N_BINS,
        );
        if dist > threshold {
            boundaries.push(i);
        }
    }
    boundaries.push(n);

    let mut segments: Vec<SegmentRange> = boundaries.windows(2)
        .map(|w| SegmentRange { start: w[0], end: w[1] })
        .collect();

    let mut merged: Vec<SegmentRange> = Vec::new();
    for seg in segments.drain(..) {
        let len = seg.end - seg.start;
        if len < min_keyframes && !merged.is_empty() {
            merged.last_mut().unwrap().end = seg.end;
        } else {
            merged.push(seg);
        }
    }

    merged
}

/// Compute per-keyframe zone boundaries from each keyframe's depth map.
///
/// Returns one `Vec<f32>` of `n_zones + 1` boundary values per keyframe.
pub fn compute_per_kf_zones(
    keyframe_depths: &[Vec<f32>],
    n_zones: usize,
) -> Vec<Vec<f32>> {
    keyframe_depths.iter()
        .map(|depths| adaptive_zone_boundaries(depths, n_zones))
        .collect()
}

/// Smooth per-keyframe zone boundaries using a weighted moving average.
///
/// Center keyframe gets weight 0.6, each neighbor gets 0.2.
/// Does NOT smooth across segment boundaries.
/// `window` of 1 means no smoothing.
pub fn smooth_zone_boundaries(
    raw: &[Vec<f32>],
    segments: &[SegmentRange],
    window: usize,
) -> Vec<Vec<f32>> {
    if window <= 1 || raw.is_empty() {
        return raw.to_vec();
    }

    let n_bounds = raw[0].len();
    let mut smoothed = raw.to_vec();

    for seg in segments {
        if seg.end - seg.start <= 1 {
            continue;
        }

        for ki in seg.start..seg.end {
            let prev = if ki > seg.start { Some(ki - 1) } else { None };
            let next = if ki + 1 < seg.end { Some(ki + 1) } else { None };

            let mut total_w = 0.6_f32;
            let mut result = vec![0.0_f32; n_bounds];
            for j in 0..n_bounds {
                result[j] = raw[ki][j] * 0.6;
            }

            if let Some(pi) = prev {
                total_w += 0.2;
                for j in 0..n_bounds {
                    result[j] += raw[pi][j] * 0.2;
                }
            }
            if let Some(ni) = next {
                total_w += 0.2;
                for j in 0..n_bounds {
                    result[j] += raw[ni][j] * 0.2;
                }
            }

            for j in 0..n_bounds {
                result[j] /= total_w;
            }
            result[0] = 0.0;
            *result.last_mut().unwrap() = 1.0;

            smoothed[ki] = result;
        }
    }

    smoothed
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

    #[test]
    fn depth_dist_identical_is_zero() {
        let a = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        assert!((depth_distribution_distance(&a, &a, 64) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn depth_dist_opposite_is_high() {
        let near = vec![0.0; 100];
        let far = vec![1.0; 100];
        let d = depth_distribution_distance(&near, &far, 64);
        assert!(d > 0.9, "expected >0.9 for opposite distributions, got {d}");
    }

    #[test]
    fn depth_dist_symmetric() {
        let a: Vec<f32> = (0..50).map(|i| i as f32 / 50.0).collect();
        let b: Vec<f32> = (0..50).map(|i| (i as f32 / 50.0).powi(2)).collect();
        let d_ab = depth_distribution_distance(&a, &b, 64);
        let d_ba = depth_distribution_distance(&b, &a, 64);
        assert!((d_ab - d_ba).abs() < 1e-6, "not symmetric: {d_ab} vs {d_ba}");
    }

    #[test]
    fn depth_dist_empty_is_zero() {
        assert!((depth_distribution_distance(&[], &[], 64) - 0.0).abs() < 1e-6);
        assert!((depth_distribution_distance(&[0.5], &[], 64) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn segments_single_uniform_clip() {
        let depths: Vec<Vec<f32>> = (0..10)
            .map(|_| (0..100).map(|i| i as f32 / 100.0).collect())
            .collect();
        let segs = detect_scene_segments(&depths, 0.15, 5);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].start, 0);
        assert_eq!(segs[0].end, 10);
    }

    #[test]
    fn segments_hard_scene_cut() {
        let mut depths: Vec<Vec<f32>> = Vec::new();
        for _ in 0..5 { depths.push(vec![0.1; 100]); }
        for _ in 0..5 { depths.push(vec![0.9; 100]); }
        let segs = detect_scene_segments(&depths, 0.15, 3);
        assert_eq!(segs.len(), 2, "expected 2 segments, got {:?}", segs);
        assert_eq!(segs[0].end, segs[1].start);
    }

    #[test]
    fn segments_short_segment_merged() {
        let mut depths: Vec<Vec<f32>> = Vec::new();
        for _ in 0..5 { depths.push(vec![0.1; 100]); }
        for _ in 0..2 { depths.push(vec![0.9; 100]); }
        for _ in 0..5 { depths.push(vec![0.1; 100]); }
        let segs = detect_scene_segments(&depths, 0.15, 5);
        assert!(segs.len() <= 2, "short segment should be merged, got {:?}", segs);
    }

    #[test]
    fn per_kf_zones_basic() {
        let depths: Vec<Vec<f32>> = vec![
            (0..100).map(|i| i as f32 / 100.0).collect(),
            vec![0.5; 100],
        ];
        let zones = compute_per_kf_zones(&depths, 4);
        assert_eq!(zones.len(), 2);
        assert_eq!(zones[0].len(), 5); // 4 zones → 5 boundaries
        assert!((zones[0][0] - 0.0).abs() < 1e-6);
        assert!((zones[0][4] - 1.0).abs() < 1e-6);
        for i in 1..4 {
            let expected = i as f32 / 4.0;
            assert!((zones[0][i] - expected).abs() < 0.05,
                "boundary {i}: expected ~{expected}, got {}", zones[0][i]);
        }
        for i in 1..4 {
            assert!((zones[1][i] - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn smooth_zones_no_change_when_window_1() {
        let raw = vec![
            vec![0.0, 0.25, 0.5, 0.75, 1.0],
            vec![0.0, 0.10, 0.20, 0.30, 1.0],
        ];
        let segments = vec![SegmentRange { start: 0, end: 2 }];
        let smoothed = smooth_zone_boundaries(&raw, &segments, 1);
        assert_eq!(smoothed, raw);
    }

    #[test]
    fn smooth_zones_dampens_outlier() {
        let raw = vec![
            vec![0.0, 0.25, 0.50, 0.75, 1.0],
            vec![0.0, 0.02, 0.04, 0.06, 1.0],
            vec![0.0, 0.25, 0.50, 0.75, 1.0],
        ];
        let segments = vec![SegmentRange { start: 0, end: 3 }];
        let smoothed = smooth_zone_boundaries(&raw, &segments, 3);
        assert!(smoothed[1][1] > 0.05, "smoothing should pull outlier up, got {}", smoothed[1][1]);
    }

    #[test]
    fn smooth_zones_respects_segment_boundary() {
        let raw = vec![
            vec![0.0, 0.10, 0.20, 0.30, 1.0],
            vec![0.0, 0.80, 0.85, 0.90, 1.0],
        ];
        let segments = vec![
            SegmentRange { start: 0, end: 1 },
            SegmentRange { start: 1, end: 2 },
        ];
        let smoothed = smooth_zone_boundaries(&raw, &segments, 3);
        assert_eq!(smoothed[0], raw[0]);
        assert_eq!(smoothed[1], raw[1]);
    }
}
