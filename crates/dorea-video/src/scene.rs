// Scene change detection based on L1 histogram distance.

/// Number of bins per channel for the histogram.
const BINS: usize = 64;
const TOTAL_BINS: usize = BINS * 3;

/// Compute the L1 histogram distance between two RGB frames.
///
/// Returns a value in [0, 1] where:
/// - 0.0 = identical frames
/// - 1.0 = completely disjoint histograms (maximum difference)
/// - Values > `SCENE_CHANGE_THRESHOLD` typically indicate a scene cut
///
/// `a` and `b` must both be RGB24 (length = width * height * 3).
pub fn histogram_distance(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len(), "frame size mismatch");

    let hist_a = build_histogram(a);
    let hist_b = build_histogram(b);

    let n_pixels = (a.len() / 3) as f64;
    if n_pixels == 0.0 {
        return 0.0;
    }

    // L1 distance between normalized per-channel histograms, averaged over 3 channels.
    // Sum |ha[i]/N - hb[i]/N| over all bins, divide by 2 to normalize to [0,1].
    let mut l1_sum = 0.0_f64;
    for i in 0..TOTAL_BINS {
        l1_sum += (hist_a[i] as f64 - hist_b[i] as f64).abs();
    }
    // l1_sum is over 3 channels combined; each channel sums to n_pixels.
    // Max l1_sum = 3 * 2 * n_pixels = 6 * n_pixels (completely disjoint histograms).
    (l1_sum / (6.0 * n_pixels)) as f32
}

fn build_histogram(frame: &[u8]) -> Vec<u32> {
    let mut hist = vec![0u32; TOTAL_BINS];
    for chunk in frame.chunks_exact(3) {
        let r_bin = (chunk[0] as usize * BINS) >> 8; // fast floor(v * BINS / 256)
        let g_bin = (chunk[1] as usize * BINS) >> 8;
        let b_bin = (chunk[2] as usize * BINS) >> 8;
        hist[r_bin] += 1;
        hist[BINS + g_bin] += 1;
        hist[BINS * 2 + b_bin] += 1;
    }
    hist
}

/// Default threshold for scene change detection.
/// Tune via env var `DOREA_SCENE_THRESHOLD`.
pub fn scene_change_threshold() -> f32 {
    std::env::var("DOREA_SCENE_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.15)
}

/// Return `true` if the histogram distance exceeds the threshold.
pub fn is_scene_change(a: &[u8], b: &[u8]) -> bool {
    histogram_distance(a, b) > scene_change_threshold()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_frames_zero_distance() {
        let frame: Vec<u8> = (0..768).map(|i| (i % 256) as u8).collect();
        let dist = histogram_distance(&frame, &frame);
        assert!(dist < 1e-6, "identical frames should have distance ~0, got {dist}");
    }

    #[test]
    fn opposite_frames_high_distance() {
        let frame_a: Vec<u8> = vec![10u8; 3 * 100];
        let frame_b: Vec<u8> = vec![245u8; 3 * 100];
        let dist = histogram_distance(&frame_a, &frame_b);
        assert!(dist > 0.1, "contrasting frames should have distance > 0.1, got {dist}");
    }

    #[test]
    fn scene_change_detection() {
        let black: Vec<u8> = vec![0u8; 3 * 1024];
        let white: Vec<u8> = vec![255u8; 3 * 1024];
        assert!(is_scene_change(&black, &white), "black→white should be a scene change");
        assert!(!is_scene_change(&black, &black), "same frame should not trigger");
    }
}
