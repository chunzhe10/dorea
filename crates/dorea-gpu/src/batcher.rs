/// Adaptive batch sizing for GPU grading.
///
/// Probes VRAM, starts at the maximum safe batch size, and adjusts at runtime:
/// - On OOM: halve batch size (floor at 1)
/// - After `grow_threshold` consecutive successes: grow by 50% (cap at max_batch)
/// - Batch size does NOT persist across runs
pub struct AdaptiveBatcher {
    batch_size: usize,
    max_batch: usize,
    min_batch: usize,
    successes: usize,
    grow_threshold: usize,
}

impl AdaptiveBatcher {
    /// Create a new batcher from VRAM probe results.
    ///
    /// `vram_free`: free VRAM in bytes (after `torch.cuda.empty_cache()`)
    /// `per_frame_bytes`: estimated bytes per frame (e.g. 56MB for 1080p)
    /// `safety_margin`: fraction reserved for fragmentation (0.15 = 15%)
    pub fn new(vram_free: usize, per_frame_bytes: usize, safety_margin: f64) -> Self {
        let usable = (vram_free as f64 * (1.0 - safety_margin)) as usize;
        let max_batch = if per_frame_bytes > 0 {
            (usable / per_frame_bytes).max(1)
        } else {
            1
        };
        Self {
            batch_size: max_batch,
            max_batch,
            min_batch: 1,
            successes: 0,
            grow_threshold: 10,
        }
    }

    /// Create a batcher with a fixed batch size (no adaptation).
    pub fn fixed(batch_size: usize) -> Self {
        Self {
            batch_size: batch_size.max(1),
            max_batch: batch_size.max(1),
            min_batch: 1,
            successes: 0,
            grow_threshold: usize::MAX,
        }
    }

    /// Current batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Report a successful batch completion.
    pub fn report_success(&mut self) {
        self.successes += 1;
        if self.successes >= self.grow_threshold && self.batch_size < self.max_batch {
            self.batch_size = ((self.batch_size as f64 * 1.5).ceil() as usize).min(self.max_batch);
            self.successes = 0;
        }
    }

    /// Report an OOM failure. Returns true if batch_size was already at minimum
    /// (meaning the frame genuinely cannot fit).
    pub fn report_oom(&mut self) -> bool {
        let was_min = self.batch_size <= self.min_batch;
        self.successes = 0;
        self.batch_size = (self.batch_size / 2).max(self.min_batch);
        was_min
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_computes_max_batch_from_vram() {
        let b = AdaptiveBatcher::new(1_000_000_000, 56_000_000, 0.15);
        assert_eq!(b.batch_size(), 15);
        assert_eq!(b.max_batch, 15);
    }

    #[test]
    fn new_with_zero_per_frame_gives_batch_1() {
        let b = AdaptiveBatcher::new(1_000_000_000, 0, 0.15);
        assert_eq!(b.batch_size(), 1);
    }

    #[test]
    fn new_with_tiny_vram_gives_batch_1() {
        let b = AdaptiveBatcher::new(10_000_000, 56_000_000, 0.15);
        assert_eq!(b.batch_size(), 1);
    }

    #[test]
    fn report_oom_halves_batch_size() {
        let mut b = AdaptiveBatcher::new(1_000_000_000, 56_000_000, 0.15);
        let initial = b.batch_size();
        assert!(!b.report_oom());
        assert_eq!(b.batch_size(), initial / 2);
    }

    #[test]
    fn report_oom_at_minimum_returns_true() {
        let mut b = AdaptiveBatcher::fixed(1);
        assert!(b.report_oom());
        assert_eq!(b.batch_size(), 1);
    }

    #[test]
    fn report_oom_resets_success_counter() {
        let mut b = AdaptiveBatcher::new(1_000_000_000, 56_000_000, 0.15);
        for _ in 0..9 {
            b.report_success();
        }
        b.report_oom();
        let size_after_oom = b.batch_size();
        for _ in 0..9 {
            b.report_success();
        }
        assert_eq!(b.batch_size(), size_after_oom);
    }

    #[test]
    fn growth_after_threshold_successes() {
        let mut b = AdaptiveBatcher::new(1_000_000_000, 56_000_000, 0.15);
        b.report_oom();
        let halved = b.batch_size();
        for _ in 0..10 {
            b.report_success();
        }
        let expected = ((halved as f64 * 1.5).ceil() as usize).min(b.max_batch);
        assert_eq!(b.batch_size(), expected);
    }

    #[test]
    fn growth_capped_at_max() {
        let mut b = AdaptiveBatcher::new(1_000_000_000, 56_000_000, 0.15);
        let max = b.max_batch;
        for _ in 0..20 {
            b.report_success();
        }
        assert_eq!(b.batch_size(), max);
    }

    #[test]
    fn fixed_batcher_does_not_grow() {
        let mut b = AdaptiveBatcher::fixed(4);
        for _ in 0..100 {
            b.report_success();
        }
        assert_eq!(b.batch_size(), 4);
    }
}
