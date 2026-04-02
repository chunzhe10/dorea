//! Core LUT data types.

/// A N×N×N LUT mapping input RGB → output RGB.
///
/// Stored as flat `Vec<f32>` for cache efficiency.
/// Index formula: `(r * size * size + g * size + b) * 3`
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LutGrid {
    pub size: usize,
    /// Length = size³ × 3
    pub data: Vec<f32>,
}

impl LutGrid {
    /// Create a zeroed LUT of given size.
    pub fn new(size: usize) -> Self {
        Self {
            size,
            data: vec![0.0_f32; size * size * size * 3],
        }
    }

    #[inline]
    fn idx(&self, ri: usize, gi: usize, bi: usize) -> usize {
        (ri * self.size * self.size + gi * self.size + bi) * 3
    }

    /// Get the output RGB triple at grid position (ri, gi, bi).
    pub fn get(&self, ri: usize, gi: usize, bi: usize) -> [f32; 3] {
        debug_assert!(ri < self.size && gi < self.size && bi < self.size);
        let i = self.idx(ri, gi, bi);
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }

    /// Set the output RGB triple at grid position (ri, gi, bi).
    pub fn set(&mut self, ri: usize, gi: usize, bi: usize, val: [f32; 3]) {
        debug_assert!(ri < self.size && gi < self.size && bi < self.size);
        let i = self.idx(ri, gi, bi);
        self.data[i] = val[0];
        self.data[i + 1] = val[1];
        self.data[i + 2] = val[2];
    }
}

/// 5 depth-stratified LUTs with their zone boundaries.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DepthLuts {
    pub luts: Vec<LutGrid>,
    /// Length = n_zones + 1, values in [0, 1]
    pub zone_boundaries: Vec<f32>,
    /// Length = n_zones — midpoint of each zone
    pub zone_centers: Vec<f32>,
}

impl DepthLuts {
    pub fn new(luts: Vec<LutGrid>, zone_boundaries: Vec<f32>) -> Self {
        // L7: validate that zone_boundaries has exactly luts.len() + 1 entries.
        assert_eq!(
            zone_boundaries.len(),
            luts.len() + 1,
            "zone_boundaries length ({}) must be luts.len() + 1 ({})",
            zone_boundaries.len(),
            luts.len() + 1
        );
        let n = luts.len();
        let zone_centers = (0..n)
            .map(|i| (zone_boundaries[i] + zone_boundaries[i + 1]) / 2.0)
            .collect();
        Self {
            luts,
            zone_boundaries,
            zone_centers,
        }
    }

    pub fn n_zones(&self) -> usize {
        self.luts.len()
    }
}
