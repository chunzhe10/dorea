//! .dorea-cal binary calibration file format (bincode).

use std::io::{Read as _, Write as _};
use std::path::Path;

use dorea_hsl::HslCorrections;
use dorea_lut::DepthLuts;

/// Current format version.
///
/// Bumped to 2: `created_at` changed from String to u64 (Unix timestamp, seconds since epoch).
pub const FORMAT_VERSION: u8 = 2;

/// A saved calibration combining depth-stratified LUTs and HSL corrections.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Calibration {
    /// Format version; current = 2
    pub version: u8,
    pub depth_luts: DepthLuts,
    pub hsl_corrections: HslCorrections,
    /// Unix timestamp (seconds since epoch) of calibration creation.
    pub created_at_unix_secs: u64,
    pub keyframe_count: usize,
    /// Human-readable description, e.g. "3 keyframes from 2026-04-01 dive"
    pub source_description: String,
}

impl Calibration {
    pub fn new(
        depth_luts: DepthLuts,
        hsl_corrections: HslCorrections,
        keyframe_count: usize,
    ) -> Self {
        Self {
            version: FORMAT_VERSION,
            depth_luts,
            hsl_corrections,
            created_at_unix_secs: unix_now(),
            keyframe_count,
            source_description: format!("{keyframe_count} keyframe(s)"),
        }
    }

    /// Save to a `.dorea-cal` file (bincode-encoded).
    pub fn save(&self, path: &Path) -> Result<(), CalError> {
        let bytes = bincode::serialize(self).map_err(|e| CalError::Serialization(e.to_string()))?;
        let mut file = std::fs::File::create(path)?;
        file.write_all(&bytes)?;
        Ok(())
    }

    /// Load from a `.dorea-cal` file.
    pub fn load(path: &Path) -> Result<Self, CalError> {
        let mut file = std::fs::File::open(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;

        // L2: check version byte before full deserialization to give a clear error.
        let version_byte = bytes
            .first()
            .copied()
            .ok_or_else(|| CalError::Serialization("empty file".into()))?;
        if version_byte != FORMAT_VERSION {
            return Err(CalError::UnsupportedVersion(version_byte));
        }

        let cal: Calibration =
            bincode::deserialize(&bytes).map_err(|e| CalError::Serialization(e.to_string()))?;
        Ok(cal)
    }
}

/// Returns the current time as seconds since the Unix epoch.
fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Errors that can occur when saving or loading a `.dorea-cal` file.
#[derive(Debug, thiserror::Error)]
pub enum CalError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Format version {0} not supported (expected {FORMAT_VERSION})")]
    UnsupportedVersion(u8),
}

#[cfg(test)]
mod tests {
    use super::*;
    use dorea_hsl::derive::{HslCorrections, QualifierCorrection};
    use dorea_lut::types::{DepthLuts, LutGrid};

    fn dummy_lut() -> LutGrid {
        let mut lut = LutGrid::new(3);
        for ri in 0..3 {
            for gi in 0..3 {
                for bi in 0..3 {
                    lut.set(ri, gi, bi, [ri as f32 / 2.0, gi as f32 / 2.0, bi as f32 / 2.0]);
                }
            }
        }
        lut
    }

    fn dummy_calibration() -> Calibration {
        let luts = vec![dummy_lut(); 5];
        let boundaries = vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let depth_luts = DepthLuts::new(luts, boundaries);

        let corrections = HslCorrections(vec![
            QualifierCorrection {
                h_center: 0.0,
                h_width: 40.0,
                h_offset: 5.0,
                s_ratio: 1.1,
                v_offset: 0.05,
                weight: 200.0,
            },
        ]);

        Calibration::new(depth_luts, corrections, 3)
    }

    #[test]
    fn test_round_trip() {
        let cal = dummy_calibration();

        // Save to a temp file
        let tmp = std::env::temp_dir().join("dorea_cal_test.dorea-cal");
        cal.save(&tmp).expect("save failed");

        // Load back
        let loaded = Calibration::load(&tmp).expect("load failed");
        assert_eq!(loaded.version, FORMAT_VERSION);
        assert_eq!(loaded.keyframe_count, cal.keyframe_count);
        assert_eq!(loaded.depth_luts.n_zones(), cal.depth_luts.n_zones());
        assert_eq!(loaded.hsl_corrections.0.len(), cal.hsl_corrections.0.len());

        // Verify LUT data integrity
        let orig_cell = cal.depth_luts.luts[0].get(1, 1, 1);
        let load_cell = loaded.depth_luts.luts[0].get(1, 1, 1);
        assert_eq!(orig_cell, load_cell);

        // Cleanup
        let _ = std::fs::remove_file(&tmp);
    }
}
