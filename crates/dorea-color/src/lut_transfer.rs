//! 1D LUT-based transfer function.
//!
//! Loads a 1D .cube file as a transfer function for cameras whose exact
//! log curve is unknown. Drop in an empirical extraction to replace the
//! S-Log3 proxy for I-Log.

use crate::TransferFunction;
use std::path::Path;

/// A transfer function defined by a 1D lookup table.
pub struct LutBased {
    forward: Vec<f32>,
    inverse: Vec<f32>,
    shoulder: f32,
    name: String,
}

impl LutBased {
    /// Build from a forward LUT (encoded → linear).
    /// The inverse is computed by table inversion.
    pub fn new(forward: Vec<f32>, shoulder: f32, name: String) -> Self {
        let inverse = invert_1d_lut(&forward);
        Self { forward, inverse, shoulder, name }
    }

    /// Load from a 1D .cube file.
    pub fn from_cube_file(path: &Path, shoulder: f32) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;

        let mut size: Option<usize> = None;
        let mut values: Vec<f32> = Vec::new();
        let mut title = path.file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "custom".to_string());

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(rest) = line.strip_prefix("TITLE") {
                title = rest.trim().trim_matches('"').to_string();
                continue;
            }
            if let Some(rest) = line.strip_prefix("LUT_1D_SIZE") {
                size = Some(rest.trim().parse::<usize>()
                    .map_err(|e| format!("bad LUT_1D_SIZE: {e}"))?);
                continue;
            }
            if line.starts_with("LUT_3D_SIZE") || line.starts_with("DOMAIN_") {
                continue;
            }
            let first_token = line.split_whitespace().next().unwrap_or("");
            if let Ok(v) = first_token.parse::<f32>() {
                values.push(v);
            }
        }

        if let Some(sz) = size {
            if values.len() != sz {
                return Err(format!(
                    "LUT_1D_SIZE={sz} but found {} values", values.len()
                ));
            }
        }
        if values.len() < 2 {
            return Err("LUT must have at least 2 entries".to_string());
        }

        Ok(Self::new(values, shoulder, title))
    }
}

impl TransferFunction for LutBased {
    fn to_linear(&self, encoded: f32) -> f32 {
        lerp_lut(&self.forward, encoded)
    }

    fn from_linear(&self, linear: f32) -> f32 {
        lerp_lut(&self.inverse, linear)
    }

    fn shoulder(&self) -> f32 {
        self.shoulder
    }

    fn name(&self) -> &'static str {
        // Leak to get 'static. LutBased lives for the pipeline duration.
        Box::leak(self.name.clone().into_boxed_str())
    }
}

fn lerp_lut(lut: &[f32], x: f32) -> f32 {
    let n = lut.len();
    if n == 0 { return x; }
    let x = x.clamp(0.0, 1.0);
    let pos = x * (n - 1) as f32;
    let lo = (pos as usize).min(n - 2);
    let hi = lo + 1;
    let frac = pos - lo as f32;
    lut[lo] * (1.0 - frac) + lut[hi] * frac
}

fn invert_1d_lut(forward: &[f32]) -> Vec<f32> {
    let n = forward.len();
    let mut inverse = vec![0.0f32; n];
    let out_min = forward[0];
    let out_max = forward[n - 1];
    let range = (out_max - out_min).max(1e-10);

    for i in 0..n {
        let target = i as f32 / (n - 1) as f32;
        let target_scaled = out_min + target * range;
        let mut lo = 0usize;
        let mut hi = n - 1;
        while lo < hi - 1 {
            let mid = (lo + hi) / 2;
            if forward[mid] <= target_scaled {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let denom = (forward[hi] - forward[lo]).max(1e-10);
        let frac = (target_scaled - forward[lo]) / denom;
        inverse[i] = (lo as f32 + frac) / (n - 1) as f32;
    }
    inverse
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_lut(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32 / (n - 1) as f32).collect()
    }

    #[test]
    fn test_identity_round_trip() {
        let lut = LutBased::new(identity_lut(256), 0.90, "identity".to_string());
        for i in 0..=10 {
            let x = i as f32 / 10.0;
            let linear = lut.to_linear(x);
            let encoded = lut.from_linear(linear);
            assert!(
                (encoded - x).abs() < 0.01,
                "Identity round-trip failed at {x}: linear={linear}, encoded={encoded}"
            );
        }
    }

    #[test]
    fn test_gamma_lut_round_trip() {
        let n = 1024;
        let forward: Vec<f32> = (0..n)
            .map(|i| (i as f32 / (n - 1) as f32).powf(2.2))
            .collect();
        let lut = LutBased::new(forward, 0.85, "gamma2.2".to_string());
        let values = [0.0, 0.1, 0.18, 0.5, 0.8, 1.0];
        for &v in &values {
            let linear = lut.to_linear(v);
            let back = lut.from_linear(linear);
            assert!(
                (back - v).abs() < 0.01,
                "Gamma round-trip failed at {v}: linear={linear}, back={back}"
            );
        }
    }

    #[test]
    fn test_lerp_lut_boundaries() {
        let lut = vec![0.0, 0.5, 1.0];
        assert!((lerp_lut(&lut, 0.0) - 0.0).abs() < 1e-6);
        assert!((lerp_lut(&lut, 0.5) - 0.5).abs() < 1e-6);
        assert!((lerp_lut(&lut, 1.0) - 1.0).abs() < 1e-6);
        assert!((lerp_lut(&lut, 0.25) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_trait_impl() {
        let lut = LutBased::new(identity_lut(256), 0.90, "test".to_string());
        let tf: &dyn crate::TransferFunction = &lut;
        assert_eq!(tf.shoulder(), 0.90);
        assert_eq!(tf.name(), "test");
    }
}
