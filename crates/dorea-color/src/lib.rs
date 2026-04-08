pub mod dlog_m;
pub mod hsv;
pub mod ilog;
pub mod lab;

/// Camera log curve abstraction. Implementations decode/encode between
/// a camera-specific log encoding and scene-linear light.
pub trait TransferFunction {
    /// Decode a log-encoded value to scene-linear light.
    fn to_linear(&self, encoded: f32) -> f32;
    /// Encode a scene-linear light value to log encoding.
    fn from_linear(&self, linear: f32) -> f32;
    /// Highlight rolloff shoulder (0.0–1.0). Higher = later compression.
    fn shoulder(&self) -> f32;
    /// Human-readable name (e.g. "D-Log M").
    fn name(&self) -> &'static str;
}

pub use dlog_m::DLogM;
pub use ilog::ILog;
