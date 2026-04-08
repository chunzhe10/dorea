//! DAG stage abstractions for the dorea grading pipeline.
//!
//! Stages: Keyframe → Feature → Calibration → Grading
//! Each stage is a free function with typed input/output structs.

pub mod keyframe;
pub mod feature;
pub mod calibration;
pub mod grading;

use std::collections::HashMap;
use std::path::PathBuf;

use dorea_gpu::GradeParams;
use dorea_video::ffmpeg::{InputEncoding, OutputCodec};
use crate::optical_flow::MotionField;

/// Resolved pipeline configuration — all CLI/config/defaults merged.
/// Passed to each stage by reference.
pub struct PipelineConfig {
    pub input: PathBuf,
    pub warmth: f32,
    pub strength: f32,
    pub contrast: f32,
    pub proxy_size: usize,
    pub depth_skip_threshold: f32,
    pub depth_max_interval: usize,
    pub fused_batch_size: usize,
    pub depth_zones: usize,
    pub base_lut_zones: usize,
    pub scene_threshold: f32,
    pub min_segment_kfs: usize,
    pub zone_smoothing_w: usize,
    pub maxine_upscale_factor: u32,
    pub interp_enabled: bool,
    pub maxine_in_fused_batch: bool,
    pub input_encoding: InputEncoding,
    pub output_codec: OutputCodec,
}

impl PipelineConfig {
    pub fn grade_params(&self) -> GradeParams {
        GradeParams {
            warmth: self.warmth,
            strength: self.strength,
            contrast: self.contrast,
        }
    }
}

/// Output of KeyframeStage → input of FeatureStage.
pub struct KeyframeStageOutput {
    pub keyframes: Vec<keyframe::KeyframeEntry>,
    pub proxy_w: usize,
    pub proxy_h: usize,
    /// Per-frame motion vectors from optical flow (None for first frame).
    pub motion_fields: Vec<Option<MotionField>>,
}

/// Output of FeatureStage → input of CalibrationStage.
pub struct FeatureStageOutput {
    pub store: feature::PagedCalibrationStore,
    pub keyframe_depths: HashMap<u64, (Vec<f32>, usize, usize)>,
    pub keyframes: Vec<keyframe::KeyframeEntry>,
    /// Per-keyframe YOLO-seg class masks (0=water, 1=diver). Keyed by frame_index.
    pub keyframe_masks: HashMap<u64, (Vec<u8>, usize, usize)>,
}

/// Output of CalibrationStage → input of GradingStage.
pub struct CalibrationStageOutput {
    pub segment_calibrations: Vec<calibration::SegmentCalibration>,
    pub smoothed_kf_zones: Vec<Vec<f32>>,
    pub kf_to_segment: Vec<usize>,
    pub keyframe_depths: HashMap<u64, (Vec<f32>, usize, usize)>,
    pub kf_index_list: Vec<(u64, bool)>,
    pub store_len: usize,
    /// Per-keyframe class masks from YOLO-seg. Keyed by frame_index.
    pub keyframe_masks: HashMap<u64, (Vec<u8>, usize, usize)>,
}
