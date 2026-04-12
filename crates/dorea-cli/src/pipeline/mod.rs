//! Direct-mode grading pipeline.

pub mod grading;

use std::path::PathBuf;
use dorea_video::ffmpeg::OutputCodec;

/// Resolved pipeline configuration passed to the grading stage.
pub struct PipelineConfig {
    pub input: PathBuf,
    pub output_codec: OutputCodec,
    pub output: PathBuf,
    pub python: PathBuf,
    pub raune_weights: PathBuf,
    pub raune_models_dir: PathBuf,
    pub raune_proxy_size: usize,
    pub batch_size: usize,
    pub proxy_w: usize,
    pub proxy_h: usize,
}
