//! Direct-mode grading pipeline.

pub mod grading;

use std::path::PathBuf;
use dorea_video::ffmpeg::{InputEncoding, OutputCodec};

/// Resolved pipeline configuration passed to the grading stage.
///
/// Holds everything `grading::run` needs: probed input, resolved codec and
/// encoding, RAUNE model paths, and proxy/batch tuning. Built once in
/// `grade::run` from CLI args + config + defaults.
pub struct PipelineConfig {
    // Input and probed metadata
    pub input: PathBuf,
    pub input_encoding: InputEncoding,
    pub output_codec: OutputCodec,
    pub output: PathBuf,

    // RAUNE subprocess parameters
    pub python: PathBuf,
    pub raune_weights: PathBuf,
    pub raune_models_dir: PathBuf,
    pub raune_proxy_size: usize,
    pub batch_size: usize,
}
