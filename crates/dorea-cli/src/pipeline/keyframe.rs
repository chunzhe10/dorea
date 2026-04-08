//! Keyframe stage: proxy decode → change detection → keyframe selection.

use anyhow::{Context, Result};
use dorea_video::ffmpeg;

use crate::change_detect::{ChangeDetector, MseDetector};
use super::{PipelineConfig, KeyframeStageOutput};

/// A keyframe collected during the proxy-decode pass.
pub struct KeyframeEntry {
    pub frame_index: u64,
    pub proxy_pixels: Vec<u8>,
    /// True if this keyframe follows a scene cut (suppresses lerp across cuts).
    pub scene_cut_before: bool,
}

/// Run the keyframe detection stage: proxy decode → change detect → select keyframes.
///
/// Spawns a single ffmpeg process at proxy resolution and streams frames through
/// the change detector. Returns keyframes with their proxy pixels.
pub fn run_keyframe_stage(cfg: &PipelineConfig, info: &ffmpeg::VideoInfo) -> Result<KeyframeStageOutput> {
    let (proxy_w, proxy_h) = dorea_video::resize::proxy_dims(info.width, info.height, cfg.proxy_size);

    let mut keyframes: Vec<KeyframeEntry> = Vec::new();
    let mut detector: Box<dyn ChangeDetector> = Box::new(MseDetector::default());
    let mut frames_since_kf = 0usize;
    let scene_cut_threshold = cfg.depth_skip_threshold * 10.0;

    let proxy_frames = ffmpeg::decode_frames_scaled(&cfg.input, info, proxy_w, proxy_h)
        .context("failed to spawn ffmpeg proxy decoder")?;

    for frame_result in proxy_frames {
        let frame = frame_result.context("proxy frame decode error")?;
        let change = detector.score(&frame.pixels);
        let scene_cut = change < f32::MAX && change > scene_cut_threshold;
        let is_keyframe = !cfg.interp_enabled
            || keyframes.is_empty()
            || scene_cut
            || frames_since_kf >= cfg.depth_max_interval
            || (change < f32::MAX && change > cfg.depth_skip_threshold);

        if is_keyframe {
            if scene_cut {
                log::info!("Scene cut at frame {} (change={:.6})", frame.index, change);
                detector.reset();
            }
            keyframes.push(KeyframeEntry {
                frame_index: frame.index,
                proxy_pixels: frame.pixels.clone(),
                scene_cut_before: scene_cut,
            });
            detector.set_reference(&frame.pixels);
            frames_since_kf = 0;
        } else {
            frames_since_kf += 1;
        }
    }
    log::info!("Pass 1 complete: {} keyframes detected", keyframes.len());

    anyhow::ensure!(
        !keyframes.is_empty(),
        "pass 1 detected no keyframes — video may be empty or undecodable"
    );

    // Motion fields are populated when optical flow is enabled (future: OpticalFlowDetector).
    // For now, no motion vectors are computed during keyframe detection.
    let motion_fields = vec![None; keyframes.len()];

    Ok(KeyframeStageOutput { keyframes, proxy_w, proxy_h, motion_fields })
}
