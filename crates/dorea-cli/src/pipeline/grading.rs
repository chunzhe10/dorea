//! Grading stage: full-res decode → depth interpolate → CUDA/CPU grade → encode.
//!
//! Structured as a 3-thread pipeline with bounded channels:
//!   Thread 1 (decoder):  ffmpeg frames → channel_a (capacity 4)
//!   Thread 2 (GPU/main): channel_a → depth interp + grade → channel_b (capacity 4)
//!   Thread 3 (encoder):  channel_b → ffmpeg encoder write
//!
//! The AdaptiveGrader is !Send — it stays on the main thread (Thread 2).

use anyhow::{Context, Result};

use dorea_video::ffmpeg::{self, Frame, FrameEncoder, VideoInfo};
use dorea_video::inference::InferenceServer;

#[cfg(feature = "cuda")]
use dorea_gpu::AdaptiveGrader;

use super::{PipelineConfig, CalibrationStageOutput};

/// Channel capacity for each of the two bounded queues (decoder→GPU, GPU→encoder).
/// Peak memory: 2 channels × 4 frames × 24 MB/frame (4K RGB24) ≈ 190 MB headroom.
const CHANNEL_CAPACITY: usize = 4;

/// Linearly interpolate between two f32 depth maps.
pub fn lerp_depth(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    let t = t.clamp(0.0, 1.0);
    a.iter().zip(b.iter())
        .map(|(&va, &vb)| va + (vb - va) * t)
        .collect()
}

/// Run the grading stage as a 3-thread pipeline: decode | grade | encode.
///
/// The GPU thread (main) owns the AdaptiveGrader (!Send). Decoder and encoder
/// run on spawned threads connected by bounded crossbeam channels.
pub fn run_grading_stage(
    cfg: &PipelineConfig,
    info: &VideoInfo,
    encoder: FrameEncoder,
    cal_out: CalibrationStageOutput,
) -> Result<u64> {
    let CalibrationStageOutput {
        segment_calibrations,
        smoothed_kf_zones,
        kf_to_segment,
        keyframe_depths,
        kf_index_list,
        store_len,
    } = cal_out;

    let params = cfg.grade_params();

    // Initialize adaptive CUDA grader on the main thread (Thread 2 — GPU owner).
    #[cfg(feature = "cuda")]
    let mut adaptive_grader = {
        let seg0 = &segment_calibrations[0];
        let base_flat: Vec<f32> = seg0.depth_luts.luts.iter()
            .flat_map(|lut| lut.data.iter().copied())
            .collect();
        let base_bounds = &seg0.depth_luts.zone_boundaries;
        let hsl = &seg0.hsl_corrections;
        let h_offsets: Vec<f32> = hsl.0.iter().map(|q| q.h_offset).collect();
        let s_ratios:  Vec<f32> = hsl.0.iter().map(|q| q.s_ratio).collect();
        let v_offsets: Vec<f32> = hsl.0.iter().map(|q| q.v_offset).collect();
        let weights:   Vec<f32> = hsl.0.iter().map(|q| q.weight).collect();
        let lut_size = seg0.depth_luts.luts[0].size;

        AdaptiveGrader::new(
            &base_flat, base_bounds, cfg.base_lut_zones,
            (&h_offsets, &s_ratios, &v_offsets, &weights),
            &params, lut_size, cfg.depth_zones,
        ).context("AdaptiveGrader init failed")?
    };

    #[cfg(feature = "cuda")]
    {
        adaptive_grader.prepare_keyframe(&smoothed_kf_zones[0])
            .context("prepare initial keyframe texture failed")?;
        adaptive_grader.swap_textures();
        if smoothed_kf_zones.len() > 1 {
            adaptive_grader.prepare_keyframe(&smoothed_kf_zones[1])
                .context("prepare second keyframe texture failed")?;
        }
        log::info!(
            "Adaptive CUDA grader initialized ({} base zones, {} runtime zones)",
            cfg.base_lut_zones, cfg.depth_zones,
        );
    }

    // Create bounded channels for the 3-thread pipeline.
    let (decoded_tx, decoded_rx) = crossbeam_channel::bounded::<Frame>(CHANNEL_CAPACITY);
    let (graded_tx, graded_rx) = crossbeam_channel::bounded::<Vec<u8>>(CHANNEL_CAPACITY);

    // Use std::thread::scope so spawned threads can borrow from this stack frame.
    let total_frames = info.frame_count;

    std::thread::scope(|s| -> Result<u64> {
        // ---------------------------------------------------------------
        // Thread 1: Decoder — reads frames from ffmpeg, sends to GPU thread
        // ---------------------------------------------------------------
        let decoder_handle = s.spawn(|| -> Result<()> {
            let frames = ffmpeg::decode_frames(&cfg.input, info)
                .context("failed to spawn ffmpeg full-res decoder")?;
            for frame_result in frames {
                let frame = frame_result.context("frame decode error")?;
                if decoded_tx.send(frame).is_err() {
                    // GPU thread dropped its receiver — likely hit an error
                    break;
                }
            }
            drop(decoded_tx);
            Ok(())
        });

        // ---------------------------------------------------------------
        // Thread 3: Encoder — receives graded frames, writes to ffmpeg
        // ---------------------------------------------------------------
        let encoder_handle = s.spawn(move || -> Result<u64> {
            let mut encoder = encoder;
            let mut frame_count = 0u64;
            for graded in graded_rx {
                encoder.write_frame(&graded).context("encoder write failed")?;
                frame_count += 1;
                if frame_count % 100 == 0 {
                    let pct = frame_count as f64 / total_frames.max(1) as f64 * 100.0;
                    log::info!("Progress: {frame_count}/{} frames ({:.1}%)", total_frames, pct);
                }
            }
            encoder.finish().context("ffmpeg encoder failed to finalize")?;
            Ok(frame_count)
        });

        // ---------------------------------------------------------------
        // GPU thread (main) — depth interp + grade
        // Wrapped in a closure so thread joins always run, even on error.
        // ---------------------------------------------------------------
        let gpu_result: Result<()> = (|| {
            let mut kf_cursor = 0usize;
            #[cfg(feature = "cuda")]
            let mut current_segment = kf_to_segment.first().copied().unwrap_or(0);

            for frame in decoded_rx {
                let fi = frame.index;

                // Advance cursor to most recent keyframe ≤ fi
                while kf_cursor + 1 < kf_index_list.len() && kf_index_list[kf_cursor + 1].0 <= fi {
                    kf_cursor += 1;

                    #[cfg(feature = "cuda")]
                    {
                        let new_seg = kf_to_segment[kf_cursor];
                        if new_seg != current_segment {
                            let seg_cal = &segment_calibrations[new_seg];
                            let base_flat: Vec<f32> = seg_cal.depth_luts.luts.iter()
                                .flat_map(|lut| lut.data.iter().copied())
                                .collect();
                            let base_bounds = &seg_cal.depth_luts.zone_boundaries;
                            let hsl = &seg_cal.hsl_corrections;
                            let h_offsets: Vec<f32> = hsl.0.iter().map(|q| q.h_offset).collect();
                            let s_ratios:  Vec<f32> = hsl.0.iter().map(|q| q.s_ratio).collect();
                            let v_offsets: Vec<f32> = hsl.0.iter().map(|q| q.v_offset).collect();
                            let weights:   Vec<f32> = hsl.0.iter().map(|q| q.weight).collect();
                            adaptive_grader.load_segment(
                                &base_flat, base_bounds,
                                (&h_offsets, &s_ratios, &v_offsets, &weights),
                            ).context("load_segment failed")?;
                            adaptive_grader.prepare_keyframe(&smoothed_kf_zones[kf_cursor])
                                .context("prepare_keyframe at segment boundary failed")?;
                            current_segment = new_seg;
                            log::info!("Segment switch to {new_seg} at keyframe {kf_cursor}");
                        }

                        adaptive_grader.swap_textures();

                        if kf_cursor + 1 < smoothed_kf_zones.len() {
                            adaptive_grader.prepare_keyframe(&smoothed_kf_zones[kf_cursor + 1])
                                .context("prepare_keyframe failed")?;
                        }
                    }
                }

                let (prev_kf_idx, _) = kf_index_list[kf_cursor];
                let (prev_depth_proxy, dpw, dph) = keyframe_depths
                    .get(&prev_kf_idx)
                    .expect("prev keyframe depth missing — logic error");
                let (dpw, dph) = (*dpw, *dph);

                let depth_proxy = if fi == prev_kf_idx {
                    prev_depth_proxy.clone()
                } else if let Some(&(next_kf_idx, scene_cut_before_next)) = kf_index_list.get(kf_cursor + 1) {
                    if scene_cut_before_next {
                        prev_depth_proxy.clone()
                    } else {
                        let (next_depth_proxy, _, _) = keyframe_depths
                            .get(&next_kf_idx)
                            .expect("next keyframe depth missing — logic error");
                        let t = (fi - prev_kf_idx) as f32 / (next_kf_idx - prev_kf_idx) as f32;
                        lerp_depth(prev_depth_proxy, next_depth_proxy, t)
                    }
                } else {
                    prev_depth_proxy.clone()
                };

                let depth = if dpw == frame.width && dph == frame.height {
                    depth_proxy
                } else {
                    InferenceServer::upscale_depth(&depth_proxy, dpw, dph, frame.width, frame.height)
                };

                let blend_t = if fi == prev_kf_idx {
                    0.0_f32
                } else if let Some(&(next_kf_idx, scene_cut)) = kf_index_list.get(kf_cursor + 1) {
                    if scene_cut { 0.0 } else {
                        ((fi - prev_kf_idx) as f32 / (next_kf_idx - prev_kf_idx) as f32).clamp(0.0, 1.0)
                    }
                } else {
                    0.0
                };

                #[cfg(feature = "cuda")]
                let graded = adaptive_grader.grade_frame_blended(
                    &frame.pixels, &depth, frame.width, frame.height, blend_t,
                ).map_err(|e| anyhow::anyhow!("Grading failed for frame {fi}: {e}"))?;

                #[cfg(not(feature = "cuda"))]
                let graded = {
                    use dorea_cal::Calibration;
                    use dorea_gpu::grade_frame;
                    let seg_idx = kf_to_segment.get(kf_cursor).copied().unwrap_or(0);
                    let cal = &segment_calibrations[seg_idx];
                    let calibration = Calibration::new(
                        cal.depth_luts.clone(), cal.hsl_corrections.clone(), store_len,
                    );
                    grade_frame(&frame.pixels, &depth, frame.width, frame.height, &calibration, &params)
                        .map_err(|e| anyhow::anyhow!("Grading failed for frame {fi}: {e}"))?
                };

                if graded_tx.send(graded).is_err() {
                    // Encoder thread dropped — likely hit an error
                    break;
                }
            }
            Ok(())
        })();

        // Signal encoder that no more frames are coming
        drop(graded_tx);

        // Always join threads — even when GPU errored — to collect all errors.
        let decoder_result = decoder_handle.join()
            .map_err(|_| anyhow::anyhow!("decoder thread panicked"))?;
        let encoder_result = encoder_handle.join()
            .map_err(|_| anyhow::anyhow!("encoder thread panicked"))?;

        // Propagate GPU error first (root cause), then thread errors.
        gpu_result.context("GPU grading failed")?;
        decoder_result.context("decoder thread failed")?;
        encoder_result
    })
}
