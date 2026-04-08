//! Grading stage: full-res decode → depth interpolate → blend_t → CUDA grade → encode.

use anyhow::{Context, Result};

use dorea_video::ffmpeg::{self, FrameEncoder, VideoInfo};
use dorea_video::inference::InferenceServer;

#[cfg(feature = "cuda")]
use dorea_gpu::AdaptiveGrader;

use super::{PipelineConfig, CalibrationStageOutput};

/// Linearly interpolate between two f32 depth maps.
pub fn lerp_depth(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    let t = t.clamp(0.0, 1.0);
    a.iter().zip(b.iter())
        .map(|(&va, &vb)| va + (vb - va) * t)
        .collect()
}

/// Run the grading stage: full-res decode → depth interpolate → grade → encode.
pub fn run_grading_stage(
    cfg: &PipelineConfig,
    info: &VideoInfo,
    mut encoder: FrameEncoder,
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

    // Initialize adaptive CUDA grader with first segment's base LUT
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

    // Build first keyframe's runtime texture
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

    // Full-resolution decode + grading
    let frames = ffmpeg::decode_frames(&cfg.input, info)
        .context("failed to spawn ffmpeg full-res decoder")?;

    let mut kf_cursor = 0usize;
    let mut frame_count = 0u64;
    #[cfg(feature = "cuda")]
    let mut current_segment = kf_to_segment.first().copied().unwrap_or(0);

    for frame_result in frames {
        let frame = frame_result.context("frame decode error")?;
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

        // Lerp depth at proxy resolution, then upscale once
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

        encoder.write_frame(&graded).context("encoder write failed")?;
        frame_count += 1;

        if frame_count % 100 == 0 {
            let pct = frame_count as f64 / info.frame_count.max(1) as f64 * 100.0;
            log::info!("Progress: {frame_count}/{} frames ({:.1}%)", info.frame_count, pct);
        }
    }

    encoder.finish().context("ffmpeg encoder failed to finalize")?;
    Ok(frame_count)
}
