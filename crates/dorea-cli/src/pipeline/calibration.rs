//! Calibration stage: zone compute → segment detect → LUT build → HSL average → grader init.

use anyhow::Result;

use dorea_hsl::derive::{derive_hsl_corrections, HslCorrections, QualifierCorrection};
use dorea_hsl::{HSL_QUALIFIERS, MIN_WEIGHT};
use dorea_lut::apply::apply_depth_luts;
use dorea_lut::build::{adaptive_zone_boundaries, compute_importance, StreamingLutBuilder};

use crate::change_detect::{detect_scene_segments, compute_per_kf_zones, smooth_zone_boundaries};
use super::{PipelineConfig, FeatureStageOutput, CalibrationStageOutput};

/// Per-segment base LUT + HSL corrections.
pub struct SegmentCalibration {
    pub depth_luts: dorea_lut::types::DepthLuts,
    pub hsl_corrections: HslCorrections,
}

/// Run the calibration stage: compute zones, detect segments, build LUTs, derive HSL.
pub fn run_calibration_stage(
    cfg: &PipelineConfig,
    feat_out: FeatureStageOutput,
) -> Result<CalibrationStageOutput> {
    let FeatureStageOutput { store, keyframe_depths, keyframes, keyframe_masks } = feat_out;

    // Step 1: Collect per-keyframe depth maps from store
    let kf_depths_store: Vec<Vec<f32>> = (0..store.len())
        .map(|i| store.read_depth(i))
        .collect();

    // Step 2: Compute per-keyframe zone boundaries
    let raw_kf_zones = compute_per_kf_zones(&kf_depths_store, cfg.depth_zones);

    // Step 3: Detect scene segments via Wasserstein-1 depth histogram distance
    let segments = detect_scene_segments(&kf_depths_store, cfg.scene_threshold, cfg.min_segment_kfs);
    log::info!("Scene segments: {} (from {} keyframes)", segments.len(), store.len());
    for (si, seg) in segments.iter().enumerate() {
        log::info!(
            "  segment {si}: keyframes {}..{} ({} KFs)",
            seg.start, seg.end, seg.end - seg.start
        );
    }

    // Step 4: Smooth zone boundaries (respecting segment boundaries)
    let smoothed_kf_zones = smooth_zone_boundaries(&raw_kf_zones, &segments, cfg.zone_smoothing_w);

    // Step 5: Build per-segment base LUTs + HSL corrections
    let n_quals = HSL_QUALIFIERS.len();

    let segment_calibrations: Vec<SegmentCalibration> = segments.iter().map(|seg| {
        let seg_depths: Vec<f32> = (seg.start..seg.end)
            .flat_map(|i| kf_depths_store[i].iter().cloned())
            .collect();
        let base_boundaries = adaptive_zone_boundaries(&seg_depths, cfg.base_lut_zones);

        let mut lut_builder = StreamingLutBuilder::new(base_boundaries);
        for i in seg.start..seg.end {
            let (w, h) = store.dims(i);
            let (pixels_u8, target_u8) = store.pixtar_slices(i);
            let depth = store.read_depth(i);
            let original: Vec<[f32; 3]> = pixels_u8.chunks_exact(3)
                .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
                .collect();
            let target: Vec<[f32; 3]> = target_u8.chunks_exact(3)
                .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
                .collect();
            let importance = compute_importance(&depth, w, h);
            lut_builder.add_frame(&original, &target, &depth, &importance);
        }
        let depth_luts = lut_builder.finish();

        // HSL corrections: weighted average over keyframes in segment
        let mut h_offset_acc    = vec![0.0_f64; n_quals];
        let mut s_ratio_acc     = vec![0.0_f64; n_quals];
        let mut v_offset_acc    = vec![0.0_f64; n_quals];
        let mut active_count    = vec![0_usize;  n_quals];
        let mut total_weight_acc = vec![0.0_f64; n_quals];

        for i in seg.start..seg.end {
            let (pixels_u8, target_u8) = store.pixtar_slices(i);
            let depth = store.read_depth(i);
            let original: Vec<[f32; 3]> = pixels_u8.chunks_exact(3)
                .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
                .collect();
            let target: Vec<[f32; 3]> = target_u8.chunks_exact(3)
                .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
                .collect();
            let lut_output = apply_depth_luts(&original, &depth, &depth_luts);
            let corrs = derive_hsl_corrections(&lut_output, &target);
            for (qi, corr) in corrs.0.iter().enumerate() {
                if corr.weight >= MIN_WEIGHT {
                    let w = corr.weight as f64;
                    h_offset_acc[qi]     += corr.h_offset as f64 * w;
                    s_ratio_acc[qi]      += corr.s_ratio  as f64 * w;
                    v_offset_acc[qi]     += corr.v_offset as f64 * w;
                    active_count[qi]     += 1;
                    total_weight_acc[qi] += w;
                }
            }
        }

        let avg_corrections: Vec<QualifierCorrection> = (0..n_quals).map(|qi| {
            let qual = &HSL_QUALIFIERS[qi];
            if active_count[qi] > 0 {
                let tw = total_weight_acc[qi];
                QualifierCorrection {
                    h_center: qual.h_center,
                    h_width:  qual.h_width,
                    h_offset: (h_offset_acc[qi] / tw) as f32,
                    s_ratio:  (s_ratio_acc[qi]  / tw) as f32,
                    v_offset: (v_offset_acc[qi] / tw) as f32,
                    weight:   tw as f32,
                }
            } else {
                QualifierCorrection {
                    h_center: qual.h_center,
                    h_width:  qual.h_width,
                    h_offset: 0.0,
                    s_ratio:  1.0,
                    v_offset: 0.0,
                    weight:   0.0,
                }
            }
        }).collect();

        SegmentCalibration {
            depth_luts,
            hsl_corrections: HslCorrections(avg_corrections),
        }
    }).collect();

    log::info!(
        "Pre-compute complete: {} segments, {} per-KF zone sets",
        segment_calibrations.len(), smoothed_kf_zones.len(),
    );

    // Build segment-to-keyframe index lookup
    let kf_to_segment: Vec<usize> = {
        let mut map = vec![0usize; store.len()];
        for (si, seg) in segments.iter().enumerate() {
            for ki in seg.start..seg.end {
                map[ki] = si;
            }
        }
        map
    };

    let kf_index_list: Vec<(u64, bool)> = keyframes.iter()
        .map(|kf| (kf.frame_index, kf.scene_cut_before))
        .collect();

    let store_len = store.len();

    Ok(CalibrationStageOutput {
        segment_calibrations,
        smoothed_kf_zones,
        kf_to_segment,
        keyframe_depths,
        kf_index_list,
        store_len,
        keyframe_masks,
    })
}
