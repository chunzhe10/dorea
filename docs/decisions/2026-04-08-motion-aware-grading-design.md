# Motion-Aware Content-Adaptive Grading Design

**Date:** 2026-04-08
**Status:** Approved
**Issues:** chunzhe10/dorea#50, #52, #51

## Overview

Three subsystems integrated into the existing 4-stage pipeline:
1. NVIDIA OFA optical flow for keyframe detection + depth map warping (#50)
2. YOLOv11n-seg binary segmentation (diver vs water) for per-class grading (#52)
3. CUDA guided filter to snap warped depth to class boundaries (#51)

## Key Decisions

- Optical flow via direct CUDA driver API (not Python) — runs on OFA hardware engine
- YOLO-seg via Python inference server — PyTorch model, fits existing pattern
- Binary classification (diver=1 / water=0) using COCO "person" zero-shot
- YOLO runs on ALL proxy frames (not keyframe-only) — cheap at 5-15ms/frame
- Guided filter as CUDA kernel — runs at proxy resolution alongside grading
- Motion vectors stored per-frame during Pass 1, consumed in Pass 2 (grading)
- Depth warping replaces linear lerp: warp prev keyframe depth using motion vectors
- Per-class LUT bank: combined texture becomes N_classes x N_zones x 97^3

## Architecture

```
Pass 1 (KeyframeStage):
  proxy decode → optical flow (OFA) → change detect → keyframe select
                                    → motion vectors stored per-frame
              → YOLO-seg (all frames) → class masks stored per-frame

Pass 2 (GradingStage):
  full-res decode → warp_depth(prev_kf_depth, motion_vectors)
                  → guided_filter(warped_depth, class_mask)  [CUDA]
                  → combined_lut_kernel(pixels, depth, class_id)  [per-class LUT]
                  → encode
```

## Implementation Phases

Phase A (#50): OpticalFlowDetector + warp_depth + motion vector storage
Phase B (#52): YOLO-seg Python model + IPC + class mask storage
Phase C (#51): Guided filter CUDA kernel + per-class LUT kernel changes + integration

Phases A and B are independent. Phase C depends on both.

## Files

| Phase | File | Change |
|-------|------|--------|
| A | `crates/dorea-cli/src/optical_flow.rs` | New — OFA bindings + OpticalFlowDetector |
| A | `crates/dorea-cli/src/pipeline/keyframe.rs` | Add optical flow to Pass 1 |
| A | `crates/dorea-cli/src/pipeline/mod.rs` | Add MotionField to boundary structs |
| A | `crates/dorea-cli/src/pipeline/grading.rs` | Replace lerp_depth with warp_depth |
| B | `python/dorea_inference/yolo_seg.py` | New — YOLOv11n-seg inference |
| B | `python/dorea_inference/server.py` | Add yolo_seg_batch handler |
| B | `python/dorea_inference/bridge.py` | Add run_yolo_seg_batch_cpu |
| B | `crates/dorea-video/src/inference_subprocess.rs` | Add YOLO-seg IPC |
| B | `crates/dorea-cli/src/pipeline/keyframe.rs` | Add YOLO-seg to Pass 1 |
| C | `crates/dorea-gpu/src/cuda/kernels/guided_filter.cu` | New — guided filter kernel |
| C | `crates/dorea-gpu/src/cuda/kernels/combined_lut.cu` | Add class_id to LUT lookup |
| C | `crates/dorea-gpu/src/cuda/combined_lut.rs` | Multi-class texture management |
| C | `crates/dorea-cli/src/pipeline/grading.rs` | Wire guided filter + per-class grading |
