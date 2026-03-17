"""Phase 3: SAM2 Subject Tracking (~1-3 min per clip, local GPU)

Generates per-frame mask sequences for each tracked subject using SAM2.
Initialised at the exact first-appearance frame from Phase 2 scene analysis,
using the bounding box as the starting prompt.

Usage:
    python 03_run_sam2.py --date 2026-03-17

Inputs:
    - working/scene_analysis/{clip_id}.json — from Phase 2
    - Raw footage files (for full-resolution frame access)

Outputs:
    - working/masks/{clip_id}/{subject_label}/frame_NNNNNN.png
      (binary alpha mask, 0 or 255 per pixel)

Dependencies:
    - SAM2 (facebookresearch/sam2)
    - PyTorch with CUDA
    - VRAM: ~3GB (SAM2-small). Only one model loaded at a time.
    - Preferred: UWSAM variant if available (better underwater performance)

Architecture doc: Section 4.3
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Run SAM2 subject tracking from scene analysis")
    parser.add_argument("--date", required=True, help="Dive date YYYY-MM-DD")
    args = parser.parse_args()

    # TODO: Implement SAM2 tracking
    # 1. Load SAM2-small model (or UWSAM if available)
    # 2. For each clip, load scene_analysis JSON
    # 3. For each subject in the clip:
    #    a. Seek to first-appearance frame
    #    b. Initialise SAM2 predictor with bounding box
    #    c. Propagate mask forward through all frames
    #    d. Export mask as PNG sequence (binary alpha)
    #    e. Handle subject exit: pad with empty masks
    #    f. Handle re-entry: new SAM2 session after 30+ frame gap
    # 4. Unload SAM2 model before next phase

    print("SAM2 tracking not yet implemented", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
