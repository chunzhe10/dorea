"""Phase 2: Claude Scene Analysis — Subject Detection (~2-5 min per clip, API cost)

Sends keyframe batches to Claude API for temporal scene analysis. Claude identifies
subjects (divers, fish species, coral), their bounding boxes, and first-appearance
frames. Uses two-pass sampling for frame-accurate detection.

Usage:
    python 02_claude_scene_analysis.py --date 2026-03-17

Inputs:
    - working/keyframes/{clip_id}/ — extracted keyframes from Phase 1

Outputs:
    - working/scene_analysis/{clip_id}.json — per-clip scene analysis
      Schema: {"clip_id": str, "frames": {frame_id: {subjects: [], new_subjects: [{label, bbox_normalised, first_appearance, confidence}]}}}

Dependencies:
    - anthropic SDK
    - No GPU required (API call)
    - Estimated cost: <$2 per dive session

Architecture doc: Section 4.2
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Run Claude scene analysis on keyframes")
    parser.add_argument("--date", required=True, help="Dive date YYYY-MM-DD")
    args = parser.parse_args()

    # TODO: Implement Claude scene analysis
    # Pass 1: Sample every 2s — Claude identifies approximate appearance windows
    # 1. Load keyframes in batches of 12 (config: claude_batch_size)
    # 2. Encode as base64, send to Claude API (claude-sonnet-4-6)
    # 3. Claude returns structured JSON with subjects, bboxes, timestamps
    #
    # Pass 2: Around each detected appearance, sample every 10 frames
    # 4. Pinpoint exact first-appearance frame per subject
    #
    # 5. Write scene_analysis JSON per clip

    print("Scene analysis not yet implemented", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
