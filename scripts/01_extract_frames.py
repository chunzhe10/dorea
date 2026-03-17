"""Phase 1: Footage Ingest & Frame Extraction (~5-10 min per dive session)

Extracts keyframes from raw footage at 2-second intervals for AI analysis.
Handles both DJI Action 4 (D-Log M) and Insta360 X5 (pre-flattened) clips.

Usage:
    python 01_extract_frames.py --date 2026-03-17

Inputs:
    - footage/raw/YYYY-MM-DD/ — DJI Action 4 clips
    - footage/flat/YYYY-MM-DD/ — Insta360 X5 flattened clips

Outputs:
    - working/keyframes/{clip_id}/frame_NNNNNN.jpg (1280px wide, JPEG 85%)

Dependencies:
    - ffmpeg (system package, CPU only, no GPU required)

Architecture doc: Section 4.1
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Extract keyframes from dive footage")
    parser.add_argument("--date", required=True, help="Dive date YYYY-MM-DD")
    args = parser.parse_args()

    # TODO: Implement frame extraction
    # 1. Scan footage/raw/{date}/ and footage/flat/{date}/ for video files
    # 2. For each clip, generate clip_id from filename
    # 3. Run: ffmpeg -i {clip} -vf fps=0.5,scale=1280:-1 -q:v 5 working/keyframes/{clip_id}/frame_%06d.jpg
    # 4. Log clip count and frame count

    print("Frame extraction not yet implemented", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
