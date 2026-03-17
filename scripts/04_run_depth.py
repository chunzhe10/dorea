"""Phase 4: Depth Anything V2 — Depth Map Generation (~1-2 min per clip, local GPU)

Performs monocular depth estimation on every frame. Output is a per-frame depth
map used as a luminance matte in Resolve to drive depth-dependent colour correction.

Bright pixels = close to camera. Dark pixels = far from camera.

Usage:
    python 04_run_depth.py --date 2026-03-17

Inputs:
    - Raw footage files (for full-resolution frame access)

Outputs:
    - working/depth/{clip_id}/frame_NNNNNN.png
      (16-bit grayscale PNG, resolution matches source)

Dependencies:
    - Depth Anything V2 Small (via transformers)
    - PyTorch with CUDA
    - VRAM: ~1.5GB
    - Note: Trained on terrestrial footage. Reduced accuracy in turbid water.

Architecture doc: Section 4.4
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Run Depth Anything V2 depth estimation")
    parser.add_argument("--date", required=True, help="Dive date YYYY-MM-DD")
    args = parser.parse_args()

    # TODO: Implement depth estimation
    # 1. Load Depth Anything V2 Small model
    # 2. For each clip, extract every frame at source resolution
    # 3. Run depth estimation per frame
    # 4. Save as 16-bit grayscale PNG (preserves precision for soft mattes)
    # 5. Unload model before next phase

    print("Depth estimation not yet implemented", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
