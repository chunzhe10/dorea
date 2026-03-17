"""Phase 0: Reference LUT Generation (One-Time Setup)

Analyses reference underwater images and generates a 33x33x33 3D .cube LUT
that captures the target colour look. This LUT is applied as Node 1 on every
clip in DaVinci Resolve.

Run once per look. Re-run when developing a new visual aesthetic.

Usage:
    python 00_generate_lut.py --references /path/to/reference/images --output /path/to/output.cube

Inputs:
    - 20-30 underwater reference images in references/ directory
    - Images should cover: different depths, lighting, subject types

Outputs:
    - luts/underwater_base.cube (33x33x33 3D LUT)

Dependencies:
    - colour-science, numpy, Pillow
    - No GPU required (CPU only)

Architecture doc: Section 4.0
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Generate underwater reference LUT from images")
    parser.add_argument("--references", required=True, help="Directory of reference images")
    parser.add_argument("--output", default="luts/underwater_base.cube", help="Output .cube file path")
    args = parser.parse_args()

    # TODO: Implement LUT generation
    # 1. Load reference images from --references directory
    # 2. Analyse colour characteristics: mean RGB per zone (shadow/midtone/highlight)
    # 3. Compute hue distribution, red channel falloff with depth, saturation targets
    # 4. Generate 33x33x33 3D LUT mapping D-Log M → target look
    # 5. Write .cube file to --output path

    print("LUT generation not yet implemented", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
