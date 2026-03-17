"""Phase 5: DaVinci Resolve Import & Setup (HOST ONLY)

WARNING: This script runs on the HOST MACHINE, not in the devcontainer.
It requires DaVinci Resolve Studio to be running and imports fusionscript.

Imports footage into Resolve, deploys the DRX template, applies the base LUT,
and attaches mask/depth sequences as Fusion mattes.

Usage:
    python 05_resolve_setup.py --date 2026-03-17

Inputs:
    - Raw footage files
    - working/masks/{clip_id}/{subject}/ — SAM2 mask sequences from Phase 3
    - working/depth/{clip_id}/ — depth map sequences from Phase 4
    - repos/dorea/luts/underwater_base.cube — reference LUT from Phase 0
    - repos/dorea/templates/underwater_grade_v1.drx — DRX node template

Outputs:
    - DaVinci Resolve project with timeline ready for creative grading
    - Each clip has 8-node structure: Base LUT, Neutral Balance, Depth Grade,
      Foreground Pop, Diver, Marine Life, Creative Look, Output

Dependencies:
    - DaVinci Resolve Studio (running on host)
    - fusionscript.so (Resolve Python API)
    - Environment vars: RESOLVE_SCRIPT_API, RESOLVE_SCRIPT_LIB, PYTHONPATH
    - No GPU required (Resolve uses GPU independently)

Architecture doc: Sections 5 and 7
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Set up DaVinci Resolve project with mattes")
    parser.add_argument("--date", required=True, help="Dive date YYYY-MM-DD")
    args = parser.parse_args()

    # TODO: Implement Resolve setup
    # Requires: RESOLVE_SCRIPT_API, RESOLVE_SCRIPT_LIB, PYTHONPATH set
    # import DaVinciResolveScript as dvr
    #
    # 1. Connect to running Resolve instance
    # 2. Create or open project (config: resolve_project_name)
    # 3. Import all footage to Media Pool
    # 4. Create timeline from imported clips
    # 5. For each clip:
    #    a. Apply DRX template (8-node structure)
    #    b. SetLUT(1, underwater_base.cube) on Node 1
    #    c. Import mask sequences as image sequences to Media Pool
    #    d. Import depth sequences as image sequences to Media Pool
    #    e. Connect mattes to corresponding nodes via Fusion page
    # 6. Queue initial optimised media render

    print("Resolve setup not yet implemented", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
