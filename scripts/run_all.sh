#!/bin/bash
# Master overnight batch script for Dorea pipeline
# Runs Phases 1-4 inside the devcontainer, then prompts for Phase 5 on host.
#
# Usage: bash scripts/run_all.sh [YYYY-MM-DD]
#        Defaults to today's date if no argument given.
#
# Requires:
#   - config.yaml configured with correct paths
#   - Footage dumped to footage/raw/ and/or footage/flat/
#
# Architecture doc: Section 10

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE="${1:-$(date +%Y-%m-%d)}"

# Activate venv if not already active
if [ -z "$VIRTUAL_ENV" ] && [ -f /opt/dorea-venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source /opt/dorea-venv/bin/activate
fi

echo "=== Dorea Pipeline — $DATE ==="
echo ""

# Phase 0 (00_generate_lut.py) is NOT included here — it is a one-time setup
# that runs once per visual look, not per dive session.
# Run manually: python scripts/00_generate_lut.py --references references/look_v1/ --output luts/underwater_base.cube

# DRX template (create_drx_template.py) is also one-time — run on the HOST with
# Resolve to generate templates/underwater_grade_v1.drx before first Phase 5 run.
# On the host: python scripts/create_drx_template.py

# LUT preview (preview_lut.py) is an optional evaluation tool — run after Phase 1
# to visually compare RAW vs LUT-graded keyframes without opening Resolve.
# Run manually: python scripts/preview_lut.py --date YYYY-MM-DD

echo "=== Phase 1: Extract keyframes ==="
python "$SCRIPT_DIR/01_extract_frames.py" --date "$DATE"

echo "=== Phase 1b: Per-clip white balance estimation ==="
python "$SCRIPT_DIR/01b_estimate_white_balance.py" --date "$DATE"

echo "=== Phase 2: Claude scene analysis ==="
python "$SCRIPT_DIR/02_claude_scene_analysis.py" --date "$DATE"

echo "=== Phase 3: SAM2 subject tracking ==="
python "$SCRIPT_DIR/03_run_sam2.py" --date "$DATE"

echo "=== Phase 4: Depth estimation ==="
python "$SCRIPT_DIR/04_run_depth.py" --date "$DATE"

echo ""
echo "=== Phases 1-4 complete ==="
echo ""
echo "Phase 5 must run on the HOST (requires DaVinci Resolve)."
echo "Open a terminal on the host and run:"
echo ""
echo "  cd <your-workspace-root>/repos/dorea"
echo "  python scripts/05_resolve_setup.py --date $DATE"
echo ""
echo "Then open Resolve — your timeline is ready for creative grading."
