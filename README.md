# Dorea

Automated underwater video AI editing pipeline. Named after the Dorado
constellation (the golden fish).

## What it does

Dorea automates the technically repetitive stages of underwater video
post-production while preserving full creative control for the human editor.

### Pipeline phases

| # | Script | What it does | GPU |
|---|--------|-------------|-----|
| 0 | `00_generate_lut.py` | Reference images → .cube LUT | No |
| 1 | `01_extract_frames.py` | ffmpeg keyframe extraction | No |
| 1b | `01b_estimate_white_balance.py` | Per-clip WB estimation from keyframes | No |
| 2 | `02_claude_scene_analysis.py` | Claude API scene + subject detection | No |
| 3 | `03_run_sam2.py` | SAM2 per-subject mask tracking | ~3GB |
| 4 | `04_run_depth.py` | Depth Anything V2 depth maps | ~1.5GB |
| 5 | `05_resolve_setup.py` | Resolve API: import, DRX, WB, mattes | No |

### Hardware requirements

- Linux workstation
- NVIDIA GPU with 6GB+ VRAM (RTX 3060 or better)
- 64GB RAM recommended
- DaVinci Resolve Studio (one-time purchase)

### Camera support

- DJI Action 4 (D-Log M)
- Insta360 X5 (pre-flattened via Insta360 Studio)

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/sam2.git

# 2. Download model weights
# SAM2: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
# Depth Anything V2: auto-downloads from HuggingFace on first run

# 3. Configure
# Edit config.yaml with your paths and API key

# 4. One-time: generate reference LUT
python scripts/00_generate_lut.py --references references/look_v1/ --output luts/underwater_base.cube

# 5. After each dive: run overnight batch
bash scripts/run_all.sh

# 6. Morning: open Resolve — timeline is ready for creative grading
```

## Architecture

See `underwater_pipeline_architecture.docx` for the full architecture document.
