# NVIDIA Maxine VFX SDK Setup

Optional dependency for AI-enhanced pre-processing. When enabled, Maxine runs on
every input frame at full resolution before RAUNE and depth inference.

## Prerequisites

- NVIDIA GPU: Turing, Ampere, Ada, or Blackwell with Tensor Cores (RTX 3060 ✓)
- Linux driver: 570.190+, 580.82+, or 590.44+
- Python 3.10+
- dorea inference venv active (`/opt/dorea-venv`)

## Installation

1. Download the Maxine Video Effects SDK from
   [NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/maxine/resources/maxine_linux_vfx_sdk_ga)

2. Install the Python bindings:

   ```bash
   source /opt/dorea-venv/bin/activate
   pip install nvvfx
   ```

3. Verify installation:

   ```bash
   python -c "import nvvfx; print('nvvfx OK')"
   ```

## Usage

Enable Maxine preprocessing when grading:

```bash
dorea grade --input footage.mp4 --maxine
```

CLI options:

| Flag | Default | Description |
|------|---------|-------------|
| `--maxine` | off | Enable Maxine preprocessing |
| `--maxine-upscale-factor N` | 2 | Super-resolution factor (2, 3, 4) |
| `--no-maxine-artifact-reduction` | — | Skip artifact reduction step |

## What it does

Maxine runs **before** RAUNE and depth inference, on every frame at full resolution:

1. Artifact reduction at original resolution (if ≤1080p input)
2. 2× AI super-resolution (e.g. 1080p → 4K intermediate)
3. Area downsample back to original resolution

The Maxine-enhanced frames are stored in a **temporary lossless video file** (~2–4 GB for
5-min 1080p) and reused in the grading pass. Maxine runs once per frame, not twice.

All downstream algorithms (RAUNE, Depth Anything, LUT calibration, GPU grading) operate
on Maxine-enhanced frames for consistent quality.

## Disk space

The temp file is created in `$TMPDIR` and deleted automatically after grading.
Estimated size: `width × height × fps × duration_s × 0.5 bits/px`.

| Clip | Resolution | Duration | Approx temp size |
|------|------------|----------|-----------------|
| Short | 1080p | 1 min | ~0.5 GB |
| Medium | 1080p | 5 min | ~2–4 GB |
| Long | 4K | 10 min | ~15–20 GB |

Ensure `$TMPDIR` has sufficient free space before running with `--maxine`.

## VRAM budget

All models (Maxine + RAUNE + Depth) are loaded simultaneously in one Python process
during Pass 1 and calibration (~4.8–5.6 GB peak on RTX 3060 6 GB).

After calibration, the inference server shuts down — Pass 2 grading uses only
the CUDA grader (~300 MB).

If you run out of VRAM:
- Try `--no-maxine-artifact-reduction` to save ~500 MB
- Or disable Maxine entirely (`dorea grade --input footage.mp4` without `--maxine`)

## Known limitations

- **8-bit bottleneck**: Maxine accepts only uint8 input. Full pipeline operates in 8-bit
  color depth when Maxine is enabled (vs 10-bit without Maxine).
- **H.265 source**: Artifact reduction is trained for H.264. Effectiveness on H.265/HEVC
  footage is unvalidated, but expected to reduce common blocking/ringing artifacts.
- **Supported scale factors**: 2, 3, 4 only (`--maxine-upscale-factor`).
  Factors 3 and 4 require more VRAM — validate with `nvidia-smi` on first use.

## CI / Testing without the SDK

Set `DOREA_MAXINE_MOCK=1` to enable passthrough mode (identity transform — no nvvfx needed):

```bash
DOREA_MAXINE_MOCK=1 dorea grade --input test.mp4 --maxine
```

All IPC paths, the temp file creation/deletion lifecycle, and keyframe detection on
Maxine-proxied frames are exercised even in mock mode.
