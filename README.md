# Dorea

Automated underwater video color grading — single-pass direct-mode pipeline.
Named after the Dorado constellation.

## Architecture

A thin Rust CLI (`dorea`) that probes input video, resolves config, and
spawns a Python subprocess (`dorea_inference.raune_filter`) which does all
the heavy lifting: PyAV decode, batched RAUNE-Net inference at proxy
resolution on CUDA, OKLab delta computation, bilinear upscale to full
resolution, fused Triton OKLab transfer, PyAV encode — all in a single
producer-consumer 3-thread pipeline.

```
crates/
├── dorea-cli    — `dorea` binary (argument parsing, config, subprocess spawn)
└── dorea-video  — ffmpeg probe, InputEncoding auto-detect, OutputCodec enum

python/dorea_inference/
├── raune_filter.py  — all runtime logic: decode → RAUNE → OKLab delta → encode
└── __init__.py
```

## Hardware requirements

- NVIDIA GPU with ≥ 6 GB VRAM (RTX 3060 or better) — required for RAUNE-Net fp16 inference + Triton OKLab transfer.
- Linux workstation or devcontainer.
- FFmpeg with HEVC/H.264 support.

## Build

```bash
cargo build --release -p dorea-cli
```

The binary lands at `target/release/dorea`.

## Run

```bash
# Auto-detects encoding + output codec. Direct mode, 1440p proxy, batch=8.
dorea path/to/clip.mp4

# Explicit output path
dorea path/to/clip.mp4 --output path/to/graded.mov

# Verbose logging
dorea path/to/clip.mp4 --verbose
```

## Configuration

Create `dorea.toml` in the current working directory (or `~/.config/dorea/config.toml`):

```toml
[models]
python           = "/opt/dorea-venv/bin/python"
raune_weights    = "/path/to/RAUNENet/weights_95.pth"
raune_models_dir = "/path/to/sea_thru_poc"

[grade]
# raune_proxy_size = 1440    # long-edge pixels; values above 1440 may OOM on 6 GB VRAM
# direct_batch_size = 8      # frames per RAUNE forward pass; max 32
```

CLI flags always override config values. See `dorea --help` for the full list.

## Tunables

- `--raune-proxy-size N` — RAUNE input long-edge in pixels. Default **1440**. Lower = faster but noisier upscale; higher = better ΔE but more VRAM + compute. Delta upscale bench (1080p vs 1440p) showed ~8% ΔE improvement at 1440p with no upscale-stage cost.
- `--direct-batch-size N` — frames per RAUNE forward pass. Default **8**. fp16 activation memory is ~½ of fp32, so batch=8 fp16 fits in the same VRAM envelope as batch=4 fp32. Values above 8 show diminishing returns; `N > 16` regresses throughput.

## Breaking changes from previous `dorea`

This release removes the `calibrate`, `preview`, and `probe` subcommands and the entire 3D LUT / depth-zones / YOLO-seg / Maxine pipeline. `dorea grade` is gone; the new form is `dorea <input>` with `<input>` as a positional argument (no `--input` flag). Existing `dorea.toml` files will parse but fields for removed features (`[maxine]`, `[preview]`, `[inference]`, most of `[grade]`) are silently ignored.
