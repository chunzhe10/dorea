# Dorea

Automated underwater video color grading pipeline. Named after the Dorado
constellation (the golden fish).

## v2 — Rust rewrite

Dorea v2 is a Rust workspace that ports the proven Python/NumPy algorithms from
the Sea-Thru POC into a fast, maintainable native binary.

### Architecture

```
crates/
├── dorea-color   — D-Log M, sRGB↔LAB, RGB↔HSV color math
├── dorea-lut     — Depth-stratified 33³ LUT build + trilinear apply
├── dorea-hsl     — 6-vector HSL qualifier derive + apply
├── dorea-cal     — .dorea-cal calibration file format (bincode)
├── dorea-video   — Phase 2: ffmpeg NVDEC/NVENC integration (placeholder)
└── dorea-cli     — `dorea` binary (calibrate / grade / preview)
```

### Pipeline (Phase 1)

```
keyframes + depth maps + RAUNE targets
         ↓
dorea calibrate   →  calibration.dorea-cal
         ↓
dorea grade       →  graded video  (Phase 3)
```

### Hardware requirements

- Linux workstation (devcontainer or bare metal)
- NVIDIA GPU with 6GB VRAM (RTX 3060 or better) — needed for RAUNE-Net inference (Phase 2)

### Camera support

- DJI Action 4 (D-Log M transfer function in `dorea-color::dlog_m`)
- Insta360 X5 (pre-flattened via Insta360 Studio)

## Quick start

```bash
# Build
cargo build --release

# Calibrate from keyframes
# NOTE: --targets expects RAUNE-Net output images (Phase 2, not yet implemented).
# For testing, you can pass the original keyframes directory as a placeholder,
# which produces an identity-like calibration useful for validating the pipeline.
# Phase 2 will add `dorea infer` to generate these automatically.
dorea calibrate \
  --keyframes /path/to/keyframes \
  --depth     /path/to/depth_maps \
  --targets   /path/to/raune_targets \
  --output    calibration.dorea-cal

# Grade (Phase 3 — not yet implemented)
dorea grade --input dive.mp4 --calibration calibration.dorea-cal --output graded.mp4
```

## Development

```bash
cargo test
cargo clippy -- -D warnings
```

## Key design decisions

- **σ=0** — no Gaussian smoothing on LUT cells (critical fix from POC)
- **NN fill** — empty LUT cells get nearest populated cell's value (brute-force L2 in index space)
- **Adaptive zone boundaries** — depth quantiles, not linspace
- **6-vector HSL qualifier** — matches DaVinci Resolve secondary color corrector

## Repository layout

```
Cargo.toml          Workspace manifest
crates/             Rust crates (see above)
luts/               Reference .cube LUT files (data, not generated)
references/         Reference still images for LUT generation
```

See `underwater_pipeline_architecture.docx` for the full architecture document.
