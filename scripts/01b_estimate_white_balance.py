"""Phase 1b: Per-Clip White Balance Estimation (~1-2 min per dive session)

Estimates white balance correction gains per clip from extracted keyframes using
a grey-world algorithm adapted for underwater footage. The per-clip gains are
applied to Node 2 ("Neutral Balance") in the Resolve grade to compensate for
varying dive conditions (depth, turbidity, ambient light).

Usage:
    python 01b_estimate_white_balance.py --date 2026-03-17
    python 01b_estimate_white_balance.py --date 2026-03-17 --verbose

Inputs:
    - working/keyframes/{date}/{clip_id}/ — extracted keyframes from Phase 1

Outputs:
    - working/white_balance/{date}/{clip_id}.json — per-clip WB correction gains
      Schema: {"clip_id": str, "white_balance": {"gain_r": float, "gain_g": float,
               "gain_b": float, "method": str, "confidence": str,
               "keyframes_analyzed": int, "mean_rgb_linear": [R, G, B]}}

Dependencies:
    - numpy, Pillow, colour-science
    - No GPU required (CPU only)

Architecture doc: Section 4.0 (colour pipeline)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from colour.models import eotf_sRGB
from PIL import Image

from pipeline_utils import (
    configure_logging,
    find_workspace_root,
    load_config,
    resolve_working_paths,
    validate_date,
)

logger = logging.getLogger(__name__)

# Pixel filtering thresholds
MIN_LUMINANCE = 0.05  # Exclude very dark pixels (noise-dominated)
MAX_LUMINANCE = 0.95  # Exclude clipped highlights
MAX_SATURATION = 0.90  # Exclude highly saturated pixels (skew neutral estimate)

# Gain safety bounds — prevent extreme corrections
GAIN_MIN = 0.5
GAIN_MAX = 2.5

# Minimum keyframes required for a reliable estimate
MIN_KEYFRAMES = 3


def load_frame_linear(path: Path) -> np.ndarray:
    """Load a JPEG keyframe and convert to linear-light float64 RGB.

    Returns array of shape (H, W, 3) in linear light, range [0, 1].
    """
    img = Image.open(path).convert("RGB")
    data = np.asarray(img, dtype=np.float64) / 255.0
    return eotf_sRGB(data)


def compute_luminance(linear_rgb: np.ndarray) -> np.ndarray:
    """Compute relative luminance from linear RGB using Rec. 709 coefficients."""
    return (0.2126 * linear_rgb[..., 0]
            + 0.7152 * linear_rgb[..., 1]
            + 0.0722 * linear_rgb[..., 2])


def compute_saturation(linear_rgb: np.ndarray) -> np.ndarray:
    """Compute saturation as (max - min) / max per pixel in linear space."""
    rgb_max = linear_rgb.max(axis=-1)
    rgb_min = linear_rgb.min(axis=-1)
    # Avoid division by zero for black pixels
    return np.where(rgb_max > 1e-10, (rgb_max - rgb_min) / rgb_max, 0.0)


def estimate_white_balance(keyframe_dir: Path) -> dict:
    """Estimate per-clip white balance gains from keyframes.

    Uses a filtered grey-world algorithm:
    1. Load all keyframes and convert to linear light
    2. Filter out pixels that would skew the estimate:
       - Very dark pixels (luminance < 5%) — noise-dominated
       - Clipped highlights (luminance > 95%) — saturated sensor
       - Highly chromatic pixels (saturation > 90%) — colored subjects, not illuminant
    3. Compute mean RGB of filtered pixels across all keyframes
    4. Derive per-channel gains to neutralize the color cast

    Returns dict with WB parameters, or None if insufficient data.
    """
    # Discover keyframe files
    frame_paths = sorted(
        f for f in keyframe_dir.iterdir()
        if f.is_file() and f.suffix == ".jpg"
    )

    if len(frame_paths) < MIN_KEYFRAMES:
        logger.warning(
            "  Only %d keyframe(s) found (need %d). Skipping WB estimation.",
            len(frame_paths), MIN_KEYFRAMES,
        )
        return None

    # Accumulate filtered pixel sums across all keyframes
    channel_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for frame_path in frame_paths:
        try:
            linear = load_frame_linear(frame_path)
        except Exception as e:
            logger.warning("  Skipping unreadable frame %s: %s", frame_path.name, e)
            continue

        # Flatten to (N, 3)
        pixels = linear.reshape(-1, 3)

        # Compute filtering masks
        lum = compute_luminance(linear).ravel()
        sat = compute_saturation(linear).ravel()

        # Keep pixels that pass all filters
        valid = (
            (lum >= MIN_LUMINANCE)
            & (lum <= MAX_LUMINANCE)
            & (sat <= MAX_SATURATION)
        )

        valid_pixels = pixels[valid]
        if len(valid_pixels) == 0:
            continue

        channel_sum += valid_pixels.sum(axis=0)
        pixel_count += len(valid_pixels)

    if pixel_count == 0:
        logger.warning("  No valid pixels after filtering. Skipping WB estimation.")
        return None

    # Mean RGB across all filtered pixels
    mean_rgb = channel_sum / pixel_count
    mean_luminance = 0.2126 * mean_rgb[0] + 0.7152 * mean_rgb[1] + 0.0722 * mean_rgb[2]

    if mean_luminance < 1e-6:
        logger.warning("  Mean luminance near zero. Skipping WB estimation.")
        return None

    # Grey-world correction: gains that would make mean RGB neutral
    # Target: each channel's contribution to luminance should match neutral balance
    # gain_ch = mean_luminance / mean_ch
    gains = np.array([
        mean_luminance / max(mean_rgb[0], 1e-10),
        mean_luminance / max(mean_rgb[1], 1e-10),
        mean_luminance / max(mean_rgb[2], 1e-10),
    ])

    # Normalize so the smallest gain (least correction needed) is 1.0
    # This preserves the strongest channel and boosts the weaker ones
    gains = gains / gains.min()

    # Clamp to safe range
    gains = np.clip(gains, GAIN_MIN, GAIN_MAX)

    # Assess confidence based on pixel coverage and gain magnitude
    valid_ratio = pixel_count / (len(frame_paths) * linear.shape[0] * linear.shape[1])
    max_gain = gains.max()

    if valid_ratio < 0.1:
        confidence = "low"
    elif max_gain > 2.0:
        confidence = "low"
    elif valid_ratio < 0.3 or max_gain > 1.5:
        confidence = "medium"
    else:
        confidence = "high"

    logger.info(
        "  WB gains (R,G,B): %.3f, %.3f, %.3f | confidence: %s | pixels: %d (%.0f%%)",
        gains[0], gains[1], gains[2], confidence, pixel_count, valid_ratio * 100,
    )

    return {
        "gain_r": round(float(gains[0]), 4),
        "gain_g": round(float(gains[1]), 4),
        "gain_b": round(float(gains[2]), 4),
        "method": "grey_world",
        "confidence": confidence,
        "keyframes_analyzed": len(frame_paths),
        "mean_rgb_linear": [round(float(v), 6) for v in mean_rgb],
    }


def discover_keyframe_dirs(keyframes_root: Path) -> list[Path]:
    """Find all clip keyframe directories under the keyframes root."""
    if not keyframes_root.is_dir():
        return []

    dirs = []
    for entry in sorted(keyframes_root.iterdir()):
        if entry.is_dir():
            jpg_files = [f for f in entry.iterdir() if f.suffix == ".jpg"]
            if jpg_files:
                dirs.append(entry)
    return dirs


def main():
    parser = argparse.ArgumentParser(
        description="Estimate per-clip white balance from keyframes"
    )
    parser.add_argument("--date", required=True, help="Dive date YYYY-MM-DD")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)
    validate_date(args.date)

    workspace_root = find_workspace_root()
    logger.info("Workspace root: %s", workspace_root)

    config = load_config(workspace_root)

    paths = resolve_working_paths(workspace_root, config, args.date)
    keyframes_root = paths["keyframes"]
    wb_output_dir = paths["white_balance"]

    # Check for keyframe directories
    clip_dirs = discover_keyframe_dirs(keyframes_root)
    if not clip_dirs:
        logger.error(
            "No keyframe directories found under %s. "
            "Run Phase 1 (01_extract_frames.py --date %s) first.",
            keyframes_root, args.date,
        )
        sys.exit(1)

    logger.info(
        "Found %d clip(s) with keyframes for date %s", len(clip_dirs), args.date
    )

    # Create output directory
    wb_output_dir.mkdir(parents=True, exist_ok=True)

    # Process each clip
    successful = 0
    skipped = 0

    for clip_dir in clip_dirs:
        clip_id = clip_dir.name
        logger.info("--- Clip: %s ---", clip_id)

        wb_params = estimate_white_balance(clip_dir)

        if wb_params is None:
            skipped += 1
            continue

        # Write output JSON
        output = {
            "clip_id": clip_id,
            "white_balance": wb_params,
        }
        output_path = wb_output_dir / f"{clip_id}.json"

        try:
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
            logger.info("  Wrote: %s", output_path)
            successful += 1
        except OSError as e:
            logger.error("  Failed to write %s: %s", output_path, e)
            skipped += 1

    # Summary
    logger.info("--- White balance estimation complete ---")
    logger.info(
        "Clips: %d estimated, %d skipped | Date: %s",
        successful, skipped, args.date,
    )

    if successful == 0 and skipped > 0:
        logger.warning("No clips produced WB estimates. Check keyframe quality.")


if __name__ == "__main__":
    main()
