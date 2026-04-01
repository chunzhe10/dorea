"""Generate reference images from D-Log M keyframes for LUT creation.

Applies principled underwater color correction to raw D-Log M footage to create
"target look" reference images. These references are then fed into
00_generate_lut.py to derive a 33x33x33 3D LUT.

The correction is based on underwater light physics:
    - Red channel recovery (water absorbs red wavelengths first)
    - Contrast expansion (D-Log M is intentionally flat for DR)
    - Saturation boost (restore vibrancy lost in log encoding)
    - White balance shift (reduce blue/cyan cast from scattering)

Usage:
    python generate_references.py --input /path/to/keyframes --output /path/to/references

    # Using all keyframes from a dive date:
    python generate_references.py \
        --input ../../working/keyframes/2025-11-01 \
        --output references/look_v1

Dependencies:
    - numpy, Pillow, colour-science
    - No GPU required (CPU only)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from colour.models import eotf_inverse_sRGB

from pipeline_utils import (
    IMAGE_EXTENSIONS,
    configure_logging,
    find_workspace_root,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# D-Log M transfer function (copied from 00_generate_lut.py for independence)
# ---------------------------------------------------------------------------

def dlog_m_to_linear(x: np.ndarray) -> np.ndarray:
    """Convert D-Log M encoded values to scene-linear light.

    D-Log M is DJI's log gamma curve. Has a linear segment in deep shadows
    and a logarithmic curve above.
    """
    x = np.asarray(x, dtype=np.float64)

    a = 0.9892
    b = 0.0108
    c = 0.256663
    d = 0.584555
    cut_encoded = 0.14

    cut_linear = (10.0 ** ((cut_encoded - d) / c) - b) / a
    slope = c * a / ((a * cut_linear + b) * np.log(10.0))
    intercept = cut_encoded - slope * cut_linear

    linear = np.where(
        x <= cut_encoded,
        (x - intercept) / slope,
        (np.power(10.0, (x - d) / c) - b) / a,
    )

    return np.clip(linear, 0.0, None)


# ---------------------------------------------------------------------------
# Underwater colour correction
# ---------------------------------------------------------------------------

# Default correction parameters based on underwater photography color science.
# These represent a "natural underwater" look for 3-5m depth tropical water.
DEFAULT_PARAMS = {
    "red_gain": 2.0,        # Compensate for water absorption of red (aggressive for 3-5m depth)
    "green_gain": 1.0,      # Neutral green
    "blue_gain": 0.75,      # Reduce cyan/blue cast from scattering
    "gamma": 0.55,          # Expand D-Log M flat contrast (< 1 = more contrast)
    "saturation": 1.3,      # Restore vibrancy lost in log encoding
    "shadow_lift": 0.005,   # Prevent shadow crushing
}


def correct_image(
    dlog_m_data: np.ndarray,
    red_gain: float = 1.35,
    green_gain: float = 1.05,
    blue_gain: float = 0.95,
    gamma: float = 0.50,
    saturation: float = 1.4,
    shadow_lift: float = 0.01,
) -> np.ndarray:
    """Apply underwater colour correction to a D-Log M image.

    Processing chain:
        1. D-Log M decode → linear light
        2. White balance (per-channel gain)
        3. Shadow lift (prevent crushing)
        4. Gamma correction (contrast expansion)
        5. Saturation boost
        6. Linear → sRGB encode
        7. Clip to [0, 1]

    Args:
        dlog_m_data: Image array (H, W, 3), values in [0, 1] representing
                     D-Log M encoded pixel values.
        red_gain: Red channel multiplier (>1 recovers red absorbed by water).
        green_gain: Green channel multiplier.
        blue_gain: Blue channel multiplier (<1 reduces cyan cast).
        gamma: Power curve exponent (<1 expands contrast, >1 compresses).
        saturation: Saturation multiplier (>1 increases vibrancy).
        shadow_lift: Additive offset for shadows (prevents crushing).

    Returns:
        Corrected image in sRGB encoding, shape (H, W, 3), range [0, 1].
    """
    # Step 1: Decode D-Log M to scene-linear light
    linear = dlog_m_to_linear(dlog_m_data)

    # Step 2: White balance — per-channel gain
    gains = np.array([red_gain, green_gain, blue_gain])
    corrected = linear * gains

    # Step 3: Shadow lift
    corrected = corrected + shadow_lift

    # Step 4: Gamma correction (contrast expansion)
    corrected = np.clip(corrected, 0.0, None)
    corrected = np.power(corrected + 1e-10, gamma) - np.power(1e-10, gamma)

    # Step 5: Saturation adjustment
    lum = (0.2126 * corrected[..., 0:1]
           + 0.7152 * corrected[..., 1:2]
           + 0.0722 * corrected[..., 2:3])
    corrected = lum + (corrected - lum) * saturation

    # Step 6: Encode linear → sRGB
    corrected = np.clip(corrected, 0.0, 1.0)
    srgb = eotf_inverse_sRGB(corrected)

    # Step 7: Final clip
    return np.clip(srgb, 0.0, 1.0)


def discover_keyframes(input_dir: Path) -> list[Path]:
    """Find all JPEG/PNG keyframe images, recursing into subdirectories."""
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(input_dir.rglob(f"*{ext}"))
        # Also check uppercase
        images.extend(input_dir.rglob(f"*{ext.upper()}"))
    # Deduplicate (rglob may find same file via case variants on case-insensitive fs)
    seen = set()
    unique = []
    for p in sorted(images):
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def process_image(input_path: Path, output_path: Path, params: dict) -> None:
    """Load a D-Log M keyframe, apply correction, save as sRGB JPEG."""
    img = Image.open(input_path).convert("RGB")
    # Normalise to 0-1 (these are D-Log M encoded, NOT sRGB)
    data = np.asarray(img, dtype=np.float64) / 255.0

    corrected = correct_image(data, **params)

    # Convert to uint8 and save
    out_uint8 = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)
    out_img = Image.fromarray(out_uint8, "RGB")
    out_img.save(output_path, "JPEG", quality=95)


def main():
    parser = argparse.ArgumentParser(
        description="Generate reference images from D-Log M keyframes"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing D-Log M keyframes (searched recursively)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for corrected reference images",
    )
    parser.add_argument(
        "--red-gain", type=float, default=DEFAULT_PARAMS["red_gain"],
        help=f"Red channel gain (default: {DEFAULT_PARAMS['red_gain']})",
    )
    parser.add_argument(
        "--green-gain", type=float, default=DEFAULT_PARAMS["green_gain"],
        help=f"Green channel gain (default: {DEFAULT_PARAMS['green_gain']})",
    )
    parser.add_argument(
        "--blue-gain", type=float, default=DEFAULT_PARAMS["blue_gain"],
        help=f"Blue channel gain (default: {DEFAULT_PARAMS['blue_gain']})",
    )
    parser.add_argument(
        "--gamma", type=float, default=DEFAULT_PARAMS["gamma"],
        help=f"Gamma correction (default: {DEFAULT_PARAMS['gamma']})",
    )
    parser.add_argument(
        "--saturation", type=float, default=DEFAULT_PARAMS["saturation"],
        help=f"Saturation multiplier (default: {DEFAULT_PARAMS['saturation']})",
    )
    parser.add_argument(
        "--shadow-lift", type=float, default=DEFAULT_PARAMS["shadow_lift"],
        help=f"Shadow lift offset (default: {DEFAULT_PARAMS['shadow_lift']})",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)

    workspace_root = find_workspace_root()
    repo_root = workspace_root / "repos" / "dorea"

    # Resolve paths relative to repo root if not absolute
    input_dir = Path(args.input)
    if not input_dir.is_absolute():
        input_dir = repo_root / input_dir
    input_dir = input_dir.resolve()

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir = output_dir.resolve()

    if not input_dir.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    # Discover keyframes
    keyframes = discover_keyframes(input_dir)
    if not keyframes:
        logger.error("No image files found in %s", input_dir)
        sys.exit(1)

    logger.info("Found %d keyframe(s) in %s", len(keyframes), input_dir)

    # Prepare output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect correction parameters
    params = {
        "red_gain": args.red_gain,
        "green_gain": args.green_gain,
        "blue_gain": args.blue_gain,
        "gamma": args.gamma,
        "saturation": args.saturation,
        "shadow_lift": args.shadow_lift,
    }

    logger.info("Correction parameters:")
    for k, v in params.items():
        logger.info("  %s: %.3f", k, v)

    # Process each keyframe
    processed = 0
    for kf in keyframes:
        out_name = f"ref_{kf.stem}.jpg"
        out_path = output_dir / out_name
        logger.info("Processing: %s → %s", kf.name, out_name)
        process_image(kf, out_path, params)
        processed += 1

    logger.info("--- Reference generation complete ---")
    logger.info("  Input frames: %d", len(keyframes))
    logger.info("  Output references: %d", processed)
    logger.info("  Output directory: %s", output_dir)
    logger.info("")
    logger.info("Next step: run LUT generation:")
    logger.info(
        "  python 00_generate_lut.py --references %s --output luts/underwater_base.cube",
        output_dir.relative_to(repo_root) if str(output_dir).startswith(str(repo_root)) else output_dir,
    )


if __name__ == "__main__":
    main()
