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
import logging
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# colour-science: used for sRGB EOTF conversions and LUT3D file writing
from colour import LUT3D
from colour.io import write_LUT_IridasCube
from colour.models import eotf_sRGB, eotf_inverse_sRGB

# Supported reference image extensions (case-insensitive matching)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# LUT grid resolution — 33x33x33 is the industry standard for .cube files
LUT_SIZE = 33

# Luminance zone boundaries (in linear light, 0.0-1.0)
SHADOW_UPPER = 0.2
MIDTONE_UPPER = 0.7
# Highlights: 0.7 - 1.0

# D-Log M transfer function constants
# Middle grey (18% reflectance) maps to ~0.39 in D-Log M
DLOG_M_MID_GREY = 0.39

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# D-Log M transfer function
# ---------------------------------------------------------------------------

def dlog_m_to_linear(x: np.ndarray) -> np.ndarray:
    """Convert D-Log M encoded values to scene-linear light.

    D-Log M is DJI's log gamma curve used in Action 4 and other cameras.
    It has a linear segment in deep shadows and a logarithmic curve above.

    The transfer function is an approximation based on published D-Log M
    characteristics:
        - Middle grey (18%) maps to ~0.39 in D-Log M
        - Very flat contrast curve designed to maximise dynamic range

    The encoding function (linear -> D-Log M) is:
        For linear >= cut_linear:  encoded = c * log10(a * linear + b) + d
        For linear < cut_linear:   encoded = slope * linear + intercept

    This function is the inverse (D-Log M -> linear).
    """
    x = np.asarray(x, dtype=np.float64)

    # D-Log M curve parameters (DJI published curve)
    a = 0.9892
    b = 0.0108
    c = 0.256663
    d = 0.584555

    # Cut point in encoded space — below this, the encoding is linear
    # The linear segment ensures continuity at the boundary.
    # Compute the cut point values for proper continuity:
    cut_encoded = 0.14

    # Log segment decode: linear = (10^((encoded - d) / c) - b) / a
    cut_linear = (10.0 ** ((cut_encoded - d) / c) - b) / a

    # Slope of the encoding function at the cut point (derivative of log segment):
    # d/d(linear) [c * log10(a*linear + b) + d] = c * a / ((a*linear + b) * ln(10))
    slope = c * a / ((a * cut_linear + b) * np.log(10.0))

    # Linear segment encoding: encoded = slope * linear + intercept
    # At cut: cut_encoded = slope * cut_linear + intercept
    intercept = cut_encoded - slope * cut_linear

    linear = np.where(
        x <= cut_encoded,
        # Linear segment inverse: linear = (encoded - intercept) / slope
        (x - intercept) / slope,
        # Log segment inverse: linear = (10^((encoded - d) / c) - b) / a
        (np.power(10.0, (x - d) / c) - b) / a,
    )

    return np.clip(linear, 0.0, None)


def linear_to_dlog_m(x: np.ndarray) -> np.ndarray:
    """Convert scene-linear light values to D-Log M encoding.

    Inverse of dlog_m_to_linear(). Used for verification.
    """
    x = np.asarray(x, dtype=np.float64)

    a = 0.9892
    b = 0.0108
    c = 0.256663
    d = 0.584555
    cut_encoded = 0.14

    # Compute the same cut/slope/intercept as the decode function
    cut_linear = (10.0 ** ((cut_encoded - d) / c) - b) / a
    slope = c * a / ((a * cut_linear + b) * np.log(10.0))
    intercept = cut_encoded - slope * cut_linear

    encoded = np.where(
        x <= cut_linear,
        slope * x + intercept,
        c * np.log10(a * x + b) + d,
    )

    return np.clip(encoded, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Image loading and analysis
# ---------------------------------------------------------------------------

def find_workspace_root() -> Path:
    """Resolve the workspace root directory.

    Uses $CORVIA_WORKSPACE if set, otherwise walks up from the script location
    to find the directory containing repos/dorea.
    """
    env_root = os.environ.get("CORVIA_WORKSPACE")
    if env_root:
        return Path(env_root)

    # Walk up from script location: scripts/ -> dorea/ -> repos/ -> workspace root
    candidate = Path(__file__).resolve().parent.parent.parent.parent
    if (candidate / "repos" / "dorea").is_dir():
        return candidate

    # Fallback: hardcoded devcontainer path
    return Path("/workspaces/dorea-workspace")


def discover_images(references_dir: Path) -> list[Path]:
    """Find all valid image files in the references directory."""
    if not references_dir.is_dir():
        return []

    images = []
    for entry in sorted(references_dir.iterdir()):
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(entry)
    return images


def load_image_linear(path: Path) -> np.ndarray:
    """Load an image and convert to linear-light float64 RGB array.

    Assumes input images are sRGB-encoded (standard for JPEG/PNG from cameras
    and editing software). Converts via the sRGB EOTF to linear light for
    accurate colour analysis.

    Returns:
        numpy array of shape (H, W, 3) in linear light, range [0, 1].
    """
    img = Image.open(path).convert("RGB")
    # Normalise to 0-1 float
    data = np.asarray(img, dtype=np.float64) / 255.0
    # Convert sRGB encoding to linear light
    linear = eotf_sRGB(data)
    return linear


def compute_luminance(linear_rgb: np.ndarray) -> np.ndarray:
    """Compute relative luminance from linear RGB using Rec. 709 coefficients."""
    return 0.2126 * linear_rgb[..., 0] + 0.7152 * linear_rgb[..., 1] + 0.0722 * linear_rgb[..., 2]


def analyse_images(image_paths: list[Path]) -> dict:
    """Analyse colour characteristics across all reference images.

    Computes per-channel statistics in shadow, midtone, and highlight zones,
    plus overall hue and saturation characteristics.

    Returns:
        Dictionary with keys:
            - shadows_mean_rgb: mean (R, G, B) in linear light for shadow zone
            - midtones_mean_rgb: mean (R, G, B) for midtone zone
            - highlights_mean_rgb: mean (R, G, B) for highlight zone
            - overall_mean_rgb: mean (R, G, B) across all pixels
            - saturation_mean: mean saturation (0-1) across all pixels
            - hue_histogram: 36-bin hue distribution (each bin = 10 degrees)
    """
    # Accumulators for zone statistics
    shadow_pixels = []
    midtone_pixels = []
    highlight_pixels = []
    all_pixels = []

    # Hue and saturation accumulators
    hue_bins = np.zeros(36, dtype=np.float64)
    saturation_sum = 0.0
    saturation_count = 0

    for path in image_paths:
        logger.debug("  Loading: %s", path.name)
        linear = load_image_linear(path)

        # Reshape to (N, 3)
        pixels = linear.reshape(-1, 3)
        all_pixels.append(pixels)

        # Compute luminance for zone classification
        lum = compute_luminance(linear).ravel()

        # Classify pixels into zones
        shadow_mask = lum < SHADOW_UPPER
        midtone_mask = (lum >= SHADOW_UPPER) & (lum < MIDTONE_UPPER)
        highlight_mask = lum >= MIDTONE_UPPER

        if shadow_mask.any():
            shadow_pixels.append(pixels[shadow_mask])
        if midtone_mask.any():
            midtone_pixels.append(pixels[midtone_mask])
        if highlight_mask.any():
            highlight_pixels.append(pixels[highlight_mask])

        # Convert to sRGB for hue/saturation analysis (HSV works in display space)
        srgb = eotf_inverse_sRGB(linear)
        srgb_uint8 = np.clip(srgb * 255, 0, 255).astype(np.uint8)
        hsv_img = np.array(Image.fromarray(srgb_uint8, "RGB").convert("HSV"))
        hsv_flat = hsv_img.reshape(-1, 3)

        # Hue histogram (Pillow HSV: H is 0-255 mapping to 0-360 degrees)
        hues_deg = hsv_flat[:, 0].astype(np.float64) * 360.0 / 255.0
        bin_indices = np.clip((hues_deg / 10.0).astype(int), 0, 35)
        hue_bins += np.bincount(bin_indices, minlength=36).astype(np.float64)

        # Saturation (Pillow HSV: S is 0-255)
        sat = hsv_flat[:, 1].astype(np.float64) / 255.0
        saturation_sum += sat.sum()
        saturation_count += len(sat)

    # Compute zone means
    def safe_mean(pixel_list: list[np.ndarray]) -> np.ndarray:
        if not pixel_list:
            return np.array([0.0, 0.0, 0.0])
        combined = np.concatenate(pixel_list, axis=0)
        return combined.mean(axis=0)

    result = {
        "shadows_mean_rgb": safe_mean(shadow_pixels),
        "midtones_mean_rgb": safe_mean(midtone_pixels),
        "highlights_mean_rgb": safe_mean(highlight_pixels),
        "overall_mean_rgb": safe_mean(all_pixels),
        "saturation_mean": saturation_sum / max(saturation_count, 1),
        "hue_histogram": hue_bins / max(hue_bins.sum(), 1),
    }

    return result


# ---------------------------------------------------------------------------
# Colour correction and LUT generation
# ---------------------------------------------------------------------------

def compute_correction_params(analysis: dict) -> dict:
    """Derive lift/gamma/gain correction parameters from reference analysis.

    Builds a per-channel colour correction that maps D-Log M neutral values
    towards the reference underwater look.

    The correction model uses:
        - Lift: offset applied to shadows (additive)
        - Gamma: power curve applied to midtones (exponent)
        - Gain: multiplier applied to highlights (multiplicative)

    Additionally applies underwater-specific corrections:
        - Red channel recovery (red is absorbed first underwater)
        - Blue/green balance adjustment
    """
    shadows = analysis["shadows_mean_rgb"]
    midtones = analysis["midtones_mean_rgb"]
    highlights = analysis["highlights_mean_rgb"]
    overall = analysis["overall_mean_rgb"]
    saturation = analysis["saturation_mean"]

    # --- Compute D-Log M decoded reference points ---
    # D-Log M middle grey sits at ~0.39 encoded, which decodes to ~0.18 linear
    # We compute what each zone's mean colour looks like as a target.

    # For a neutral D-Log M image, all channels would decode similarly.
    # The reference images tell us what the target colour balance should be.

    # Normalise zone means relative to overall brightness
    overall_lum = compute_luminance(overall.reshape(1, 1, 3)).item()
    if overall_lum < 1e-6:
        overall_lum = 0.18  # fallback to middle grey

    # --- Lift (shadow correction) ---
    # Lift shifts the black point. Underwater shadows tend to be blue/green.
    # Small offset to push shadows toward the reference look.
    shadow_lum = compute_luminance(shadows.reshape(1, 1, 3)).item()
    if shadow_lum < 1e-6:
        shadow_lum = 0.01
    lift = shadows / max(shadow_lum, 0.01) * 0.02  # Small lift per channel
    lift = np.clip(lift, -0.05, 0.05)

    # --- Gain (highlight correction) ---
    # Gain scales the white point. We want highlights to match the reference.
    highlight_lum = compute_luminance(highlights.reshape(1, 1, 3)).item()
    if highlight_lum < 1e-6:
        highlight_lum = 0.8
    # Compute per-channel gain relative to luminance
    gain = highlights / highlight_lum
    # Normalise so the brightest channel has gain ~1.0
    gain = gain / max(gain.max(), 1e-6)
    # Ensure gain stays in reasonable range
    gain = np.clip(gain, 0.5, 1.5)

    # --- Gamma (midtone correction) ---
    # Gamma adjusts the midtone response. We need to expand D-Log M's flat
    # contrast into the reference look's richer tones.
    midtone_lum = compute_luminance(midtones.reshape(1, 1, 3)).item()
    if midtone_lum < 1e-6:
        midtone_lum = 0.18

    # Per-channel gamma: ratio of midtone channel to midtone luminance
    # tells us the colour balance in midtones
    mid_balance = midtones / max(midtone_lum, 1e-6)
    mid_balance = mid_balance / max(mid_balance.max(), 1e-6)

    # D-Log M needs an S-curve expansion. The base gamma controls contrast.
    # A gamma < 1 expands contrast (makes midtones brighter), > 1 compresses.
    # Typical D-Log M to Rec.709 needs gamma ~0.45-0.55 for good contrast.
    base_gamma = 0.50

    # Adjust per-channel gamma based on midtone colour balance
    gamma = np.full(3, base_gamma) * (1.0 / np.clip(mid_balance, 0.5, 2.0))
    gamma = np.clip(gamma, 0.3, 0.8)

    # --- Underwater-specific: red channel recovery ---
    # Water absorbs red light first, so underwater images are blue/green biased.
    # We boost the red channel to compensate, scaled by how much red is missing.
    red_ratio = overall[0] / max(overall.mean(), 1e-6)
    red_deficit = max(0.0, 1.0 - red_ratio)

    # Apply graduated red recovery
    red_boost = 1.0 + red_deficit * 0.3  # Up to 30% boost
    gain[0] *= red_boost

    # --- Underwater-specific: blue/green balance ---
    # Underwater images are often overly cyan. Slight green reduction if needed.
    green_excess = overall[1] / max(overall.mean(), 1e-6) - 1.0
    if green_excess > 0.05:
        gain[1] *= (1.0 - green_excess * 0.15)

    # --- Saturation ---
    # Reference saturation guides how vibrant the output should be.
    # A multiplier > 1 increases saturation, < 1 decreases.
    # Underwater images often need a saturation boost.
    sat_target = np.clip(saturation * 1.3, 0.3, 0.9)

    # Re-clip gain to safe bounds
    gain = np.clip(gain, 0.5, 1.8)

    params = {
        "lift": lift,
        "gamma": gamma,
        "gain": gain,
        "saturation_factor": sat_target,
    }

    logger.info("Correction parameters:")
    logger.info("  Lift  (R,G,B): %.4f, %.4f, %.4f", *lift)
    logger.info("  Gamma (R,G,B): %.4f, %.4f, %.4f", *gamma)
    logger.info("  Gain  (R,G,B): %.4f, %.4f, %.4f", *gain)
    logger.info("  Saturation factor: %.3f", sat_target)
    logger.info("  Red deficit: %.3f (boost: %.2fx)", red_deficit, red_boost)

    return params


def apply_correction(rgb: np.ndarray, params: dict) -> np.ndarray:
    """Apply lift/gamma/gain colour correction to linear RGB values.

    The processing chain is:
        1. D-Log M decode (input is in D-Log M space)
        2. Apply lift (shadow offset)
        3. Apply gain (highlight scaling)
        4. Apply gamma (midtone contrast curve)
        5. Saturation adjustment
        6. Encode back to output space (clipped to 0-1)

    Args:
        rgb: Array of shape (..., 3) with D-Log M encoded values (0-1).
        params: Correction parameters from compute_correction_params().

    Returns:
        Corrected RGB values in output space, clipped to [0, 1].
    """
    lift = params["lift"]
    gamma = params["gamma"]
    gain = params["gain"]
    sat_factor = params["saturation_factor"]

    # Step 1: Decode D-Log M to linear light
    linear = dlog_m_to_linear(rgb)

    # Step 2: Apply lift (shadow offset)
    corrected = linear + lift

    # Step 3: Apply gain (highlight scaling)
    corrected = corrected * gain

    # Clamp to avoid negative values before gamma
    corrected = np.clip(corrected, 0.0, None)

    # Step 4: Apply gamma (midtone contrast — power curve)
    # Safe power: avoid issues with zero values
    corrected = np.power(corrected + 1e-10, gamma) - np.power(1e-10, gamma)

    # Step 5: Saturation adjustment
    # Compute luminance and adjust saturation relative to it
    lum = (0.2126 * corrected[..., 0:1]
           + 0.7152 * corrected[..., 1:2]
           + 0.0722 * corrected[..., 2:3])
    corrected = lum + (corrected - lum) * sat_factor

    # Step 6: Clip to valid range
    corrected = np.clip(corrected, 0.0, 1.0)

    return corrected


def generate_lut(params: dict) -> LUT3D:
    """Generate a 33x33x33 3D LUT applying the colour correction.

    The LUT maps D-Log M input values to corrected output values.
    Each grid point in the 33^3 cube is processed through the correction chain.

    Returns:
        colour.LUT3D instance ready for writing to .cube file.
    """
    logger.info("Generating %dx%dx%d 3D LUT...", LUT_SIZE, LUT_SIZE, LUT_SIZE)

    # Create a unity LUT — colour.LUT3D expects table[R, G, B, channel]
    table = np.zeros((LUT_SIZE, LUT_SIZE, LUT_SIZE, 3), dtype=np.float64)

    # Generate the grid of input values
    steps = np.linspace(0.0, 1.0, LUT_SIZE)

    # Fill the grid with input D-Log M values
    # colour.LUT3D indexing: table[R_idx, G_idx, B_idx, channel]
    # .cube output: R changes fastest (handled by colour.io writer)
    for ri, r_val in enumerate(steps):
        for gi, g_val in enumerate(steps):
            table[ri, gi, :, 0] = r_val        # R value (constant for this slice)
            table[ri, gi, :, 1] = g_val        # G value (constant for this row)
            table[ri, gi, :, 2] = steps         # B values

    # Apply correction to all grid points at once (vectorised)
    original_shape = table.shape
    flat_rgb = table.reshape(-1, 3)

    logger.info("Applying colour correction to %d grid points...", len(flat_rgb))
    corrected = apply_correction(flat_rgb, params)

    table = corrected.reshape(original_shape)

    lut = LUT3D(
        table=table,
        name="Dorea Underwater Base",
        comments=[
            "Generated by dorea pipeline",
            "D-Log M to underwater reference look",
        ],
    )

    return lut


# ---------------------------------------------------------------------------
# .cube file output
# ---------------------------------------------------------------------------

def write_cube_file(lut: LUT3D, output_path: Path, look_name: str) -> None:
    """Write a 3D LUT to an Adobe/Iridas .cube file.

    Uses colour-science's built-in writer for correct formatting,
    then prepends our custom header comments.
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Update the LUT name to include the look name
    lut.name = look_name

    # Add header comments
    lut.comments = [
        "Generated by dorea pipeline",
        f"Reference look: {look_name}",
    ]

    # Write using colour-science's Iridas .cube writer
    write_LUT_IridasCube(lut, str(output_path), decimals=6)

    file_size = output_path.stat().st_size
    logger.info("LUT written: %s (%.1f KB)", output_path, file_size / 1024)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate underwater reference LUT from images"
    )
    parser.add_argument(
        "--references",
        required=True,
        help="Directory of reference images (JPEG, PNG, TIFF)",
    )
    parser.add_argument(
        "--output",
        default="luts/underwater_base.cube",
        help="Output .cube file path (default: luts/underwater_base.cube)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Configure logging — same style as other pipeline scripts
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    workspace_root = find_workspace_root()
    repo_root = workspace_root / "repos" / "dorea"
    logger.info("Workspace root: %s", workspace_root)

    # --- Resolve references directory ---
    references_dir = Path(args.references)
    if not references_dir.is_absolute():
        references_dir = repo_root / references_dir
    references_dir = references_dir.resolve()

    if not references_dir.is_dir():
        logger.error("References directory does not exist: %s", references_dir)
        sys.exit(1)

    # --- Resolve output path ---
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path = output_path.resolve()

    # --- Discover reference images ---
    image_paths = discover_images(references_dir)

    if not image_paths:
        logger.error(
            "No valid image files found in %s. "
            "Supported formats: %s",
            references_dir, ", ".join(sorted(IMAGE_EXTENSIONS)),
        )
        sys.exit(1)

    logger.info("Found %d reference image(s) in %s", len(image_paths), references_dir)
    for p in image_paths:
        logger.debug("  %s", p.name)

    # --- Analyse reference images ---
    logger.info("Analysing colour characteristics...")
    analysis = analyse_images(image_paths)

    logger.info("Reference colour analysis:")
    logger.info(
        "  Shadows  (R,G,B): %.4f, %.4f, %.4f",
        *analysis["shadows_mean_rgb"],
    )
    logger.info(
        "  Midtones (R,G,B): %.4f, %.4f, %.4f",
        *analysis["midtones_mean_rgb"],
    )
    logger.info(
        "  Highlights (R,G,B): %.4f, %.4f, %.4f",
        *analysis["highlights_mean_rgb"],
    )
    logger.info(
        "  Overall  (R,G,B): %.4f, %.4f, %.4f",
        *analysis["overall_mean_rgb"],
    )
    logger.info("  Mean saturation: %.3f", analysis["saturation_mean"])

    # Log dominant hue
    hue_hist = analysis["hue_histogram"]
    dominant_bin = int(np.argmax(hue_hist))
    dominant_hue_deg = dominant_bin * 10
    logger.info(
        "  Dominant hue: ~%d degrees (%.1f%% of pixels)",
        dominant_hue_deg, hue_hist[dominant_bin] * 100,
    )

    # --- Compute correction parameters ---
    logger.info("Computing colour correction parameters...")
    params = compute_correction_params(analysis)

    # --- Generate LUT ---
    lut = generate_lut(params)

    # --- Derive look name from references directory ---
    look_name = references_dir.name
    if look_name in (".", ""):
        look_name = "underwater_base"

    # --- Write output ---
    write_cube_file(lut, output_path, look_name)

    # --- Summary ---
    logger.info("--- LUT generation complete ---")
    logger.info("  Reference images: %d", len(image_paths))
    logger.info("  LUT size: %dx%dx%d", LUT_SIZE, LUT_SIZE, LUT_SIZE)
    logger.info("  Output: %s", output_path)


if __name__ == "__main__":
    main()
