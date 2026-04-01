"""LUT Preview: Apply .cube LUT to keyframes and generate side-by-side comparisons.

Enables fast visual evaluation of colour correction without opening Resolve.
Applies the generated .cube LUT to Phase 1 keyframes and outputs side-by-side
comparisons (RAW | GRADED) for A/B testing.

Usage:
    python preview_lut.py --date 2026-03-17
    python preview_lut.py --date 2026-03-17 --clip DJI_0042
    python preview_lut.py --date 2026-03-17 --lut luts/custom_look.cube
    python preview_lut.py --date 2026-03-17 --force  # regenerate existing

Inputs:
    - working/keyframes/{date}/{clip_id}/frame_NNNNNN.jpg (from Phase 1)
    - LUT path from config.yaml resolve_lut_path (or custom --lut path)

Outputs:
    - working/previews/{date}/{clip_id}/graded/frame_NNNNNN.jpg
    - working/previews/{date}/{clip_id}/comparison/frame_NNNNNN.jpg

Dependencies:
    - ffmpeg (system package, CPU only)

Note:
    Video loop generation around keyframes (for motion evaluation) is planned
    for a follow-up issue.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from pipeline_utils import (
    configure_logging,
    find_workspace_root,
    get_progress_bar,
    load_config,
    resolve_working_paths,
    validate_date,
)

logger = logging.getLogger(__name__)


def _escape_ffmpeg_filter_value(path: str) -> str:
    """Escape special characters for ffmpeg filter option values.

    ffmpeg's filter parser treats \\, ', :, ;, [, ] as delimiters.
    These must be backslash-escaped when used in filter option values.
    """
    for ch in ("\\", "'", ":", ";", "[", "]"):
        path = path.replace(ch, f"\\{ch}")
    return path


def discover_keyframes(keyframes_dir: Path, clip_filter: str | None = None) -> dict[str, list[Path]]:
    """Discover extracted keyframes grouped by clip_id.

    Args:
        keyframes_dir: Path to working/keyframes/{date}/
        clip_filter: If set, only return frames for this clip_id.

    Returns:
        Dict mapping clip_id to sorted list of frame paths.
    """
    if not keyframes_dir.is_dir():
        return {}

    clips = {}
    for clip_dir in sorted(keyframes_dir.iterdir()):
        if not clip_dir.is_dir():
            continue
        if clip_filter and clip_dir.name != clip_filter:
            continue

        frames = sorted(
            f for f in clip_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if frames:
            clips[clip_dir.name] = frames

    return clips


def apply_lut(frame_path: Path, output_path: Path, lut_path: Path) -> bool:
    """Apply a .cube LUT to a single frame using ffmpeg.

    Converts to RGB before LUT application for correct colour processing,
    matching how DaVinci Resolve applies the same LUT.

    Returns True on success, False on failure.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    escaped_lut = _escape_ffmpeg_filter_value(str(lut_path))
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(frame_path),
        "-vf", f"format=rgb24,lut3d=file='{escaped_lut}'",
        "-q:v", "2",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("LUT application failed for %s: %s", frame_path.name, e.stderr.strip())
        return False


def generate_comparison(raw_path: Path, graded_path: Path, output_path: Path) -> bool:
    """Generate a side-by-side comparison image (RAW | GRADED) with labels.

    Uses ffmpeg hstack filter with drawtext overlay for labels.
    Falls back to unlabeled hstack if drawtext fails (e.g. missing fonts).
    Returns True on success, False on failure.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try labeled comparison first (drawtext requires fontconfig).
    filter_labeled = (
        "[0]drawtext=text='RAW':font='monospace':"
        "x=10:y=10:fontsize=28:fontcolor=white:"
        "borderw=2:bordercolor=black[left];"
        "[1]drawtext=text='GRADED':font='monospace':"
        "x=10:y=10:fontsize=28:fontcolor=white:"
        "borderw=2:bordercolor=black[right];"
        "[left][right]hstack=inputs=2"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(raw_path),
        "-i", str(graded_path),
        "-filter_complex", filter_labeled,
        "-q:v", "2",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError:
        pass

    # Fallback: unlabeled hstack (no font dependency)
    logger.debug("drawtext failed, falling back to unlabeled comparison for %s", raw_path.name)
    filter_plain = "[0][1]hstack=inputs=2"

    cmd_fallback = [
        "ffmpeg",
        "-y",
        "-i", str(raw_path),
        "-i", str(graded_path),
        "-filter_complex", filter_plain,
        "-q:v", "2",
        str(output_path),
    ]

    try:
        subprocess.run(cmd_fallback, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Comparison failed for %s: %s", raw_path.name, e.stderr.strip())
        return False


def resolve_lut_path(workspace_root: Path, config: dict, lut_arg: str | None) -> Path:
    """Resolve the LUT .cube file path.

    Priority: --lut argument > config resolve_lut_path > default.
    """
    if lut_arg:
        lut_path = Path(lut_arg)
        if not lut_path.is_absolute():
            lut_path = workspace_root / "repos" / "dorea" / lut_path
        return lut_path.resolve()

    config_lut = config.get("resolve_lut_path", "repos/dorea/luts/underwater_base.cube")
    lut_path = workspace_root / config_lut
    return lut_path.resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Apply LUT to keyframes and generate side-by-side comparisons"
    )
    parser.add_argument("--date", required=True, help="Dive date YYYY-MM-DD")
    parser.add_argument(
        "--clip", default=None,
        help="Process only this clip_id (default: all clips for the date)",
    )
    parser.add_argument(
        "--lut", default=None,
        help="Path to .cube LUT file (relative to repos/dorea/, default: from config.yaml)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate existing previews (default: skip if output exists)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging",
    )
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)
    validate_date(args.date)

    workspace_root = find_workspace_root()
    logger.info("Workspace root: %s", workspace_root)

    config = load_config(workspace_root)
    paths = resolve_working_paths(workspace_root, config, args.date)

    # Resolve LUT path
    lut_path = resolve_lut_path(workspace_root, config, args.lut)
    if not lut_path.is_file():
        logger.error("LUT file not found: %s", lut_path)
        logger.error(
            "Run 00_generate_lut.py first, or specify --lut with a valid .cube file."
        )
        sys.exit(1)
    logger.info("Using LUT: %s", lut_path)

    # Discover keyframes
    keyframes_dir = paths["keyframes"]
    clips = discover_keyframes(keyframes_dir, clip_filter=args.clip)

    if not clips:
        if args.clip:
            logger.error(
                "No keyframes found for clip '%s' in %s",
                args.clip, keyframes_dir,
            )
        else:
            logger.error(
                "No keyframes found in %s. Run 01_extract_frames.py --date %s first.",
                keyframes_dir, args.date,
            )
        sys.exit(1)

    total_frames = sum(len(frames) for frames in clips.values())
    logger.info(
        "Found %d clip(s), %d frame(s) total",
        len(clips), total_frames,
    )

    # Process each clip
    previews_dir = paths["previews"]
    graded_ok = 0
    graded_failed = 0
    comparison_ok = 0
    comparison_failed = 0
    skipped = 0

    all_frames = [
        (clip_id, frame)
        for clip_id, frames in clips.items()
        for frame in frames
    ]

    for clip_id, frame_path in get_progress_bar(all_frames, desc="Generating previews"):
        graded_dir = previews_dir / clip_id / "graded"
        comparison_dir = previews_dir / clip_id / "comparison"

        graded_path = graded_dir / frame_path.name
        comparison_path = comparison_dir / frame_path.name

        # Skip existing outputs unless --force
        if not args.force and graded_path.is_file() and comparison_path.is_file():
            skipped += 1
            graded_ok += 1
            comparison_ok += 1
            continue

        # Step 1: Apply LUT
        if not apply_lut(frame_path, graded_path, lut_path):
            graded_failed += 1
            continue
        graded_ok += 1

        # Step 2: Generate side-by-side comparison
        if generate_comparison(frame_path, graded_path, comparison_path):
            comparison_ok += 1
        else:
            comparison_failed += 1

    # Summary
    logger.info("--- Preview generation complete ---")
    logger.info(
        "Graded: %d/%d | Comparisons: %d/%d | Skipped: %d",
        graded_ok, total_frames, comparison_ok, total_frames, skipped,
    )
    if graded_failed > 0 or comparison_failed > 0:
        logger.warning(
            "Failures — LUT: %d, comparison: %d",
            graded_failed, comparison_failed,
        )

    # Print path to first comparison for easy viewing
    first_clip = next(iter(clips))
    first_comparison = previews_dir / first_clip / "comparison" / clips[first_clip][0].name
    logger.info("Output: %s", previews_dir)
    if first_comparison.is_file():
        logger.info("First comparison: %s", first_comparison)

    if graded_failed > 0 and graded_ok == 0:
        logger.error("All frames failed. Check ffmpeg errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
