"""Phase 1: Footage Ingest & Frame Extraction (~5-10 min per dive session)

Extracts keyframes from raw footage at 2-second intervals for AI analysis.
Handles both DJI Action 4 (D-Log M) and Insta360 X5 (pre-flattened) clips.

Usage:
    python 01_extract_frames.py --date 2026-03-17

Inputs:
    - footage/raw/YYYY-MM-DD/ — DJI Action 4 clips
    - footage/flat/YYYY-MM-DD/ — Insta360 X5 flattened clips

Outputs:
    - working/keyframes/{date}/{clip_id}/frame_NNNNNN.jpg (1280px wide, JPEG 85%)

Dependencies:
    - ffmpeg (system package, CPU only, no GPU required)

Architecture doc: Section 4.1
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from pipeline_utils import (
    VIDEO_EXTENSIONS,
    check_legacy_working_dir,
    configure_logging,
    deduplicate_clips,
    discover_clips,
    find_workspace_root,
    get_progress_bar,
    load_config,
    resolve_working_paths,
    validate_date,
)

logger = logging.getLogger(__name__)


def extract_frames(clip_path: Path, output_dir: Path, fps: float = 0.5) -> int:
    """Extract keyframes from a single clip using ffmpeg.

    Returns the number of frames extracted, or -1 on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(clip_path),
        "-vf", f"fps={fps},scale=1280:-1",
        "-q:v", "5",
        str(output_dir / "frame_%06d.jpg"),
    ]

    logger.info("Processing: %s", clip_path.name)
    logger.debug("Command: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error("ffmpeg failed for %s: %s", clip_path.name, e.stderr.strip())
        return -1

    # Count extracted frames
    frame_count = len([
        f for f in output_dir.iterdir()
        if f.is_file() and f.suffix == ".jpg"
    ])
    return frame_count


def main():
    parser = argparse.ArgumentParser(description="Extract keyframes from dive footage")
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

    # Resolve footage directories for the given date
    raw_dir = workspace_root / config["footage_raw"] / args.date
    flat_dir = workspace_root / config["footage_flat"] / args.date
    paths = resolve_working_paths(workspace_root, config, args.date)
    keyframes_dir = paths["keyframes"]

    check_legacy_working_dir(workspace_root, config, args.date)

    logger.info("Scanning for footage dated %s", args.date)
    logger.debug("  Raw dir:  %s (exists: %s)", raw_dir, raw_dir.is_dir())
    logger.debug("  Flat dir: %s (exists: %s)", flat_dir, flat_dir.is_dir())

    # Check that at least one footage directory exists
    if not raw_dir.is_dir() and not flat_dir.is_dir():
        logger.error(
            "No footage directories found for date %s. "
            "Expected at least one of:\n  %s\n  %s",
            args.date, raw_dir, flat_dir,
        )
        sys.exit(1)

    # Discover all video clips (deduplicate raw/flat by stem)
    clips = deduplicate_clips(discover_clips(raw_dir) + discover_clips(flat_dir))

    if not clips:
        logger.error(
            "No video files found in footage directories for date %s. "
            "Supported extensions: %s",
            args.date, ", ".join(sorted(VIDEO_EXTENSIONS)),
        )
        sys.exit(1)

    logger.info("Found %d clip(s) to process", len(clips))

    # Compute fps from config sample rate
    sample_rate = config.get("frame_sample_rate_seconds", 2)
    fps = 1.0 / sample_rate

    # Process each clip
    total_frames = 0
    successful_clips = 0
    failed_clips = 0

    for clip_path in get_progress_bar(clips, desc="Extracting frames"):
        clip_id = clip_path.stem
        output_dir = keyframes_dir / clip_id

        frame_count = extract_frames(clip_path, output_dir, fps=fps)

        if frame_count < 0:
            failed_clips += 1
            logger.warning("Skipping %s due to ffmpeg error", clip_path.name)
            continue

        successful_clips += 1
        total_frames += frame_count
        logger.info(
            "  %s: %d frame(s) extracted → %s",
            clip_id, frame_count, output_dir,
        )

    # Summary
    logger.info("--- Extraction complete ---")
    logger.info(
        "Clips: %d processed, %d failed | Total frames: %d",
        successful_clips, failed_clips, total_frames,
    )

    if failed_clips > 0 and successful_clips == 0:
        logger.error("All clips failed. Check ffmpeg errors above.")
        sys.exit(1)

    if failed_clips > 0:
        logger.warning("%d clip(s) failed. Check errors above.", failed_clips)


if __name__ == "__main__":
    main()
