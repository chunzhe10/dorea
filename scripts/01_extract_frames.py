"""Phase 1: Footage Ingest & Frame Extraction (~5-10 min per dive session)

Extracts keyframes from raw footage at 2-second intervals for AI analysis.
Handles both DJI Action 4 (D-Log M) and Insta360 X5 (pre-flattened) clips.

Usage:
    python 01_extract_frames.py --date 2026-03-17

Inputs:
    - footage/raw/YYYY-MM-DD/ — DJI Action 4 clips
    - footage/flat/YYYY-MM-DD/ — Insta360 X5 flattened clips

Outputs:
    - working/keyframes/{clip_id}/frame_NNNNNN.jpg (1280px wide, JPEG 85%)

Dependencies:
    - ffmpeg (system package, CPU only, no GPU required)

Architecture doc: Section 4.1
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

# Video file extensions to scan (case-insensitive matching via explicit variants)
VIDEO_EXTENSIONS = {".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI"}

logger = logging.getLogger(__name__)


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


def load_config(workspace_root: Path) -> dict:
    """Load pipeline config from repos/dorea/config.yaml."""
    config_path = workspace_root / "repos" / "dorea" / "config.yaml"
    if not config_path.is_file():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def discover_clips(footage_dir: Path) -> list[Path]:
    """Find all video files in a footage directory (non-recursive)."""
    if not footage_dir.is_dir():
        return []

    clips = []
    for entry in sorted(footage_dir.iterdir()):
        if entry.is_file() and entry.suffix in VIDEO_EXTENSIONS:
            clips.append(entry)
    return clips


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

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    workspace_root = find_workspace_root()
    logger.info("Workspace root: %s", workspace_root)

    config = load_config(workspace_root)

    # Resolve footage directories for the given date
    raw_dir = workspace_root / config["footage_raw"] / args.date
    flat_dir = workspace_root / config["footage_flat"] / args.date
    keyframes_dir = workspace_root / config["working_dir"] / "keyframes"

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

    # Discover all video clips
    clips = discover_clips(raw_dir) + discover_clips(flat_dir)

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

    for clip_path in clips:
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
