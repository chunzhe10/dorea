"""Shared utilities for the dorea pipeline scripts.

Extracted from scripts 00-05 to eliminate duplication. All pipeline scripts
import their common functions from here.
"""

import logging
import os
import re
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Video file extensions to scan (case-insensitive matching via explicit variants)
VIDEO_EXTENSIONS = {".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI"}

# Supported reference image extensions (case-insensitive matching)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Date format pattern for --date arguments
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


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


def configure_logging(verbose: bool = False) -> None:
    """Configure logging with standard format used across all pipeline scripts."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def validate_date(date_str: str) -> str:
    """Validate that a date string is in YYYY-MM-DD format.

    Exits with error if invalid. Returns the date string if valid.
    """
    if not DATE_PATTERN.match(date_str):
        logger.error("Invalid date format: '%s'. Expected YYYY-MM-DD.", date_str)
        sys.exit(1)
    return date_str


def discover_clips(footage_dir: Path) -> list[Path]:
    """Find all video files in a footage directory (non-recursive)."""
    if not footage_dir.is_dir():
        return []

    clips = []
    for entry in sorted(footage_dir.iterdir()):
        if entry.is_file() and entry.suffix in VIDEO_EXTENSIONS:
            clips.append(entry)
    return clips


def discover_images(references_dir: Path) -> list[Path]:
    """Find all valid image files in a directory."""
    if not references_dir.is_dir():
        return []

    images = []
    for entry in sorted(references_dir.iterdir()):
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(entry)
    return images


def deduplicate_clips(clips: list[Path], prefer: str = "raw") -> list[Path]:
    """Deduplicate clips by stem, preferring raw footage (D-Log M dynamic range).

    When the same clip exists in both raw/ and flat/ directories, keeps only
    the preferred version. Warns on duplicates.

    Args:
        clips: List of clip paths (may contain duplicates by stem).
        prefer: Which to keep on conflict — "raw" or "flat".

    Returns:
        Deduplicated list preserving original order of kept clips.
    """
    seen: dict[str, Path] = {}
    result = []

    for clip in clips:
        stem = clip.stem
        if stem in seen:
            existing = seen[stem]
            # Determine which is raw and which is flat by parent dir name
            existing_is_raw = "raw" in existing.parts
            clip_is_raw = "raw" in clip.parts

            if prefer == "raw":
                keep_new = clip_is_raw and not existing_is_raw
            else:
                keep_new = not clip_is_raw and existing_is_raw

            if keep_new:
                logger.warning(
                    "Duplicate clip '%s': keeping %s over %s",
                    stem, clip, existing,
                )
                result = [p for p in result if p.stem != stem]
                result.append(clip)
                seen[stem] = clip
            else:
                logger.warning(
                    "Duplicate clip '%s': keeping %s over %s",
                    stem, existing, clip,
                )
        else:
            seen[stem] = clip
            result.append(clip)

    return result


def resolve_working_paths(workspace_root: Path, config: dict, date: str) -> dict:
    """Resolve date-scoped working directory paths.

    Returns a dict with keys:
        - keyframes: working/keyframes/{date}/
        - scene_analysis: working/scene_analysis/{date}/
        - masks: working/masks/{date}/
        - depth: working/depth/{date}/
        - working_dir: working/
    """
    working_dir = workspace_root / config["working_dir"]
    return {
        "working_dir": working_dir,
        "keyframes": working_dir / "keyframes" / date,
        "scene_analysis": working_dir / "scene_analysis" / date,
        "masks": working_dir / "masks" / date,
        "depth": working_dir / "depth" / date,
    }


def check_legacy_working_dir(workspace_root: Path, config: dict, date: str) -> None:
    """Warn if non-date-scoped working data exists (from before date-scoped dirs).

    Checks for clip directories directly under working/{type}/ that aren't
    date directories (YYYY-MM-DD pattern).
    """
    working_dir = workspace_root / config["working_dir"]
    subdirs = ["keyframes", "scene_analysis", "masks", "depth"]

    for subdir_name in subdirs:
        subdir = working_dir / subdir_name
        if not subdir.is_dir():
            continue
        for entry in subdir.iterdir():
            if entry.is_dir() and not DATE_PATTERN.match(entry.name):
                logger.warning(
                    "Legacy (non-date-scoped) data found: %s. "
                    "Consider moving to %s/%s/%s/",
                    entry, subdir_name, date, entry.name,
                )
                break  # One warning per subdir is enough


def get_progress_bar(iterable, total=None, desc=None, **kwargs):
    """Return a tqdm progress bar if stderr is a TTY, otherwise the plain iterable.

    This keeps log-based progress as fallback for batch/headless runs.
    """
    try:
        import tqdm
        if sys.stderr.isatty():
            return tqdm.tqdm(iterable, total=total, desc=desc, file=sys.stderr, **kwargs)
    except ImportError:
        pass
    return iterable
