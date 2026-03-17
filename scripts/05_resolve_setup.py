"""Phase 5: DaVinci Resolve Import & Setup (HOST ONLY)

WARNING: This script runs on the HOST MACHINE, not in the devcontainer.
It requires DaVinci Resolve Studio to be running and imports fusionscript.

Imports footage into Resolve, deploys the DRX template, applies the base LUT,
and attaches mask/depth sequences as External Mattes on the appropriate nodes.

Usage:
    python 05_resolve_setup.py --date 2026-03-17
    python 05_resolve_setup.py --date 2026-03-17 --verbose

Inputs:
    - Raw footage files from footage/raw/{date}/ and footage/flat/{date}/
    - working/masks/{clip_id}/{subject_label}/ — SAM2 mask PNG sequences from Phase 3
    - working/depth/{clip_id}/ — depth map PNG sequences from Phase 4
    - repos/dorea/luts/underwater_base.cube — reference LUT from Phase 0
    - repos/dorea/templates/underwater_grade_v1.drx — DRX node graph template

Outputs:
    - DaVinci Resolve project with timeline ready for creative grading
    - Each clip has 8-node structure: Base LUT, Neutral Balance, Depth Grade,
      Foreground Pop, Diver, Marine Life, Creative Look, Output

Dependencies:
    - DaVinci Resolve Studio (running on host)
    - fusionscript.so (Resolve Python API)
    - Environment vars: RESOLVE_SCRIPT_API, RESOLVE_SCRIPT_LIB, PYTHONPATH
    - No GPU required (Resolve uses GPU independently)

Architecture doc: Sections 5 and 7
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

# Video file extensions to scan (case-insensitive matching via explicit variants)
VIDEO_EXTENSIONS = {".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI"}

logger = logging.getLogger(__name__)

# --- Node structure constants (Section 5.1) ---
# 8-node DRX template structure. Node indices are 1-based.
NODE_BASE_LUT = 1
NODE_NEUTRAL_BALANCE = 2
NODE_DEPTH_GRADE = 3
NODE_FOREGROUND_POP = 4
NODE_DIVER = 5
NODE_MARINE_LIFE = 6
NODE_CREATIVE_LOOK = 7
NODE_OUTPUT = 8

NODE_NAMES = {
    NODE_BASE_LUT: "Base LUT",
    NODE_NEUTRAL_BALANCE: "Neutral Balance",
    NODE_DEPTH_GRADE: "Depth Grade",
    NODE_FOREGROUND_POP: "Foreground Pop",
    NODE_DIVER: "Diver",
    NODE_MARINE_LIFE: "Marine Life",
    NODE_CREATIVE_LOOK: "Creative Look",
    NODE_OUTPUT: "Output",
}

# Marine life identifiers for subject-to-node mapping
MARINE_LIFE_KEYWORDS = {
    "wrasse", "fish", "turtle", "ray", "shark", "octopus", "squid",
    "clownfish", "grouper", "barracuda", "moray", "eel", "seahorse",
    "jellyfish", "cuttlefish", "lionfish", "parrotfish", "angelfish",
    "butterflyfish", "triggerfish", "pufferfish", "boxfish", "nudibranch",
    "lobster", "crab", "shrimp", "whale", "dolphin", "manta", "stingray",
    "coral", "anemone", "sponge", "sea_fan", "sea_star", "starfish",
    "urchin", "cucumber", "bommie",
}


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


def discover_footage(raw_dir: Path, flat_dir: Path) -> list[Path]:
    """Find all video files in raw and flat footage directories.

    Returns a sorted list of video file paths.
    """
    clips = []
    for footage_dir in [raw_dir, flat_dir]:
        if not footage_dir.is_dir():
            continue
        for entry in sorted(footage_dir.iterdir()):
            if entry.is_file() and entry.suffix in VIDEO_EXTENSIONS:
                clips.append(entry)
    return sorted(clips, key=lambda p: p.name)


def load_scene_analysis(scene_analysis_dir: Path, clip_id: str) -> list[dict]:
    """Load scene analysis subjects for a clip.

    Returns the list of subject dicts, or empty list if not found.
    """
    json_path = scene_analysis_dir / f"{clip_id}.json"
    if not json_path.is_file():
        logger.debug("No scene analysis file for clip %s at %s", clip_id, json_path)
        return []

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load scene analysis for %s: %s", clip_id, e)
        return []

    if not isinstance(data, dict) or "subjects" not in data:
        logger.warning("Invalid scene analysis format for %s", clip_id)
        return []

    return data["subjects"]


def classify_subject(label: str) -> int | None:
    """Map a subject label to its target node index.

    Returns:
        NODE_DIVER (5) for diver subjects,
        NODE_MARINE_LIFE (6) for marine life subjects,
        NODE_FOREGROUND_POP (4) for other identifiable subjects,
        or None if the label cannot be classified.
    """
    label_lower = label.lower()

    # Diver detection
    if "diver" in label_lower:
        return NODE_DIVER

    # Marine life detection — check if any keyword appears in the label
    for keyword in MARINE_LIFE_KEYWORDS:
        if keyword in label_lower:
            return NODE_MARINE_LIFE

    # Default: foreground subject
    return NODE_FOREGROUND_POP


def assign_subjects_to_nodes(subjects: list[dict]) -> dict[int, dict]:
    """Assign subjects to grade nodes based on their labels.

    Each node (4, 5, 6) gets at most one subject. The first matching subject
    for each category wins; additional subjects are logged as unassigned.

    Returns:
        Dict mapping node_index -> subject dict for assigned subjects.
    """
    assignments: dict[int, dict] = {}
    unassigned: list[dict] = []

    for subject in subjects:
        label = subject.get("label", "unknown")
        target_node = classify_subject(label)

        if target_node is None:
            unassigned.append(subject)
            continue

        if target_node not in assignments:
            assignments[target_node] = subject
            logger.debug(
                "  Assigned '%s' -> Node %d (%s)",
                label, target_node, NODE_NAMES.get(target_node, "?"),
            )
        else:
            # Node already taken — log as unassigned
            unassigned.append(subject)
            logger.debug(
                "  Subject '%s' unassigned (Node %d already has '%s')",
                label, target_node, assignments[target_node].get("label", "?"),
            )

    if unassigned:
        labels = [s.get("label", "?") for s in unassigned]
        logger.info(
            "  %d subject(s) unassigned (can be manually added): %s",
            len(unassigned), ", ".join(labels),
        )

    return assignments


def find_mask_sequence_dir(masks_base: Path, clip_id: str, label: str) -> Path | None:
    """Find the mask sequence directory for a specific subject.

    Expected path: working/masks/{clip_id}/{label}/
    Returns the path if it exists and contains PNG files, else None.
    """
    mask_dir = masks_base / clip_id / label
    if not mask_dir.is_dir():
        return None

    # Verify it contains at least one PNG file
    png_files = [f for f in mask_dir.iterdir() if f.is_file() and f.suffix == ".png"]
    if not png_files:
        return None

    return mask_dir


def find_depth_sequence_dir(depth_base: Path, clip_id: str) -> Path | None:
    """Find the depth map sequence directory for a clip.

    Expected path: working/depth/{clip_id}/
    Returns the path if it exists and contains PNG files, else None.
    """
    depth_dir = depth_base / clip_id
    if not depth_dir.is_dir():
        return None

    # Verify it contains at least one PNG file
    png_files = [f for f in depth_dir.iterdir() if f.is_file() and f.suffix == ".png"]
    if not png_files:
        return None

    return depth_dir


# ---------------------------------------------------------------------------
# Resolve API wrappers — each wraps a single API interaction with error handling
# ---------------------------------------------------------------------------


def connect_to_resolve():
    """Connect to a running DaVinci Resolve instance.

    Returns the Resolve object, or None if connection fails.
    """
    try:
        import DaVinciResolveScript as dvr
    except ImportError:
        logger.error(
            "Cannot import DaVinciResolveScript. Ensure that:\n"
            "  1. DaVinci Resolve Studio is installed\n"
            "  2. Environment variables are set:\n"
            "     - RESOLVE_SCRIPT_API (e.g., /opt/resolve/Developer/Scripting)\n"
            "     - RESOLVE_SCRIPT_LIB (e.g., /opt/resolve/libs/Fusion/fusionscript.so)\n"
            "     - PYTHONPATH includes the Resolve scripting modules directory\n"
            "  3. This script is run from the HOST machine, not the devcontainer"
        )
        return None

    try:
        resolve = dvr.scriptapp("Resolve")
    except Exception as e:
        logger.error("Failed to connect to Resolve scripting API: %s", e)
        return None

    if resolve is None:
        logger.error(
            "DaVinci Resolve is not running or not responding. "
            "Please start Resolve Studio before running this script."
        )
        return None

    logger.info("Connected to DaVinci Resolve")
    return resolve


def get_or_create_project(resolve, project_name: str):
    """Open an existing project or create a new one.

    Returns the Project object, or None on failure.
    """
    try:
        project_manager = resolve.GetProjectManager()
    except Exception as e:
        logger.error("Failed to get ProjectManager: %s", e)
        return None

    if project_manager is None:
        logger.error("ProjectManager is None — Resolve may not be fully loaded")
        return None

    # Try to load existing project first
    try:
        current = project_manager.GetCurrentProject()
        if current is not None and current.GetName() == project_name:
            logger.info("Using current project: %s", project_name)
            return current
    except Exception:
        pass

    # Try loading the project by name
    try:
        project = project_manager.LoadProject(project_name)
        if project is not None:
            logger.info("Opened existing project: %s", project_name)
            return project
    except Exception as e:
        logger.debug("Could not load project '%s': %s", project_name, e)

    # Create new project
    try:
        project = project_manager.CreateProject(project_name)
        if project is not None:
            logger.info("Created new project: %s", project_name)
            return project
    except Exception as e:
        logger.error("Failed to create project '%s': %s", project_name, e)
        return None

    logger.error("Failed to create or open project '%s'", project_name)
    return None


def create_footage_subfolder(media_pool, root_folder, date: str):
    """Create a 'Footage/{date}' subfolder in the Media Pool.

    Returns the subfolder object, or the root folder on failure.
    """
    # Try to find or create 'Footage' folder
    footage_folder = None
    try:
        subfolders = root_folder.GetSubFolderList()
        for sf in subfolders:
            if sf.GetName() == "Footage":
                footage_folder = sf
                break
    except Exception as e:
        logger.debug("Error listing subfolders: %s", e)

    if footage_folder is None:
        try:
            footage_folder = media_pool.AddSubFolder(root_folder, "Footage")
            if footage_folder is None:
                logger.warning("Could not create 'Footage' subfolder, using root")
                return root_folder
        except Exception as e:
            logger.warning("Error creating 'Footage' subfolder: %s — using root", e)
            return root_folder

    # Create date subfolder under Footage
    date_folder = None
    try:
        subfolders = footage_folder.GetSubFolderList()
        for sf in subfolders:
            if sf.GetName() == date:
                date_folder = sf
                break
    except Exception:
        pass

    if date_folder is None:
        try:
            date_folder = media_pool.AddSubFolder(footage_folder, date)
            if date_folder is None:
                logger.warning(
                    "Could not create date subfolder '%s', using 'Footage'", date
                )
                return footage_folder
        except Exception as e:
            logger.warning(
                "Error creating date subfolder '%s': %s — using 'Footage'", date, e
            )
            return footage_folder

    logger.info("Media Pool folder: Footage/%s", date)
    return date_folder


def import_footage_to_media_pool(media_pool, target_folder, clip_paths: list[Path]):
    """Import footage clips into the specified Media Pool folder.

    Returns list of imported MediaPoolItem objects, or empty list on failure.
    """
    try:
        media_pool.SetCurrentFolder(target_folder)
    except Exception as e:
        logger.warning("Could not set current folder: %s", e)

    file_paths = [str(p) for p in clip_paths]

    try:
        clips = media_pool.ImportMedia(file_paths)
    except Exception as e:
        logger.error("Failed to import footage: %s", e)
        return []

    if clips is None:
        logger.error("ImportMedia returned None — check file paths and Resolve logs")
        return []

    if isinstance(clips, list):
        logger.info("Imported %d clip(s) to Media Pool", len(clips))
        return clips

    # Some API versions return a dict or other iterable
    try:
        clip_list = list(clips.values()) if isinstance(clips, dict) else list(clips)
        logger.info("Imported %d clip(s) to Media Pool", len(clip_list))
        return clip_list
    except Exception:
        logger.info("Imported clip(s) to Media Pool (count unknown)")
        return [clips] if clips else []


def create_timeline(media_pool, timeline_name: str, clips):
    """Create a timeline from imported clips.

    Returns the Timeline object, or None on failure.
    """
    if not clips:
        logger.error("No clips provided for timeline creation")
        return None

    try:
        timeline = media_pool.CreateTimelineFromClips(timeline_name, clips)
    except Exception as e:
        logger.error("Failed to create timeline '%s': %s", timeline_name, e)
        return None

    if timeline is None:
        logger.error(
            "CreateTimelineFromClips returned None for '%s'. "
            "Clips may already be on a timeline or be incompatible.",
            timeline_name,
        )
        return None

    logger.info("Created timeline: %s", timeline_name)
    return timeline


def apply_drx_template(timeline_item, drx_path: str) -> bool:
    """Apply a DRX grade template to a timeline item.

    The DRX file deploys the 8-node grade structure.
    Returns True on success, False on failure.
    """
    try:
        result = timeline_item.ApplyGradeFromDRX(drx_path, 0)
        if result:
            logger.debug("  Applied DRX template")
            return True
        else:
            logger.warning("  ApplyGradeFromDRX returned False")
            return False
    except Exception as e:
        logger.warning("  Failed to apply DRX template: %s", e)
        return False


def set_node_lut(timeline_item, node_index: int, lut_path: str) -> bool:
    """Set a LUT on a specific node of a timeline item.

    Node indices are 1-based.
    Returns True on success, False on failure.
    """
    try:
        result = timeline_item.SetLUT(node_index, lut_path)
        if result:
            logger.debug(
                "  Set LUT on Node %d (%s)",
                node_index, NODE_NAMES.get(node_index, "?"),
            )
            return True
        else:
            logger.warning(
                "  SetLUT returned False for Node %d", node_index
            )
            return False
    except Exception as e:
        logger.warning("  Failed to set LUT on Node %d: %s", node_index, e)
        return False


def import_image_sequence(media_pool, sequence_dir: Path):
    """Import an image sequence directory into the Media Pool.

    Resolve treats a directory of sequentially-named images as an image sequence
    when imported via ImportMedia with the directory path.

    Returns the imported MediaPoolItem, or None on failure.
    """
    try:
        result = media_pool.ImportMedia([str(sequence_dir)])
    except Exception as e:
        logger.warning("  Failed to import image sequence from %s: %s", sequence_dir, e)
        return None

    if result is None:
        return None

    # Extract the single imported item
    if isinstance(result, list) and len(result) > 0:
        return result[0]
    if isinstance(result, dict):
        values = list(result.values())
        return values[0] if values else None
    return result


def attach_external_matte(timeline_item, node_index: int, matte_path: str) -> bool:
    """Attach an external matte (image sequence) to a specific node.

    Uses the Resolve API to set the external matte on the given node.
    The matte_path should point to the first frame or directory of the sequence.

    Returns True on success, False on failure.
    """
    # The Resolve API provides multiple approaches for attaching mattes.
    # The most reliable approach is via SetNodeCacheMode or using the
    # Fusion page. We use the Color page's external matte functionality.
    #
    # Method: Use the timeline item's SetNodeExternalMatte API if available,
    # otherwise fall back to property-based approaches.
    try:
        # Primary approach: Use the node-level external matte API
        # SetNodeEnabled and similar APIs are node-index-based (1-based)
        result = timeline_item.SetNodeExternalMatte(node_index, matte_path)
        if result:
            logger.debug(
                "  Attached external matte to Node %d (%s)",
                node_index, NODE_NAMES.get(node_index, "?"),
            )
            return True
    except AttributeError:
        # API method may not exist in this Resolve version
        logger.debug("  SetNodeExternalMatte not available, trying alternative")
    except Exception as e:
        logger.debug("  SetNodeExternalMatte failed: %s", e)

    # Fallback approach: Use the clip-level external matte property
    # This attaches the matte to the clip's overall external matte slot,
    # which can then be routed to specific nodes in the Color page.
    try:
        result = timeline_item.SetClipProperty("ExternalMatte", matte_path)
        if result:
            logger.debug(
                "  Attached external matte via SetClipProperty for Node %d (%s)",
                node_index, NODE_NAMES.get(node_index, "?"),
            )
            return True
    except Exception as e:
        logger.debug("  SetClipProperty fallback failed: %s", e)

    logger.warning(
        "  Could not attach external matte to Node %d (%s) — "
        "manual attachment may be required in Resolve Color page",
        node_index, NODE_NAMES.get(node_index, "?"),
    )
    return False


def get_first_frame_path(sequence_dir: Path) -> str | None:
    """Get the path to the first PNG frame in a sequence directory.

    Returns the path as a string, or None if no frames found.
    """
    png_files = sorted(
        f for f in sequence_dir.iterdir()
        if f.is_file() and f.suffix == ".png"
    )
    if png_files:
        return str(png_files[0])
    return None


def process_timeline_item(
    timeline_item,
    media_pool,
    clip_id: str,
    drx_path: str | None,
    lut_path: str | None,
    masks_base: Path,
    depth_base: Path,
    scene_analysis_dir: Path,
) -> dict:
    """Process a single timeline item: apply DRX, LUT, and attach mattes.

    Returns a summary dict with counts of applied elements.
    """
    result = {
        "drx_applied": False,
        "lut_set": False,
        "depth_attached": False,
        "mattes_attached": 0,
        "mattes_failed": 0,
        "warnings": [],
    }

    # Step 1: Apply DRX template (deploys 8-node structure)
    if drx_path:
        result["drx_applied"] = apply_drx_template(timeline_item, drx_path)
        if not result["drx_applied"]:
            result["warnings"].append("DRX template not applied")
    else:
        result["warnings"].append("DRX template file not found, skipped")

    # Step 2: Set LUT on Node 1 (Base LUT)
    if lut_path:
        result["lut_set"] = set_node_lut(timeline_item, NODE_BASE_LUT, lut_path)
        if not result["lut_set"]:
            result["warnings"].append("LUT not set on Node 1")
    else:
        result["warnings"].append("LUT file not found, skipped")

    # Step 3: Find and attach depth sequence to Node 3
    depth_dir = find_depth_sequence_dir(depth_base, clip_id)
    if depth_dir:
        depth_matte_path = get_first_frame_path(depth_dir)
        if depth_matte_path:
            # Import depth sequence to media pool
            depth_item = import_image_sequence(media_pool, depth_dir)
            if depth_item:
                logger.debug("  Imported depth sequence for %s", clip_id)

            result["depth_attached"] = attach_external_matte(
                timeline_item, NODE_DEPTH_GRADE, depth_matte_path
            )
            if result["depth_attached"]:
                logger.info("  Depth matte -> Node %d (%s)", NODE_DEPTH_GRADE, NODE_NAMES[NODE_DEPTH_GRADE])
        else:
            result["warnings"].append("Depth directory exists but contains no PNG frames")
    else:
        logger.debug("  No depth sequence found for %s", clip_id)

    # Step 4: Load scene analysis and assign subjects to nodes
    subjects = load_scene_analysis(scene_analysis_dir, clip_id)
    if not subjects:
        logger.debug("  No subjects from scene analysis for %s", clip_id)
        return result

    assignments = assign_subjects_to_nodes(subjects)

    # Step 5: Attach mask mattes for assigned subjects
    for node_index, subject in assignments.items():
        label = subject.get("label", "unknown")
        mask_dir = find_mask_sequence_dir(masks_base, clip_id, label)

        if mask_dir is None:
            logger.warning(
                "  No mask sequence found for subject '%s' (expected at %s)",
                label, masks_base / clip_id / label,
            )
            result["mattes_failed"] += 1
            continue

        # Import mask sequence to media pool
        mask_item = import_image_sequence(media_pool, mask_dir)
        if mask_item:
            logger.debug("  Imported mask sequence for subject '%s'", label)

        mask_path = get_first_frame_path(mask_dir)
        if mask_path is None:
            logger.warning("  Mask directory for '%s' contains no PNG frames", label)
            result["mattes_failed"] += 1
            continue

        attached = attach_external_matte(timeline_item, node_index, mask_path)
        if attached:
            logger.info(
                "  Mask '%s' -> Node %d (%s)",
                label, node_index, NODE_NAMES.get(node_index, "?"),
            )
            result["mattes_attached"] += 1
        else:
            result["mattes_failed"] += 1

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Set up DaVinci Resolve project with footage, grades, and mattes"
    )
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

    # Resolve paths from config
    raw_dir = workspace_root / config["footage_raw"] / args.date
    flat_dir = workspace_root / config["footage_flat"] / args.date
    working_dir = workspace_root / config["working_dir"]
    masks_base = working_dir / "masks"
    depth_base = working_dir / "depth"
    scene_analysis_dir = working_dir / "scene_analysis"

    # Resolve template and LUT paths
    lut_rel = config.get("resolve_lut_path", "repos/dorea/luts/underwater_base.cube")
    drx_rel = config.get("resolve_drx_path", "repos/dorea/templates/underwater_grade_v1.drx")
    lut_path = workspace_root / lut_rel
    drx_path = workspace_root / drx_rel

    project_name = config.get("resolve_project_name", "Dive_2026")

    # Validate paths
    logger.info("Date: %s", args.date)
    logger.info("Project name: %s", project_name)
    logger.debug("Raw footage dir:  %s (exists: %s)", raw_dir, raw_dir.is_dir())
    logger.debug("Flat footage dir: %s (exists: %s)", flat_dir, flat_dir.is_dir())
    logger.debug("Masks dir:        %s (exists: %s)", masks_base, masks_base.is_dir())
    logger.debug("Depth dir:        %s (exists: %s)", depth_base, depth_base.is_dir())
    logger.debug("LUT path:         %s (exists: %s)", lut_path, lut_path.is_file())
    logger.debug("DRX path:         %s (exists: %s)", drx_path, drx_path.is_file())

    # Check LUT and DRX availability (warn, don't exit)
    lut_path_str = str(lut_path) if lut_path.is_file() else None
    drx_path_str = str(drx_path) if drx_path.is_file() else None

    if lut_path_str is None:
        logger.warning(
            "LUT file not found at %s. Node 1 LUT will not be set. "
            "Run Phase 0 (00_generate_lut.py) to generate it.",
            lut_path,
        )

    if drx_path_str is None:
        logger.warning(
            "DRX template not found at %s. 8-node grade structure will not be "
            "deployed. Place the template file manually or create it in Resolve.",
            drx_path,
        )

    # Discover footage
    clips = discover_footage(raw_dir, flat_dir)
    if not clips:
        logger.error(
            "No footage files found for date %s. "
            "Expected video files in:\n  %s\n  %s",
            args.date, raw_dir, flat_dir,
        )
        sys.exit(1)

    logger.info("Found %d footage clip(s)", len(clips))
    for clip in clips:
        logger.debug("  %s", clip)

    # --- Connect to Resolve ---
    resolve = connect_to_resolve()
    if resolve is None:
        sys.exit(1)

    # --- Project setup ---
    project = get_or_create_project(resolve, project_name)
    if project is None:
        sys.exit(1)

    # --- Media Pool setup ---
    try:
        media_pool = project.GetMediaPool()
    except Exception as e:
        logger.error("Failed to get Media Pool: %s", e)
        sys.exit(1)

    if media_pool is None:
        logger.error("GetMediaPool returned None")
        sys.exit(1)

    try:
        root_folder = media_pool.GetRootFolder()
    except Exception as e:
        logger.error("Failed to get root folder: %s", e)
        sys.exit(1)

    # Create subfolder structure
    target_folder = create_footage_subfolder(media_pool, root_folder, args.date)

    # --- Import footage ---
    imported_clips = import_footage_to_media_pool(media_pool, target_folder, clips)
    if not imported_clips:
        logger.error("No clips were imported to Media Pool. Cannot continue.")
        sys.exit(1)

    # --- Create timeline ---
    timeline_name = f"Dive_{args.date}"
    timeline = create_timeline(media_pool, timeline_name, imported_clips)
    if timeline is None:
        logger.error("Failed to create timeline. Cannot continue.")
        sys.exit(1)

    # --- Process each clip on the timeline ---
    try:
        timeline_items = timeline.GetItemListInTrack("video", 1)
    except Exception as e:
        logger.error("Failed to get timeline items: %s", e)
        sys.exit(1)

    if not timeline_items:
        logger.error("No items found on timeline video track 1")
        sys.exit(1)

    logger.info("Processing %d timeline item(s)", len(timeline_items))

    total_drx = 0
    total_lut = 0
    total_depth = 0
    total_mattes = 0
    total_matte_failures = 0
    all_warnings = []

    for item in timeline_items:
        try:
            clip_name = item.GetName()
        except Exception:
            clip_name = "unknown"

        # Derive clip_id from the timeline item name (strip extension)
        clip_id = Path(clip_name).stem if clip_name else "unknown"
        logger.info("--- Clip: %s (clip_id: %s) ---", clip_name, clip_id)

        item_result = process_timeline_item(
            timeline_item=item,
            media_pool=media_pool,
            clip_id=clip_id,
            drx_path=drx_path_str,
            lut_path=lut_path_str,
            masks_base=masks_base,
            depth_base=depth_base,
            scene_analysis_dir=scene_analysis_dir,
        )

        if item_result["drx_applied"]:
            total_drx += 1
        if item_result["lut_set"]:
            total_lut += 1
        if item_result["depth_attached"]:
            total_depth += 1
        total_mattes += item_result["mattes_attached"]
        total_matte_failures += item_result["mattes_failed"]
        all_warnings.extend(item_result["warnings"])

    # --- Summary ---
    logger.info("--- Resolve setup complete ---")
    logger.info("Project: %s | Timeline: %s", project_name, timeline_name)
    logger.info(
        "Clips: %d imported, %d on timeline",
        len(imported_clips), len(timeline_items),
    )
    logger.info(
        "Grades: %d DRX applied, %d LUT set",
        total_drx, total_lut,
    )
    logger.info(
        "Mattes: %d depth attached, %d subject mattes attached, %d failed",
        total_depth, total_mattes, total_matte_failures,
    )

    if all_warnings:
        unique_warnings = sorted(set(all_warnings))
        logger.info("Warnings (%d):", len(unique_warnings))
        for w in unique_warnings:
            logger.warning("  %s", w)

    if total_matte_failures > 0:
        logger.warning(
            "%d matte attachment(s) failed. "
            "Check Resolve's Color page to manually attach missing mattes.",
            total_matte_failures,
        )


if __name__ == "__main__":
    main()
