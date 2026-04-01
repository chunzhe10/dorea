"""Create the 8-node DRX grade template for the dorea underwater pipeline.

This script must be run on the HOST MACHINE with DaVinci Resolve Studio running.
It programmatically creates the 8-node serial grade structure defined in the
pipeline architecture and exports it as a .drx file.

The 8-node structure (matching 05_resolve_setup.py constants):
    Node 1: Base LUT         - Receives underwater color correction LUT
    Node 2: Neutral Balance   - Manual white/color balance adjustments
    Node 3: Depth Grade       - Receives depth map as external matte
    Node 4: Foreground Pop    - Receives primary subject mask
    Node 5: Diver             - Receives diver mask
    Node 6: Marine Life       - Receives marine life mask
    Node 7: Creative Look     - Manual creative grading
    Node 8: Output            - Final output adjustments

Usage (on host, with Resolve running):
    python create_drx_template.py
    python create_drx_template.py --output /custom/path/template.drx
    python create_drx_template.py --verbose

Prerequisites:
    - DaVinci Resolve Studio running on the host
    - Environment variables set:
        RESOLVE_SCRIPT_API  (e.g., /opt/resolve/Developer/Scripting)
        RESOLVE_SCRIPT_LIB  (e.g., /opt/resolve/libs/Fusion/fusionscript.so)
        PYTHONPATH includes the Resolve scripting modules directory
    - NOT run from the devcontainer (Resolve is host-only)

Architecture doc: Section 5.1
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

from pipeline_utils import configure_logging, find_workspace_root, load_config

logger = logging.getLogger(__name__)

# Node labels must match the constants in 05_resolve_setup.py
NODE_LABELS = {
    1: "Base LUT",
    2: "Neutral Balance",
    3: "Depth Grade",
    4: "Foreground Pop",
    5: "Diver",
    6: "Marine Life",
    7: "Creative Look",
    8: "Output",
}

TOTAL_NODES = 8
TEMP_PROJECT_NAME = "_DRX_Template_Generator"


def create_test_image(output_path: Path) -> Path:
    """Create a minimal black PNG image for the temporary timeline.

    We need a real clip/image to create a timeline item that can hold a grade.
    A 64x64 black PNG is sufficient — the grade structure is resolution-independent.
    """
    try:
        from PIL import Image
        img = Image.new("RGB", (64, 64), color=(0, 0, 0))
        img.save(str(output_path))
        return output_path
    except ImportError:
        pass

    # Fallback: write a minimal valid PNG manually (1x1 black pixel)
    # PNG signature + IHDR + IDAT + IEND
    import struct
    import zlib

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        raw = chunk_type + data
        return struct.pack(">I", len(data)) + raw + struct.pack(">I", zlib.crc32(raw) & 0xFFFFFFFF)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)  # 1x1, 8-bit RGB
    ihdr = _chunk(b"IHDR", ihdr_data)
    raw_row = b"\x00" + b"\x00\x00\x00"  # filter byte + 1 RGB pixel (black)
    idat = _chunk(b"IDAT", zlib.compress(raw_row))
    iend = _chunk(b"IEND", b"")

    output_path.write_bytes(signature + ihdr + idat + iend)
    return output_path


def connect_to_resolve():
    """Connect to a running DaVinci Resolve instance."""
    try:
        import DaVinciResolveScript as dvr
    except ImportError:
        logger.error(
            "Cannot import DaVinciResolveScript. This script must run on the HOST.\n"
            "Ensure environment variables are set:\n"
            "  RESOLVE_SCRIPT_API, RESOLVE_SCRIPT_LIB, PYTHONPATH"
        )
        return None

    try:
        resolve = dvr.scriptapp("Resolve")
    except Exception as e:
        logger.error("Failed to connect to Resolve: %s", e)
        return None

    if resolve is None:
        logger.error("DaVinci Resolve is not running. Start Resolve Studio first.")
        return None

    logger.info("Connected to DaVinci Resolve")
    return resolve


def create_temp_project(resolve):
    """Create a temporary project for DRX generation.

    Returns (project_manager, project) tuple, or (None, None) on failure.
    """
    try:
        pm = resolve.GetProjectManager()
    except Exception as e:
        logger.error("Failed to get ProjectManager: %s", e)
        return None, None

    if pm is None:
        logger.error("ProjectManager is None")
        return None, None

    # Save reference to current project so we can switch back
    try:
        current_project = pm.GetCurrentProject()
        current_name = current_project.GetName() if current_project else None
    except Exception:
        current_name = None

    # Delete old temp project if it exists (from a previous failed run)
    try:
        pm.DeleteProject(TEMP_PROJECT_NAME)
    except Exception:
        pass

    try:
        project = pm.CreateProject(TEMP_PROJECT_NAME)
    except Exception as e:
        logger.error("Failed to create temp project: %s", e)
        return pm, None

    if project is None:
        logger.error("CreateProject returned None for '%s'", TEMP_PROJECT_NAME)
        return pm, None

    logger.info("Created temp project: %s", TEMP_PROJECT_NAME)
    return pm, project


def setup_timeline_with_test_clip(project, test_image_path: Path):
    """Import a test image and create a timeline with one clip.

    Returns (media_pool, timeline, timeline_item) or (None, None, None).
    """
    try:
        media_pool = project.GetMediaPool()
    except Exception as e:
        logger.error("Failed to get MediaPool: %s", e)
        return None, None, None

    if media_pool is None:
        logger.error("MediaPool is None")
        return None, None, None

    # Import the test image
    try:
        clips = media_pool.ImportMedia([str(test_image_path)])
    except Exception as e:
        logger.error("Failed to import test image: %s", e)
        return media_pool, None, None

    if not clips:
        logger.error("ImportMedia returned empty result")
        return media_pool, None, None

    clip_list = list(clips.values()) if isinstance(clips, dict) else list(clips) if not isinstance(clips, list) else clips

    # Create timeline
    try:
        timeline = media_pool.CreateTimelineFromClips("_DRX_Gen", clip_list)
    except Exception as e:
        logger.error("Failed to create timeline: %s", e)
        return media_pool, None, None

    if timeline is None:
        logger.error("CreateTimelineFromClips returned None")
        return media_pool, None, None

    # Get the timeline item
    try:
        items = timeline.GetItemListInTrack("video", 1)
    except Exception as e:
        logger.error("Failed to get timeline items: %s", e)
        return media_pool, timeline, None

    if not items:
        logger.error("No items on timeline video track")
        return media_pool, timeline, None

    return media_pool, timeline, items[0]


def build_node_graph(timeline_item) -> bool:
    """Add serial nodes to create the 8-node grade structure.

    A new timeline item starts with 1 node. We add 7 more serial nodes
    and label each one.

    Returns True if all 8 nodes were created successfully.
    """
    # Check starting node count
    try:
        initial_count = timeline_item.GetNumNodes()
        logger.debug("Initial node count: %d", initial_count)
    except Exception as e:
        logger.error("GetNumNodes() not available: %s", e)
        logger.error("Your Resolve version may not support node graph scripting.")
        return False

    # Add serial nodes until we have 8
    nodes_to_add = TOTAL_NODES - initial_count
    if nodes_to_add < 0:
        logger.warning("Timeline item already has %d nodes (expected 1)", initial_count)
        return False

    for i in range(nodes_to_add):
        try:
            result = timeline_item.AddSerialNode()
            if not result:
                logger.error(
                    "AddSerialNode() returned False on node %d of %d",
                    i + 1, nodes_to_add,
                )
                return False
        except AttributeError:
            logger.error(
                "AddSerialNode() not available. Your Resolve version may not "
                "support programmatic node creation. Create the 8-node grade "
                "manually in the Color page, then run this script with --export-only."
            )
            return False
        except Exception as e:
            logger.error("Failed to add serial node %d: %s", i + 1, e)
            return False

    # Verify node count
    try:
        final_count = timeline_item.GetNumNodes()
        if final_count != TOTAL_NODES:
            logger.error(
                "Expected %d nodes, got %d", TOTAL_NODES, final_count
            )
            return False
        logger.info("Created %d-node serial grade structure", final_count)
    except Exception:
        logger.info("Added %d serial nodes (verification not available)", nodes_to_add)

    # Label each node
    label_failures = 0
    for node_index, label in NODE_LABELS.items():
        try:
            result = timeline_item.LabelNode(node_index, label)
            if result:
                logger.debug("  Node %d: %s", node_index, label)
            else:
                logger.debug("  LabelNode(%d, '%s') returned False", node_index, label)
                label_failures += 1
        except AttributeError:
            if node_index == 1:
                logger.info(
                    "LabelNode() not available in this Resolve version. "
                    "Node structure is correct but labels must be added manually."
                )
            label_failures += 1
            break
        except Exception as e:
            logger.debug("  Failed to label node %d: %s", node_index, e)
            label_failures += 1

    if label_failures == 0:
        logger.info("All %d nodes labeled successfully", TOTAL_NODES)
    elif label_failures < TOTAL_NODES:
        logger.warning(
            "%d of %d node labels failed. Add missing labels manually in Color page.",
            label_failures, TOTAL_NODES,
        )

    return True


def export_drx(timeline_item, output_path: Path) -> bool:
    """Export the current grade as a DRX file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = timeline_item.ExportGradeAsDRX(str(output_path), 0)
        if result:
            logger.info("Exported DRX template: %s", output_path)
            return True
        else:
            logger.error("ExportGradeAsDRX returned False")
            return False
    except AttributeError:
        logger.error(
            "ExportGradeAsDRX() not available. Your Resolve version may not "
            "support DRX export via scripting API."
        )
        return False
    except Exception as e:
        logger.error("Failed to export DRX: %s", e)
        return False


def cleanup_temp_project(project_manager):
    """Delete the temporary project used for DRX generation."""
    if project_manager is None:
        return

    try:
        # Switch away from the temp project first
        project_manager.CloseProject(project_manager.GetCurrentProject())
    except Exception:
        pass

    try:
        project_manager.DeleteProject(TEMP_PROJECT_NAME)
        logger.info("Cleaned up temp project: %s", TEMP_PROJECT_NAME)
    except Exception as e:
        logger.warning(
            "Could not delete temp project '%s': %s. "
            "Delete it manually in Resolve's Project Manager.",
            TEMP_PROJECT_NAME, e,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Create the 8-node DRX grade template for underwater video processing"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output path for the DRX file (default: from config.yaml)",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help=(
            "Skip node creation — only export the current grade of the first "
            "timeline item. Use this if you manually created the 8-node grade."
        ),
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't delete the temporary project after export",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)

    # Determine output path
    if args.output:
        output_path = args.output.resolve()
    else:
        workspace_root = find_workspace_root()
        config = load_config(workspace_root)
        drx_rel = config.get(
            "resolve_drx_path", "repos/dorea/templates/underwater_grade_v1.drx"
        )
        output_path = workspace_root / drx_rel

    logger.info("Output path: %s", output_path)

    # Connect to Resolve
    resolve = connect_to_resolve()
    if resolve is None:
        sys.exit(1)

    if args.export_only:
        # Export from the current project's first timeline item
        try:
            project = resolve.GetProjectManager().GetCurrentProject()
            timeline = project.GetCurrentTimeline()
            items = timeline.GetItemListInTrack("video", 1)
            item = items[0]
        except Exception as e:
            logger.error("Failed to get current timeline item for --export-only: %s", e)
            sys.exit(1)

        if not export_drx(item, output_path):
            sys.exit(1)
        logger.info("Done. DRX exported from current project.")
        return

    # Full automated flow
    project_manager = None
    try:
        # Create test image in temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            test_image = Path(tmpdir) / "black_frame.png"
            create_test_image(test_image)
            logger.debug("Created test image: %s", test_image)

            # Create temp project
            project_manager, project = create_temp_project(resolve)
            if project is None:
                sys.exit(1)

            # Set up timeline
            media_pool, timeline, item = setup_timeline_with_test_clip(
                project, test_image
            )
            if item is None:
                logger.error("Failed to create timeline item. Cannot build node graph.")
                sys.exit(1)

            # Build the 8-node grade structure
            if not build_node_graph(item):
                logger.error(
                    "Failed to create 8-node grade structure.\n"
                    "Manual alternative:\n"
                    "  1. Open any project in Resolve Color page\n"
                    "  2. Select a clip\n"
                    "  3. Right-click -> Add Node -> Add Serial (repeat 7 times)\n"
                    "  4. Label nodes: %s\n"
                    "  5. Right-click -> Export -> Export LUT/Grade as DRX\n"
                    "  6. Save to: %s",
                    ", ".join(NODE_LABELS.values()),
                    output_path,
                )
                sys.exit(1)

            # Export the DRX
            if not export_drx(item, output_path):
                sys.exit(1)

    finally:
        # Clean up temp project
        if not args.no_cleanup and project_manager is not None:
            cleanup_temp_project(project_manager)

    logger.info("Done. DRX template created at: %s", output_path)
    logger.info(
        "Next step: run Phase 5 on the host:\n"
        "  python scripts/05_resolve_setup.py --date <YYYY-MM-DD>"
    )


if __name__ == "__main__":
    main()
