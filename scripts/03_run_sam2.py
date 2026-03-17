"""Phase 3: SAM2 Subject Tracking (~1-3 min per clip, local GPU)

Generates per-frame mask sequences for each tracked subject using SAM2.
Initialised at the exact first-appearance frame from Phase 2 scene analysis,
using the bounding box as the starting prompt.

Usage:
    python 03_run_sam2.py --date 2026-03-17
    python 03_run_sam2.py --date 2026-03-17 --verbose

Inputs:
    - working/scene_analysis/{date}/{clip_id}.json — from Phase 2
    - Raw footage files (for full-resolution frame access)

Outputs:
    - working/masks/{date}/{clip_id}/{subject_label}/frame_NNNNNN.png
      (binary alpha mask, 0 or 255 per pixel)

Dependencies:
    - SAM2 (facebookresearch/sam2)
    - PyTorch with CUDA
    - ffmpeg (for full-resolution frame extraction)
    - VRAM: ~3GB (SAM2-small). Only one model loaded at a time.

Architecture doc: Section 4.3
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pipeline_utils import (
    VIDEO_EXTENSIONS,
    configure_logging,
    find_workspace_root,
    get_progress_bar,
    load_config,
    resolve_working_paths,
    validate_date,
)

logger = logging.getLogger(__name__)

# SAM2 model config name mapping
SAM2_CONFIG_MAP = {
    "sam2_tiny": "sam2.1_hiera_t",
    "sam2_small": "sam2.1_hiera_s",
    "sam2_base_plus": "sam2.1_hiera_b+",
    "sam2_large": "sam2.1_hiera_l",
}

# Minimum mask area fraction (of total frame pixels) before considering subject "exited"
MASK_EXIT_THRESHOLD = 0.0005  # 0.05% of frame area

# Number of consecutive empty frames before declaring subject has exited
EXIT_FRAME_GAP = 30


# SAM2 model download URLs (Meta hosted)
SAM2_DOWNLOAD_URLS = {
    "sam2_tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    "sam2_small": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    "sam2_base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "sam2_large": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}


def download_sam2_weights(model_key: str, target_path: Path) -> bool:
    """Download SAM2 model weights from Meta's servers.

    Downloads to a .tmp file then atomically renames to prevent partial downloads.
    Logs progress every 10%.

    Returns True on success, False on failure.
    """
    url = SAM2_DOWNLOAD_URLS.get(model_key)
    if not url:
        logger.error(
            "No download URL for model '%s'. Valid options: %s",
            model_key, ", ".join(SAM2_DOWNLOAD_URLS.keys()),
        )
        return False

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".pt.tmp")

    logger.info("Downloading SAM2 weights: %s", url)
    logger.info("Target: %s", target_path)

    try:
        response = urllib.request.urlopen(url)
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        last_pct = -10
        chunk_size = 1024 * 1024  # 1MB chunks

        with open(tmp_path, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = int(downloaded / total_size * 100)
                    if pct >= last_pct + 10:
                        logger.info(
                            "  Download progress: %d%% (%d / %d MB)",
                            pct, downloaded // (1024 * 1024),
                            total_size // (1024 * 1024),
                        )
                        last_pct = pct

        # Atomic rename
        tmp_path.rename(target_path)
        logger.info("Download complete: %s", target_path)
        return True

    except Exception as e:
        logger.error("Download failed: %s", e)
        if tmp_path.is_file():
            tmp_path.unlink()
        logger.error(
            "Manual download: %s -> %s",
            url, target_path,
        )
        return False


def discover_scene_analyses(scene_analysis_dir: Path) -> list[Path]:
    """Find all scene_analysis JSON files.

    Returns sorted list of JSON file paths.
    """
    if not scene_analysis_dir.is_dir():
        return []

    jsons = []
    for entry in sorted(scene_analysis_dir.iterdir()):
        if entry.is_file() and entry.suffix == ".json":
            jsons.append(entry)
    return jsons


def load_scene_analysis(json_path: Path) -> dict | None:
    """Load and validate a scene_analysis JSON file.

    Returns the parsed dict or None if invalid.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load %s: %s", json_path, e)
        return None

    if not isinstance(data, dict):
        logger.error("Invalid scene_analysis format in %s: expected dict", json_path)
        return None

    if "clip_id" not in data:
        logger.error("Missing 'clip_id' in %s", json_path)
        return None

    if "subjects" not in data or not isinstance(data["subjects"], list):
        logger.error("Missing or invalid 'subjects' in %s", json_path)
        return None

    return data


def find_source_video(
    clip_id: str, raw_dir: Path, flat_dir: Path
) -> Path | None:
    """Find the source video file for a given clip_id.

    Searches raw_dir first (preserves D-Log M dynamic range), then flat_dir.
    Returns the path or None if not found.
    """
    for footage_dir in [raw_dir, flat_dir]:
        if not footage_dir.is_dir():
            continue
        for entry in footage_dir.iterdir():
            if entry.is_file() and entry.suffix in VIDEO_EXTENSIONS:
                if entry.stem == clip_id:
                    logger.debug(
                        "Source video for '%s': %s (raw preferred)", clip_id, entry
                    )
                    return entry
    return None


def get_video_info(video_path: Path) -> dict | None:
    """Get video resolution and frame count using ffprobe.

    Returns dict with 'width', 'height', 'frame_count' or None on failure.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "v:0",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        logger.error("ffprobe failed for %s: %s", video_path, e)
        return None

    streams = data.get("streams", [])
    if not streams:
        logger.error("No video streams found in %s", video_path)
        return None

    stream = streams[0]
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))

    # Try to get frame count from nb_frames, fall back to counting via ffprobe
    frame_count = 0
    nb_frames = stream.get("nb_frames")
    if nb_frames and nb_frames != "N/A":
        frame_count = int(nb_frames)
    else:
        # Count frames using ffprobe
        count_cmd = [
            "ffprobe",
            "-v", "quiet",
            "-count_frames",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-print_format", "json",
            str(video_path),
        ]
        try:
            count_result = subprocess.run(
                count_cmd, capture_output=True, text=True, check=True, timeout=120
            )
            count_data = json.loads(count_result.stdout)
            nb_read = count_data.get("streams", [{}])[0].get("nb_read_frames")
            if nb_read and nb_read != "N/A":
                frame_count = int(nb_read)
        except (subprocess.CalledProcessError, json.JSONDecodeError, subprocess.TimeoutExpired):
            pass

    if width == 0 or height == 0:
        logger.error("Could not determine video resolution for %s", video_path)
        return None

    return {"width": width, "height": height, "frame_count": frame_count}


def extract_frames_to_dir(
    video_path: Path, output_dir: Path, max_short_edge: int = 1024
) -> int:
    """Extract all frames from a video to a directory as JPEG files using ffmpeg.

    SAM2 expects a directory of JPEG frames named sequentially (e.g., 000000.jpg,
    000001.jpg, ...). We extract starting from frame 0 of the video.

    Frames are downscaled so the short edge is at most max_short_edge pixels.
    SAM2 internally resizes to 1024px, so extracting at native 4K wastes memory
    and causes OOM on 6GB VRAM cards with long clips.

    Returns the number of frames extracted, or -1 on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scale so the short edge is at most max_short_edge pixels.
    # If already small enough, pass through unchanged.
    # For landscape (w>h): scale height to max_short_edge, width=-2 (auto)
    # For portrait (h>w): scale width to max_short_edge, height=-2 (auto)
    scale_filter = (
        f"scale='if(lte(min(iw,ih),{max_short_edge}),iw,"
        f"if(lte(iw,ih),{max_short_edge},-2))'"
        f":'if(lte(min(iw,ih),{max_short_edge}),ih,"
        f"if(gt(iw,ih),{max_short_edge},-2))'"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", scale_filter,
        "-start_number", "0",
        "-q:v", "2",  # high quality JPEG
        str(output_dir / "%06d.jpg"),
    ]

    logger.debug("Extracting frames: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error("ffmpeg frame extraction failed: %s", e.stderr.strip()[:500])
        return -1

    # Count extracted frames
    frame_count = len([
        f for f in output_dir.iterdir()
        if f.is_file() and f.suffix == ".jpg"
    ])
    return frame_count


def bbox_normalised_to_pixel(
    bbox_normalised: list[float], width: int, height: int
) -> list[float]:
    """Convert normalised [x_min, y_min, x_max, y_max] (0-1) to pixel coordinates.

    Returns [x_min, y_min, x_max, y_max] in pixel space.
    """
    x_min, y_min, x_max, y_max = bbox_normalised
    return np.array([
        x_min * width,
        y_min * height,
        x_max * width,
        y_max * height,
    ], dtype=np.float32)


def save_mask_as_png(
    mask: np.ndarray, output_path: Path
) -> None:
    """Save a binary mask as a single-channel PNG (0 or 255 per pixel).

    Args:
        mask: Boolean or float numpy array of shape (H, W).
        output_path: Path to save the PNG file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to binary alpha: 0 or 255
    if mask.dtype == bool:
        alpha = (mask.astype(np.uint8)) * 255
    else:
        # Threshold at 0.0 for logit masks
        alpha = ((mask > 0).astype(np.uint8)) * 255

    img = Image.fromarray(alpha, mode="L")
    img.save(str(output_path))


def save_empty_mask(width: int, height: int, output_path: Path) -> None:
    """Save an all-black (empty) mask as a single-channel PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    alpha = np.zeros((height, width), dtype=np.uint8)
    img = Image.fromarray(alpha, mode="L")
    img.save(str(output_path))


def load_sam2_model(
    config_name: str, checkpoint_path: Path, device: str
):
    """Load the SAM2 video predictor model.

    Args:
        config_name: SAM2 config identifier (e.g., "sam2.1_hiera_s").
        checkpoint_path: Path to the model weights file.
        device: CUDA device string (e.g., "cuda:0").

    Returns:
        SAM2VideoPredictor instance.
    """
    from sam2.build_sam import build_sam2_video_predictor

    # Ensure hydra can find SAM2.1 configs in the configs/sam2.1/ subdirectory
    # by re-initializing with the configs subpackage as search path
    from hydra.core.global_hydra import GlobalHydra
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    from hydra import initialize_config_dir
    import sam2 as _sam2_pkg
    sam2_configs_dir = str(Path(_sam2_pkg.__file__).parent / "configs" / "sam2.1")
    initialize_config_dir(config_dir=sam2_configs_dir, version_base="1.2")

    logger.info("Loading SAM2 model: config=%s, weights=%s", config_name, checkpoint_path)

    if torch.cuda.is_available():
        vram_before = torch.cuda.memory_allocated(device) / (1024 ** 3)
        logger.info("VRAM before model load: %.2f GB", vram_before)

    predictor = build_sam2_video_predictor(
        config_file=config_name,
        ckpt_path=str(checkpoint_path),
        device=device,
    )

    if torch.cuda.is_available():
        vram_after = torch.cuda.memory_allocated(device) / (1024 ** 3)
        logger.info(
            "VRAM after model load: %.2f GB (model uses ~%.2f GB)",
            vram_after, vram_after - vram_before,
        )

    return predictor


def unload_sam2_model(predictor, device: str) -> None:
    """Explicitly unload the SAM2 model and free GPU memory."""
    logger.info("Unloading SAM2 model")
    del predictor

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        vram_after = torch.cuda.memory_allocated(device) / (1024 ** 3)
        logger.info("VRAM after model unload: %.2f GB", vram_after)


def track_subject(
    predictor,
    frame_dir: str,
    subject: dict,
    video_width: int,
    video_height: int,
    total_frames: int,
    mask_output_dir: Path,
    device: str,
) -> int:
    """Track a single subject through the video and export mask PNGs.

    Args:
        predictor: SAM2VideoPredictor instance.
        frame_dir: Path to directory of extracted JPEG frames.
        subject: Subject dict with label, first_appearance_frame, bbox_normalised.
        video_width: Video width in pixels.
        video_height: Video height in pixels.
        total_frames: Total number of frames in the video.
        mask_output_dir: Output directory for this subject's masks.
        device: CUDA device string.

    Returns:
        Number of mask frames written.
    """
    label = subject["label"]
    first_frame = subject["first_appearance_frame"]
    bbox_norm = subject["bbox_normalised"]

    # Convert normalised bbox to pixel coordinates
    bbox_pixel = bbox_normalised_to_pixel(bbox_norm, video_width, video_height)

    logger.info(
        "  Subject '%s': first_frame=%d, bbox_norm=%s, bbox_pixel=%s",
        label, first_frame, bbox_norm,
        [f"{v:.1f}" for v in bbox_pixel],
    )

    # Clamp first_frame to valid range (0-indexed for SAM2)
    # Scene analysis frame numbers are 1-indexed (from ffmpeg frame_NNNNNN naming
    # where frame_000001 is the first frame). SAM2's frame directory uses 0-indexed
    # filenames (000000.jpg, 000001.jpg, ...).
    # Convert: scene analysis frame N -> SAM2 frame index N-1
    sam2_first_frame_idx = max(0, first_frame - 1)
    if sam2_first_frame_idx >= total_frames:
        logger.warning(
            "  Subject '%s': first_frame %d exceeds video length %d, clamping",
            label, first_frame, total_frames,
        )
        sam2_first_frame_idx = total_frames - 1

    mask_output_dir.mkdir(parents=True, exist_ok=True)
    masks_written = 0
    consecutive_empty = 0
    subject_exited = False

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Initialise SAM2 state for this video
        state = predictor.init_state(
            video_path=frame_dir,
            offload_video_to_cpu=True,  # save VRAM on 6GB card
        )

        # Add bounding box prompt at the first appearance frame
        # SAM2 expects pixel coordinates when normalize_coords=True (default)
        frame_idx, obj_ids, masks = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=sam2_first_frame_idx,
            obj_id=1,  # single object per tracking session
            box=bbox_pixel,
        )

        # Propagate forward through remaining frames
        for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(state):
            # video_res_masks shape: (num_objects, 1, H, W) — logit scores
            # We track one object, so index [0, 0]
            mask_logits = video_res_masks[0, 0]  # (H, W)
            mask_binary = (mask_logits > 0.0).cpu().numpy()

            # Check mask area for subject exit detection
            mask_area_fraction = mask_binary.sum() / mask_binary.size

            if mask_area_fraction < MASK_EXIT_THRESHOLD:
                consecutive_empty += 1
                if consecutive_empty >= EXIT_FRAME_GAP and not subject_exited:
                    logger.info(
                        "  Subject '%s': exited at frame %d "
                        "(empty for %d consecutive frames)",
                        label, frame_idx, consecutive_empty,
                    )
                    subject_exited = True
            else:
                if subject_exited:
                    # Subject re-entered after exit — for now, continue tracking
                    # (re-entry with new SAM2 session would require additional logic
                    # to detect the re-entry frame and re-initialise; the current
                    # propagation may lose accuracy, but SAM2's built-in tracking
                    # handles many re-entry cases adequately)
                    logger.info(
                        "  Subject '%s': possible re-entry at frame %d",
                        label, frame_idx,
                    )
                    subject_exited = False
                consecutive_empty = 0

            # Convert SAM2 0-indexed frame back to 1-indexed for output naming
            output_frame_num = frame_idx + 1
            output_path = mask_output_dir / f"frame_{output_frame_num:06d}.png"

            if subject_exited:
                # Pad with empty mask after subject exit
                save_empty_mask(video_width, video_height, output_path)
            else:
                save_mask_as_png(mask_binary, output_path)

            masks_written += 1

        # Reset state to free memory for next subject
        predictor.reset_state(state)

    return masks_written


def process_clip(
    predictor,
    clip_id: str,
    subjects: list[dict],
    video_path: Path,
    masks_base_dir: Path,
    device: str,
) -> dict:
    """Process all subjects in a single clip.

    Extracts frames to a temp directory, then runs SAM2 tracking for each subject
    independently (one tracking session per subject).

    Returns dict with per-subject results.
    """
    # Get video info
    video_info = get_video_info(video_path)
    if video_info is None:
        logger.error("Could not get video info for %s, skipping clip", video_path)
        return {"subjects_tracked": 0, "total_masks": 0, "errors": len(subjects)}

    video_width = video_info["width"]
    video_height = video_info["height"]
    logger.info(
        "Clip %s: %dx%d, source=%s",
        clip_id, video_width, video_height, video_path.name,
    )

    # Extract all frames to a temporary directory for SAM2
    # SAM2 init_state expects a directory of sequentially-named JPEG files
    with tempfile.TemporaryDirectory(prefix=f"sam2_{clip_id}_") as tmp_dir:
        frame_dir = Path(tmp_dir) / "frames"
        logger.info("Extracting frames from %s to temp dir...", video_path.name)
        frame_count = extract_frames_to_dir(video_path, frame_dir)

        if frame_count <= 0:
            logger.error(
                "Failed to extract frames from %s, skipping clip", video_path.name
            )
            return {"subjects_tracked": 0, "total_masks": 0, "errors": len(subjects)}

        logger.info(
            "Clip %s: %d frames extracted, %d subject(s) to track",
            clip_id, frame_count, len(subjects),
        )

        subjects_tracked = 0
        total_masks = 0
        errors = 0

        for subj_idx, subject in enumerate(get_progress_bar(
            subjects, desc=f"Tracking {clip_id}"
        )):
            label = subject.get("label", f"unknown_{subj_idx}")
            logger.info(
                "Tracking subject %d/%d: '%s'",
                subj_idx + 1, len(subjects), label,
            )

            mask_output_dir = masks_base_dir / clip_id / label

            try:
                mask_count = track_subject(
                    predictor=predictor,
                    frame_dir=str(frame_dir),
                    subject=subject,
                    video_width=video_width,
                    video_height=video_height,
                    total_frames=frame_count,
                    mask_output_dir=mask_output_dir,
                    device=device,
                )
                subjects_tracked += 1
                total_masks += mask_count
                logger.info(
                    "  Subject '%s': %d mask frames written → %s",
                    label, mask_count, mask_output_dir,
                )
            except Exception as e:
                logger.error(
                    "  SAM2 tracking failed for subject '%s': %s", label, e,
                    exc_info=True,
                )
                errors += 1
                continue

    return {
        "subjects_tracked": subjects_tracked,
        "total_masks": total_masks,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run SAM2 subject tracking from scene analysis"
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

    # Resolve paths (date-scoped)
    paths = resolve_working_paths(workspace_root, config, args.date)
    scene_analysis_dir = paths["scene_analysis"]
    masks_base_dir = paths["masks"]
    raw_dir = workspace_root / config["footage_raw"] / args.date
    flat_dir = workspace_root / config["footage_flat"] / args.date

    # Resolve SAM2 config
    sam2_model_key = config.get("sam2_model", "sam2_small")
    sam2_config_name = SAM2_CONFIG_MAP.get(sam2_model_key)
    if sam2_config_name is None:
        logger.error(
            "Unknown sam2_model '%s' in config. Valid options: %s",
            sam2_model_key, ", ".join(SAM2_CONFIG_MAP.keys()),
        )
        sys.exit(1)

    sam2_weights_path = workspace_root / config.get(
        "sam2_weights", "models/sam2/sam2.1_hiera_small.pt"
    )
    device = config.get("gpu_device", "cuda:0")

    logger.info("SAM2 config: model=%s (%s)", sam2_model_key, sam2_config_name)
    logger.info("SAM2 weights: %s", sam2_weights_path)
    logger.info("GPU device: %s", device)

    # Check that model weights exist; auto-download if missing
    if not sam2_weights_path.is_file():
        logger.warning("SAM2 model weights not found at %s", sam2_weights_path)
        if not download_sam2_weights(sam2_model_key, sam2_weights_path):
            sys.exit(1)

    # Check for CUDA availability
    if not torch.cuda.is_available():
        logger.error(
            "CUDA is not available. SAM2 requires a GPU. "
            "Check your PyTorch installation and GPU drivers."
        )
        sys.exit(1)

    # Discover scene_analysis JSON files
    scene_files = discover_scene_analyses(scene_analysis_dir)
    if not scene_files:
        logger.error(
            "No scene_analysis JSON files found under %s. "
            "Run Phase 2 (02_claude_scene_analysis.py --date %s) first.",
            scene_analysis_dir, args.date,
        )
        sys.exit(1)

    logger.info(
        "Found %d scene_analysis file(s) for processing", len(scene_files)
    )

    # Check that at least one footage directory exists
    if not raw_dir.is_dir() and not flat_dir.is_dir():
        logger.error(
            "No footage directories found for date %s. "
            "Expected at least one of:\n  %s\n  %s",
            args.date, raw_dir, flat_dir,
        )
        sys.exit(1)

    # Load SAM2 model once for all clips
    predictor = None
    try:
        predictor = load_sam2_model(sam2_config_name, sam2_weights_path, device)
    except Exception as e:
        logger.error("Failed to load SAM2 model: %s", e, exc_info=True)
        sys.exit(1)

    # Process each clip
    total_clips_processed = 0
    total_clips_skipped = 0
    total_clips_failed = 0
    total_subjects_tracked = 0
    total_masks_generated = 0

    try:
        for scene_file in scene_files:
            clip_id = scene_file.stem
            logger.info("--- Processing clip: %s ---", clip_id)

            # Load scene analysis
            scene_data = load_scene_analysis(scene_file)
            if scene_data is None:
                total_clips_failed += 1
                continue

            subjects = scene_data["subjects"]
            if not subjects:
                logger.info(
                    "Clip %s: no subjects found in scene analysis, skipping",
                    clip_id,
                )
                total_clips_skipped += 1
                continue

            logger.info(
                "Clip %s: %d subject(s) to track", clip_id, len(subjects)
            )

            # Find the source video for this clip
            video_path = find_source_video(clip_id, raw_dir, flat_dir)
            if video_path is None:
                logger.error(
                    "Source video not found for clip '%s' in:\n  %s\n  %s",
                    clip_id, raw_dir, flat_dir,
                )
                total_clips_failed += 1
                continue

            logger.info("Source video: %s", video_path)

            # Process all subjects in this clip
            result = process_clip(
                predictor=predictor,
                clip_id=clip_id,
                subjects=subjects,
                video_path=video_path,
                masks_base_dir=masks_base_dir,
                device=device,
            )

            total_clips_processed += 1
            total_subjects_tracked += result["subjects_tracked"]
            total_masks_generated += result["total_masks"]

            if result["errors"] > 0:
                logger.warning(
                    "Clip %s: %d subject(s) failed tracking",
                    clip_id, result["errors"],
                )

    finally:
        # Always unload the model, even if processing fails
        if predictor is not None:
            unload_sam2_model(predictor, device)

    # Summary
    logger.info("--- SAM2 tracking complete ---")
    logger.info(
        "Clips: %d processed, %d skipped (no subjects), %d failed",
        total_clips_processed, total_clips_skipped, total_clips_failed,
    )
    logger.info(
        "Subjects tracked: %d | Total mask frames: %d",
        total_subjects_tracked, total_masks_generated,
    )
    logger.info("Masks output: %s", masks_base_dir)

    if total_clips_failed > 0 and total_clips_processed == 0:
        logger.error("All clips failed. Check errors above.")
        sys.exit(1)

    if total_clips_failed > 0:
        logger.warning("%d clip(s) failed. Check errors above.", total_clips_failed)


if __name__ == "__main__":
    main()
