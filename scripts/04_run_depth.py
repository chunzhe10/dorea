"""Phase 4: Depth Anything V2 — Depth Map Generation (~1-2 min per clip, local GPU)

Performs monocular depth estimation on every frame. Output is a per-frame depth
map used as a luminance matte in Resolve to drive depth-dependent colour correction.

Bright pixels = close to camera. Dark pixels = far from camera.

Physics basis: Water absorbs red/orange first (1-2m), green dominates at 5-10m,
blue dominates at 15m+. Depth maps enable physically-accurate colour correction:
foreground gets warmth/red lift, background gets red/orange recovery.

Usage:
    python 04_run_depth.py --date 2026-03-17

Inputs:
    - Raw footage files (for full-resolution frame access)

Outputs:
    - working/depth/{clip_id}/frame_NNNNNN.png
      (16-bit grayscale PNG, resolution matches source)

Dependencies:
    - Depth Anything V2 Small (via transformers)
    - PyTorch with CUDA
    - VRAM: ~1.5GB
    - Note: Trained on terrestrial footage. Reduced accuracy in turbid water.

Architecture doc: Section 4.4
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

# Video file extensions to scan (case-insensitive matching via explicit variants)
VIDEO_EXTENSIONS = {".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI"}

logger = logging.getLogger(__name__)

# HuggingFace model ID for Depth Anything V2 Small
HF_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


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


def load_depth_model(
    weights_path: Path, device: str
) -> tuple:
    """Load the Depth Anything V2 model and image processor.

    If weights_path is a local directory containing model files, load from there.
    Otherwise fall back to HuggingFace model ID for download.

    Returns (processor, model) tuple.
    """
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    # Determine model source: local directory or HuggingFace ID
    if weights_path.is_dir() and (weights_path / "config.json").is_file():
        model_source = str(weights_path)
        logger.info("Loading depth model from local weights: %s", weights_path)
    else:
        model_source = HF_MODEL_ID
        logger.info(
            "Local weights not found at %s; loading from HuggingFace: %s",
            weights_path, HF_MODEL_ID,
        )

    try:
        processor = AutoImageProcessor.from_pretrained(model_source)
        model = AutoModelForDepthEstimation.from_pretrained(model_source)
        model.to(device)
        model.eval()
    except Exception as e:
        logger.error("Failed to load depth model from %s: %s", model_source, e)
        sys.exit(1)

    # Log VRAM usage after loading
    if device.startswith("cuda"):
        allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
        logger.info(
            "Depth model loaded on %s | VRAM: %.0f MB allocated, %.0f MB reserved",
            device, allocated_mb, reserved_mb,
        )

    return processor, model


def unload_depth_model(model, device: str) -> None:
    """Explicitly unload the depth model and free GPU memory."""
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        logger.info(
            "Depth model unloaded | VRAM after cleanup: %.0f MB allocated",
            allocated_mb,
        )
    else:
        logger.info("Depth model unloaded")


def run_depth_inference(
    image: Image.Image, processor, model, device: str
) -> np.ndarray:
    """Run depth estimation on a single PIL Image.

    Returns a float32 numpy array of shape (H, W) with values in [0.0, 1.0],
    where 1.0 = close to camera (bright) and 0.0 = far from camera (dark).
    The array is resized to match the original image dimensions.
    """
    original_size = image.size  # (width, height)

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract predicted depth — shape is (1, H, W) or (1, 1, H, W)
    predicted_depth = outputs.predicted_depth

    # Squeeze batch/channel dimensions to get (H, W)
    depth = predicted_depth.squeeze().cpu().numpy()

    # Normalise to [0.0, 1.0] range
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max - depth_min > 0:
        depth_normalised = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_normalised = np.zeros_like(depth)

    # Depth Anything outputs inverse depth (closer = higher values), which
    # matches our convention: bright = close, dark = far. No inversion needed.

    # Resize to match original source resolution for pixel-accurate matte alignment
    w, h = original_size
    if depth_normalised.shape[0] != h or depth_normalised.shape[1] != w:
        depth_normalised = cv2.resize(
            depth_normalised, (w, h), interpolation=cv2.INTER_LINEAR
        )

    return depth_normalised


def save_depth_map_16bit(depth_float: np.ndarray, output_path: Path) -> None:
    """Save a float32 depth map as a 16-bit grayscale PNG.

    Input: float32 array in [0.0, 1.0]
    Output: 16-bit PNG where 65535 = close (bright), 0 = far (dark)
    """
    # Scale to uint16 range
    depth_uint16 = (depth_float * 65535.0).clip(0, 65535).astype(np.uint16)

    # Save using OpenCV (reliable 16-bit PNG support)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(output_path),
        depth_uint16,
        [cv2.IMWRITE_PNG_COMPRESSION, 3],  # moderate compression (0-9)
    )


def process_clip(
    clip_path: Path,
    output_dir: Path,
    processor,
    model,
    device: str,
) -> tuple[int, int]:
    """Extract every frame from a video clip and run depth estimation.

    Returns (frames_processed, frames_failed).
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        logger.error("Failed to open video: %s", clip_path)
        return 0, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(
        "  %s: %d frames, %.1f fps, %dx%d",
        clip_path.name, total_frames, fps, width, height,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    frames_processed = 0
    frames_failed = 0
    frame_idx = 0
    clip_start_time = time.monotonic()
    last_log_time = clip_start_time

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame_number = frame_idx  # 1-based, sequential

        output_path = output_dir / f"frame_{frame_number:06d}.png"

        try:
            # Convert BGR (OpenCV) to RGB PIL Image
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Run depth inference
            depth_map = run_depth_inference(pil_image, processor, model, device)

            # Save as 16-bit PNG
            save_depth_map_16bit(depth_map, output_path)
            frames_processed += 1

        except Exception as e:
            frames_failed += 1
            logger.error(
                "  Depth inference failed for frame %d of %s: %s",
                frame_number, clip_path.name, e,
            )
            continue

        # Log progress every 10 seconds or at milestones
        now = time.monotonic()
        if now - last_log_time >= 10.0 or frame_number == total_frames:
            elapsed = now - clip_start_time
            fps_rate = frames_processed / elapsed if elapsed > 0 else 0
            logger.info(
                "  Progress: %d/%d frames (%.1f%%) | %.2f frames/sec",
                frame_number, total_frames,
                (frame_number / total_frames * 100) if total_frames > 0 else 0,
                fps_rate,
            )
            last_log_time = now

    cap.release()

    elapsed = time.monotonic() - clip_start_time
    fps_rate = frames_processed / elapsed if elapsed > 0 else 0
    logger.info(
        "  Completed: %d depth maps in %.1fs (%.2f frames/sec)%s",
        frames_processed, elapsed, fps_rate,
        f" | {frames_failed} failed" if frames_failed > 0 else "",
    )

    return frames_processed, frames_failed


def main():
    parser = argparse.ArgumentParser(
        description="Run Depth Anything V2 depth estimation on dive footage"
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

    # Resolve footage directories for the given date
    raw_dir = workspace_root / config["footage_raw"] / args.date
    flat_dir = workspace_root / config["footage_flat"] / args.date
    depth_output_root = workspace_root / config["working_dir"] / "depth"

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

    # Resolve model config
    device = config.get("gpu_device", "cuda:0")
    weights_path = workspace_root / config.get("depth_weights", "models/depth_anything_v2_small/")

    # Check CUDA availability
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.error(
            "CUDA is not available but gpu_device is '%s'. "
            "Install CUDA-enabled PyTorch or set gpu_device to 'cpu' in config.yaml.",
            device,
        )
        sys.exit(1)

    if device.startswith("cuda"):
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_mem / (1024 * 1024)
        logger.info("GPU: %s (%.0f MB VRAM)", gpu_name, total_vram)

    # Load depth model
    logger.info("Loading Depth Anything V2 model...")
    pipeline_start_time = time.monotonic()
    processor, model = load_depth_model(weights_path, device)

    # Process each clip
    total_depth_maps = 0
    total_failed = 0
    successful_clips = 0
    failed_clips = 0

    try:
        for clip_path in clips:
            clip_id = clip_path.stem
            output_dir = depth_output_root / clip_id

            logger.info("--- Processing clip: %s ---", clip_id)

            frames_processed, frames_failed = process_clip(
                clip_path=clip_path,
                output_dir=output_dir,
                processor=processor,
                model=model,
                device=device,
            )

            if frames_processed == 0 and frames_failed == 0:
                # Could not open video
                failed_clips += 1
                logger.warning("Skipping %s due to video open error", clip_path.name)
                continue

            if frames_processed > 0:
                successful_clips += 1
            else:
                failed_clips += 1

            total_depth_maps += frames_processed
            total_failed += frames_failed
    finally:
        # Always unload model, even if processing fails partway through
        logger.info("Unloading depth model...")
        unload_depth_model(model, device)

    pipeline_elapsed = time.monotonic() - pipeline_start_time

    # Summary
    logger.info("--- Depth estimation complete ---")
    logger.info(
        "Clips: %d processed, %d failed | "
        "Depth maps: %d generated, %d failed | "
        "Total time: %.1fs",
        successful_clips, failed_clips,
        total_depth_maps, total_failed,
        pipeline_elapsed,
    )

    if failed_clips > 0 and successful_clips == 0:
        logger.error("All clips failed. Check errors above.")
        sys.exit(1)

    if failed_clips > 0:
        logger.warning("%d clip(s) failed. Check errors above.", failed_clips)


if __name__ == "__main__":
    main()
