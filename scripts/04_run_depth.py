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
    - working/depth/{date}/{clip_id}/frame_NNNNNN.png
      (16-bit grayscale PNG at native model resolution ~518px;
       Resolve auto-scales mattes to timeline resolution)

Dependencies:
    - Depth Anything V2 Small (via transformers)
    - PyTorch with CUDA
    - VRAM: ~1.5GB
    - Note: Trained on terrestrial footage. Reduced accuracy in turbid water.

Architecture doc: Section 4.4
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from pipeline_utils import (
    VIDEO_EXTENSIONS,
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

# Map config model names to HuggingFace model IDs
DEPTH_MODEL_MAP = {
    "depth_anything_v2_small": "depth-anything/Depth-Anything-V2-Small-hf",
    "depth_anything_v2_base": "depth-anything/Depth-Anything-V2-Base-hf",
    "depth_anything_v2_large": "depth-anything/Depth-Anything-V2-Large-hf",
}
DEFAULT_MODEL = "depth_anything_v2_small"


def load_depth_model(
    weights_path: Path, device: str, model_name: str = DEFAULT_MODEL
) -> tuple:
    """Load the Depth Anything V2 model and image processor.

    If weights_path is a local directory containing model files, load from there.
    Otherwise fall back to HuggingFace model ID for download.

    Returns (processor, model) tuple.
    """
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    hf_model_id = DEPTH_MODEL_MAP.get(model_name)
    if not hf_model_id:
        logger.error(
            "Unknown depth model '%s'. Supported: %s",
            model_name, ", ".join(DEPTH_MODEL_MAP.keys()),
        )
        sys.exit(1)

    # Determine model source: local directory or HuggingFace ID
    if weights_path.is_dir() and (weights_path / "config.json").is_file():
        model_source = str(weights_path)
        logger.info("Loading depth model from local weights: %s", weights_path)
    else:
        model_source = hf_model_id
        logger.info(
            "Local weights not found at %s; loading from HuggingFace: %s",
            weights_path, hf_model_id,
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
    The array is at the model's native resolution (~518px for V2 Small).
    Resolve auto-scales external mattes to timeline resolution via GPU.
    """
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
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

    return depth_normalised


def save_depth_map(
    depth_float: np.ndarray, output_path: Path, output_format: str = "png16"
) -> None:
    """Save a float32 depth map as a grayscale PNG.

    Args:
        depth_float: float32 array in [0.0, 1.0]
        output_path: Destination file path
        output_format: "png16" for 16-bit (default) or "png8" for 8-bit

    Output: PNG where bright = close, dark = far.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "png8":
        depth_out = (depth_float * 255.0).clip(0, 255).astype(np.uint8)
    else:
        # Default: 16-bit for matte precision (avoids banding on smooth UW gradients)
        depth_out = (depth_float * 65535.0).clip(0, 65535).astype(np.uint16)

    success = cv2.imwrite(
        str(output_path),
        depth_out,
        [cv2.IMWRITE_PNG_COMPRESSION, 3],  # moderate compression (0-9)
    )
    if not success:
        raise RuntimeError(f"cv2.imwrite failed for {output_path}")


def process_clip(
    clip_path: Path,
    output_dir: Path,
    processor,
    model,
    device: str,
    output_format: str = "png16",
    output_resolution: str = "native",
) -> tuple[int, int]:
    """Extract every frame from a video clip and run depth estimation.

    Args:
        output_format: "png16" or "png8" — controls bit depth of saved PNGs.
        output_resolution: "native" to keep model output resolution, or
            "WxH" (e.g. "1920x1080") to resize before saving.

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

    # Use tqdm for TTY, fall back to 10-second log intervals for headless
    use_tqdm = sys.stderr.isatty()
    pbar = None
    if use_tqdm:
        try:
            import tqdm
            pbar = tqdm.tqdm(
                total=total_frames, desc=f"  {clip_path.stem}",
                file=sys.stderr, unit="frame",
            )
        except ImportError:
            pass

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

            # Run depth inference (returns native model resolution)
            depth_map = run_depth_inference(pil_image, processor, model, device)

            # Optional resize if depth_output_resolution is not "native"
            if output_resolution != "native":
                try:
                    target_w, target_h = (int(x) for x in output_resolution.split("x"))
                    if depth_map.shape[1] != target_w or depth_map.shape[0] != target_h:
                        depth_map = cv2.resize(
                            depth_map, (target_w, target_h),
                            interpolation=cv2.INTER_LINEAR,
                        )
                except ValueError:
                    pass  # Invalid format logged once at startup; skip per-frame

            # Save depth map
            save_depth_map(depth_map, output_path, output_format=output_format)
            frames_processed += 1

        except Exception as e:
            frames_failed += 1
            logger.error(
                "  Depth inference failed for frame %d of %s: %s",
                frame_number, clip_path.name, e,
            )
            continue

        if pbar is not None:
            pbar.update(1)
        else:
            # Log progress every 10 seconds for headless/batch runs
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

    if pbar is not None:
        pbar.close()
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

    configure_logging(verbose=args.verbose)

    validate_date(args.date)

    workspace_root = find_workspace_root()
    logger.info("Workspace root: %s", workspace_root)

    config = load_config(workspace_root)

    # Resolve footage directories for the given date
    raw_dir = workspace_root / config["footage_raw"] / args.date
    flat_dir = workspace_root / config["footage_flat"] / args.date
    paths = resolve_working_paths(workspace_root, config, args.date)
    depth_output_root = paths["depth"]

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

    # Resolve model config
    device = config.get("gpu_device", "cuda:0")
    model_name = config.get("depth_model", DEFAULT_MODEL)
    weights_path = workspace_root / config.get("depth_weights", "models/depth_anything_v2_small/")
    output_format = config.get("depth_output_format", "png16")
    output_resolution = config.get("depth_output_resolution", "native")

    # Validate config values
    if output_format not in ("png16", "png8"):
        logger.warning(
            "Unknown depth_output_format '%s', defaulting to 'png16'", output_format
        )
        output_format = "png16"

    if output_resolution != "native":
        try:
            rw, rh = (int(x) for x in output_resolution.split("x"))
            logger.info("Depth output resolution: %dx%d", rw, rh)
        except ValueError:
            logger.warning(
                "Invalid depth_output_resolution '%s' (expected 'native' or 'WxH'). "
                "Defaulting to 'native'.", output_resolution,
            )
            output_resolution = "native"

    if output_resolution == "native":
        logger.info("Depth output resolution: native (model default)")
    logger.info("Depth output format: %s", output_format)

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
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        logger.info("GPU: %s (%.0f MB VRAM)", gpu_name, total_vram)

    # Load depth model
    logger.info("Loading Depth Anything V2 model...")
    pipeline_start_time = time.monotonic()
    processor, model = load_depth_model(weights_path, device, model_name=model_name)

    # Process each clip
    total_depth_maps = 0
    total_failed = 0
    successful_clips = 0
    failed_clips = 0

    try:
        for clip_path in get_progress_bar(clips, desc="Depth estimation"):
            clip_id = clip_path.stem
            output_dir = depth_output_root / clip_id

            logger.info("--- Processing clip: %s ---", clip_id)

            frames_processed, frames_failed = process_clip(
                clip_path=clip_path,
                output_dir=output_dir,
                processor=processor,
                model=model,
                device=device,
                output_format=output_format,
                output_resolution=output_resolution,
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
