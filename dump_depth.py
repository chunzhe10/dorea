#!/usr/bin/env python3
"""Dump depth map for a single keyframe as a grayscale PNG.

Usage: python dump_depth.py <video_path> [frame_index]
"""
import sys
import subprocess
import numpy as np
from PIL import Image

def extract_frame(video_path, timestamp, width, height):
    cmd = [
        "ffmpeg", "-y", "-ss", str(timestamp),
        "-i", video_path, "-vframes", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "pipe:1"
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width, 3)

def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "/workspaces/dorea-workspace/footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4"

    # Probe video
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
        capture_output=True, text=True
    )
    import json
    streams = json.loads(probe.stdout)["streams"]
    vs = [s for s in streams if s["codec_type"] == "video"][0]
    width, height = int(vs["width"]), int(vs["height"])
    print(f"Video: {width}x{height}")

    # Extract frame 0 at full resolution
    frame = extract_frame(video_path, 0.0, width, height)
    print(f"Extracted frame: {frame.shape}")

    # Load depth model
    sys.path.insert(0, "/workspaces/dorea-workspace/repos/dorea/python")
    from dorea_inference.depth_anything import DepthAnythingInference

    print("Loading Depth Anything V2...")
    model = DepthAnythingInference(device="cuda")

    # Run at 1518px (matching pipeline config)
    print("Running depth at max_size=1518...")
    depth = model.infer(frame, max_size=1518)
    print(f"Depth map shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]")

    # Normalize to 0-255 and save as grayscale PNG
    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)

    out_path = "/workspaces/dorea-workspace/working/depth_raw.png"
    Image.fromarray(depth_norm).save(out_path)
    print(f"Saved raw depth map: {out_path}")

    # Also save a colorized version using matplotlib colormap
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        colored = (cm.inferno(depth_norm / 255.0)[:, :, :3] * 255).astype(np.uint8)
        out_color = "/workspaces/dorea-workspace/working/depth_colored.png"
        Image.fromarray(colored).save(out_color)
        print(f"Saved colored depth map: {out_color}")
    except ImportError:
        print("matplotlib not available — skipping colored depth map")

if __name__ == "__main__":
    main()
