#!/usr/bin/env python3
"""Investigate grading artifacts by dumping intermediate pipeline outputs.

Extracts frame 0 and shows:
1. Original frame
2. RAUNE-enhanced frame
3. Depth map overlaid on original
4. Per-zone depth visualization (which zone each pixel falls into)
5. Graded frame
6. Diff between original and graded (amplified)
"""
import sys, json, subprocess
import numpy as np
from PIL import Image

sys.path.insert(0, "/workspaces/dorea-workspace/repos/dorea/python")

VIDEO = "/workspaces/dorea-workspace/footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4"
GRADED = "/workspaces/dorea-workspace/footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s_graded.mp4"
OUT = "/workspaces/dorea-workspace/working"

def extract_frame(path, ts, w, h):
    cmd = ["ffmpeg", "-y", "-ss", str(ts), "-i", path, "-vframes", "1",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True)
    return np.frombuffer(r.stdout, dtype=np.uint8).reshape(h, w, 3)

def probe(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
                       capture_output=True, text=True)
    vs = [s for s in json.loads(r.stdout)["streams"] if s["codec_type"] == "video"][0]
    return int(vs["width"]), int(vs["height"])

def save(arr, name):
    path = f"{OUT}/{name}.png"
    Image.fromarray(arr).save(path)
    print(f"  Saved: {path}")

def main():
    w, h = probe(VIDEO)
    print(f"Video: {w}x{h}")

    # 1. Original frame 0
    print("Extracting original frame 0...")
    orig = extract_frame(VIDEO, 0.0, w, h)
    save(orig, "01_original")

    # 2. Graded frame 0
    print("Extracting graded frame 0...")
    graded = extract_frame(GRADED, 0.0, w, h)
    save(graded, "02_graded")

    # 3. Diff (amplified 5x)
    print("Computing diff...")
    diff = np.abs(orig.astype(np.int16) - graded.astype(np.int16)).astype(np.uint8)
    diff_amp = np.clip(diff.astype(np.uint16) * 5, 0, 255).astype(np.uint8)
    save(diff_amp, "03_diff_5x")

    # 4. Per-channel diff
    for ch, name in enumerate(["red", "green", "blue"]):
        ch_diff = np.abs(orig[:,:,ch].astype(np.int16) - graded[:,:,ch].astype(np.int16))
        ch_img = np.clip(ch_diff.astype(np.uint16) * 5, 0, 255).astype(np.uint8)
        save(ch_img, f"04_diff_{name}")

    # 5. Depth map at proxy resolution with zone boundaries
    print("Running depth estimation...")
    from dorea_inference.depth_anything import DepthAnythingInference
    model = DepthAnythingInference(device="cuda")
    depth = model.infer(orig, max_size=1518)
    dh, dw = depth.shape
    print(f"  Depth: {dw}x{dh}, range [{depth.min():.3f}, {depth.max():.3f}]")

    # Save raw depth
    depth_u8 = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
    save(depth_u8, "05_depth_raw")

    # 6. Zone visualization — show which of 8 zones each pixel falls into
    n_zones = 8
    # Compute adaptive zone boundaries (quantile-based, matching pipeline)
    flat = depth.flatten()
    boundaries = np.quantile(flat, np.linspace(0, 1, n_zones + 1))
    print(f"  Zone boundaries: {[f'{b:.3f}' for b in boundaries]}")

    zone_map = np.zeros_like(depth, dtype=np.uint8)
    for z in range(n_zones):
        mask = (depth >= boundaries[z]) & (depth < boundaries[z + 1])
        zone_map[mask] = int(255 * z / (n_zones - 1))
    # Last zone catches the upper boundary
    zone_map[depth >= boundaries[-1]] = 255
    save(zone_map, "06_zones")

    # 7. Zone boundaries overlaid — show edges between zones
    import cv2
    zone_edges = cv2.Canny(zone_map, 50, 150)
    # Upscale edges to full res for overlay
    zone_edges_full = cv2.resize(zone_edges, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay = orig.copy()
    overlay[zone_edges_full > 0] = [255, 0, 0]  # red edges
    save(overlay, "07_zone_edges_on_original")

    # 8. Histogram of depth values
    print(f"\nDepth histogram (8 bins):")
    for z in range(n_zones):
        count = np.sum((depth >= boundaries[z]) & (depth < boundaries[z + 1]))
        pct = count / depth.size * 100
        bar = "#" * int(pct / 2)
        print(f"  zone {z} [{boundaries[z]:.3f}-{boundaries[z+1]:.3f}]: {pct:5.1f}% {bar}")

    # 9. Check for banding — look at graded frame's color histogram
    print(f"\nGraded frame color statistics:")
    for ch, name in enumerate(["R", "G", "B"]):
        vals = graded[:,:,ch].flatten()
        unique = len(np.unique(vals))
        print(f"  {name}: mean={vals.mean():.1f}, std={vals.std():.1f}, unique_values={unique}/256")

    # 10. Check for macro-blocking — compute local variance in 8x8 blocks
    gray_graded = np.mean(graded.astype(np.float32), axis=2)
    bh, bw = h // 8, w // 8
    block_vars = np.zeros((bh, bw))
    for by in range(bh):
        for bx in range(bw):
            block = gray_graded[by*8:(by+1)*8, bx*8:(bx+1)*8]
            block_vars[by, bx] = block.var()

    low_var_pct = np.sum(block_vars < 1.0) / block_vars.size * 100
    print(f"\n8x8 block variance analysis:")
    print(f"  Blocks with variance < 1.0 (flat/banded): {low_var_pct:.1f}%")
    print(f"  Mean block variance: {block_vars.mean():.1f}")
    print(f"  Min/Max block variance: {block_vars.min():.2f} / {block_vars.max():.1f}")

    # Save block variance as heatmap
    bv_norm = ((block_vars - block_vars.min()) / (block_vars.max() - block_vars.min() + 1e-8) * 255).astype(np.uint8)
    bv_img = cv2.resize(bv_norm, (w, h), interpolation=cv2.INTER_NEAREST)
    save(bv_img, "08_block_variance")

    print(f"\nAll outputs in {OUT}/")

if __name__ == "__main__":
    main()
