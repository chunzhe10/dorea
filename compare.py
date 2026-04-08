#!/usr/bin/env python3
import subprocess, json, numpy as np
from PIL import Image

VIDEO = "/workspaces/dorea-workspace/footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4"
GRADED = "/workspaces/dorea-workspace/footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s_graded.mp4"
OUT = "/workspaces/dorea-workspace/working"

def probe(p):
    r = subprocess.run(["ffprobe","-v","quiet","-print_format","json","-show_streams",p], capture_output=True, text=True)
    vs = [s for s in json.loads(r.stdout)["streams"] if s["codec_type"]=="video"][0]
    return int(vs["width"]), int(vs["height"])

def extract(p, ts, w, h):
    cmd = ["ffmpeg","-y","-ss",str(ts),"-i",p,"-vframes","1","-f","rawvideo","-pix_fmt","rgb24","-s",f"{w}x{h}","pipe:1"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, dtype=np.uint8).reshape(h,w,3)

w, h = probe(VIDEO)
graded = extract(GRADED, 0.0, w, h)
Image.fromarray(graded).save(f"{OUT}/09_graded_32zones_smooth.png")
print(f"Saved {OUT}/09_graded_32zones_smooth.png")
