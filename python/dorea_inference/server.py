"""Dorea inference subprocess server.

Reads JSON-lines requests from stdin, writes JSON-lines responses to stdout.
See protocol.py for message format.

Usage:
    python -m dorea_inference.server [OPTIONS]

Options:
    --raune-weights PATH        Path to RAUNE-Net weights .pth
    --raune-models-dir PATH     Path to sea_thru_poc directory (contains models/raune_net.py)
    --depth-model PATH          Path to Depth Anything V2 model dir or HF id
    --device cpu|cuda           Compute device (default: cuda if available, else cpu)
    --no-raune                  Skip loading RAUNE-Net (depth only)
    --no-depth                  Skip loading Depth Anything (RAUNE only)
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Optional

from . import __version__
from .protocol import (
    PongResponse,
    RauneResult,
    DepthResult,
    DepthBatchResult,
    RauneDepthBatchResult,
    ErrorResponse,
    OkResponse,
    decode_png,
    decode_raw_rgb,
    encode_png,
)


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m dorea_inference.server",
        description="Dorea inference subprocess server",
    )
    p.add_argument("--raune-weights", default=None)
    p.add_argument("--raune-models-dir", default=None)
    p.add_argument("--depth-model", default=None)
    p.add_argument("--device", default=None, choices=["cpu", "cuda"])
    p.add_argument("--no-raune", action="store_true")
    p.add_argument("--no-depth", action="store_true")
    return p.parse_args(argv)


def _choose_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main(argv: Optional[list] = None) -> None:
    args = _parse_args(argv)
    device = _choose_device(args.device)

    # Write startup message to stderr (not stdout — stdout is the IPC channel)
    print(f"[dorea-inference] v{__version__} starting on device={device}", file=sys.stderr, flush=True)

    # Lazy-load models to avoid startup cost when not needed
    raune_model = None
    depth_model = None

    if not args.no_raune:
        try:
            from .raune_net import RauneNetInference
            raune_model = RauneNetInference(
                weights_path=args.raune_weights,
                device=device,
                raune_models_dir=args.raune_models_dir,
            )
            print("[dorea-inference] RAUNE-Net loaded", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[dorea-inference] WARNING: RAUNE-Net failed to load: {e}", file=sys.stderr, flush=True)

    if not args.no_depth:
        try:
            from .depth_anything import DepthAnythingInference
            depth_model = DepthAnythingInference(
                model_path=args.depth_model,
                device=device,
            )
            print("[dorea-inference] Depth Anything V2 loaded", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[dorea-inference] WARNING: Depth Anything V2 failed to load: {e}", file=sys.stderr, flush=True)

    print("[dorea-inference] ready", file=sys.stderr, flush=True)

    # Main request loop
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            resp = ErrorResponse(id=None, message=f"invalid JSON: {e}")
            try:
                print(json.dumps(resp.to_dict()), flush=True)
            except BrokenPipeError:
                break
            continue

        if not isinstance(req, dict):
            resp = ErrorResponse(id=None, message=f"expected JSON object, got {type(req).__name__}")
            try:
                print(json.dumps(resp.to_dict()), flush=True)
            except BrokenPipeError:
                break
            continue

        req_type = req.get("type", "")
        req_id = req.get("id")

        try:
            if req_type == "ping":
                resp = PongResponse(version=__version__)
            elif req_type == "raune":
                if raune_model is None:
                    raise RuntimeError("RAUNE-Net model not loaded (--no-raune or load failed)")
                fmt = req.get("format", "png")
                if fmt == "raw_rgb":
                    img = decode_raw_rgb(req["image_b64"], int(req["width"]), int(req["height"]))
                else:
                    img = decode_png(req["image_b64"])
                max_size = int(req.get("max_size", 1024))
                result = raune_model.infer(img, max_size=max_size)
                resp = RauneResult(
                    id=req_id,
                    image_b64=encode_png(result),
                    width=result.shape[1],
                    height=result.shape[0],
                )
            elif req_type == "depth":
                if depth_model is None:
                    raise RuntimeError("Depth Anything model not loaded (--no-depth or load failed)")
                fmt = req.get("format", "png")
                if fmt == "raw_rgb":
                    img = decode_raw_rgb(req["image_b64"], int(req["width"]), int(req["height"]))
                else:
                    img = decode_png(req["image_b64"])
                max_size = int(req.get("max_size", 518))
                depth = depth_model.infer(img, max_size=max_size)
                resp = DepthResult.from_array(req_id, depth)
            elif req_type == "depth_batch":
                if depth_model is None:
                    raise RuntimeError("Depth Anything model not loaded (--no-depth or load failed)")
                items = req.get("items", [])
                imgs = []
                for item in items:
                    fmt = item.get("format", "png")
                    if fmt == "raw_rgb":
                        imgs.append(decode_raw_rgb(item["image_b64"], int(item["width"]), int(item["height"])))
                    else:
                        imgs.append(decode_png(item["image_b64"]))
                max_size = int(items[0].get("max_size", 518)) if items else 518
                depths = depth_model.infer_batch(imgs, max_size=max_size)
                results = [
                    DepthResult.from_array(item.get("id"), depth).to_dict()
                    for item, depth in zip(items, depths)
                ]
                resp = DepthBatchResult(results=results)
            elif req_type == "raune_depth_batch":
                if raune_model is None:
                    raise RuntimeError("RAUNE-Net not loaded — pass --raune-weights")
                if depth_model is None:
                    raise RuntimeError("Depth Anything not loaded — pass --depth-model")

                items = req.get("items", [])
                imgs = []
                for item in items:
                    fmt = item.get("format", "png")
                    if fmt == "raw_rgb":
                        imgs.append(decode_raw_rgb(item["image_b64"], int(item["width"]), int(item["height"])))
                    else:
                        imgs.append(decode_png(item["image_b64"]))

                raune_max = int(items[0].get("raune_max_size", 1024)) if items else 1024
                depth_max = int(items[0].get("depth_max_size", 518)) if items else 518

                # RAUNE → enhanced tensors stay on GPU
                enhanced_batch, enh_w, enh_h = raune_model.infer_batch_gpu(imgs, max_size=raune_max)

                # Depth on enhanced tensors — no dtoh between models
                depth_maps = depth_model.infer_batch_from_tensors(enhanced_batch, depth_max_size=depth_max)

                # dtoh enhanced frames for output
                enhanced_np = (
                    enhanced_batch.permute(0, 2, 3, 1).cpu().numpy() * 255
                ).astype("uint8")  # (N, H, W, 3)

                results = []
                for i, (item, depth) in enumerate(zip(items, depth_maps)):
                    depth_result = DepthResult.from_array(item.get("id"), depth)
                    results.append({
                        "id": item.get("id"),
                        "image_b64": encode_png(enhanced_np[i]),
                        "enhanced_width": int(enh_w),
                        "enhanced_height": int(enh_h),
                        **{k: v for k, v in depth_result.to_dict().items()
                           if k in ("depth_f32_b64", "depth_width", "depth_height", "type")},
                    })
                resp = RauneDepthBatchResult(results=results)
            elif req_type == "shutdown":
                resp = OkResponse()
                try:
                    print(json.dumps(resp.to_dict()), flush=True)
                except BrokenPipeError:
                    pass
                break
            else:
                resp = ErrorResponse(id=req_id, message=f"unknown request type: {req_type!r}")

        except Exception as e:
            tb = traceback.format_exc()
            print(f"[dorea-inference] ERROR on {req_type}: {e}\n{tb}", file=sys.stderr, flush=True)
            resp = ErrorResponse(id=req_id, message=str(e))

        try:
            print(json.dumps(resp.to_dict()), flush=True)
        except BrokenPipeError:
            break

    print("[dorea-inference] exiting", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
