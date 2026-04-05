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

import numpy as np

from . import __version__
from .protocol import (
    PongResponse,
    RauneResult,
    DepthResult,
    DepthBatchResult,
    RauneDepthBatchResult,
    EnhanceResult,
    ErrorResponse,
    OkResponse,
    decode_png,
    decode_raw_rgb,
    encode_raw_rgb,
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
    p.add_argument("--maxine", action="store_true",
                   help="Enable Maxine enhancement (requires nvvfx SDK or DOREA_MAXINE_MOCK=1)")
    p.add_argument("--maxine-upscale-factor", type=int, default=2,
                   help="Maxine super-resolution upscale factor (default: 2)")
    p.add_argument("--no-maxine-artifact-reduction", action="store_true",
                   help="Disable artifact reduction before upscale")
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
        from .raune_net import RauneNetInference
        raune_model = RauneNetInference(
            weights_path=args.raune_weights,
            device=device,
            raune_models_dir=args.raune_models_dir,
        )
        print("[dorea-inference] RAUNE-Net loaded", file=sys.stderr, flush=True)

    if not args.no_depth:
        from .depth_anything import DepthAnythingInference
        depth_model = DepthAnythingInference(
            model_path=args.depth_model,
            device=device,
        )
        print("[dorea-inference] Depth Anything V2 loaded", file=sys.stderr, flush=True)

    maxine_enhancer = None
    if args.maxine:
        from .maxine_enhancer import MaxineEnhancer
        maxine_enhancer = MaxineEnhancer(upscale_factor=args.maxine_upscale_factor)
        print(
            f"[dorea-inference] Maxine enhancer loaded "
            f"(upscale_factor={args.maxine_upscale_factor}, "
            f"artifact_reduction={not args.no_maxine_artifact_reduction})",
            file=sys.stderr, flush=True,
        )

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
                max_size = int(req.get("max_size", 1080))
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

                import torch

                items = req.get("items", [])
                imgs = []
                for item in items:
                    fmt = item.get("format", "png")
                    if fmt == "raw_rgb":
                        imgs.append(decode_raw_rgb(item["image_b64"], int(item["width"]), int(item["height"])))
                    else:
                        imgs.append(decode_png(item["image_b64"]))

                raune_max = int(items[0].get("raune_max_size", 1080)) if items else 1080
                depth_max = int(items[0].get("depth_max_size", 518)) if items else 518

                # Sub-batch to avoid GPU OOM: process in smaller chunks even if the request is larger
                # RTX 3060 6GB: safe limit ~4-6 images per batch at 1024p
                sub_batch_size = 4

                # Process RAUNE + Depth in sub-batches, accumulate results
                all_enhanced_np = []
                all_depths = []
                enh_w, enh_h = 0, 0

                for batch_start in range(0, len(imgs), sub_batch_size):
                    batch_end = min(batch_start + sub_batch_size, len(imgs))
                    imgs_chunk = imgs[batch_start:batch_end]

                    # RAUNE → enhanced tensors stay on GPU
                    enhanced_batch, enh_w, enh_h = raune_model.infer_batch_gpu(imgs_chunk, max_size=raune_max)

                    # Maxine upscale (optional) — insert between RAUNE and Depth
                    enable_maxine = req.get("enable_maxine", False)
                    if enable_maxine:
                        if maxine_enhancer is None:
                            raise RuntimeError(
                                "Maxine upscaling requested but enhancer failed to load — "
                                "check NVIDIA CUDA, VFX SDK, and dependencies"
                            )
                        # Convert batch to uint8 RGB for Maxine (dtoh)
                        enhanced_np = (
                            enhanced_batch.permute(0, 2, 3, 1).cpu().numpy() * 255
                        ).astype("uint8")  # (N, H, W, 3)

                        # Frame-by-frame Maxine enhance (single-frame API) — propagate errors
                        for i in range(enhanced_np.shape[0]):
                            # Call _enhance_impl directly to avoid silent error swallowing
                            try:
                                enhanced_np[i] = maxine_enhancer._enhance_impl(
                                    enhanced_np[i],
                                    width=enh_w,
                                    height=enh_h,
                                )
                            except Exception as e:
                                raise RuntimeError(
                                    f"Maxine enhance failed on frame {batch_start + i}/{len(imgs)}: {e}"
                                ) from e

                        # Convert back to float32 tensor on GPU (htod)
                        enhanced_batch = (
                            torch.from_numpy(enhanced_np.transpose(0, 3, 1, 2) / 255.0)
                            .float()
                            .cuda()
                        )
                        # Note: enh_w, enh_h remain unchanged (Maxine returns same resolution)

                    # Depth on enhanced tensors — no dtoh between models
                    depth_maps = depth_model.infer_batch_from_tensors(enhanced_batch, depth_max_size=depth_max)
                    all_depths.extend(depth_maps)

                    # dtoh enhanced frames for output
                    enhanced_np = (
                        enhanced_batch.permute(0, 2, 3, 1).cpu().numpy() * 255
                    ).astype("uint8")  # (N, H, W, 3)
                    all_enhanced_np.append(enhanced_np)

                    # Free GPU memory before next iteration
                    del enhanced_batch
                    torch.cuda.empty_cache()

                # Concatenate all sub-batches
                enhanced_np = np.concatenate(all_enhanced_np, axis=0)

                results = []
                for i, (item, depth) in enumerate(zip(items, all_depths)):
                    depth_result = DepthResult.from_array(item.get("id"), depth)
                    results.append({
                        "id": item.get("id"),
                        "image_b64": encode_png(enhanced_np[i]),
                        "enhanced_width": int(enh_w),
                        "enhanced_height": int(enh_h),
                        "depth_f32_b64": depth_result.depth_f32_b64,
                        "depth_width": depth_result.width,
                        "depth_height": depth_result.height,
                    })
                resp = RauneDepthBatchResult(results=results)
            elif req_type == "enhance":
                fmt = req.get("format", "raw_rgb")
                if fmt == "raw_rgb":
                    img = decode_raw_rgb(req["image_b64"], int(req["width"]), int(req["height"]))
                else:
                    img = decode_png(req["image_b64"])
                if maxine_enhancer is None:
                    # Maxine not available — return original frame unchanged
                    resp = EnhanceResult.from_array(req_id, img)
                else:
                    enhanced = maxine_enhancer.enhance(
                        img,
                        width=int(req["width"]),
                        height=int(req["height"]),
                        artifact_reduce=not req.get("no_artifact_reduce", False),
                    )
                    resp = EnhanceResult.from_array(req_id, enhanced)
            elif req_type == "unload_maxine":
                maxine_enhancer = None
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                print("[dorea-inference] Maxine unloaded — VRAM freed", file=sys.stderr, flush=True)
                resp = OkResponse()

            elif req_type == "load_raune":
                # Unload existing model first to avoid OOM on 6 GB VRAM budget
                if raune_model is not None:
                    del raune_model
                    raune_model = None
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
                from .raune_net import RauneNetInference
                raune_model = RauneNetInference(
                    weights_path=req.get("weights"),
                    device=device,
                    raune_models_dir=req.get("models_dir"),
                )
                print("[dorea-inference] RAUNE-Net loaded (on demand)", file=sys.stderr, flush=True)
                resp = OkResponse()

            elif req_type == "load_depth":
                # Unload existing model first to avoid OOM on 6 GB VRAM budget
                if depth_model is not None:
                    del depth_model
                    depth_model = None
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
                from .depth_anything import DepthAnythingInference
                depth_model = DepthAnythingInference(
                    model_path=req.get("model_path"),
                    device=device,
                )
                print("[dorea-inference] Depth Anything V2 loaded (on demand)", file=sys.stderr, flush=True)
                resp = OkResponse()

            elif req_type == "shutdown":
                if maxine_enhancer is not None:
                    stats = maxine_enhancer.stats()
                    print(f"[dorea-inference] {stats}", file=sys.stderr, flush=True)
                    if maxine_enhancer._total_count == 0:
                        raise RuntimeError(
                            "Maxine was enabled but processed 0 frames — "
                            "this indicates Maxine was never called during inference"
                        )
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
