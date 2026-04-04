"""Depth Anything V2 Small inference wrapper.

Supports two loading modes:
1. HuggingFace hub: depth-anything/Depth-Anything-V2-Small-hf (requires internet)
2. Local weights: path to .pth checkpoint (model architecture loaded from transformers)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


# __file__ is python/dorea_inference/depth_anything.py → parents[4] = workspace root
_DEFAULT_DEPTH_MODEL = Path(__file__).parents[4] / "models" / "depth_anything_v2_small"

# ViT-S patch size; input resolution must be multiple of this.
_PATCH_SIZE = 14


def _round_to_patch(size: int, patch: int = _PATCH_SIZE) -> int:
    """Round size down to nearest multiple of patch size (min 1 patch)."""
    return max(patch, (size // patch) * patch)


def _resize_for_depth(
    img: "PIL.Image.Image", max_size: int
) -> "PIL.Image.Image":
    """Resize image so max(W,H) ≤ max_size and both dims are multiples of _PATCH_SIZE."""
    from PIL import Image as _Image
    w, h = img.size
    scale = min(max_size / max(w, h), 1.0)
    tw = _round_to_patch(int(round(w * scale)))
    th = _round_to_patch(int(round(h * scale)))
    return img.resize((tw, th), _Image.Resampling.BILINEAR)


class DepthAnythingInference:
    """Run Depth Anything V2 Small for monocular depth estimation."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        import torch
        from transformers import AutoModelForDepthEstimation

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False. "
                "Cannot run depth inference on CPU — GPU is required for dorea grade."
            )

        self.device = torch.device(device)

        path = Path(model_path) if model_path else _DEFAULT_DEPTH_MODEL
        model_id_or_path = str(path)
        has_local = path.is_dir() and any(
            (path / f).exists() for f in ("config.json", "pytorch_model.bin", "model.safetensors")
        )
        if not has_local:
            import sys
            print(
                f"[dorea_inference] WARNING: local depth model not found at {path}; "
                "falling back to HuggingFace hub download (requires internet access). "
                "Pass --depth-model to suppress this.",
                file=sys.stderr,
            )
            model_id_or_path = "depth-anything/Depth-Anything-V2-Small-hf"

        self.model = AutoModelForDepthEstimation.from_pretrained(model_id_or_path)
        self.model = self.model.to(self.device)
        self.model.eval()

    # ImageNet normalization constants (Depth Anything V2 uses these)
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def infer(self, img_rgb: np.ndarray, max_size: int = 518) -> np.ndarray:
        """Run depth estimation on uint8 HxWx3 RGB image.

        Returns float32 HxW depth map normalized to [0, 1] at inference resolution.
        """
        import torch
        from PIL import Image as _Image

        pil = _Image.fromarray(img_rgb)
        capped = _resize_for_depth(pil, max_size)

        # Direct tensor construction — bypass AutoImageProcessor
        arr = np.array(capped).astype(np.float32) / 255.0
        arr = (arr - self._MEAN) / self._STD
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values=tensor)

        depth = outputs.predicted_depth.squeeze(0).cpu().numpy()

        d_min, d_max = float(depth.min()), float(depth.max())
        if d_max - d_min < 1e-6:
            depth = np.zeros_like(depth, dtype=np.float32)
        else:
            depth = ((depth - d_min) / (d_max - d_min)).astype(np.float32)

        return depth

    def infer_batch(self, imgs: "list[np.ndarray]", max_size: int = 518) -> "list[np.ndarray]":
        """Run depth estimation on a batch of uint8 HxWx3 RGB images.

        All images must have the same source dimensions (guaranteed when called
        with proxy-size keyframes from a single video). Stacks into one
        (N, 3, H, W) forward pass. Each output normalized independently to [0,1].

        Returns list of (H, W) float32 depth maps at inference resolution.
        Falls back to sequential infer() if images have different post-resize dims.
        """
        if not imgs:
            return []

        import torch
        from PIL import Image as _Image

        pils = [_Image.fromarray(img) for img in imgs]
        resized = [_resize_for_depth(pil, max_size) for pil in pils]

        dims = set(r.size for r in resized)  # (tw, th) tuples
        if len(dims) > 1:
            import sys
            print(
                f"[dorea_inference] WARNING: infer_batch: images have different post-resize "
                f"dims {dims}; falling back to sequential inference (performance penalty)",
                file=sys.stderr,
            )
            return [self.infer(img, max_size) for img in imgs]

        arrays = []
        for r in resized:
            arr = np.array(r).astype(np.float32) / 255.0
            arr = (arr - self._MEAN) / self._STD
            arrays.append(torch.from_numpy(arr).permute(2, 0, 1))

        batch = torch.stack(arrays)
        if self.device.type == "cuda":
            batch = batch.pin_memory().to(self.device, non_blocking=True)
        else:
            batch = batch.to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values=batch)

        depths = outputs.predicted_depth.cpu().numpy()  # (N, H, W)

        result = []
        for i in range(len(imgs)):
            depth = depths[i]
            d_min, d_max = float(depth.min()), float(depth.max())
            if d_max - d_min < 1e-6:
                depth = np.zeros_like(depth, dtype=np.float32)
            else:
                depth = ((depth - d_min) / (d_max - d_min)).astype(np.float32)
            result.append(depth)

        return result

    def infer_gpu(self, img_rgb: np.ndarray, max_size: int = 518) -> "torch.Tensor":
        """Run depth estimation, return on-device f32 tensor (not copied to CPU).

        Returns a 2D float32 CUDA tensor normalized to [0, 1] at inference resolution.
        The caller must keep a reference to prevent GC.
        """
        import torch
        from PIL import Image as _Image

        pil = _Image.fromarray(img_rgb)
        capped = _resize_for_depth(pil, max_size)

        # Direct tensor construction — bypass AutoImageProcessor
        arr = np.array(capped).astype(np.float32) / 255.0
        arr = (arr - self._MEAN) / self._STD
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values=tensor)

        depth = outputs.predicted_depth.squeeze(0)  # stays on device

        d_min = float(depth.min())
        d_max = float(depth.max())
        if d_max - d_min < 1e-6:
            depth = torch.zeros_like(depth)
        else:
            depth = (depth - d_min) / (d_max - d_min)

        return depth.to(torch.float32).contiguous()

    def infer_batch_from_tensors(
        self,
        enhanced: "torch.Tensor",
        depth_max_size: int = 518,
    ) -> "list[np.ndarray]":
        """Run depth estimation on a batch of RAUNE-output GPU tensors.

        Args:
            enhanced: (N, 3, H_r, W_r) float32 in [0, 1], on self.device.
                      This is the direct output of RauneNetInference.infer_batch_gpu().
            depth_max_size: max long-edge for depth model (default 518).

        Returns list of (H_d, W_d) float32 depth maps normalized to [0, 1].
        The resize and re-normalisation happen on-device — no dtoh between models.
        """
        if enhanced.device != self.device:
            raise ValueError(
                f"infer_batch_from_tensors: enhanced tensor is on {enhanced.device} "
                f"but model is on {self.device}. Caller must keep tensors on the same device."
            )

        import torch
        import torch.nn.functional as F

        N, _, H_r, W_r = enhanced.shape

        # GPU resize: keep aspect, snap to multiples of _PATCH_SIZE
        scale = min(depth_max_size / max(H_r, W_r), 1.0)
        H_d = max(_PATCH_SIZE, int(H_r * scale) // _PATCH_SIZE * _PATCH_SIZE)
        W_d = max(_PATCH_SIZE, int(W_r * scale) // _PATCH_SIZE * _PATCH_SIZE)

        resized = F.interpolate(
            enhanced, size=(H_d, W_d), mode="bilinear", align_corners=False
        )  # (N, 3, H_d, W_d) in [0, 1]

        # Re-normalise [0,1] → ImageNet stats Depth Anything expects
        mean = torch.tensor(self._MEAN, dtype=torch.float32, device=enhanced.device).view(1, 3, 1, 1)
        std  = torch.tensor(self._STD,  dtype=torch.float32, device=enhanced.device).view(1, 3, 1, 1)
        depth_input = (resized - mean) / std  # (N, 3, H_d, W_d)

        with torch.no_grad():
            outputs = self.model(pixel_values=depth_input)

        depths_raw = outputs.predicted_depth.cpu().numpy()  # (N, H_d, W_d)

        result = []
        for i in range(N):
            d = depths_raw[i]
            d_min, d_max = float(d.min()), float(d.max())
            if d_max - d_min < 1e-6:
                result.append(np.zeros_like(d, dtype=np.float32))
            else:
                result.append(((d - d_min) / (d_max - d_min)).astype(np.float32))
        return result
