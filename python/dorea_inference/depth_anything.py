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
