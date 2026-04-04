"""RAUNE-Net inference wrapper.

Loads the RAUNE-Net model from the working/sea_thru_poc/models/ directory
(which must be on sys.path, or provided as a separate path argument).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# Default location relative to workspace root.
# __file__ is python/dorea_inference/raune_net.py → parents[4] = workspace root
_DEFAULT_RAUNE_MODELS_DIR = (
    Path(__file__).parents[4] / "working" / "sea_thru_poc" / "models" / "RAUNE-Net"
)
_DEFAULT_WEIGHTS = (
    Path(__file__).parents[4]
    / "working" / "sea_thru_poc" / "models" / "RAUNE-Net"
    / "pretrained" / "RAUNENet" / "test" / "weights_95.pth"
)


def _resize_maintain_aspect(
    img: "PIL.Image.Image", max_size: int
) -> Tuple["PIL.Image.Image", int, int]:
    """Resize image so max(width, height) <= max_size, maintain aspect ratio."""
    from PIL import Image as _Image
    w, h = img.size
    scale = min(max_size / max(w, h), 1.0)
    tw = int(round(w * scale))
    th = int(round(h * scale))
    resized = img.resize((tw, th), _Image.Resampling.BILINEAR)
    return resized, tw, th


class RauneNetInference:
    """Run RAUNE-Net underwater image enhancement."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cpu",
        raune_models_dir: Optional[str] = None,
    ) -> None:
        import torch

        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        # Add the RAUNE-Net directory to sys.path so we can import models.raune_net.
        # Accepts either the RAUNE-Net dir directly (has models/raune_net.py) or the
        # parent sea_thru_poc dir (auto-descends to models/RAUNE-Net/).
        given = Path(raune_models_dir) if raune_models_dir else _DEFAULT_RAUNE_MODELS_DIR
        if (given / "models" / "raune_net.py").exists():
            models_dir = given
        elif (given / "models" / "RAUNE-Net" / "models" / "raune_net.py").exists():
            models_dir = given / "models" / "RAUNE-Net"
        else:
            models_dir = given  # let the import fail with a clear error below

        if str(models_dir) not in sys.path:
            sys.path.insert(0, str(models_dir))

        try:
            from models.raune_net import RauneNet  # type: ignore
        except ImportError as e:
            raise ImportError(
                f"Could not import RauneNet from {models_dir}/models/raune_net.py: {e}\n"
                f"Pass --raune-models-dir pointing to the RAUNE-Net checkout directory "
                f"(the one that contains a models/raune_net.py file)."
            ) from e

        weights = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS
        if not weights.exists():
            raise FileNotFoundError(
                f"RAUNE-Net weights not found at {weights}. "
                "Pass --raune-weights to specify the .pth file."
            )

        self.model = RauneNet(input_nc=3, output_nc=3, n_blocks=30, n_down=2).to(self.device)
        state = torch.load(str(weights), map_location=self.device, weights_only=False)
        self.model.load_state_dict(state)
        self.model.eval()

    def infer(self, img_rgb: np.ndarray, max_size: int = 1024) -> np.ndarray:
        """Run RAUNE-Net on an uint8 HxWx3 RGB image.

        Returns uint8 HxWx3 RGB array at inference resolution (≤ max_size).
        """
        import torch
        import torchvision.transforms as transforms
        from PIL import Image as _Image

        pil = _Image.fromarray(img_rgb)
        resized, tw, th = _resize_maintain_aspect(pil, max_size)

        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        tensor = normalize(transforms.ToTensor()(resized)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)

        # De-normalize: model output in [-1, 1], convert to [0, 1]
        out = ((out.squeeze(0) + 1.0) / 2.0).clamp(0.0, 1.0)
        result = (out.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return result

    def infer_gpu(self, img_rgb: np.ndarray, max_size: int = 1024) -> "torch.Tensor":
        """Run RAUNE-Net, return on-device uint8 tensor (not copied to CPU).

        Returns a 3D uint8 CUDA tensor (HxWx3) at inference resolution.
        The caller must keep a reference to prevent GC.
        """
        import torch
        import torchvision.transforms as transforms
        from PIL import Image as _Image

        pil = _Image.fromarray(img_rgb)
        resized, tw, th = _resize_maintain_aspect(pil, max_size)

        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        tensor = normalize(transforms.ToTensor()(resized)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)

        # De-normalize: model output in [-1, 1], convert to [0, 255] uint8
        out = ((out.squeeze(0) + 1.0) / 2.0).clamp(0.0, 1.0)
        result = (out.permute(1, 2, 0) * 255).to(torch.uint8).contiguous()

        return result  # stays on device

    def infer_batch(self, imgs: "list[np.ndarray]", max_size: int = 1024) -> "list[np.ndarray]":
        """Run RAUNE-Net on a batch of uint8 HxWx3 RGB images.

        All images must resize to the same dims (guaranteed for frames from one video).
        Falls back to sequential infer() if post-resize dims differ.
        Returns list of uint8 HxWx3 arrays.
        """
        if not imgs:
            return []

        import torch
        import torchvision.transforms as transforms
        from PIL import Image as _Image

        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        tensors = []
        wh_set = set()
        for img in imgs:
            r, tw, th = _resize_maintain_aspect(_Image.fromarray(img), max_size)
            wh_set.add((tw, th))
            tensors.append(normalize(transforms.ToTensor()(r)))

        if len(wh_set) > 1:
            import sys
            print(
                f"[dorea_inference] WARNING: infer_batch(raune): mixed post-resize dims "
                f"{wh_set}; falling back to sequential",
                file=sys.stderr,
            )
            return [self.infer(img, max_size) for img in imgs]

        batch = torch.stack(tensors)  # (N, 3, H, W)
        if self.device.type == "cuda":
            batch = batch.pin_memory().to(self.device, non_blocking=True)
        else:
            batch = batch.to(self.device)

        with torch.no_grad():
            out = self.model(batch)  # (N, 3, H, W) in [-1, 1]

        out = ((out + 1.0) / 2.0).clamp(0.0, 1.0)
        out_np = out.cpu().numpy()  # (N, 3, H, W) float32
        return [(out_np[i].transpose(1, 2, 0) * 255).astype(np.uint8) for i in range(len(imgs))]

    def infer_batch_gpu(self, imgs: "list[np.ndarray]", max_size: int = 1024) -> "tuple[torch.Tensor, int, int]":
        """Run RAUNE-Net batch, returning enhanced tensors on device (no dtoh).

        Returns (batch_tensor, out_w, out_h) where batch_tensor is (N, 3, H, W)
        float32 in [0, 1], still on self.device. Caller must not let it be GC'd.
        Falls back to sequential per-image forward passes (results stacked on device) if dims differ.
        """
        import torch
        import torchvision.transforms as transforms
        from PIL import Image as _Image

        if not imgs:
            return torch.zeros(0, 3, 0, 0, device=self.device), 0, 0

        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        tensors = []
        wh_set = set()
        out_w, out_h = 0, 0
        for img in imgs:
            r, tw, th = _resize_maintain_aspect(_Image.fromarray(img), max_size)
            wh_set.add((tw, th))
            out_w, out_h = tw, th
            tensors.append(normalize(transforms.ToTensor()(r)))

        if len(wh_set) > 1:
            # Different sizes: run sequentially, collect results as GPU tensors
            results = []
            for img in imgs:
                r, tw, th = _resize_maintain_aspect(_Image.fromarray(img), max_size)
                t = normalize(transforms.ToTensor()(r)).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out = self.model(t)
                out = ((out.squeeze(0) + 1.0) / 2.0).clamp(0.0, 1.0)
                results.append(out)
            stacked = torch.stack(results)
            _, _, H, W = stacked.shape
            return stacked, W, H

        batch = torch.stack(tensors)
        if self.device.type == "cuda":
            batch = batch.pin_memory().to(self.device, non_blocking=True)
        else:
            batch = batch.to(self.device)

        with torch.no_grad():
            out = self.model(batch)  # (N, 3, H, W) in [-1, 1]

        out = ((out + 1.0) / 2.0).clamp(0.0, 1.0)  # [0, 1], still on device
        return out, out_w, out_h
