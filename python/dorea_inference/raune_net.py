"""RAUNE-Net inference wrapper.

Loads the RAUNE-Net model from the working/sea_thru_poc/models/ directory
(which must be on sys.path, or provided as a separate path argument).
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# Default location relative to workspace root.
_DEFAULT_RAUNE_MODELS_DIR = Path(__file__).parents[5] / "working" / "sea_thru_poc"
_DEFAULT_WEIGHTS = (
    Path(__file__).parents[5]
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
    resized = img.resize((tw, th), _Image.BILINEAR)
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

        # Add the sea_thru_poc directory to sys.path so we can import models.raune_net
        models_dir = Path(raune_models_dir) if raune_models_dir else _DEFAULT_RAUNE_MODELS_DIR
        if str(models_dir) not in sys.path:
            sys.path.insert(0, str(models_dir))

        try:
            from models.raune_net import RauneNet  # type: ignore
        except ImportError as e:
            raise ImportError(
                f"Could not import RauneNet from {models_dir}/models/raune_net.py: {e}\n"
                f"Ensure --raune-models-dir points to the sea_thru_poc directory."
            ) from e

        weights = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS
        if not weights.exists():
            raise FileNotFoundError(
                f"RAUNE-Net weights not found at {weights}. "
                "Pass --raune-weights to specify the .pth file."
            )

        self.model = RauneNet().to(self.device)
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
