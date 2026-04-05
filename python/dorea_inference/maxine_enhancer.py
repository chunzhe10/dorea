"""NVIDIA Maxine VFX SDK wrapper for AI enhancement (super-resolution + artifact reduction).

Requires the nvvfx package (NVIDIA Maxine Video Effects SDK Python bindings).
Install separately from NGC — not bundled with dorea. See docs/guides/maxine-setup.md.

Set DOREA_MAXINE_MOCK=1 to enable mock mode for CI testing without the SDK.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

_MOCK_MODE = os.environ.get("DOREA_MAXINE_MOCK", "") == "1"

# Attempt nvvfx import (unless mock mode)
_nvvfx = None
if not _MOCK_MODE:
    try:
        import nvvfx as _nvvfx
    except ImportError:
        _nvvfx = None


def _log(msg: str) -> None:
    print(f"[dorea-inference] {msg}", file=sys.stderr, flush=True)


class MaxineEnhancer:
    """AI enhancement via Maxine VideoSuperRes.

    If DOREA_MAXINE_MOCK=1 is set, all enhance() calls return the input unchanged.
    """

    def __init__(self, upscale_factor: int = 2) -> None:
        self.upscale_factor = upscale_factor
        self._mock = _MOCK_MODE
        self._total_count = 0
        self._sr_effect = None

        if self._mock:
            _log("Maxine mock mode enabled (DOREA_MAXINE_MOCK=1)")
            return

        if _nvvfx is None:
            raise RuntimeError(
                "Maxine SDK not found. Install from NGC (nvidia-maxine-vfx-sdk): "
                "see docs/guides/maxine-setup.md, or unset --maxine"
            )

        _log(f"Maxine VideoSuperRes initialized (upscale_factor={upscale_factor})")

    def _init_effects(self, width: int, height: int) -> None:
        """Lazily initialize Maxine VideoSuperRes with known input dimensions."""
        out_w = width * self.upscale_factor
        out_h = height * self.upscale_factor

        self._sr_effect = _nvvfx.VideoSuperRes()
        self._sr_effect.output_width = out_w
        self._sr_effect.output_height = out_h
        self._sr_effect.load()

        _log(f"Maxine VideoSuperRes loaded: {width}x{height}→{out_w}x{out_h}")

    def enhance(
        self,
        rgb_u8: np.ndarray,
        width: int,
        height: int,
        artifact_reduce: bool = True,
    ) -> np.ndarray:
        """Enhance a single RGB u8 frame via VideoSuperRes. Returns RGB u8 at original resolution.

        Raises on failure — callers must not swallow errors.
        """
        if self._mock:
            self._total_count += 1
            return rgb_u8

        return self._enhance_impl(rgb_u8, width, height)

    def _enhance_impl(
        self,
        rgb_u8: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Internal enhancement via VideoSuperRes — exceptions propagate."""
        import torch

        if self._sr_effect is None:
            self._init_effects(width, height)

        # nvvfx SDK expects: (3, H, W) RGB float32 in [0, 1] on CUDA
        rgb_hwc = rgb_u8.reshape(height, width, 3).astype(np.float32) / 255.0
        rgb_chw = np.ascontiguousarray(np.transpose(rgb_hwc, (2, 0, 1)))  # (3, H, W)
        tensor = torch.from_numpy(rgb_chw).cuda()

        # Super-resolution → returns VideoSuperResOutput with DLPack capsule
        output = self._sr_effect.run(tensor)

        # DLPack capsule → torch tensor, clone before next call invalidates it
        upscaled = torch.from_dlpack(output.image).clone()  # (3, H', W') float32 [0, 1]

        # Convert to uint8 RGB (H', W', 3)
        upscaled_hwc = upscaled.permute(1, 2, 0).cpu().numpy()
        upscaled_rgb = (np.clip(upscaled_hwc, 0.0, 1.0) * 255).astype(np.uint8)

        # Downsample back to original resolution
        result = cv2.resize(
            upscaled_rgb, (width, height), interpolation=cv2.INTER_AREA,
        )

        self._total_count += 1
        return result

    def stats(self) -> str:
        """Return summary string for shutdown logging."""
        if self._total_count == 0:
            return "Maxine: no frames processed"
        return f"Maxine: enhanced {self._total_count} frames"
