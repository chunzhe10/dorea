"""NVIDIA Maxine VFX SDK wrapper for AI enhancement (super-resolution + artifact reduction).

Requires the nvvfx package (NVIDIA Maxine Video Effects SDK Python bindings).
Install separately from NGC — not bundled with dorea. See docs/guides/maxine-setup.md.

Set DOREA_MAXINE_MOCK=1 to enable mock mode for CI testing without the SDK.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("dorea-inference")

_MOCK_MODE = os.environ.get("DOREA_MAXINE_MOCK", "") == "1"

# Attempt nvvfx import (unless mock mode)
_nvvfx = None
if not _MOCK_MODE:
    try:
        import nvvfx as _nvvfx
    except ImportError:
        _nvvfx = None


class MaxineEnhancer:
    """AI enhancement via Maxine VideoSuperRes + ArtifactReduction.

    If DOREA_MAXINE_MOCK=1 is set, all enhance() calls return the input unchanged.
    """

    def __init__(self, upscale_factor: int = 2) -> None:
        self.upscale_factor = upscale_factor
        self._mock = _MOCK_MODE
        self._passthrough_count = 0
        self._total_count = 0
        self._sr_effect = None
        self._ar_effect = None

        if self._mock:
            log.info("Maxine mock mode enabled (DOREA_MAXINE_MOCK=1)")
            return

        if _nvvfx is None:
            raise RuntimeError(
                "Maxine SDK not found. Install from NGC (nvidia-maxine-vfx-sdk): "
                "see docs/guides/maxine-setup.md, or unset --maxine"
            )

        log.info("Maxine VideoSuperRes initialized (upscale_factor=%d)", upscale_factor)
        # Effects are initialized lazily on first enhance() call because we need
        # the input dimensions to configure the output size.

    def _init_effects(self, width: int, height: int) -> None:
        """Lazily initialize Maxine VideoSuperRes with known input dimensions."""
        import torch

        out_w = width * self.upscale_factor
        out_h = height * self.upscale_factor

        self._sr_effect = _nvvfx.VideoSuperRes()
        # Set output dimensions as properties (not constructor args)
        self._sr_effect.output_width = out_w
        self._sr_effect.output_height = out_h
        stream = torch.cuda.current_stream()
        self._sr_effect.set_cuda_stream(stream.cuda_stream)
        self._sr_effect.load()

        log.info(
            "Maxine VideoSuperRes loaded: %dx%d→%dx%d",
            width, height, out_w, out_h,
        )

    def enhance(
        self,
        rgb_u8: np.ndarray,
        width: int,
        height: int,
        artifact_reduce: bool = True,
    ) -> np.ndarray:
        """Enhance a single RGB u8 frame via VideoSuperRes. Returns RGB u8 at original resolution.

        On any failure, logs the error and returns the original frame unchanged.
        Note: artifact_reduce parameter is ignored (ArtifactReduction not available in nvvfx SDK).
        """
        self._total_count += 1

        if self._mock:
            return rgb_u8

        try:
            return self._enhance_impl(rgb_u8, width, height)
        except Exception as e:
            self._passthrough_count += 1
            log.warning("Maxine enhance failed (frame passthrough): %s", e)
            return rgb_u8

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

        # RGB → BGR (Maxine expects BGR interleaved, float32, 3 channels)
        bgr = cv2.cvtColor(rgb_u8.reshape(height, width, 3), cv2.COLOR_RGB2BGR)

        # Convert uint8 → float32, then (H, W, 3) → (3, H, W) for Maxine SDK
        bgr_f32 = bgr.astype("float32")
        bgr_chw = np.transpose(bgr_f32, (2, 0, 1))  # (3, H, W)
        tensor = torch.from_numpy(bgr_chw).cuda()

        # Super-resolution → upscaled intermediate
        tensor = self._sr_effect.run(tensor)

        # Download and convert (3, H', W') → (H', W', 3)
        upscaled_chw = tensor.cpu().numpy()
        upscaled_bgr = np.transpose(upscaled_chw, (1, 2, 0))  # (H', W', 3)

        # Ensure uint8 before cv2 operations
        if upscaled_bgr.dtype != np.uint8:
            upscaled_bgr = (np.clip(upscaled_bgr, 0, 255)).astype(np.uint8)

        downscaled_bgr = cv2.resize(
            upscaled_bgr, (width, height), interpolation=cv2.INTER_AREA,
        )

        # BGR → RGB, ensure uint8 output
        result = cv2.cvtColor(downscaled_bgr, cv2.COLOR_BGR2RGB)

        if result.dtype != np.uint8:
            result = (np.clip(result, 0, 255)).astype(np.uint8)

        return result

    def stats(self) -> str:
        """Return summary string for shutdown logging."""
        if self._total_count == 0:
            return "Maxine: no frames processed"
        return (
            f"Maxine: enhanced {self._total_count} frames"
            + (f", {self._passthrough_count} passthrough failures"
               if self._passthrough_count > 0 else "")
        )
