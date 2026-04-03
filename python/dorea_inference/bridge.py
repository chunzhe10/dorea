"""PyO3-callable entry points for Rust-Python inference bridge.

Rust calls these functions via PyO3 embedding. The TensorGuard class
prevents Python's GC from reclaiming GPU tensors while Rust holds
their device pointers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class TensorGuard:
    """Prevent GC of a GPU tensor while Rust holds its device pointer.

    Rust holds a Py<TensorGuard> which prevents Python from collecting
    this object. Call release() explicitly when Rust is done with the pointer.
    """

    def __init__(self, tensor: "torch.Tensor") -> None:
        self.tensor = tensor
        self.data_ptr = tensor.data_ptr()
        self.numel = tensor.numel()
        self.shape = tuple(tensor.shape)
        self.dtype = str(tensor.dtype)

    def release(self) -> None:
        """Explicitly release the tensor. Called by Rust when done."""
        self.tensor = None
        self.data_ptr = 0
        self.numel = 0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_depth_model = None
_raune_model = None


def load_depth_model(model_path: Optional[str] = None, device: str = "cuda") -> None:
    """Load the Depth Anything V2 model. Called once at init."""
    global _depth_model
    from .depth_anything import DepthAnythingInference
    _depth_model = DepthAnythingInference(model_path=model_path, device=device)


def load_raune_model(
    weights_path: Optional[str] = None,
    device: str = "cuda",
    raune_models_dir: Optional[str] = None,
) -> None:
    """Load the RAUNE-Net model. Called once at init."""
    global _raune_model
    from .raune_net import RauneNetInference
    _raune_model = RauneNetInference(
        weights_path=weights_path, device=device, raune_models_dir=raune_models_dir,
    )


# ---------------------------------------------------------------------------
# GPU inference (returns TensorGuard with on-device tensor)
# ---------------------------------------------------------------------------

def run_depth_gpu(frame_rgb: np.ndarray, max_size: int = 518) -> TensorGuard:
    """Run depth inference, return TensorGuard holding the on-device result."""
    if _depth_model is None:
        raise RuntimeError("Depth model not loaded — call load_depth_model() first")
    tensor = _depth_model.infer_gpu(frame_rgb, max_size=max_size)
    return TensorGuard(tensor)


def run_raune_gpu(frame_rgb: np.ndarray, max_size: int = 1024) -> TensorGuard:
    """Run RAUNE-Net inference, return TensorGuard holding the on-device result."""
    if _raune_model is None:
        raise RuntimeError("RAUNE-Net model not loaded — call load_raune_model() first")
    tensor = _raune_model.infer_gpu(frame_rgb, max_size=max_size)
    return TensorGuard(tensor)


# ---------------------------------------------------------------------------
# CPU inference (returns numpy arrays — fallback path)
# ---------------------------------------------------------------------------

def run_depth_cpu(frame_rgb: np.ndarray, max_size: int = 518) -> np.ndarray:
    """Run depth inference on CPU, return numpy array."""
    if _depth_model is None:
        raise RuntimeError("Depth model not loaded — call load_depth_model() first")
    return _depth_model.infer(frame_rgb, max_size=max_size)


def run_depth_batch_cpu(imgs: "list[np.ndarray]", max_size: int = 518) -> "list[np.ndarray]":
    """Run batch depth inference, returning list of numpy arrays."""
    if _depth_model is None:
        raise RuntimeError("Depth model not loaded — call load_depth_model() first")
    return _depth_model.infer_batch(imgs, max_size=max_size)


def run_raune_cpu(frame_rgb: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """Run RAUNE-Net on CPU, return numpy array."""
    if _raune_model is None:
        raise RuntimeError("RAUNE-Net model not loaded — call load_raune_model() first")
    return _raune_model.infer(frame_rgb, max_size=max_size)


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

def unload_models() -> None:
    """Release model references so they can be garbage-collected."""
    global _depth_model, _raune_model
    _depth_model = None
    _raune_model = None


# ---------------------------------------------------------------------------
# VRAM query (called by Rust for adaptive batching)
# ---------------------------------------------------------------------------

def vram_free_bytes() -> int:
    """Return free VRAM in bytes after flushing PyTorch's cache.

    Uses torch.cuda.mem_get_info() which accounts for the caching allocator.
    """
    import torch
    if not torch.cuda.is_available():
        return 0
    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info()
    return free
