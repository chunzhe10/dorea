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
_maxine_model = None


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


def load_maxine_model(upscale_factor: int = 2) -> None:
    """Load the Maxine enhancer. Always CUDA-backed — no device selection."""
    global _maxine_model
    from .maxine_enhancer import MaxineEnhancer
    _maxine_model = MaxineEnhancer(upscale_factor=upscale_factor)


def unload_maxine() -> None:
    """Release Maxine model reference and free CUDA cache."""
    global _maxine_model
    _maxine_model = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_maxine(frame_rgb: np.ndarray, artifact_reduce: bool = True) -> np.ndarray:
    """Run Maxine enhancement (CUDA-backed), return same-resolution numpy uint8 array."""
    if _maxine_model is None:
        raise RuntimeError("Maxine model not loaded — call load_maxine_model() first")
    h, w = frame_rgb.shape[:2]
    return _maxine_model.enhance(frame_rgb, width=w, height=h, artifact_reduce=artifact_reduce)


# ---------------------------------------------------------------------------
# GPU inference (returns TensorGuard with on-device tensor)
# ---------------------------------------------------------------------------

def run_depth_gpu(frame_rgb: np.ndarray, max_size: int = 518) -> TensorGuard:
    """Run depth inference, return TensorGuard holding the on-device result."""
    if _depth_model is None:
        raise RuntimeError("Depth model not loaded — call load_depth_model() first")
    tensor = _depth_model.infer_gpu(frame_rgb, max_size=max_size)
    return TensorGuard(tensor)


def run_raune_gpu(frame_rgb: np.ndarray, max_size: int = 1080) -> TensorGuard:
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


def run_raune_cpu(frame_rgb: np.ndarray, max_size: int = 1080) -> np.ndarray:
    """Run RAUNE-Net on CPU, return numpy array."""
    if _raune_model is None:
        raise RuntimeError("RAUNE-Net model not loaded — call load_raune_model() first")
    return _raune_model.infer(frame_rgb, max_size=max_size)


def run_raune_depth_batch_cpu(
    imgs: "list[np.ndarray]",
    raune_max_size: int = 1080,
    depth_max_size: int = 518,
    enable_maxine: bool = False,
) -> "list[tuple[np.ndarray, np.ndarray]]":
    """Run fused RAUNE + Depth batch inference.

    For each image: RAUNE enhancement → (optional Maxine upscale) → Depth estimation.
    Returns list of (enhanced_u8, depth_f32) numpy array pairs.

    Uses sub-batching (4 frames) to fit within 6 GB VRAM.
    """
    import torch

    if _raune_model is None:
        raise RuntimeError("RAUNE-Net model not loaded — call load_raune_model() first")
    if _depth_model is None:
        raise RuntimeError("Depth model not loaded — call load_depth_model() first")

    sub_batch_size = 4
    results: list[tuple[np.ndarray, np.ndarray]] = []

    for batch_start in range(0, len(imgs), sub_batch_size):
        imgs_chunk = imgs[batch_start:batch_start + sub_batch_size]

        # RAUNE → enhanced tensors stay on GPU
        enhanced_batch, enh_w, enh_h = _raune_model.infer_batch_gpu(imgs_chunk, max_size=raune_max_size)

        # Maxine upscale (optional) — insert between RAUNE and Depth
        if enable_maxine and _maxine_model is not None:
            enhanced_np = (
                enhanced_batch.permute(0, 2, 3, 1).cpu().numpy() * 255
            ).astype("uint8")  # (N, H, W, 3)

            for i in range(enhanced_np.shape[0]):
                enhanced_np[i] = _maxine_model._enhance_impl(
                    enhanced_np[i], width=enh_w, height=enh_h,
                )

            enhanced_batch = (
                torch.from_numpy(enhanced_np.transpose(0, 3, 1, 2) / 255.0)
                .float()
                .cuda()
            )

        # Depth on ORIGINAL frames — RAUNE enhancement distorts depth estimation
        depth_maps = _depth_model.infer_batch(imgs_chunk, max_size=depth_max_size)

        # dtoh enhanced frames
        enhanced_np = (
            enhanced_batch.permute(0, 2, 3, 1).cpu().numpy() * 255
        ).astype("uint8")  # (N, H, W, 3)

        for i in range(len(imgs_chunk)):
            results.append((enhanced_np[i], depth_maps[i]))

        del enhanced_batch
        torch.cuda.empty_cache()

    return results


_yolo_seg_model = None


def load_yolo_seg_model(model_path: Optional[str] = None, device: str = "cuda") -> None:
    """Load YOLOv11n-seg for binary diver/water segmentation."""
    global _yolo_seg_model
    from .yolo_seg import YoloSegInference
    _yolo_seg_model = YoloSegInference(model_path=model_path, device=device)


def run_yolo_seg_batch_cpu(imgs: "list[np.ndarray]") -> "list[np.ndarray]":
    """Run YOLO-seg on a batch of frames, returning class masks."""
    if _yolo_seg_model is None:
        raise RuntimeError("YOLO-seg not loaded — call load_yolo_seg_model() first")
    return _yolo_seg_model.infer_batch(imgs)


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

def unload_models() -> None:
    """Release model references so they can be garbage-collected."""
    global _depth_model, _raune_model, _maxine_model, _yolo_seg_model
    _depth_model = None
    _raune_model = None
    _maxine_model = None
    _yolo_seg_model = None


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
