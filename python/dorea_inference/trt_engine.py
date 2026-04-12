"""TensorRT engine wrapper for RAUNE-Net inference.

Handles engine building from ONNX, disk caching with automatic invalidation,
and zero-copy inference with PyTorch CUDA tensors.
"""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

try:
    import tensorrt as trt
except ImportError:
    trt = None


def _require_tensorrt():
    if trt is None:
        raise RuntimeError(
            "tensorrt is required for --tensorrt mode. "
            "Install with: pip install tensorrt-cu12"
        )


def _streaming_sha256(path: str) -> str:
    """SHA-256 hash of a file using streaming reads (constant memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class RauneTRTEngine:
    """TensorRT engine for RAUNE-Net inference."""

    @staticmethod
    def cache_key(
        onnx_path: str,
        compute_cap: tuple[int, int],
        trt_version: str,
        batch_size: int,
        height: int,
        width: int,
        fp16: bool,
    ) -> str:
        """Deterministic cache key for engine invalidation."""
        onnx_hash = _streaming_sha256(onnx_path)
        sig = json.dumps({
            "onnx": onnx_hash,
            "sm": f"{compute_cap[0]}.{compute_cap[1]}",
            "trt": trt_version,
            "batch": batch_size,
            "h": height,
            "w": width,
            "fp16": fp16,
        }, sort_keys=True)
        return hashlib.sha256(sig.encode()).hexdigest()[:16]

    @staticmethod
    def build_engine(
        onnx_path: str,
        engine_path: str,
        batch_size: int,
        height: int,
        width: int,
        fp16: bool = True,
    ) -> None:
        """Build TRT engine from ONNX and serialize to disk."""
        _require_tensorrt()

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network()
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"[trt-engine] ONNX parse error: {parser.get_error(i)}",
                          file=sys.stderr)
                raise RuntimeError("Failed to parse ONNX model")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 31)  # 2GB

        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

        profile = builder.create_optimization_profile()
        profile.set_shape(
            "input",
            min=(1, 3, height, width),
            opt=(batch_size, 3, height, width),
            max=(batch_size, 3, height, width),
        )
        config.add_optimization_profile(profile)

        print(f"[trt-engine] Building engine: {batch_size}x3x{height}x{width}, "
              f"fp16={fp16}, workspace=2GB. This takes 2-5 minutes...",
              file=sys.stderr, flush=True)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed")

        # Atomic write: write to temp file, then rename
        parent = Path(engine_path).parent
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(serialized)
            os.rename(tmp_path, engine_path)
        except BaseException:
            os.unlink(tmp_path)
            raise

        print(f"[trt-engine] Engine saved to {engine_path} "
              f"({Path(engine_path).stat().st_size / 1e6:.1f} MB)",
              file=sys.stderr, flush=True)

    @classmethod
    def get_or_build(
        cls,
        onnx_path: str,
        cache_dir: str,
        batch_size: int,
        height: int,
        width: int,
        fp16: bool = True,
    ) -> "RauneTRTEngine":
        """Load cached engine or build from ONNX."""
        _require_tensorrt()

        compute_cap = torch.cuda.get_device_capability()
        trt_version = trt.__version__
        key = cls.cache_key(onnx_path, compute_cap, trt_version,
                            batch_size, height, width, fp16)
        engine_path = str(Path(cache_dir) / f"{key}.engine")

        if not Path(engine_path).exists():
            print(f"[trt-engine] Cache miss (key={key}), building engine...",
                  file=sys.stderr, flush=True)
            cls.build_engine(onnx_path, engine_path, batch_size, height, width, fp16)
        else:
            print(f"[trt-engine] Cache hit (key={key}), loading engine...",
                  file=sys.stderr, flush=True)

        return cls(engine_path)

    def __init__(self, engine_path: str):
        """Deserialize engine from disk."""
        _require_tensorrt()

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine from {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        # Cache engine I/O dtypes for tensor casting in infer()
        _dtype_map = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT8: torch.int8,
            trt.DataType.INT32: torch.int32,
        }
        self._input_dtype = _dtype_map[self.engine.get_tensor_dtype("input")]
        self._output_dtype = _dtype_map[self.engine.get_tensor_dtype("output")]

        # Pre-allocate output tensor (reused across calls to avoid cudaMalloc)
        self._output_buf: torch.Tensor | None = None

    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run inference on a CUDA tensor.

        Input: (B,3,H,W) CUDA tensor (any float dtype, must be contiguous).
        Returns: (B,3,H',W') tensor in the same dtype as input.

        Note: RAUNE-Net may trim spatial dimensions (e.g. H'=H-2) due to
        padding/conv architecture. The output shape is determined by the engine.
        """
        caller_dtype = input_tensor.dtype
        B, C, H, W = input_tensor.shape

        # Ensure contiguous memory layout for data_ptr()
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()

        # Cast to engine's expected input dtype if needed
        if input_tensor.dtype != self._input_dtype:
            input_tensor = input_tensor.to(self._input_dtype)

        self.context.set_input_shape("input", (B, C, H, W))

        # Reuse output buffer if shape matches, else allocate
        out_shape = tuple(self.context.get_tensor_shape("output"))
        if self._output_buf is None or self._output_buf.shape != out_shape:
            self._output_buf = torch.empty(
                out_shape, dtype=self._output_dtype, device=input_tensor.device
            )

        self.context.set_tensor_address("input", input_tensor.data_ptr())
        self.context.set_tensor_address("output", self._output_buf.data_ptr())

        # Use CUDA event for inter-stream dependency (no CPU stall)
        event = torch.cuda.current_stream().record_event()
        self.stream.wait_event(event)

        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()

        # Clone output so the buffer can be reused on next call
        output = self._output_buf.clone()

        # Cast back to caller's dtype
        if output.dtype != caller_dtype:
            output = output.to(caller_dtype)

        return output
