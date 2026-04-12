"""TensorRT engine wrapper for RAUNE-Net inference.

Handles engine building from ONNX, disk caching with automatic invalidation,
and zero-copy inference with PyTorch CUDA tensors.
"""

import hashlib
import json
import sys
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
        onnx_hash = hashlib.md5(Path(onnx_path).read_bytes()).hexdigest()
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
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"[trt-engine] ONNX parse error: {parser.get_error(i)}",
                          file=sys.stderr)
                raise RuntimeError("Failed to parse ONNX model")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        profile = builder.create_optimization_profile()
        profile.set_shape(
            "input",
            min=(1, 3, height, width),
            opt=(batch_size, 3, height, width),
            max=(batch_size, 3, height, width),
        )
        config.add_optimization_profile(profile)

        print(f"[trt-engine] Building engine: {batch_size}x3x{height}x{width}, "
              f"fp16={fp16}. This takes 2-5 minutes...",
              file=sys.stderr, flush=True)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed")

        Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(serialized)

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

    # Map TRT DataType to torch dtype
    _TRT_TO_TORCH = None

    @staticmethod
    def _trt_dtype_to_torch(trt_dtype) -> torch.dtype:
        """Convert TensorRT dtype to PyTorch dtype."""
        if RauneTRTEngine._TRT_TO_TORCH is None:
            RauneTRTEngine._TRT_TO_TORCH = {
                trt.DataType.FLOAT: torch.float32,
                trt.DataType.HALF: torch.float16,
                trt.DataType.INT8: torch.int8,
                trt.DataType.INT32: torch.int32,
            }
        return RauneTRTEngine._TRT_TO_TORCH[trt_dtype]

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
        self._input_dtype = self._trt_dtype_to_torch(
            self.engine.get_tensor_dtype("input")
        )
        self._output_dtype = self._trt_dtype_to_torch(
            self.engine.get_tensor_dtype("output")
        )

    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run inference on a CUDA tensor.

        Input: (B,3,H,W) CUDA tensor (any float dtype).
        Returns: (B,3,H',W') tensor in the same dtype as input.

        Note: RAUNE-Net may trim spatial dimensions (e.g. H'=H-2) due to
        padding/conv architecture. The output shape is determined by the engine.
        The engine's internal I/O dtype (typically fp32) may differ from the
        caller's tensor dtype; casting is handled automatically.
        """
        caller_dtype = input_tensor.dtype
        B, C, H, W = input_tensor.shape

        # Cast to engine's expected input dtype if needed
        if input_tensor.dtype != self._input_dtype:
            input_tensor = input_tensor.to(self._input_dtype)

        self.context.set_input_shape("input", (B, C, H, W))

        # Query the actual output shape from the execution context
        out_shape = tuple(self.context.get_tensor_shape("output"))
        output = torch.empty(out_shape, dtype=self._output_dtype,
                             device=input_tensor.device)

        self.context.set_tensor_address("input", input_tensor.data_ptr())
        self.context.set_tensor_address("output", output.data_ptr())

        # Synchronize the default stream so the input tensor is fully
        # materialized before TRT reads it on self.stream.
        torch.cuda.current_stream().synchronize()

        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()

        # Cast back to caller's dtype
        if output.dtype != caller_dtype:
            output = output.to(caller_dtype)

        return output
