"""Tests for TensorRT engine: ONNX export, engine build, inference accuracy."""

import time

import pytest
import torch
import numpy as np
from pathlib import Path

# Skip entire module if tensorrt is not installed
trt = pytest.importorskip("tensorrt")
onnx = pytest.importorskip("onnx")

MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "raune_net"
WEIGHTS = MODELS_DIR / "weights_95.pth"


def _load_pytorch_model():
    """Load RAUNE-Net in fp32 for reference."""
    import sys
    if str(MODELS_DIR) not in sys.path:
        sys.path.insert(0, str(MODELS_DIR))
    from models.raune_net import RauneNet

    model = RauneNet(input_nc=3, output_nc=3, n_blocks=30, n_down=2, ngf=64)
    state = torch.load(str(WEIGHTS), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@pytest.mark.skipif(not WEIGHTS.exists(), reason="RAUNE weights not found")
class TestOnnxExport:
    def test_export_produces_valid_onnx(self, tmp_path):
        from dorea_inference.export_onnx import export_raune_onnx

        onnx_path = tmp_path / "raune.onnx"
        export_raune_onnx(
            weights=str(WEIGHTS),
            models_dir=str(MODELS_DIR),
            output=str(onnx_path),
        )
        assert onnx_path.exists()
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)

    def test_export_output_matches_pytorch(self, tmp_path):
        from dorea_inference.export_onnx import export_raune_onnx
        import onnxruntime as ort

        onnx_path = tmp_path / "raune.onnx"
        export_raune_onnx(
            weights=str(WEIGHTS),
            models_dir=str(MODELS_DIR),
            output=str(onnx_path),
        )

        # PyTorch reference
        pt_model = _load_pytorch_model()
        x = torch.randn(1, 3, 270, 480)
        with torch.no_grad():
            pt_out = pt_model(x).numpy()

        # ONNX Runtime reference
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"input": x.numpy()})[0]

        np.testing.assert_allclose(pt_out, ort_out, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not WEIGHTS.exists(), reason="RAUNE weights not found")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTRTEngine:
    @pytest.fixture
    def onnx_path(self, tmp_path):
        from dorea_inference.export_onnx import export_raune_onnx
        path = tmp_path / "raune.onnx"
        export_raune_onnx(str(WEIGHTS), str(MODELS_DIR), str(path))
        return path

    def test_build_engine(self, onnx_path, tmp_path):
        from dorea_inference.trt_engine import RauneTRTEngine

        engine_path = tmp_path / "raune.engine"
        RauneTRTEngine.build_engine(
            onnx_path=str(onnx_path),
            engine_path=str(engine_path),
            batch_size=1,
            height=270,
            width=480,
            fp16=True,
        )
        assert engine_path.exists()
        assert engine_path.stat().st_size > 1_000_000

    def test_inference_matches_pytorch(self, onnx_path, tmp_path):
        from dorea_inference.trt_engine import RauneTRTEngine

        engine_path = tmp_path / "raune.engine"
        RauneTRTEngine.build_engine(
            onnx_path=str(onnx_path),
            engine_path=str(engine_path),
            batch_size=1,
            height=270,
            width=480,
            fp16=True,
        )

        engine = RauneTRTEngine(str(engine_path))

        x = torch.randn(1, 3, 270, 480, device="cuda", dtype=torch.float16)
        trt_out = engine.infer(x)

        # PyTorch reference (fp16)
        pt_model = _load_pytorch_model().cuda().half()
        with torch.no_grad():
            pt_out = pt_model(x)

        # Resize to match if shapes differ (RAUNE may trim pixels)
        if trt_out.shape != pt_out.shape:
            import torch.nn.functional as F
            trt_out = F.interpolate(trt_out.float(), size=pt_out.shape[2:],
                                    mode="bilinear", align_corners=False).half()

        # FP16 TRT vs FP16 PyTorch — PSNR > 30dB (wider tolerance for TRT kernel fusion)
        mse = torch.mean((trt_out.float() - pt_out.float()) ** 2)
        if mse > 0:
            psnr = 10 * torch.log10(4.0 / mse)
            assert psnr > 30, f"PSNR {psnr:.1f}dB < 30dB threshold"

    def test_cache_key_changes_with_gpu(self, onnx_path):
        from dorea_inference.trt_engine import RauneTRTEngine

        key1 = RauneTRTEngine.cache_key(str(onnx_path), (8, 6), "10.9.0", 1, 540, 960, True)
        key2 = RauneTRTEngine.cache_key(str(onnx_path), (8, 9), "10.9.0", 1, 540, 960, True)
        assert key1 != key2

    def test_cache_key_changes_with_precision(self, onnx_path):
        from dorea_inference.trt_engine import RauneTRTEngine

        key_fp16 = RauneTRTEngine.cache_key(str(onnx_path), (8, 6), "10.9.0", 1, 540, 960, True)
        key_fp32 = RauneTRTEngine.cache_key(str(onnx_path), (8, 6), "10.9.0", 1, 540, 960, False)
        assert key_fp16 != key_fp32


@pytest.mark.skipif(not WEIGHTS.exists(), reason="RAUNE weights not found")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestIntegration:
    def test_trt_batch_inference(self, tmp_path):
        from dorea_inference.export_onnx import export_raune_onnx
        from dorea_inference.trt_engine import RauneTRTEngine

        onnx_path = tmp_path / "raune.onnx"
        export_raune_onnx(str(WEIGHTS), str(MODELS_DIR), str(onnx_path))

        engine = RauneTRTEngine.get_or_build(
            onnx_path=str(onnx_path),
            cache_dir=str(tmp_path / "cache"),
            batch_size=4,
            height=270,
            width=480,
            fp16=True,
        )

        x = torch.randn(4, 3, 270, 480, device="cuda", dtype=torch.float16)
        out = engine.infer(x)
        assert out.shape[0] == 4
        assert out.shape[1] == 3

    def test_cache_hit_loads_fast(self, tmp_path):
        from dorea_inference.export_onnx import export_raune_onnx
        from dorea_inference.trt_engine import RauneTRTEngine

        onnx_path = tmp_path / "raune.onnx"
        export_raune_onnx(str(WEIGHTS), str(MODELS_DIR), str(onnx_path))
        cache_dir = str(tmp_path / "cache")

        # First build (slow)
        RauneTRTEngine.get_or_build(str(onnx_path), cache_dir, 1, 270, 480)

        # Second load (fast)
        t0 = time.time()
        RauneTRTEngine.get_or_build(str(onnx_path), cache_dir, 1, 270, 480)
        elapsed = time.time() - t0
        assert elapsed < 10, f"Cache load took {elapsed:.1f}s — expected <10s"
