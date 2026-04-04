"""Tests for infer_batch methods — CPU mode, no GPU or real weights required."""
import numpy as np
import pytest
import torch


def _make_rgb(h: int, w: int) -> np.ndarray:
    return np.random.default_rng(42).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _fake_raune_model():
    """Identity RauneNetInference on CPU, no weights file needed."""
    import types, sys
    fake_mod = types.ModuleType("models")
    fake_mod.raune_net = types.ModuleType("models.raune_net")

    class _IdentityNet(torch.nn.Module):
        def forward(self, x):
            return x  # identity; output in same range as input ([-1,1] after normalize)

    fake_mod.raune_net.RauneNet = lambda **_: _IdentityNet()
    sys.modules.setdefault("models", fake_mod)
    sys.modules.setdefault("models.raune_net", fake_mod.raune_net)

    from dorea_inference.raune_net import RauneNetInference
    m = RauneNetInference.__new__(RauneNetInference)
    m.device = torch.device("cpu")
    m.model = _IdentityNet()
    m.model.eval()
    return m


class TestRauneInferBatch:
    def test_returns_same_count(self):
        model = _fake_raune_model()
        imgs = [_make_rgb(60, 80) for _ in range(5)]
        results = model.infer_batch(imgs, max_size=80)
        assert len(results) == 5

    def test_output_dtype_and_channels(self):
        model = _fake_raune_model()
        results = model.infer_batch([_make_rgb(60, 80)], max_size=80)
        assert results[0].dtype == np.uint8
        assert results[0].ndim == 3
        assert results[0].shape[2] == 3

    def test_empty_returns_empty(self):
        model = _fake_raune_model()
        assert model.infer_batch([], max_size=80) == []

    def test_mixed_dims_falls_back_to_sequential(self):
        model = _fake_raune_model()
        imgs = [_make_rgb(60, 80), _make_rgb(40, 60)]  # different sizes
        results = model.infer_batch(imgs, max_size=160)
        assert len(results) == 2


class TestDepthInferBatchFromTensors:
    def test_accepts_gpu_tensor_shape(self):
        """infer_batch_from_tensors returns one depth map per input tensor."""
        from dorea_inference.depth_anything import DepthAnythingInference

        class _FakeDepthModel(torch.nn.Module):
            class _Out:
                def __init__(self, t): self.predicted_depth = t
            def forward(self, pixel_values=None):
                N, _, H, W = pixel_values.shape
                return self._Out(pixel_values[:, 0, :, :])  # (N, H, W) dummy

        model = DepthAnythingInference.__new__(DepthAnythingInference)
        model.device = torch.device("cpu")
        model.model = _FakeDepthModel()

        # Simulate RAUNE output: (3, 3, H, W) float32 in [0, 1]
        fake_enhanced = torch.rand(3, 3, 56, 84)  # 3 frames, not yet patch-aligned
        depths = model.infer_batch_from_tensors(fake_enhanced, depth_max_size=56)
        assert len(depths) == 3
        assert all(isinstance(d, np.ndarray) for d in depths)
        assert all(d.dtype == np.float32 for d in depths)

    def test_output_dims_are_patch_aligned(self):
        """Output depth map dimensions are multiples of 14."""
        from dorea_inference.depth_anything import DepthAnythingInference

        class _FakeDepthModel(torch.nn.Module):
            class _Out:
                def __init__(self, t): self.predicted_depth = t
            def forward(self, pixel_values=None):
                N, _, H, W = pixel_values.shape
                return self._Out(pixel_values[:, 0, :, :])

        model = DepthAnythingInference.__new__(DepthAnythingInference)
        model.device = torch.device("cpu")
        model.model = _FakeDepthModel()

        fake_enhanced = torch.rand(1, 3, 100, 180)
        depths = model.infer_batch_from_tensors(fake_enhanced, depth_max_size=98)
        H, W = depths[0].shape
        assert H % 14 == 0, f"height {H} not multiple of 14"
        assert W % 14 == 0, f"width {W} not multiple of 14"


class TestRauneDepthBatchProtocol:
    def test_raune_depth_batch_result_roundtrip(self):
        """RauneDepthBatchResult serialises/deserialises correctly."""
        from dorea_inference.protocol import RauneDepthBatchResult, DepthResult, RauneResult, encode_png
        import json

        dummy_img = np.zeros((14, 14, 3), dtype=np.uint8)
        dummy_depth = np.zeros((14, 14), dtype=np.float32)

        item = {
            "id": "kf0000",
            "image_b64": encode_png(dummy_img),
            "enhanced_width": 14,
            "enhanced_height": 14,
            **DepthResult.from_array("kf0000", dummy_depth).to_dict(),
        }
        resp = RauneDepthBatchResult(results=[item])
        d = resp.to_dict()
        assert d["type"] == "raune_depth_batch_result"
        assert len(d["results"]) == 1
        assert d["results"][0]["id"] == "kf0000"
        # Round-trip through JSON
        assert json.loads(json.dumps(d))["type"] == "raune_depth_batch_result"
