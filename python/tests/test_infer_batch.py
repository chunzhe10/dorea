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
