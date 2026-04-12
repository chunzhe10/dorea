"""Tests for raune_filter _process_batch and related functions."""

import numpy as np
import pytest
import torch
import torch.nn as nn


class IdentityModel(nn.Module):
    """Mock RAUNE that returns input unchanged (identity function).

    Expects input in [-1, 1] (RAUNE normalization range).
    Returns input unchanged, so the OKLab delta will be zero
    and the output should match the input.
    """

    def forward(self, x):
        return x


def _make_transfer_fn():
    """Return the PyTorch OKLab transfer function (no Triton dependency)."""
    from dorea_inference.raune_filter import pytorch_oklab_transfer
    return pytorch_oklab_transfer


@pytest.fixture
def identity_model():
    """Identity model on GPU in fp16 (matching real RAUNE dtype)."""
    model = IdentityModel().cuda().half()
    model.eval()
    return model


@pytest.fixture
def model_dtype(identity_model):
    return next(identity_model.parameters(), torch.tensor(0, dtype=torch.float16)).dtype


class TestProcessBatch8Bit:
    """Baseline: current 8-bit _process_batch behavior."""

    def test_output_shape_and_dtype(self, identity_model, model_dtype):
        from dorea_inference.raune_filter import _process_batch

        # 16x16 uint8 frame with a gradient
        frame = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
        transfer_fn = _make_transfer_fn()

        results = _process_batch(
            [frame], identity_model, None,
            fw=16, fh=16, pw=8, ph=8,
            transfer_fn=transfer_fn,
            model_dtype=model_dtype,
        )

        assert len(results) == 1
        assert results[0].shape == (16, 16, 3)
        # Before 10-bit change: uint8. After: uint16.
        # This test will be updated in Task 5.
        assert results[0].dtype == np.uint8

    def test_identity_model_preserves_values(self, identity_model, model_dtype):
        """With identity RAUNE (zero delta), output ≈ input within quantization."""
        from dorea_inference.raune_filter import _process_batch

        # Uniform mid-gray frame — avoids edge effects from OKLab nonlinearity
        frame = np.full((16, 16, 3), 128, dtype=np.uint8)
        transfer_fn = _make_transfer_fn()

        results = _process_batch(
            [frame], identity_model, None,
            fw=16, fh=16, pw=8, ph=8,
            transfer_fn=transfer_fn,
            model_dtype=model_dtype,
        )

        # Identity model → zero delta → output should be close to input.
        # Allow ±2 for OKLab round-trip quantization at 8-bit.
        diff = np.abs(results[0].astype(np.int16) - frame.astype(np.int16))
        assert diff.max() <= 2, f"max diff {diff.max()}, expected ≤2"
