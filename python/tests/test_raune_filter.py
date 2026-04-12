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


class TestProcessBatch16Bit:
    """16-bit _process_batch with int32 GPU cache."""

    def test_output_shape_and_dtype(self, identity_model, model_dtype):
        """Output must be uint16 HWC numpy arrays."""
        from dorea_inference.raune_filter import _process_batch

        frame = np.full((16, 16, 3), 32768, dtype=np.uint16)
        transfer_fn = _make_transfer_fn()

        results = _process_batch(
            [frame], identity_model, None,
            fw=16, fh=16, pw=8, ph=8,
            transfer_fn=transfer_fn,
            model_dtype=model_dtype,
        )

        assert len(results) == 1
        assert results[0].shape == (16, 16, 3)
        assert results[0].dtype == np.uint16

    def test_identity_preserves_midgray(self, identity_model, model_dtype):
        """With identity RAUNE (zero delta), mid-gray output ≈ input."""
        from dorea_inference.raune_filter import _process_batch

        # Mid-gray in 16-bit
        val = 32768
        frame = np.full((16, 16, 3), val, dtype=np.uint16)
        transfer_fn = _make_transfer_fn()

        results = _process_batch(
            [frame], identity_model, None,
            fw=16, fh=16, pw=8, ph=8,
            transfer_fn=transfer_fn,
            model_dtype=model_dtype,
        )

        diff = np.abs(results[0].astype(np.int32) - frame.astype(np.int32))
        # Allow ±128 (≈ 2 LSBs at 16-bit, ~0.2%) for OKLab round-trip
        assert diff.max() <= 128, f"max diff {diff.max()}, expected ≤128"

    def test_values_above_32767_not_corrupted(self, identity_model, model_dtype):
        """Values > 32767 must not become negative (int16 sign corruption check)."""
        from dorea_inference.raune_filter import _process_batch

        # Frame with value 60000 — would be negative if treated as int16
        frame = np.full((16, 16, 3), 60000, dtype=np.uint16)
        transfer_fn = _make_transfer_fn()

        results = _process_batch(
            [frame], identity_model, None,
            fw=16, fh=16, pw=8, ph=8,
            transfer_fn=transfer_fn,
            model_dtype=model_dtype,
        )

        # If int16 corruption occurred, values would be near 0 or wrapped
        assert results[0].min() > 50000, (
            f"min={results[0].min()}, expected >50000 — likely int16 sign corruption"
        )

    def test_8bit_input_zero_extended(self, identity_model, model_dtype):
        """8-bit source zero-extended to uint16 should produce valid output."""
        from dorea_inference.raune_filter import _process_batch

        # Simulate rgb48le zero-extension of 8-bit source: val * 257
        val_8bit = 128
        val_16bit = val_8bit * 257  # = 32896
        frame = np.full((16, 16, 3), val_16bit, dtype=np.uint16)
        transfer_fn = _make_transfer_fn()

        results = _process_batch(
            [frame], identity_model, None,
            fw=16, fh=16, pw=8, ph=8,
            transfer_fn=transfer_fn,
            model_dtype=model_dtype,
        )

        assert results[0].dtype == np.uint16
        diff = np.abs(results[0].astype(np.int32) - frame.astype(np.int32))
        assert diff.max() <= 128, f"max diff {diff.max()}, expected ≤128"

    def test_rounding_edge_values(self, identity_model, model_dtype):
        """Verify +0.5 rounding works for edge values (0, max)."""
        from dorea_inference.raune_filter import _process_batch

        # Black frame — should stay near 0
        black = np.zeros((8, 8, 3), dtype=np.uint16)
        # Near-white frame — should stay near 65535
        white = np.full((8, 8, 3), 65535, dtype=np.uint16)
        transfer_fn = _make_transfer_fn()

        results_black = _process_batch(
            [black], identity_model, None,
            fw=8, fh=8, pw=4, ph=4,
            transfer_fn=transfer_fn,
            model_dtype=model_dtype,
        )
        results_white = _process_batch(
            [white], identity_model, None,
            fw=8, fh=8, pw=4, ph=4,
            transfer_fn=transfer_fn,
            model_dtype=model_dtype,
        )

        # Black should be very close to 0
        assert results_black[0].max() <= 128, f"black max={results_black[0].max()}"
        # White should be very close to 65535
        assert results_white[0].min() >= 65000, f"white min={results_white[0].min()}"

    def test_batch_of_multiple_frames(self, identity_model, model_dtype):
        """Multiple frames in a batch should all be processed."""
        from dorea_inference.raune_filter import _process_batch

        frames = [
            np.full((16, 16, 3), v, dtype=np.uint16)
            for v in [16384, 32768, 49152]
        ]
        transfer_fn = _make_transfer_fn()

        results = _process_batch(
            frames, identity_model, None,
            fw=16, fh=16, pw=8, ph=8,
            transfer_fn=transfer_fn,
            model_dtype=model_dtype,
        )

        assert len(results) == 3
        for r in results:
            assert r.shape == (16, 16, 3)
            assert r.dtype == np.uint16
