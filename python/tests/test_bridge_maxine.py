"""Tests for Maxine support in the PyO3 bridge module."""
import os
import numpy as np
import pytest

os.environ["DOREA_MAXINE_MOCK"] = "1"


def test_load_and_run_maxine_cpu():
    from dorea_inference import bridge
    bridge.load_maxine_model(upscale_factor=2)
    frame = np.zeros((8, 10, 3), dtype=np.uint8)
    result = bridge.run_maxine_cpu(frame, artifact_reduce=True)
    assert result.shape == (8, 10, 3)
    assert result.dtype == np.uint8
    # cleanup
    bridge.unload_maxine()


def test_unload_maxine_clears_model():
    from dorea_inference import bridge
    bridge.load_maxine_model(upscale_factor=2)
    bridge.unload_maxine()
    with pytest.raises(RuntimeError, match="not loaded"):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        bridge.run_maxine_cpu(frame)


def test_unload_models_also_clears_maxine():
    from dorea_inference import bridge
    bridge.load_maxine_model(upscale_factor=2)
    bridge.unload_models()
    with pytest.raises(RuntimeError, match="not loaded"):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        bridge.run_maxine_cpu(frame)
