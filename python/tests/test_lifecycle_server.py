"""Tests for dynamic model lifecycle IPC commands."""
import base64
import io
import json
import os
import sys

import numpy as np
import pytest

os.environ["DOREA_MAXINE_MOCK"] = "1"


def _run_server(requests: list, extra_argv: list = None) -> list:
    from dorea_inference.server import main
    lines = [json.dumps(r) for r in requests] + ['{"type": "shutdown"}']
    stdin_data = "\n".join(lines) + "\n"
    captured = io.StringIO()
    old_stdin, old_stdout = sys.stdin, sys.stdout
    sys.stdin = io.StringIO(stdin_data)
    sys.stdout = captured
    argv = ["--no-raune", "--no-depth"] + (extra_argv or [])
    try:
        main(argv=argv)
    except SystemExit:
        pass
    finally:
        sys.stdin, sys.stdout = old_stdin, old_stdout
    output = captured.getvalue().strip()
    return [json.loads(line) for line in output.split("\n") if line.strip()]


def test_unload_maxine_returns_ok():
    responses = _run_server([{"type": "unload_maxine"}], extra_argv=["--maxine"])
    assert responses[0]["type"] == "ok"


def test_load_raune_returns_ok():
    import dorea_inference.raune_net as rn_mod
    original_class = rn_mod.RauneNetInference

    class _FakeRaune:
        def __init__(self, **kwargs):
            pass  # no-op — avoid real weight loading in test environment

    rn_mod.RauneNetInference = _FakeRaune
    try:
        responses = _run_server([{"type": "load_raune", "weights": None, "models_dir": None}])
        assert responses[0]["type"] == "ok"
    finally:
        rn_mod.RauneNetInference = original_class


def test_load_depth_returns_ok():
    import dorea_inference.depth_anything as da_mod
    original_class = da_mod.DepthAnythingInference

    class _FakeDepth:
        def __init__(self, **kwargs):
            pass  # no-op — avoid real weight loading in test environment

    da_mod.DepthAnythingInference = _FakeDepth
    try:
        responses = _run_server([{"type": "load_depth", "model_path": None}])
        assert responses[0]["type"] == "ok"
    finally:
        da_mod.DepthAnythingInference = original_class


def test_maxine_graceful_skip_when_unavailable():
    """Server should start successfully even if Maxine fails to load."""
    import dorea_inference.maxine_enhancer as me_mod
    original_class = me_mod.MaxineEnhancer

    class BrokenMaxine:
        def __init__(self, **kwargs):
            raise RuntimeError("nvvfx not found — simulated")

    me_mod.MaxineEnhancer = BrokenMaxine
    try:
        responses = _run_server([{"type": "ping"}], extra_argv=["--maxine"])
        assert responses[0]["type"] == "pong"
    finally:
        me_mod.MaxineEnhancer = original_class
