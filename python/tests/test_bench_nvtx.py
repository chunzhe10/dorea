"""Tests that raune_filter.py has NVTX markers at the expected spans."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_RAUNE_FILTER = Path(__file__).resolve().parents[1] / "dorea_inference" / "raune_filter.py"
_REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_SPANS = [
    "upload",
    "proxy_downscale",
    "trt_inference",
    "oklab_delta",
    "apply_transfer",
    "download",
]


def _find_nvtx_spans(source: str) -> list[str]:
    """Parse the source and return all string literals passed to nvtx.range_push."""
    tree = ast.parse(source)
    spans = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr != "range_push":
            continue
        if not node.args:
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            spans.append(arg.value)
    return spans


def test_all_expected_spans_present():
    source = _RAUNE_FILTER.read_text()
    spans = _find_nvtx_spans(source)
    for expected in EXPECTED_SPANS:
        assert expected in spans, (
            f"NVTX span '{expected}' missing from raune_filter.py. "
            f"Found: {spans}"
        )


def test_push_pop_balance():
    """Every range_push must have a matching range_pop."""
    source = _RAUNE_FILTER.read_text()
    pushes = source.count("torch.cuda.nvtx.range_push")
    pops = source.count("torch.cuda.nvtx.range_pop")
    assert pushes == pops, f"Unbalanced NVTX: {pushes} pushes vs {pops} pops"
    assert pushes >= 6, f"Expected at least 6 NVTX pushes, found {pushes}"


# Paths for the integration smoke test. Set via env vars with sensible
# fallbacks. Tests that depend on these are skipped if the files are absent.
_TEST_CLIP = os.environ.get("DOREA_TEST_CLIP", "/tmp/test_clip_30f.mp4")
_TEST_WEIGHTS = os.environ.get(
    "DOREA_RAUNE_WEIGHTS",
    "/workspaces/dorea-workspace/models/raune_net/weights_95.pth",
)
_TEST_MODELS_DIR = os.environ.get(
    "DOREA_MODELS_DIR",
    "/workspaces/dorea-workspace/models/raune_net",
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    not Path(_TEST_CLIP).is_file(),
    reason=f"test clip not found at {_TEST_CLIP} (set DOREA_TEST_CLIP)",
)
@pytest.mark.skipif(
    not Path(_TEST_WEIGHTS).is_file(),
    reason=f"weights not found at {_TEST_WEIGHTS} (set DOREA_RAUNE_WEIGHTS)",
)
class TestNvtxRuntime:
    """Verify NVTX markers don't break raune_filter.

    In production mode (no DOREA_BENCH env var) the stage timing line
    should emit gpu_kernel=0.0 — meaning the CUDA event path was NOT taken,
    so there's no hot-path synchronize cost. In bench mode (DOREA_BENCH=1)
    the gpu_kernel value should be non-zero.
    """

    def test_production_mode_gpu_kernel_is_zero(self):
        """Without DOREA_BENCH, CUDA events should NOT run and gpu_kernel=0.0."""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_REPO_ROOT / "python")
        # Explicitly unset DOREA_BENCH (in case it leaked from parent)
        env.pop("DOREA_BENCH", None)

        result = subprocess.run(
            [sys.executable, "-m", "dorea_inference.raune_filter",
             "--weights", _TEST_WEIGHTS,
             "--models-dir", _TEST_MODELS_DIR,
             "--full-width", "3840", "--full-height", "2160",
             "--proxy-width", "960", "--proxy-height", "540",
             "--batch-size", "4",
             "--input", _TEST_CLIP,
             "--output", "/tmp/test_nvtx_prod.mov",
             "--output-codec", "prores_ks",
             "--tensorrt"],
            env=env, capture_output=True, text=True,
            cwd=str(_REPO_ROOT),
            timeout=120,
        )
        assert result.returncode == 0, f"raune_filter failed: {result.stderr[-2000:]}"
        assert "gpu_kernel=" in result.stderr
        # In production mode, gpu_kernel should be 0.0 (events not recorded)
        assert "gpu_kernel=0.0" in result.stderr, (
            "Expected gpu_kernel=0.0 in production mode. "
            "If non-zero, the CUDA event path is running in the hot path."
        )

    def test_bench_mode_gpu_kernel_is_nonzero(self):
        """With DOREA_BENCH=1, CUDA events should run and gpu_kernel > 0."""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_REPO_ROOT / "python")
        env["DOREA_BENCH"] = "1"

        result = subprocess.run(
            [sys.executable, "-m", "dorea_inference.raune_filter",
             "--weights", _TEST_WEIGHTS,
             "--models-dir", _TEST_MODELS_DIR,
             "--full-width", "3840", "--full-height", "2160",
             "--proxy-width", "960", "--proxy-height", "540",
             "--batch-size", "4",
             "--input", _TEST_CLIP,
             "--output", "/tmp/test_nvtx_bench.mov",
             "--output-codec", "prores_ks",
             "--tensorrt"],
            env=env, capture_output=True, text=True,
            cwd=str(_REPO_ROOT),
            timeout=120,
        )
        assert result.returncode == 0, f"raune_filter failed: {result.stderr[-2000:]}"
        # Parse gpu_kernel value from stderr
        import re
        m = re.search(r"gpu_kernel=([\d.]+)", result.stderr)
        assert m, "gpu_kernel= missing from stage timing"
        value = float(m.group(1))
        assert value > 0, f"Expected gpu_kernel > 0 in bench mode, got {value}"
