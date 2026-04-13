"""Tests that raune_filter.py has NVTX markers at the expected spans."""

import ast
from pathlib import Path

_RAUNE_FILTER = Path(__file__).resolve().parents[1] / "dorea_inference" / "raune_filter.py"

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


class TestNvtxRuntime:
    """Verify NVTX markers don't break raune_filter and new metrics are emitted."""

    def test_raune_filter_still_runs(self):
        """End-to-end smoke test: raune_filter with NVTX markers completes
        successfully and emits gpu_kernel= in the stage timing line."""
        import os
        import subprocess

        env = os.environ.copy()
        env["PYTHONPATH"] = str(
            Path(__file__).resolve().parents[1]  # python/
        )
        result = subprocess.run(
            ["/opt/dorea-venv/bin/python", "-m", "dorea_inference.raune_filter",
             "--weights", "/workspaces/dorea-workspace/models/raune_net/weights_95.pth",
             "--models-dir", "/workspaces/dorea-workspace/models/raune_net",
             "--full-width", "3840", "--full-height", "2160",
             "--proxy-width", "960", "--proxy-height", "540",
             "--batch-size", "4",
             "--input", "/tmp/test_clip_30f.mp4",
             "--output", "/tmp/test_nvtx_smoke.mov",
             "--output-codec", "prores_ks",
             "--tensorrt"],
            env=env, capture_output=True, text=True,
            cwd="/workspaces/dorea-workspace/repos/dorea",
            timeout=120,
        )
        assert result.returncode == 0, f"raune_filter failed: {result.stderr}"
        assert "gpu_kernel=" in result.stderr, "Expected gpu_kernel in timing output"
        assert "gpu_thread=" in result.stderr, "Expected gpu_thread in timing output"
