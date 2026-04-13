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
