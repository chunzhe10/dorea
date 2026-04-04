"""Tests for enhance request handler in mock mode (DOREA_MAXINE_MOCK=1)."""
import base64
import json
import os
import sys
import io
import numpy as np
import pytest

# Force mock mode before any imports
os.environ["DOREA_MAXINE_MOCK"] = "1"


def _make_enhance_req(width: int, height: int, req_id: str = "f001") -> dict:
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    b64 = base64.b64encode(pixels.tobytes()).decode("ascii")
    return {
        "type": "enhance",
        "id": req_id,
        "format": "raw_rgb",
        "image_b64": b64,
        "width": width,
        "height": height,
        "no_artifact_reduce": False,
        "upscale_factor": 2,
    }


def _run_server_with_reqs(requests: list, extra_argv: list = None) -> list:
    """Run main() with a sequence of requests, return parsed responses."""
    from dorea_inference.server import main

    lines = [json.dumps(r) for r in requests] + ['{"type": "shutdown"}']
    stdin_data = "\n".join(lines) + "\n"

    captured = io.StringIO()
    old_stdin = sys.stdin
    old_stdout = sys.stdout
    sys.stdin = io.StringIO(stdin_data)
    sys.stdout = captured

    argv = ["--no-raune", "--no-depth"] + (extra_argv or [])
    try:
        main(argv=argv)
    except SystemExit:
        pass
    finally:
        sys.stdin = old_stdin
        sys.stdout = old_stdout

    output = captured.getvalue().strip()
    return [json.loads(line) for line in output.split("\n") if line.strip()]


def test_enhance_handler_returns_enhance_result():
    req = _make_enhance_req(4, 6)
    responses = _run_server_with_reqs([req], extra_argv=["--maxine"])
    # First response is the enhance_result; second is ok (shutdown)
    enhance_resp = responses[0]
    assert enhance_resp["type"] == "enhance_result"
    assert enhance_resp["id"] == "f001"
    assert enhance_resp["width"] == 4
    assert enhance_resp["height"] == 6
    raw = base64.b64decode(enhance_resp["image_b64"])
    assert len(raw) == 4 * 6 * 3


def test_enhance_handler_without_maxine_flag_errors():
    """enhance request without --maxine should return error (no enhancer loaded)."""
    req = _make_enhance_req(4, 6)
    responses = _run_server_with_reqs([req])  # no --maxine
    assert responses[0]["type"] == "error"
    assert "not loaded" in responses[0]["message"].lower() or "maxine" in responses[0]["message"].lower()


def test_enhance_handler_passthrough_preserves_dimensions():
    """Mock mode returns same dimensions as input."""
    req = _make_enhance_req(16, 9)
    responses = _run_server_with_reqs([req], extra_argv=["--maxine"])
    r = responses[0]
    assert r["width"] == 16
    assert r["height"] == 9
