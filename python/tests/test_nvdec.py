"""Tests for NV12/P010 → RGB float32 conversion kernel."""

import time

import pytest
import torch

from dorea_inference.nv12_to_rgb import nv12_to_rgb


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_p010(y_10bit: int, cb_10bit: int, cr_10bit: int, H: int = 4, W: int = 4):
    """Pack 10-bit values into P010 uint16 planes (upper 10 bits)."""
    y_plane = torch.full((H, W), y_10bit << 6, dtype=torch.int32).to(torch.uint16).cuda()
    cb_packed = torch.full((H // 2, W // 2), cb_10bit << 6, dtype=torch.int32).to(torch.uint16)
    cr_packed = torch.full((H // 2, W // 2), cr_10bit << 6, dtype=torch.int32).to(torch.uint16)
    uv_plane = torch.stack([cb_packed, cr_packed], dim=-1).cuda()
    return y_plane, uv_plane


def _make_nv12(y_8bit: int, cb_8bit: int, cr_8bit: int, H: int = 4, W: int = 4):
    """Pack 8-bit values into NV12 uint8 planes."""
    y_plane = torch.full((H, W), y_8bit, dtype=torch.uint8).cuda()
    cb_packed = torch.full((H // 2, W // 2), cb_8bit, dtype=torch.uint8)
    cr_packed = torch.full((H // 2, W // 2), cr_8bit, dtype=torch.uint8)
    uv_plane = torch.stack([cb_packed, cr_packed], dim=-1).cuda()
    return y_plane, uv_plane


def _bt709_ref(y: int, cb: int, cr: int, bits: int = 10):
    """Scalar BT.709 limited-range YUV→RGB reference conversion."""
    if bits == 10:
        y_off, uv_off, y_rng, uv_rng = 64, 512, 876.0, 896.0
    else:
        y_off, uv_off, y_rng, uv_rng = 16, 128, 219.0, 224.0
    yp = (y - y_off) / y_rng
    cbp = (cb - uv_off) / uv_rng
    crp = (cr - uv_off) / uv_rng
    r = max(0.0, min(1.0, yp + 1.5748 * crp))
    g = max(0.0, min(1.0, yp - 0.1873 * cbp - 0.4681 * crp))
    b = max(0.0, min(1.0, yp + 1.8556 * cbp))
    return r, g, b


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestNV12ToRGB:
    """Test suite for NV12/P010 → RGB conversion."""

    def test_p010_white(self):
        """P010 10-bit limited-range white (Y=940, Cb=Cr=512) → (1, 1, 1)."""
        y_plane, uv_plane = _make_p010(940, 512, 512)
        rgb = nv12_to_rgb(y_plane, uv_plane)
        r_ref, g_ref, b_ref = _bt709_ref(940, 512, 512, bits=10)
        pixel = rgb[1, 1]
        assert abs(pixel[0].item() - r_ref) < 1e-4
        assert abs(pixel[1].item() - g_ref) < 1e-4
        assert abs(pixel[2].item() - b_ref) < 1e-4

    def test_p010_black(self):
        """P010 10-bit limited-range black (Y=64, Cb=Cr=512) → (0, 0, 0)."""
        y_plane, uv_plane = _make_p010(64, 512, 512)
        rgb = nv12_to_rgb(y_plane, uv_plane)
        r_ref, g_ref, b_ref = _bt709_ref(64, 512, 512, bits=10)
        pixel = rgb[1, 1]
        assert abs(pixel[0].item() - r_ref) < 1e-4
        assert abs(pixel[1].item() - g_ref) < 1e-4
        assert abs(pixel[2].item() - b_ref) < 1e-4

    def test_nv12_white(self):
        """NV12 8-bit limited-range white (Y=235, Cb=Cr=128) → (1, 1, 1)."""
        y_plane, uv_plane = _make_nv12(235, 128, 128)
        rgb = nv12_to_rgb(y_plane, uv_plane)
        r_ref, g_ref, b_ref = _bt709_ref(235, 128, 128, bits=8)
        pixel = rgb[1, 1]
        assert abs(pixel[0].item() - r_ref) < 1e-4
        assert abs(pixel[1].item() - g_ref) < 1e-4
        assert abs(pixel[2].item() - b_ref) < 1e-4

    def test_output_shape_and_range(self):
        """Output is (H, W, 3) float32 in [0, 1]."""
        H, W = 64, 64
        y_vals = torch.randint(0, 1024, (H, W), dtype=torch.int32)
        y_plane = (y_vals << 6).to(torch.uint16).cuda()
        cb_vals = torch.randint(0, 1024, (H // 2, W // 2), dtype=torch.int32)
        cr_vals = torch.randint(0, 1024, (H // 2, W // 2), dtype=torch.int32)
        uv_plane = torch.stack([
            (cb_vals << 6).to(torch.uint16),
            (cr_vals << 6).to(torch.uint16),
        ], dim=-1).cuda()
        rgb = nv12_to_rgb(y_plane, uv_plane)
        assert rgb.shape == (H, W, 3)
        assert rgb.dtype == torch.float32
        assert rgb.min().item() >= 0.0
        assert rgb.max().item() <= 1.0

    def test_4k_throughput(self):
        """4K P010 conversion completes in <2 ms (after warmup)."""
        H, W = 2160, 3840
        y_vals = torch.randint(64, 941, (H, W), dtype=torch.int32)
        y_plane = (y_vals << 6).to(torch.uint16).cuda()
        cb_vals = torch.randint(64, 961, (H // 2, W // 2), dtype=torch.int32)
        cr_vals = torch.randint(64, 961, (H // 2, W // 2), dtype=torch.int32)
        uv_plane = torch.stack([
            (cb_vals << 6).to(torch.uint16),
            (cr_vals << 6).to(torch.uint16),
        ], dim=-1).cuda()

        # Warmup (includes Triton JIT compile)
        for _ in range(5):
            nv12_to_rgb(y_plane, uv_plane)
        torch.cuda.synchronize()

        # Benchmark
        n_iters = 100
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            nv12_to_rgb(y_plane, uv_plane)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        ms_per_frame = (elapsed / n_iters) * 1000
        assert ms_per_frame < 2.0, f"4K conversion took {ms_per_frame:.3f} ms (target: <2 ms)"
