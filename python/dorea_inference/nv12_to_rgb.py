"""NV12/P010 → RGB float32 conversion for NVDEC-decoded frames.

Handles both 8-bit NV12 (uint8) and 10-bit P010 (uint16) input.
Uses Triton kernel when available, falls back to PyTorch.
"""

import torch

_USE_TRITON = False
try:
    import triton
    import triton.language as tl
    _USE_TRITON = True
except ImportError:
    pass


if _USE_TRITON:
    @triton.jit
    def _nv12_to_rgb_kernel(
        y_ptr, uv_ptr, rgb_ptr,
        H: tl.constexpr, W: tl.constexpr,
        Y_OFFSET: tl.constexpr, UV_OFFSET: tl.constexpr,
        Y_RANGE: tl.constexpr, UV_RANGE: tl.constexpr,
        SHIFT: tl.constexpr,
        MASK: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        n_pixels = H * W
        mask = offs < n_pixels

        row = offs // W
        col = offs % W

        y_raw = tl.load(y_ptr + offs, mask=mask, other=0).to(tl.int32)
        y_val = (y_raw >> SHIFT) & MASK

        uv_row = row // 2
        uv_col = (col // 2) * 2
        uv_base = uv_row * W + uv_col

        cb_raw = tl.load(uv_ptr + uv_base, mask=mask, other=0).to(tl.int32)
        cr_raw = tl.load(uv_ptr + uv_base + 1, mask=mask, other=0).to(tl.int32)
        cb_val = (cb_raw >> SHIFT) & MASK
        cr_val = (cr_raw >> SHIFT) & MASK

        y_f = (y_val - Y_OFFSET).to(tl.float32) / Y_RANGE
        cb_f = (cb_val - UV_OFFSET).to(tl.float32) / UV_RANGE
        cr_f = (cr_val - UV_OFFSET).to(tl.float32) / UV_RANGE

        r = y_f + 1.5748 * cr_f
        g = y_f - 0.1873 * cb_f - 0.4681 * cr_f
        b = y_f + 1.8556 * cb_f

        r = tl.minimum(tl.maximum(r, 0.0), 1.0)
        g = tl.minimum(tl.maximum(g, 0.0), 1.0)
        b = tl.minimum(tl.maximum(b, 0.0), 1.0)

        out_base = offs * 3
        tl.store(rgb_ptr + out_base, r, mask=mask)
        tl.store(rgb_ptr + out_base + 1, g, mask=mask)
        tl.store(rgb_ptr + out_base + 2, b, mask=mask)


def nv12_to_rgb(y_plane: torch.Tensor, uv_plane: torch.Tensor) -> torch.Tensor:
    """Convert NV12 (uint8) or P010 (uint16) planes to RGB float32 [0,1].

    Args:
        y_plane:  (H, W) uint8 or uint16 on CUDA
        uv_plane: (H/2, W/2, 2) uint8 or uint16 on CUDA (interleaved Cb, Cr)

    Returns:
        (H, W, 3) float32 on CUDA, RGB in [0, 1]
    """
    H, W = y_plane.shape
    is_10bit = y_plane.dtype == torch.uint16

    if is_10bit:
        Y_OFF, UV_OFF, Y_RNG, UV_RNG, SHIFT, MASK = 64, 512, 876.0, 896.0, 6, 0x3FF
    else:
        Y_OFF, UV_OFF, Y_RNG, UV_RNG, SHIFT, MASK = 16, 128, 219.0, 224.0, 0, 0xFF

    if _USE_TRITON:
        uv_flat = uv_plane.reshape(H // 2, W).contiguous()
        y_c = y_plane.contiguous()
        rgb = torch.empty((H, W, 3), dtype=torch.float32, device=y_plane.device)
        n_pixels = H * W
        BLOCK_SIZE = 1024
        grid = ((n_pixels + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        _nv12_to_rgb_kernel[grid](
            y_c, uv_flat, rgb, H, W,
            Y_OFF, UV_OFF, Y_RNG, UV_RNG, SHIFT, MASK,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return rgb
    else:
        return _nv12_to_rgb_pytorch(y_plane, uv_plane, Y_OFF, UV_OFF, Y_RNG, UV_RNG, SHIFT, MASK)


def _nv12_to_rgb_pytorch(y_plane, uv_plane, y_off, uv_off, y_rng, uv_rng, shift, mask_val):
    """PyTorch fallback for NV12/P010→RGB."""
    H, W = y_plane.shape
    y = ((y_plane.to(torch.int32) >> shift) & mask_val).float()
    cb = ((uv_plane[:, :, 0].to(torch.int32) >> shift) & mask_val).float()
    cr = ((uv_plane[:, :, 1].to(torch.int32) >> shift) & mask_val).float()
    cb_full = cb.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)[:H, :W]
    cr_full = cr.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)[:H, :W]
    y_f = (y - y_off) / y_rng
    cb_f = (cb_full - uv_off) / uv_rng
    cr_f = (cr_full - uv_off) / uv_rng
    r = y_f + 1.5748 * cr_f
    g = y_f - 0.1873 * cb_f - 0.4681 * cr_f
    b = y_f + 1.8556 * cb_f
    return torch.stack([r, g, b], dim=-1).clamp(0.0, 1.0)
