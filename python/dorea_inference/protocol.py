"""IPC protocol types for Rust ↔ Python inference subprocess communication.

Protocol: JSON lines over stdin/stdout.
- One JSON object per line, terminated by newline.
- Parent (Rust) writes requests; child (Python server) writes responses.
- EOF on stdin = parent exited; server should shut down cleanly.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Request types (Rust → Python)
# ---------------------------------------------------------------------------

@dataclass
class PingRequest:
    type: str = "ping"

    def to_dict(self) -> dict:
        return {"type": "ping"}


@dataclass
class RauneRequest:
    """Run RAUNE-Net on a single image."""
    id: str
    image_b64: str   # base64-encoded PNG bytes
    max_size: int = 1024
    type: str = "raune"

    def to_dict(self) -> dict:
        return {"type": self.type, "id": self.id,
                "image_b64": self.image_b64, "max_size": self.max_size}


@dataclass
class DepthRequest:
    """Run Depth Anything V2 on a single image."""
    id: str
    image_b64: str   # base64-encoded PNG bytes
    max_size: int = 518
    type: str = "depth"

    def to_dict(self) -> dict:
        return {"type": self.type, "id": self.id,
                "image_b64": self.image_b64, "max_size": self.max_size}


@dataclass
class ShutdownRequest:
    type: str = "shutdown"

    def to_dict(self) -> dict:
        return {"type": "shutdown"}


# ---------------------------------------------------------------------------
# Response types (Python → Rust)
# ---------------------------------------------------------------------------

@dataclass
class PongResponse:
    version: str
    type: str = "pong"

    def to_dict(self) -> dict:
        return {"type": self.type, "version": self.version}


@dataclass
class RauneResult:
    id: str
    image_b64: str   # base64-encoded PNG bytes at inference resolution
    width: int
    height: int
    type: str = "raune_result"

    def to_dict(self) -> dict:
        return {"type": self.type, "id": self.id,
                "image_b64": self.image_b64,
                "width": self.width, "height": self.height}


@dataclass
class DepthResult:
    id: str
    depth_f32_b64: str  # base64-encoded raw f32 little-endian array, row-major
    width: int
    height: int
    type: str = "depth_result"

    def to_dict(self) -> dict:
        return {"type": self.type, "id": self.id,
                "depth_f32_b64": self.depth_f32_b64,
                "width": self.width, "height": self.height}

    @staticmethod
    def from_array(id: str, depth: "np.ndarray") -> "DepthResult":
        """Encode a 2-D float32 depth array as DepthResult."""
        import numpy as np
        arr = np.ascontiguousarray(depth, dtype="<f4")
        raw = arr.tobytes()
        return DepthResult(
            id=id,
            depth_f32_b64=base64.b64encode(raw).decode("ascii"),
            width=arr.shape[1],
            height=arr.shape[0],
        )


@dataclass
class DepthBatchResult:
    results: list  # list of DepthResult.to_dict()
    type: str = "depth_batch_result"

    def to_dict(self) -> dict:
        return {"type": self.type, "results": self.results}


@dataclass
class RauneDepthBatchResult:
    """Fused RAUNE+depth batch response.

    Each result dict contains:
      id, image_b64 (PNG, enhanced frame), enhanced_width, enhanced_height,
      depth_f32_b64, depth_width, depth_height, type="depth_result"
    """
    results: list
    type: str = "raune_depth_batch_result"

    def to_dict(self) -> dict:
        return {"type": self.type, "results": self.results}


@dataclass
class ErrorResponse:
    id: Optional[str]
    message: str
    type: str = "error"

    def to_dict(self) -> dict:
        return {"type": self.type, "id": self.id, "message": self.message}


@dataclass
class OkResponse:
    type: str = "ok"

    def to_dict(self) -> dict:
        return {"type": "ok"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_png(image_rgb_uint8: "np.ndarray") -> str:
    """Encode uint8 HxWx3 RGB array as base64 PNG."""
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(image_rgb_uint8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_png(b64: str) -> "np.ndarray":
    """Decode base64 PNG to uint8 HxWx3 RGB array."""
    import io
    import numpy as np
    from PIL import Image
    data = base64.b64decode(b64)
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def decode_raw_rgb(b64: str, width: int, height: int) -> "np.ndarray":
    """Decode base64 raw interleaved RGB uint8 to HxWx3 array."""
    import numpy as np
    raw = base64.b64decode(b64)
    expected = width * height * 3
    if len(raw) != expected:
        raise ValueError(f"raw_rgb size mismatch: got {len(raw)}, expected {expected}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()


def decode_depth_f32(b64: str, width: int, height: int) -> "np.ndarray":
    """Decode base64 raw f32 LE array to float32 HxW array."""
    import numpy as np
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype="<f4").reshape(height, width)
    return arr
