"""Tests for EnhanceResult protocol type and encode_raw_rgb helper."""
import base64
import numpy as np
import pytest

from dorea_inference.protocol import encode_raw_rgb, decode_raw_rgb, EnhanceResult


def test_encode_raw_rgb_roundtrip():
    """encode_raw_rgb output should be decodable by decode_raw_rgb."""
    img = np.array([[[10, 20, 30], [40, 50, 60]],
                    [[70, 80, 90], [100, 110, 120]]], dtype=np.uint8)
    h, w = img.shape[:2]
    b64 = encode_raw_rgb(img)
    recovered = decode_raw_rgb(b64, w, h)
    np.testing.assert_array_equal(recovered, img)


def test_encode_raw_rgb_shape_check():
    """encode_raw_rgb must reject non-HxWx3 arrays."""
    bad = np.zeros((4, 4), dtype=np.uint8)  # 2D, not 3D
    with pytest.raises(ValueError, match="HxWx3"):
        encode_raw_rgb(bad)


def test_enhance_result_from_array_roundtrip():
    """EnhanceResult.from_array should encode and store correct dimensions."""
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    img[0, 0] = [255, 128, 64]
    result = EnhanceResult.from_array("frame_001", img)
    assert result.width == 20
    assert result.height == 10
    assert result.id == "frame_001"
    assert result.type == "enhance_result"
    # Decode the b64 to verify pixel data
    raw = base64.b64decode(result.image_b64)
    assert len(raw) == 10 * 20 * 3
    assert raw[0] == 255
    assert raw[1] == 128
    assert raw[2] == 64


def test_enhance_result_to_dict_fields():
    """to_dict must include all required fields."""
    img = np.zeros((4, 6, 3), dtype=np.uint8)
    d = EnhanceResult.from_array("x", img).to_dict()
    assert d["type"] == "enhance_result"
    assert d["id"] == "x"
    assert d["width"] == 6
    assert d["height"] == 4
    assert "image_b64" in d
