"""Tests for preview_lut.py — LUT preview generation."""

from pathlib import Path
from unittest.mock import patch

import pytest

from preview_lut import (
    _escape_ffmpeg_filter_value,
    discover_keyframes,
    resolve_lut_path,
)


class TestEscapeFfmpegFilterValue:
    def test_plain_path_unchanged(self):
        assert _escape_ffmpeg_filter_value("/opt/luts/base.cube") == "/opt/luts/base.cube"

    def test_escapes_colon(self):
        assert _escape_ffmpeg_filter_value("/mnt/c:/luts/test.cube") == "/mnt/c\\:/luts/test.cube"

    def test_escapes_semicolon(self):
        assert _escape_ffmpeg_filter_value("path;name.cube") == "path\\;name.cube"

    def test_escapes_brackets(self):
        assert _escape_ffmpeg_filter_value("test[1].cube") == "test\\[1\\].cube"

    def test_escapes_backslash(self):
        assert _escape_ffmpeg_filter_value("path\\to\\file.cube") == "path\\\\to\\\\file.cube"

    def test_escapes_single_quote(self):
        assert _escape_ffmpeg_filter_value("it's.cube") == "it\\'s.cube"


class TestDiscoverKeyframes:
    def test_empty_directory(self, tmp_path):
        assert discover_keyframes(tmp_path) == {}

    def test_nonexistent_directory(self, tmp_path):
        assert discover_keyframes(tmp_path / "nonexistent") == {}

    def test_discovers_clip_frames(self, tmp_path):
        clip_dir = tmp_path / "DJI_0042"
        clip_dir.mkdir()
        (clip_dir / "frame_000001.jpg").write_bytes(b"fake")
        (clip_dir / "frame_000002.jpg").write_bytes(b"fake")

        result = discover_keyframes(tmp_path)
        assert "DJI_0042" in result
        assert len(result["DJI_0042"]) == 2

    def test_clip_filter(self, tmp_path):
        for name in ("DJI_0042", "DJI_0043"):
            clip_dir = tmp_path / name
            clip_dir.mkdir()
            (clip_dir / "frame_000001.jpg").write_bytes(b"fake")

        result = discover_keyframes(tmp_path, clip_filter="DJI_0042")
        assert "DJI_0042" in result
        assert "DJI_0043" not in result

    def test_ignores_non_image_files(self, tmp_path):
        clip_dir = tmp_path / "clip1"
        clip_dir.mkdir()
        (clip_dir / "frame_000001.jpg").write_bytes(b"fake")
        (clip_dir / "metadata.json").write_bytes(b"{}")

        result = discover_keyframes(tmp_path)
        assert len(result["clip1"]) == 1

    def test_skips_empty_clip_dirs(self, tmp_path):
        (tmp_path / "empty_clip").mkdir()
        assert discover_keyframes(tmp_path) == {}

    def test_supports_png(self, tmp_path):
        clip_dir = tmp_path / "clip1"
        clip_dir.mkdir()
        (clip_dir / "frame_000001.png").write_bytes(b"fake")

        result = discover_keyframes(tmp_path)
        assert len(result["clip1"]) == 1


class TestResolveLutPath:
    def test_explicit_absolute_path(self, tmp_path):
        lut_file = tmp_path / "custom.cube"
        lut_file.touch()
        config = {"resolve_lut_path": "repos/dorea/luts/default.cube"}

        result = resolve_lut_path(tmp_path, config, str(lut_file))
        assert result == lut_file.resolve()

    def test_explicit_relative_path(self, tmp_path):
        lut_dir = tmp_path / "repos" / "dorea" / "luts"
        lut_dir.mkdir(parents=True)
        (lut_dir / "custom.cube").touch()
        config = {}

        result = resolve_lut_path(tmp_path, config, "luts/custom.cube")
        assert result == (lut_dir / "custom.cube").resolve()

    def test_config_fallback(self, tmp_path):
        config = {"resolve_lut_path": "repos/dorea/luts/from_config.cube"}

        result = resolve_lut_path(tmp_path, config, None)
        assert result.name == "from_config.cube"

    def test_default_fallback(self, tmp_path):
        config = {}

        result = resolve_lut_path(tmp_path, config, None)
        assert "underwater_base.cube" in result.name
