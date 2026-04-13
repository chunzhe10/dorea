"""Tests for scripts/bench/run.py stderr parser (pure function, no subprocess)."""

import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[2] / "scripts" / "bench"
sys.path.insert(0, str(_BENCH))


VALID_STDERR = """\
[raune-filter] RAUNE converted to fp16 (65 InstanceNorm layers kept in fp32, model_dtype=torch.float16)
[raune-filter] Using TensorRT FP16 engine
[raune-filter] Using Triton fused kernel
[raune-filter] single-process: 3840x2160, proxy=960x540, batch=4, codec=prores_ks, fps=59.940, dtype=torch.float16
[raune-filter] 16 frames (53%, 4.7 fps)
[raune-filter] done: 30 frames in 6.4s (4.69 fps)
[raune-filter] stage timing (busy ms/frame): decode=21.4 gpu_thread=190.2 gpu_kernel=190.1 encode=44.2 wall=213.0
"""


class TestParseRauneStderr:
    def test_happy_path(self):
        from run import parse_raune_stderr
        m = parse_raune_stderr(VALID_STDERR, warmup_frames=4)
        assert m.total_frames == 30
        assert m.wall_ms_per_frame == 213.0
        assert m.gpu_kernel_ms_per_frame == 190.1
        assert m.gpu_thread_wall_ms_per_frame == 190.2
        assert m.decode_thread_wall_ms_per_frame == 21.4
        assert m.encode_thread_wall_ms_per_frame == 44.2
        # Steady fps should be close to reported fps (~4.69). The approximation
        # uses mean wall as warmup estimate, so if warmup frames actually took
        # the same time as mean, steady_fps ≈ reported_fps. With the small
        # arithmetic drift from integer rounding, it lands slightly below.
        assert 4.5 < m.steady_state_fps < 5.0

    def test_no_warmup_uses_reported_fps(self):
        from run import parse_raune_stderr
        m = parse_raune_stderr(VALID_STDERR, warmup_frames=0)
        assert m.steady_state_fps == 4.69

    def test_warmup_larger_than_frames_fallback(self):
        from run import parse_raune_stderr
        m = parse_raune_stderr(VALID_STDERR, warmup_frames=1000)
        # Fallback to reported_fps when warmup > total_frames
        assert m.steady_state_fps == 4.69

    def test_missing_timing_line_raises(self):
        from run import parse_raune_stderr
        stderr_no_timing = """\
[raune-filter] done: 30 frames in 6.4s (4.69 fps)
"""
        with pytest.raises(ValueError, match="stage timing"):
            parse_raune_stderr(stderr_no_timing, warmup_frames=4)

    def test_missing_done_line_raises(self):
        from run import parse_raune_stderr
        stderr_no_done = """\
[raune-filter] stage timing (busy ms/frame): decode=21.4 gpu_thread=190.2 gpu_kernel=190.1 encode=44.2 wall=213.0
"""
        with pytest.raises(ValueError, match="'done' line"):
            parse_raune_stderr(stderr_no_done, warmup_frames=4)

    def test_empty_stderr_raises(self):
        from run import parse_raune_stderr
        with pytest.raises(ValueError, match="stage timing"):
            parse_raune_stderr("", warmup_frames=4)

    def test_truncated_stderr_raises(self):
        from run import parse_raune_stderr
        # Truncated mid-timing line
        stderr = "[raune-filter] stage timing (busy ms/frame): decode=21.4 gpu_thread"
        with pytest.raises(ValueError, match="stage timing"):
            parse_raune_stderr(stderr, warmup_frames=4)

    def test_garbage_between_lines_ok(self):
        """Log lines may be interspersed with other output (e.g. warnings)."""
        from run import parse_raune_stderr
        stderr = (
            "UserWarning: some torch warning\n"
            "tracing: xyz\n"
            + VALID_STDERR +
            "closing container\n"
        )
        m = parse_raune_stderr(stderr, warmup_frames=4)
        assert m.total_frames == 30

    def test_fields_are_floats_not_strings(self):
        from run import parse_raune_stderr
        m = parse_raune_stderr(VALID_STDERR, warmup_frames=4)
        assert isinstance(m.wall_ms_per_frame, float)
        assert isinstance(m.gpu_kernel_ms_per_frame, float)
        assert isinstance(m.total_frames, int)


class TestTailHelper:
    def test_tail_short_text(self):
        from run import _tail
        assert _tail("hello\nworld", 50) == "hello\nworld"

    def test_tail_truncates(self):
        from run import _tail
        text = "\n".join(str(i) for i in range(100))
        result = _tail(text, 10)
        lines = result.splitlines()
        assert len(lines) == 10
        assert lines[-1] == "99"
        assert lines[0] == "90"

    def test_tail_empty(self):
        from run import _tail
        assert _tail("", 50) == ""
