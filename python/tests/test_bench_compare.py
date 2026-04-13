"""Tests for scripts/bench/compare.py — Welch's t-test + hardware guard."""

import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[2] / "scripts" / "bench"
sys.path.insert(0, str(_BENCH))


def _make_result(fps_samples, gpu_kernel_samples, driver="550.54.14", gpu="NVIDIA GeForce RTX 3060"):
    from schema import (
        BenchResult, MetricAggregate, PinningApplied,
        RunConfig, SCHEMA_VERSION, SystemState, ThroughputMetrics, VramStats,
    )

    sys_state = SystemState(
        kernel_version="6.8.0", cpu_governor="performance", cpu_turbo_enabled=True,
        thp_enabled="madvise", swappiness=60, gpu_name=gpu, gpu_compute_cap="8.6",
        driver_version=driver, cuda_version="12.4",
        gpu_clock_graphics_mhz=1777, gpu_clock_memory_mhz=7500,
        gpu_clock_graphics_max_mhz=1777, gpu_clock_memory_max_mhz=7500,
        gpu_power_limit_w=170, gpu_persistence_mode=False, gpu_ecc_enabled=False,
        pcie_link_gen_current=3, pcie_link_width_current=16,
        pcie_link_gen_max=4, pcie_link_width_max=16, pcie_health_gbps=11.8,
        in_container=True, python_version="3.13.0", torch_version="2.6.0",
        trt_version="10.16.1",
    )
    config = RunConfig(
        schema_version=SCHEMA_VERSION, git_sha="abc1234", git_dirty=False,
        label="test", clip_path="/tmp/clip.mp4",
        clip_frames_total=300, clip_frames_warmup=8, clip_frames_measured=292,
        proxy_w=1440, proxy_h=810, full_w=3840, full_h=2160,
        batch_size=4, tensorrt=True, torch_compile=False, n_samples=len(fps_samples),
        system_state=sys_state, pinning_applied=None,
    )
    agg_dummy = MetricAggregate.from_samples([0.0] * len(fps_samples))
    throughput = ThroughputMetrics(
        steady_state_fps=MetricAggregate.from_samples(fps_samples),
        time_to_first_frame_ms=agg_dummy,
        wall_ms_per_frame=agg_dummy,
        gpu_kernel_ms_per_frame=MetricAggregate.from_samples(gpu_kernel_samples),
        gpu_thread_wall_ms_per_frame=agg_dummy,
        decode_thread_wall_ms_per_frame=agg_dummy,
        encode_thread_wall_ms_per_frame=agg_dummy,
    )
    return BenchResult(
        schema_version=SCHEMA_VERSION, config=config, throughput=throughput,
        nvtx_spans=[], vram=VramStats(peak_allocated_mb=agg_dummy, peak_reserved_mb=agg_dummy),
    )


class TestCompare:
    def test_significant_improvement_flagged(self):
        from compare import compare_results

        baseline = _make_result(
            fps_samples=[2.77, 2.78, 2.76, 2.79, 2.77],
            gpu_kernel_samples=[298.5, 298.6, 298.4, 298.7, 298.5],
        )
        candidate = _make_result(
            fps_samples=[2.92, 2.94, 2.93, 2.95, 2.93],
            gpu_kernel_samples=[289.2, 289.3, 289.1, 289.4, 289.2],
        )
        report = compare_results(baseline, candidate, label_a="base", label_b="cand")
        # FPS delta ~+5.6%, p should be very small
        assert "steady_state_fps" in report
        lines = report.splitlines()
        fps_line = next(l for l in lines if "steady_state_fps" in l and "|" in l)
        assert "***" in fps_line, f"Expected *** in fps_line: {fps_line}"

    def test_no_change_not_flagged(self):
        from compare import compare_results

        baseline = _make_result(
            fps_samples=[2.77, 2.79, 2.78, 2.76, 2.78],
            gpu_kernel_samples=[298.5, 298.7, 298.6, 298.4, 298.5],
        )
        candidate = _make_result(
            fps_samples=[2.78, 2.77, 2.79, 2.76, 2.78],
            gpu_kernel_samples=[298.4, 298.6, 298.7, 298.5, 298.5],
        )
        report = compare_results(baseline, candidate, label_a="base", label_b="cand")
        lines = report.splitlines()
        fps_line = next(l for l in lines if "steady_state_fps" in l and "|" in l)
        assert "***" not in fps_line

    def test_hardware_mismatch_raises(self):
        from compare import compare_results, HardwareMismatchError

        baseline = _make_result([2.77]*5, [298.5]*5, driver="550.54.14")
        candidate = _make_result([2.77]*5, [298.5]*5, driver="550.54.15")
        with pytest.raises(HardwareMismatchError, match="driver"):
            compare_results(baseline, candidate, label_a="a", label_b="b")

    def test_hardware_mismatch_force(self):
        from compare import compare_results

        baseline = _make_result([2.77]*5, [298.5]*5, driver="550.54.14")
        candidate = _make_result([2.77]*5, [298.5]*5, driver="550.54.15")
        report = compare_results(baseline, candidate, label_a="a", label_b="b", force=True)
        assert "WARNING" in report
