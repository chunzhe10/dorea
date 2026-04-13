"""Tests for bench harness JSON schema (v2)."""

import json
import sys
from pathlib import Path

import pytest
import torch

_BENCH_DIR = Path(__file__).resolve().parents[2] / "scripts" / "bench"
sys.path.insert(0, str(_BENCH_DIR))


def _sample_aggregate():
    from schema import MetricAggregate
    return MetricAggregate.from_samples([2.77, 2.81, 2.75, 2.83, 2.78])


def _sample_system_state():
    from schema import SystemState
    return SystemState(
        kernel_version="6.8.0-generic",
        cpu_governor="performance",
        cpu_turbo_enabled=True,
        thp_enabled="[always] madvise never",
        swappiness=60,
        gpu_name="NVIDIA GeForce RTX 3060",
        gpu_compute_cap="8.6",
        driver_version="550.54.14",
        cuda_version="12.4",
        gpu_clock_graphics_mhz=1777,
        gpu_clock_memory_mhz=7500,
        gpu_clock_graphics_max_mhz=1777,
        gpu_clock_memory_max_mhz=7500,
        gpu_power_limit_w=170,
        gpu_persistence_mode=False,
        gpu_ecc_enabled=False,
        pcie_link_gen_current=3,
        pcie_link_width_current=16,
        pcie_link_gen_max=4,
        pcie_link_width_max=16,
        pcie_health_gbps=11.8,
        in_container=True,
        python_version="3.13.0",
        torch_version="2.6.0",
        trt_version="10.16.1",
    )


class TestMetricAggregate:
    def test_from_samples_single_value_raises(self):
        from schema import MetricAggregate
        with pytest.raises(ValueError, match="N >= 2"):
            MetricAggregate.from_samples([5.0])

    def test_from_samples_known_values(self):
        from schema import MetricAggregate
        m = MetricAggregate.from_samples([1.0, 2.0, 3.0, 4.0, 5.0])
        assert m.n_samples == 5
        assert m.mean == 3.0
        assert abs(m.stddev - 1.5811) < 1e-3
        assert m.median == 3.0
        # t(0.975, 4) ≈ 2.776, stderr ≈ 0.707, CI half-width ≈ 1.963
        assert abs(m.ci95_lo - (3.0 - 1.963)) < 0.1
        assert abs(m.ci95_hi - (3.0 + 1.963)) < 0.1

    def test_from_samples_even_length_median(self):
        from schema import MetricAggregate
        # Median of [1,2,3,4] should be 2.5, not 2 or 3
        m = MetricAggregate.from_samples([1.0, 2.0, 3.0, 4.0])
        assert m.median == 2.5

    def test_raw_samples_retained(self):
        from schema import MetricAggregate
        raw = [1.1, 2.2, 3.3]
        m = MetricAggregate.from_samples(raw)
        assert m.raw == raw

    def test_empty_raises(self):
        from schema import MetricAggregate
        with pytest.raises(ValueError, match="at least one sample"):
            MetricAggregate.from_samples([])

    def test_nan_rejected(self):
        from schema import MetricAggregate
        with pytest.raises(ValueError, match="NaN"):
            MetricAggregate.from_samples([float("nan"), 1.0, 2.0])

    def test_inf_rejected(self):
        from schema import MetricAggregate
        with pytest.raises(ValueError, match="NaN"):
            MetricAggregate.from_samples([float("inf"), 1.0, 2.0])


class TestSchemaRoundTrip:
    def test_bench_result_round_trip(self, tmp_path):
        from schema import (
            BenchResult, RunConfig, ThroughputMetrics,
            NvtxSpanAggregate, SCHEMA_VERSION,
        )

        # Use distinct aggregates per metric so a field-assignment bug in
        # _from_dict would be caught by the round trip.
        fps_agg = __import__("schema").MetricAggregate.from_samples(
            [2.77, 2.81, 2.75, 2.83, 2.78]
        )
        gpu_agg = __import__("schema").MetricAggregate.from_samples(
            [324.5, 325.1, 323.8, 324.9, 324.3]
        )
        wall_agg = __import__("schema").MetricAggregate.from_samples(
            [354.4, 355.1, 353.8, 354.9, 354.3]
        )
        decode_agg = __import__("schema").MetricAggregate.from_samples(
            [24.5, 24.7, 24.3, 24.6, 24.4]
        )
        encode_agg = __import__("schema").MetricAggregate.from_samples(
            [47.0, 47.1, 46.9, 47.0, 47.0]
        )
        thread_agg = __import__("schema").MetricAggregate.from_samples(
            [324.6, 325.2, 323.9, 325.0, 324.4]
        )

        result = BenchResult(
            schema_version=SCHEMA_VERSION,
            config=RunConfig(
                schema_version=SCHEMA_VERSION,
                git_sha="abc1234",
                git_dirty=False,
                label="test",
                clip_path="/tmp/clip.mp4",
                clip_frames_total=300,
                clip_frames_warmup=8,
                clip_frames_measured=292,
                proxy_w=1440,
                proxy_h=810,
                full_w=3840,
                full_h=2160,
                batch_size=4,
                tensorrt=True,
                torch_compile=False,
                n_samples=5,
                system_state=_sample_system_state(),
                pinning_applied=None,
            ),
            throughput=ThroughputMetrics(
                steady_state_fps=fps_agg,
                wall_ms_per_frame=wall_agg,
                gpu_kernel_ms_per_frame=gpu_agg,
                gpu_thread_wall_ms_per_frame=thread_agg,
                decode_thread_wall_ms_per_frame=decode_agg,
                encode_thread_wall_ms_per_frame=encode_agg,
            ),
            nvtx_spans=[],
        )

        path = tmp_path / "result.json"
        result.save(path)
        loaded = BenchResult.load(path)
        assert loaded == result
        assert loaded.schema_version == SCHEMA_VERSION
        assert loaded.config.git_sha == "abc1234"
        # Check fields were assigned correctly (distinct values would catch swap bugs)
        assert loaded.throughput.steady_state_fps.mean == fps_agg.mean
        assert loaded.throughput.gpu_kernel_ms_per_frame.mean == gpu_agg.mean
        assert loaded.throughput.decode_thread_wall_ms_per_frame.mean == decode_agg.mean

    def test_schema_version_mismatch_rejected(self, tmp_path):
        from schema import BenchResult
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"schema_version": 999}))
        with pytest.raises(ValueError, match="schema_version"):
            BenchResult.load(path)

    def test_schema_version_missing_rejected(self, tmp_path):
        from schema import BenchResult
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"no_version_field": True}))
        with pytest.raises(ValueError, match="schema_version"):
            BenchResult.load(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestSysStateCapture:
    def test_capture_returns_systemstate(self):
        from sysstate import capture_system_state
        from schema import SystemState

        state = capture_system_state()
        assert isinstance(state, SystemState)
        assert state.gpu_name  # non-empty
        assert state.driver_version  # non-empty
        assert state.pcie_link_gen_current in (1, 2, 3, 4, 5)
        assert state.pcie_health_gbps > 0
        assert state.python_version.startswith("3.")
        assert state.torch_version  # non-empty
