"""Schema for bench harness JSON results.

Pinned at schema_version=1. Changes to dataclass fields require a version bump
and migration logic in BenchResult.load().
"""

import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass
class MetricAggregate:
    """Aggregated metric across N samples."""
    n_samples: int
    mean: float
    stddev: float
    median: float
    ci95_lo: float
    ci95_hi: float
    raw: list[float]

    @classmethod
    def from_samples(cls, samples: list[float]) -> "MetricAggregate":
        if not samples:
            raise ValueError("MetricAggregate requires at least one sample")
        n = len(samples)
        sorted_s = sorted(samples)
        mean = sum(samples) / n
        if n == 1:
            stddev = 0.0
            ci95_lo = mean
            ci95_hi = mean
        else:
            variance = sum((x - mean) ** 2 for x in samples) / (n - 1)
            stddev = math.sqrt(variance)
            stderr = stddev / math.sqrt(n)
            from scipy import stats
            tcrit = stats.t.ppf(0.975, n - 1)
            ci95_lo = mean - tcrit * stderr
            ci95_hi = mean + tcrit * stderr
        median = sorted_s[n // 2] if n % 2 == 1 else (sorted_s[n // 2 - 1] + sorted_s[n // 2]) / 2
        return cls(
            n_samples=n,
            mean=mean,
            stddev=stddev,
            median=median,
            ci95_lo=ci95_lo,
            ci95_hi=ci95_hi,
            raw=list(samples),
        )


@dataclass
class SystemState:
    kernel_version: str
    cpu_governor: str
    cpu_turbo_enabled: bool
    thp_enabled: str
    swappiness: int
    gpu_name: str
    gpu_compute_cap: str
    driver_version: str
    cuda_version: str
    gpu_clock_graphics_mhz: int
    gpu_clock_memory_mhz: int
    gpu_clock_graphics_max_mhz: int
    gpu_clock_memory_max_mhz: int
    gpu_power_limit_w: int
    gpu_persistence_mode: bool
    gpu_ecc_enabled: bool
    pcie_link_gen_current: int
    pcie_link_width_current: int
    pcie_link_gen_max: int
    pcie_link_width_max: int
    pcie_health_gbps: float
    in_container: bool
    python_version: str
    torch_version: str
    trt_version: str | None


@dataclass
class PinningApplied:
    gpu_clocks_locked: bool
    gpu_mem_clocks_locked: bool
    persistence_mode: bool
    cpu_governor_set: str | None
    page_caches_dropped: bool
    clip_prewarmed: bool


@dataclass
class RunConfig:
    schema_version: int
    git_sha: str
    git_dirty: bool
    label: str
    clip_path: str
    clip_frames_total: int
    clip_frames_warmup: int
    clip_frames_measured: int
    proxy_w: int
    proxy_h: int
    full_w: int
    full_h: int
    batch_size: int
    tensorrt: bool
    torch_compile: bool
    n_samples: int
    system_state: SystemState
    pinning_applied: PinningApplied | None


@dataclass
class ThroughputMetrics:
    steady_state_fps: MetricAggregate
    time_to_first_frame_ms: MetricAggregate
    wall_ms_per_frame: MetricAggregate
    gpu_kernel_ms_per_frame: MetricAggregate
    gpu_thread_wall_ms_per_frame: MetricAggregate
    decode_thread_wall_ms_per_frame: MetricAggregate
    encode_thread_wall_ms_per_frame: MetricAggregate


@dataclass
class NvtxSpanAggregate:
    name: str
    count_per_sample: MetricAggregate
    p50_ms: MetricAggregate
    p95_ms: MetricAggregate
    total_ms: MetricAggregate


@dataclass
class VramStats:
    peak_allocated_mb: MetricAggregate
    peak_reserved_mb: MetricAggregate


@dataclass
class BenchResult:
    schema_version: int
    config: RunConfig
    throughput: ThroughputMetrics
    nvtx_spans: list[NvtxSpanAggregate]
    vram: VramStats
    # Phase 2 slots (None in Phase 1)
    top_ops: list | None = None
    dmon_samples: list | None = None
    dram_throughput_pct: MetricAggregate | None = None
    sm_throughput_pct: MetricAggregate | None = None
    pyspy_flamegraph_path: str | None = None
    torch_trace_path: str | None = None
    nsys_trace_path: str | None = None
    ncu_report_path: str | None = None

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "BenchResult":
        with open(path, "r") as f:
            data = json.load(f)
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version mismatch: file={data.get('schema_version')}, "
                f"current={SCHEMA_VERSION}. Re-run the bench to upgrade."
            )
        return _from_dict(cls, data)


def _from_dict(cls: type, data: Any) -> Any:
    """Recursively reconstruct dataclasses from dict (typing-aware)."""
    if data is None:
        return None
    if not is_dataclass(cls):
        return data
    kwargs = {}
    import typing
    hints = typing.get_type_hints(cls)
    for fname, ftype in hints.items():
        if fname not in data:
            continue
        value = data[fname]
        origin = typing.get_origin(ftype)
        args = typing.get_args(ftype)
        if origin is typing.Union:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                ftype = non_none[0]
                origin = typing.get_origin(ftype)
                args = typing.get_args(ftype)
        if origin is list and args and is_dataclass(args[0]):
            kwargs[fname] = [_from_dict(args[0], item) for item in (value or [])]
        elif is_dataclass(ftype):
            kwargs[fname] = _from_dict(ftype, value)
        else:
            kwargs[fname] = value
    return cls(**kwargs)
