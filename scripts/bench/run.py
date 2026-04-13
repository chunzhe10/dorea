"""Bench harness orchestrator for raune_filter.

Runs raune_filter.py N times under identical config, parses per-run timing
from stderr, aggregates via MetricAggregate, writes versioned JSON.

Usage:
    python scripts/bench/run.py --clip <path> --proxy WxH --label <name>

See docs/decisions/2026-04-13-raune-filter-profiling-harness-design.md
"""

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))

from schema import (
    BenchResult, MetricAggregate, NvtxSpanAggregate, PinningApplied,
    RunConfig, SCHEMA_VERSION, SystemState, ThroughputMetrics, VramStats,
)
from sysstate import capture_system_state


_TIMING_RE = re.compile(
    r"\[raune-filter\] stage timing \(busy ms/frame\): "
    r"decode=([\d.]+) gpu_thread=([\d.]+) gpu_kernel=([\d.]+) "
    r"encode=([\d.]+) wall=([\d.]+)"
)
_DONE_RE = re.compile(
    r"\[raune-filter\] done: (\d+) frames in ([\d.]+)s \(([\d.]+) fps\)"
)


def _git_sha() -> str:
    out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    return out


def _git_dirty() -> bool:
    rc = subprocess.call(["git", "diff", "--quiet", "HEAD"])
    return rc != 0


def _run_once(args, sample_idx: int) -> dict:
    """Run raune_filter once, return per-sample metrics dict."""
    python = "/opt/dorea-venv/bin/python"
    # PYTHONPATH points to repos/dorea/python/
    env_pythonpath = str(_THIS.parent.parent.parent / "python")

    cmd = [
        python, "-m", "dorea_inference.raune_filter",
        "--weights", args.weights,
        "--models-dir", args.models_dir,
        "--full-width", str(args.full_w),
        "--full-height", str(args.full_h),
        "--proxy-width", str(args.proxy_w),
        "--proxy-height", str(args.proxy_h),
        "--batch-size", str(args.batch_size),
        "--input", args.clip,
        "--output", f"/tmp/bench_out_{sample_idx}.mov",
        "--output-codec", "prores_ks",
    ]
    if args.tensorrt:
        cmd.append("--tensorrt")
    if args.torch_compile:
        cmd.append("--torch-compile")

    env = os.environ.copy()
    env["PYTHONPATH"] = env_pythonpath

    t_start = time.perf_counter()
    proc = subprocess.run(
        cmd, env=env, capture_output=True, text=True, check=False,
    )
    wall_time_s = time.perf_counter() - t_start

    if proc.returncode != 0:
        print(f"[bench] raune_filter failed (exit {proc.returncode}):", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError("raune_filter subprocess failed")

    stderr = proc.stderr
    m = _TIMING_RE.search(stderr)
    if not m:
        raise RuntimeError(f"Could not parse stage timing from stderr:\n{stderr}")
    decode_ms, gpu_thread_ms, gpu_kernel_ms, encode_ms, wall_ms = map(float, m.groups())

    m_done = _DONE_RE.search(stderr)
    if not m_done:
        raise RuntimeError(f"Could not parse done line from stderr:\n{stderr}")
    total_frames = int(m_done.group(1))
    reported_wall_s = float(m_done.group(2))
    reported_fps = float(m_done.group(3))

    # Startup cost = subprocess wall time - reported processing wall time
    # (includes TRT engine load, CUDA init, Triton JIT, etc.)
    startup_s = max(0.0, wall_time_s - reported_wall_s)
    time_to_first_frame_ms = startup_s * 1000 + wall_ms  # startup + first frame

    # Steady state: exclude warmup frames from the fps calculation
    warmup = args.warmup_frames
    if warmup > 0 and total_frames > warmup:
        measured_frames = total_frames - warmup
        warmup_wall_s = warmup * wall_ms / 1000
        steady_wall_s = max(reported_wall_s - warmup_wall_s, 1e-6)
        steady_state_fps = measured_frames / steady_wall_s
    else:
        steady_state_fps = reported_fps

    return {
        "total_frames": total_frames,
        "steady_state_fps": steady_state_fps,
        "time_to_first_frame_ms": time_to_first_frame_ms,
        "wall_ms_per_frame": wall_ms,
        "gpu_kernel_ms_per_frame": gpu_kernel_ms,
        "gpu_thread_wall_ms_per_frame": gpu_thread_ms,
        "decode_thread_wall_ms_per_frame": decode_ms,
        "encode_thread_wall_ms_per_frame": encode_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="raune_filter bench harness")
    parser.add_argument("--clip", required=True, help="Input video path")
    parser.add_argument("--proxy", required=True,
                        help="Proxy resolution as WxH (e.g. 1440x810)")
    parser.add_argument("--label", required=True, help="Run label")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup-frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--full-w", type=int, default=3840)
    parser.add_argument("--full-h", type=int, default=2160)
    parser.add_argument("--weights", default="/workspaces/dorea-workspace/models/raune_net/weights_95.pth")
    parser.add_argument("--models-dir", default="/workspaces/dorea-workspace/models/raune_net")
    parser.add_argument("--tensorrt", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--output-dir", default=str(_THIS.parent / "results"))
    args = parser.parse_args()

    m = re.match(r"(\d+)x(\d+)", args.proxy)
    if not m:
        print(f"error: --proxy must be WxH, got {args.proxy}", file=sys.stderr)
        return 1
    args.proxy_w = int(m.group(1))
    args.proxy_h = int(m.group(2))

    if not Path(args.clip).exists():
        print(f"error: clip not found: {args.clip}", file=sys.stderr)
        return 1

    # Probe clip for frame count
    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_packets", "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0", args.clip,
    ]).decode().strip()
    clip_frames = int(probe)

    print(f"[bench] Capturing system state...", file=sys.stderr)
    sys_state = capture_system_state()
    print(f"[bench] GPU: {sys_state.gpu_name}, driver {sys_state.driver_version}, "
          f"PCIe gen{sys_state.pcie_link_gen_current} x{sys_state.pcie_link_width_current} "
          f"({sys_state.pcie_health_gbps:.1f} GB/s measured)", file=sys.stderr)

    expected_gbps = {3: 12.0, 4: 25.0}.get(sys_state.pcie_link_gen_current, 0)
    if expected_gbps and sys_state.pcie_health_gbps < 0.8 * expected_gbps:
        print(f"[bench] WARNING: PCIe bandwidth {sys_state.pcie_health_gbps:.1f} GB/s "
              f"is below 80% of expected {expected_gbps:.1f} GB/s", file=sys.stderr)

    per_sample = []
    for i in range(args.samples):
        print(f"[bench] Running sample {i+1}/{args.samples}...", file=sys.stderr)
        sample = _run_once(args, i)
        per_sample.append(sample)
        print(f"[bench]   steady_state_fps={sample['steady_state_fps']:.2f} "
              f"gpu_kernel={sample['gpu_kernel_ms_per_frame']:.1f}ms/f",
              file=sys.stderr)

    def _agg(key: str) -> MetricAggregate:
        return MetricAggregate.from_samples([s[key] for s in per_sample])

    throughput = ThroughputMetrics(
        steady_state_fps=_agg("steady_state_fps"),
        time_to_first_frame_ms=_agg("time_to_first_frame_ms"),
        wall_ms_per_frame=_agg("wall_ms_per_frame"),
        gpu_kernel_ms_per_frame=_agg("gpu_kernel_ms_per_frame"),
        gpu_thread_wall_ms_per_frame=_agg("gpu_thread_wall_ms_per_frame"),
        decode_thread_wall_ms_per_frame=_agg("decode_thread_wall_ms_per_frame"),
        encode_thread_wall_ms_per_frame=_agg("encode_thread_wall_ms_per_frame"),
    )

    # Phase 1: VRAM zeroed (requires raune_filter to emit peak_vram_mb, deferred)
    vram = VramStats(
        peak_allocated_mb=MetricAggregate.from_samples([0.0] * args.samples),
        peak_reserved_mb=MetricAggregate.from_samples([0.0] * args.samples),
    )

    # Phase 1: empty NVTX spans (populated in Phase 2 via torch.profiler)
    nvtx_spans = []

    config = RunConfig(
        schema_version=SCHEMA_VERSION,
        git_sha=_git_sha(),
        git_dirty=_git_dirty(),
        label=args.label,
        clip_path=str(Path(args.clip).resolve()),
        clip_frames_total=clip_frames,
        clip_frames_warmup=args.warmup_frames,
        clip_frames_measured=clip_frames - args.warmup_frames,
        proxy_w=args.proxy_w,
        proxy_h=args.proxy_h,
        full_w=args.full_w,
        full_h=args.full_h,
        batch_size=args.batch_size,
        tensorrt=args.tensorrt,
        torch_compile=args.torch_compile,
        n_samples=args.samples,
        system_state=sys_state,
        pinning_applied=None,
    )

    result = BenchResult(
        schema_version=SCHEMA_VERSION,
        config=config,
        throughput=throughput,
        nvtx_spans=nvtx_spans,
        vram=vram,
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    safe_label = re.sub(r"[^a-zA-Z0-9_-]", "_", args.label)
    out_path = Path(args.output_dir) / f"{ts}_{config.git_sha}_{safe_label}.json"
    result.save(out_path)
    print(f"[bench] Wrote {out_path}", file=sys.stderr)
    ci_half = throughput.steady_state_fps.ci95_hi - throughput.steady_state_fps.mean
    print(f"[bench] steady_state_fps: {throughput.steady_state_fps.mean:.3f} "
          f"± {ci_half:.3f} (95% CI, N={args.samples})",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
