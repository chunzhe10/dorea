"""Bench harness orchestrator for raune_filter.

Runs raune_filter.py N times under identical config, parses per-run timing
from stderr, aggregates via MetricAggregate, writes versioned JSON.

Usage:
    python scripts/bench/run.py --clip <path> --proxy WxH --label <name>

Environment variables:
    DOREA_RAUNE_WEIGHTS  Default weights path (overridable via --weights)
    DOREA_MODELS_DIR     Default models dir (overridable via --models-dir)
    DOREA_TEST_CLIP      Default clip path (overridable via --clip)

See docs/decisions/2026-04-13-raune-filter-profiling-harness-design.md
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))

from schema import (
    BenchResult, MetricAggregate, RunConfig, SCHEMA_VERSION,
    ThroughputMetrics,
)
from sysstate import capture_system_state


# ────────────────────────────────────────────────────────────────────────
# Stderr parsing (pure functions — no subprocess/IO, fully unit testable)
# ────────────────────────────────────────────────────────────────────────

_TIMING_RE = re.compile(
    r"\[raune-filter\] stage timing \(busy ms/frame\): "
    r"decode=([\d.]+) gpu_thread=([\d.]+) gpu_kernel=([\d.]+) "
    r"encode=([\d.]+) wall=([\d.]+)"
)
_DONE_RE = re.compile(
    r"\[raune-filter\] done: (\d+) frames in ([\d.]+)s \(([\d.]+) fps\)"
)


class SampleMetrics(NamedTuple):
    total_frames: int
    steady_state_fps: float
    wall_ms_per_frame: float
    gpu_kernel_ms_per_frame: float
    gpu_thread_wall_ms_per_frame: float
    decode_thread_wall_ms_per_frame: float
    encode_thread_wall_ms_per_frame: float


def parse_raune_stderr(stderr: str, warmup_frames: int) -> SampleMetrics:
    """Parse raune_filter stderr into per-sample metrics.

    Raises ValueError on missing/malformed lines.
    """
    m = _TIMING_RE.search(stderr)
    if not m:
        raise ValueError("Could not parse stage timing line from stderr")
    decode_ms, gpu_thread_ms, gpu_kernel_ms, encode_ms, wall_ms = map(float, m.groups())

    m_done = _DONE_RE.search(stderr)
    if not m_done:
        raise ValueError("Could not parse 'done' line from stderr")
    total_frames = int(m_done.group(1))
    reported_wall_s = float(m_done.group(2))
    reported_fps = float(m_done.group(3))

    # Steady state: exclude warmup frames from the fps calculation.
    # NOTE: this uses the per-frame mean wall time as an approximation for
    # warmup cost. The first frames are actually slower than mean, so this
    # under-subtracts warmup and biases steady_state_fps pessimistically.
    # Acceptable for Phase 1; proper fix is to have raune_filter emit the
    # warmup and steady-state wall times separately.
    if warmup_frames > 0 and total_frames > warmup_frames:
        measured_frames = total_frames - warmup_frames
        warmup_wall_s = warmup_frames * wall_ms / 1000
        steady_wall_s = max(reported_wall_s - warmup_wall_s, 1e-6)
        steady_state_fps = measured_frames / steady_wall_s
    else:
        steady_state_fps = reported_fps

    return SampleMetrics(
        total_frames=total_frames,
        steady_state_fps=steady_state_fps,
        wall_ms_per_frame=wall_ms,
        gpu_kernel_ms_per_frame=gpu_kernel_ms,
        gpu_thread_wall_ms_per_frame=gpu_thread_ms,
        decode_thread_wall_ms_per_frame=decode_ms,
        encode_thread_wall_ms_per_frame=encode_ms,
    )


def _tail(text: str, n_lines: int = 50) -> str:
    """Return the last n_lines of text."""
    lines = text.splitlines()
    return "\n".join(lines[-n_lines:])


# ────────────────────────────────────────────────────────────────────────
# Git state
# ────────────────────────────────────────────────────────────────────────

def _git_sha(cwd: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(cwd),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_dirty(cwd: Path) -> bool:
    """Check if working tree is dirty (including untracked files)."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(cwd),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return bool(out)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# ────────────────────────────────────────────────────────────────────────
# Sample orchestration
# ────────────────────────────────────────────────────────────────────────

# Max allowed subprocess wall time per sample. 5 minutes covers even slow
# cold-start runs (TRT engine build can take 2-5 min if the cache is cold).
_SAMPLE_TIMEOUT_S = 300


def _run_once(args, sample_idx: int, output_dir: Path) -> SampleMetrics:
    """Run raune_filter once, return parsed per-sample metrics."""
    python = args.python
    env_pythonpath = str(_THIS.parent.parent.parent / "python")

    out_file = output_dir / f"bench_out_{sample_idx}.mov"

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
        "--output", str(out_file),
        "--output-codec", "prores_ks",
    ]
    if args.tensorrt:
        cmd.append("--tensorrt")
    if args.torch_compile:
        cmd.append("--torch-compile")

    env = os.environ.copy()
    env["PYTHONPATH"] = env_pythonpath
    # Enable CUDA event timing in raune_filter's _process_batch
    env["DOREA_BENCH"] = "1"

    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, check=False,
            timeout=_SAMPLE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as e:
        stderr_tail = _tail((e.stderr or "").decode() if isinstance(e.stderr, bytes) else (e.stderr or ""), 50)
        raise RuntimeError(
            f"raune_filter subprocess timed out after {_SAMPLE_TIMEOUT_S}s "
            f"(sample {sample_idx}). Stderr tail:\n{stderr_tail}"
        ) from e

    if proc.returncode != 0:
        stderr_tail = _tail(proc.stderr, 50)
        raise RuntimeError(
            f"raune_filter subprocess failed (exit {proc.returncode}, "
            f"sample {sample_idx}). Stderr tail:\n{stderr_tail}"
        )

    try:
        return parse_raune_stderr(proc.stderr, args.warmup_frames)
    except ValueError as e:
        stderr_tail = _tail(proc.stderr, 50)
        raise RuntimeError(
            f"Failed to parse raune_filter output (sample {sample_idx}): {e}\n"
            f"Stderr tail:\n{stderr_tail}"
        ) from e


# ────────────────────────────────────────────────────────────────────────
# Path resolution
# ────────────────────────────────────────────────────────────────────────

def _default_python() -> str:
    """Default to the currently running Python. User can override via --python."""
    return sys.executable


def _default_weights() -> str | None:
    return os.environ.get("DOREA_RAUNE_WEIGHTS")


def _default_models_dir() -> str | None:
    return os.environ.get("DOREA_MODELS_DIR")


def _default_clip() -> str | None:
    return os.environ.get("DOREA_TEST_CLIP")


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="raune_filter bench harness",
        epilog=(
            "Environment variables for defaults:\n"
            "  DOREA_RAUNE_WEIGHTS   weights .pth path\n"
            "  DOREA_MODELS_DIR      models dir (contains raune_net/)\n"
            "  DOREA_TEST_CLIP       input clip path\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--clip", default=_default_clip(),
                        help="Input video path (default: $DOREA_TEST_CLIP)")
    parser.add_argument("--proxy", required=True,
                        help="Proxy resolution as WxH (e.g. 1440x810)")
    parser.add_argument("--label", required=True, help="Run label")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup-frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--full-w", type=int, default=3840)
    parser.add_argument("--full-h", type=int, default=2160)
    parser.add_argument("--weights", default=_default_weights(),
                        help="Weights path (default: $DOREA_RAUNE_WEIGHTS)")
    parser.add_argument("--models-dir", default=_default_models_dir(),
                        help="Models dir (default: $DOREA_MODELS_DIR)")
    parser.add_argument("--python", default=_default_python(),
                        help="Python interpreter (default: sys.executable)")
    parser.add_argument("--tensorrt", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--output-dir", default=str(_THIS.parent / "results"))
    args = parser.parse_args()

    # Validate required-if-no-default args
    missing = []
    if not args.clip:
        missing.append("--clip (or set DOREA_TEST_CLIP)")
    if not args.weights:
        missing.append("--weights (or set DOREA_RAUNE_WEIGHTS)")
    if not args.models_dir:
        missing.append("--models-dir (or set DOREA_MODELS_DIR)")
    if missing:
        print("error: missing required arguments:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 1

    m = re.match(r"(\d+)x(\d+)", args.proxy)
    if not m:
        print(f"error: --proxy must be WxH, got {args.proxy}", file=sys.stderr)
        return 1
    args.proxy_w = int(m.group(1))
    args.proxy_h = int(m.group(2))

    if args.samples < 2:
        print(f"error: --samples must be >= 2 for statistical comparison, "
              f"got {args.samples}", file=sys.stderr)
        return 1

    clip_path = Path(args.clip)
    if not clip_path.is_file():
        print(f"error: clip not found or not a file: {args.clip}", file=sys.stderr)
        return 1

    if not Path(args.weights).is_file():
        print(f"error: weights not found: {args.weights}", file=sys.stderr)
        return 1

    if not Path(args.models_dir).is_dir():
        print(f"error: models-dir not found: {args.models_dir}", file=sys.stderr)
        return 1

    # Probe clip for frame count
    try:
        probe = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_frames", "-show_entries", "stream=nb_read_frames",
             "-of", "csv=p=0", str(clip_path)],
            timeout=60,
        ).decode().strip()
    except FileNotFoundError:
        print("error: ffprobe not found in PATH", file=sys.stderr)
        return 1
    except subprocess.TimeoutExpired:
        print("error: ffprobe timed out probing clip frame count", file=sys.stderr)
        return 1

    try:
        clip_frames = int(probe)
    except ValueError:
        print(f"error: ffprobe returned non-integer frame count: {probe!r}",
              file=sys.stderr)
        return 1

    if clip_frames <= args.warmup_frames:
        print(f"error: clip has {clip_frames} frames but --warmup-frames is "
              f"{args.warmup_frames}. Need more frames than warmup.", file=sys.stderr)
        return 1

    print(f"[bench] Capturing system state...", file=sys.stderr)
    sys_state = capture_system_state()
    print(f"[bench] GPU: {sys_state.gpu_name}, driver {sys_state.driver_version}, "
          f"PCIe gen{sys_state.pcie_link_gen_current} x{sys_state.pcie_link_width_current} "
          f"({sys_state.pcie_health_gbps:.1f} GB/s measured)", file=sys.stderr)

    expected_gbps = {3: 12.0, 4: 25.0}.get(sys_state.pcie_link_gen_current, 0)
    if expected_gbps and sys_state.pcie_health_gbps < 0.8 * expected_gbps:
        print(f"[bench] WARNING: PCIe bandwidth {sys_state.pcie_health_gbps:.1f} GB/s "
              f"is below 80% of expected {expected_gbps:.1f} GB/s", file=sys.stderr)

    # Run samples in a TemporaryDirectory so per-sample .mov files are
    # cleaned up automatically, even on crash or Ctrl-C.
    per_sample: list[SampleMetrics] = []
    with tempfile.TemporaryDirectory(prefix="dorea_bench_") as tmpdir:
        tmp_path = Path(tmpdir)
        try:
            for i in range(args.samples):
                print(f"[bench] Running sample {i+1}/{args.samples}...", file=sys.stderr)
                sample = _run_once(args, i, tmp_path)
                per_sample.append(sample)
                print(f"[bench]   steady_state_fps={sample.steady_state_fps:.2f} "
                      f"gpu_kernel={sample.gpu_kernel_ms_per_frame:.1f}ms/f",
                      file=sys.stderr)
        except KeyboardInterrupt:
            print("\n[bench] Interrupted by user. Partial results discarded.",
                  file=sys.stderr)
            return 130

    def _agg(attr: str) -> MetricAggregate:
        return MetricAggregate.from_samples([getattr(s, attr) for s in per_sample])

    throughput = ThroughputMetrics(
        steady_state_fps=_agg("steady_state_fps"),
        wall_ms_per_frame=_agg("wall_ms_per_frame"),
        gpu_kernel_ms_per_frame=_agg("gpu_kernel_ms_per_frame"),
        gpu_thread_wall_ms_per_frame=_agg("gpu_thread_wall_ms_per_frame"),
        decode_thread_wall_ms_per_frame=_agg("decode_thread_wall_ms_per_frame"),
        encode_thread_wall_ms_per_frame=_agg("encode_thread_wall_ms_per_frame"),
    )

    # Phase 1: empty NVTX spans (populated in Phase 2 via torch.profiler)
    nvtx_spans: list = []

    # _THIS is scripts/bench/run.py; three .parent calls reach the dorea repo root
    dorea_repo_root = _THIS.parent.parent.parent
    config = RunConfig(
        schema_version=SCHEMA_VERSION,
        git_sha=_git_sha(dorea_repo_root),
        git_dirty=_git_dirty(dorea_repo_root),
        label=args.label,
        clip_path=str(clip_path.resolve()),
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
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    safe_label = re.sub(r"[^a-zA-Z0-9_-]", "_", args.label) or "unlabeled"
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
