"""Compare two BenchResult JSON files with Welch's t-test.

Hard-errors on hardware/driver mismatch unless --force.
Emits a markdown report to stdout or a file.
"""

import argparse
import sys
from pathlib import Path

from scipy import stats

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))

from schema import BenchResult, MetricAggregate


class HardwareMismatchError(Exception):
    """Raised when two results were measured on different hardware/drivers."""


def _check_hardware(a: BenchResult, b: BenchResult) -> list[str]:
    """Return list of human-readable mismatches. Empty list = match."""
    mismatches = []
    sa = a.config.system_state
    sb = b.config.system_state
    checks = [
        ("gpu_name", sa.gpu_name, sb.gpu_name),
        ("compute_cap", sa.gpu_compute_cap, sb.gpu_compute_cap),
        ("driver_version", sa.driver_version, sb.driver_version),
        ("cuda_version", sa.cuda_version, sb.cuda_version),
        ("torch_version", sa.torch_version, sb.torch_version),
        ("trt_version", sa.trt_version, sb.trt_version),
        ("pcie_link_gen_current", sa.pcie_link_gen_current, sb.pcie_link_gen_current),
        ("pcie_link_width_current", sa.pcie_link_width_current, sb.pcie_link_width_current),
    ]
    for name, va, vb in checks:
        if va != vb:
            mismatches.append(f"{name}: {va} → {vb}")
    return mismatches


def _welch(a: MetricAggregate, b: MetricAggregate) -> tuple[float, float]:
    """Return (delta_pct, p_value). delta_pct is (b-a)/a * 100."""
    if a.mean == 0:
        delta_pct = 0.0 if b.mean == 0 else float("inf")
    else:
        delta_pct = (b.mean - a.mean) / a.mean * 100
    if len(a.raw) < 2 or len(b.raw) < 2:
        return delta_pct, 1.0
    res = stats.ttest_ind(a.raw, b.raw, equal_var=False)
    p_value = float(res.pvalue)
    return delta_pct, p_value


def _sig_marker(delta_pct: float, p_value: float) -> str:
    abs_delta = abs(delta_pct)
    if p_value < 0.01 and abs_delta > 2.0:
        return "***"
    if p_value < 0.05 and abs_delta > 1.0:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _fmt_metric(name: str, a: MetricAggregate, b: MetricAggregate) -> str:
    delta, p = _welch(a, b)
    sig = _sig_marker(delta, p)
    a_ci = a.ci95_hi - a.mean
    b_ci = b.ci95_hi - b.mean
    sign = "+" if delta >= 0 else ""
    return (f"| {name} | {a.mean:.3f} ± {a_ci:.3f} | {b.mean:.3f} ± {b_ci:.3f} "
            f"| {sign}{delta:.1f}% | {p:.3g} | {sig} |")


def compare_results(
    a: BenchResult,
    b: BenchResult,
    label_a: str,
    label_b: str,
    force: bool = False,
) -> str:
    """Return a markdown report comparing two BenchResults."""
    mismatches = _check_hardware(a, b)
    lines = [f"# Benchmark: {label_a} vs {label_b}", ""]

    if mismatches:
        if not force:
            raise HardwareMismatchError(
                "Hardware/state mismatch:\n  " + "\n  ".join(mismatches) +
                "\nUse force=True to override (results will be misleading)."
            )
        lines.append("**WARNING: Hardware mismatch — results may be misleading**")
        for m in mismatches:
            lines.append(f"  - {m}")
        lines.append("")

    lines.append("**Config:**")
    ca, cb = a.config, b.config
    if ca.tensorrt != cb.tensorrt:
        lines.append(f"- tensorrt: {ca.tensorrt} → {cb.tensorrt}")
    if ca.torch_compile != cb.torch_compile:
        lines.append(f"- torch_compile: {ca.torch_compile} → {cb.torch_compile}")
    if (ca.proxy_w, ca.proxy_h) != (cb.proxy_w, cb.proxy_h):
        lines.append(f"- proxy: {ca.proxy_w}x{ca.proxy_h} → {cb.proxy_w}x{cb.proxy_h}")
    if ca.batch_size != cb.batch_size:
        lines.append(f"- batch_size: {ca.batch_size} → {cb.batch_size}")
    lines.append(f"- samples: {ca.n_samples} each")
    lines.append("")

    lines.append("## Throughput (means ± 95% CI)")
    lines.append("")
    lines.append(f"| Metric | {label_a} | {label_b} | Δ | p | sig |")
    lines.append("|---|---:|---:|---:|---:|:---:|")
    ta, tb = a.throughput, b.throughput
    lines.append(_fmt_metric("steady_state_fps", ta.steady_state_fps, tb.steady_state_fps))
    lines.append(_fmt_metric("gpu_kernel_ms_per_frame", ta.gpu_kernel_ms_per_frame, tb.gpu_kernel_ms_per_frame))
    lines.append(_fmt_metric("gpu_thread_wall_ms_per_frame", ta.gpu_thread_wall_ms_per_frame, tb.gpu_thread_wall_ms_per_frame))
    lines.append(_fmt_metric("wall_ms_per_frame", ta.wall_ms_per_frame, tb.wall_ms_per_frame))
    lines.append(_fmt_metric("decode_thread_wall_ms_per_frame", ta.decode_thread_wall_ms_per_frame, tb.decode_thread_wall_ms_per_frame))
    lines.append(_fmt_metric("encode_thread_wall_ms_per_frame", ta.encode_thread_wall_ms_per_frame, tb.encode_thread_wall_ms_per_frame))
    lines.append(_fmt_metric("time_to_first_frame_ms", ta.time_to_first_frame_ms, tb.time_to_first_frame_ms))
    lines.append("")

    lines.append("**Significance:** `***` p<0.01 ∧ |Δ|>2%, `**` p<0.05 ∧ |Δ|>1%, `*` p<0.05, blank = within noise")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two bench results")
    parser.add_argument("baseline", help="Path to baseline JSON")
    parser.add_argument("candidate", help="Path to candidate JSON")
    parser.add_argument("--output", help="Write report to file (default: stdout)")
    parser.add_argument("--force", action="store_true",
                        help="Allow comparison across different hardware")
    args = parser.parse_args()

    a = BenchResult.load(Path(args.baseline))
    b = BenchResult.load(Path(args.candidate))

    try:
        report = compare_results(a, b, args.baseline, args.candidate, force=args.force)
    except HardwareMismatchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.output:
        Path(args.output).write_text(report)
        print(f"[compare] Wrote {args.output}", file=sys.stderr)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
