"""Compare two BenchResult JSON files with rigorous statistical methods.

Uses:
  - Permutation test (no distributional assumption) on raw samples
  - Holm-Bonferroni multiple-comparison correction across compared metrics
  - TOST-style equivalence classification against a ROPE (region of
    practical equivalence) to avoid false positives on tiny real differences
  - Split hardware check: fatal fields block comparison; advisory fields
    print a warning but continue

Replaces the v1 Welch's t-test approach. Welch's test assumes iid normal
samples, and calibration demonstrated that back-to-back bench samples are
autocorrelated (shared thermal state) so the nominal Type I rate was not
achieved. The permutation test makes no distributional assumptions, and the
ROPE avoids false-positive "significant" findings on sub-noise deltas.

See docs/decisions/2026-04-13-raune-filter-profiling-harness-design.md and
docs/learnings/2026-04-13-bench-harness-noise-floor.md for context.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))

from schema import BenchResult, MetricAggregate


# Region of Practical Equivalence: deltas smaller than this in absolute value
# are considered "practically equivalent" regardless of p-value. Set from
# the calibration run on this workstation (see noise floor learning doc).
DEFAULT_ROPE_PCT = 1.5

# Permutation test iteration count. 9999 gives reasonable p-value granularity.
PERMUTATION_RESAMPLES = 9999


class HardwareMismatchError(Exception):
    """Raised when two results were measured on incompatible hardware."""


# Fatal fields: a mismatch invalidates the comparison entirely.
FATAL_FIELDS = [
    ("gpu_name", "GPU model"),
    ("gpu_compute_cap", "compute capability"),
]

# Advisory fields: a mismatch warrants a warning but doesn't block.
# (pcie_link_gen_current is deliberately omitted — NVML reports it as
# gen1/x8 when the GPU is idle, so sampling is nondeterministic. The
# measured pcie_health_gbps is the honest signal.)
ADVISORY_FIELDS = [
    ("cuda_version", "CUDA version"),
    ("torch_version", "PyTorch version"),
    ("trt_version", "TensorRT version"),
]


def _major_driver(version: str | None) -> str:
    """Strip patch level from driver string. '550.54.14' -> '550.54'."""
    if not version:
        return version or ""
    parts = version.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else version


def _check_hardware(a: BenchResult, b: BenchResult) -> tuple[list[str], list[str]]:
    """Return (fatal_mismatches, advisory_mismatches)."""
    sa = a.config.system_state
    sb = b.config.system_state

    fatal = []
    for attr, label in FATAL_FIELDS:
        va, vb = getattr(sa, attr), getattr(sb, attr)
        if va != vb:
            fatal.append(f"{label}: {va} → {vb}")

    # Major driver counts as fatal, patch-level difference is advisory
    maj_a = _major_driver(sa.driver_version)
    maj_b = _major_driver(sb.driver_version)
    if maj_a != maj_b:
        fatal.append(f"major driver version: {maj_a} → {maj_b}")

    advisory = []
    if sa.driver_version != sb.driver_version and maj_a == maj_b:
        advisory.append(f"driver patch: {sa.driver_version} → {sb.driver_version}")
    for attr, label in ADVISORY_FIELDS:
        va, vb = getattr(sa, attr), getattr(sb, attr)
        if va != vb:
            advisory.append(f"{label}: {va} → {vb}")

    return fatal, advisory


def _permutation_test(
    a: MetricAggregate, b: MetricAggregate, n_resamples: int = PERMUTATION_RESAMPLES
) -> tuple[float, float]:
    """Return (delta_pct, p_value) via two-sided permutation test.

    delta_pct is ((b.mean - a.mean) / a.mean) * 100.
    """
    if a.mean == 0:
        delta_pct = 0.0 if b.mean == 0 else float("inf")
    else:
        delta_pct = (b.mean - a.mean) / a.mean * 100

    a_samples = np.asarray(a.raw, dtype=float)
    b_samples = np.asarray(b.raw, dtype=float)

    def _stat(x, y, axis):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    result = stats.permutation_test(
        (a_samples, b_samples),
        _stat,
        n_resamples=n_resamples,
        alternative="two-sided",
        permutation_type="independent",
        vectorized=True,
    )
    return delta_pct, float(result.pvalue)


def _holm_correct(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni step-down correction.

    Given N raw p-values, return N adjusted p-values in the original order.
    """
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda t: t[1])
    running_max = 0.0
    adjusted_by_rank = []
    for rank, (_orig_idx, p) in enumerate(indexed):
        mult = n - rank
        p_adj = min(1.0, p * mult)
        running_max = max(running_max, p_adj)
        adjusted_by_rank.append(running_max)
    result = [0.0] * n
    for (orig_idx, _p), p_adj in zip(indexed, adjusted_by_rank):
        result[orig_idx] = p_adj
    return result


def _classify(delta_pct: float, p_holm: float, rope_pct: float) -> str:
    """Classify a metric delta against the ROPE.

    Returns one of:
      - 'EQUIVALENT' : |delta| < rope, change is within noise budget
      - 'DIFFERENT'  : p_holm < 0.01 AND |delta| > rope, confident real change
      - 'UNCLEAR'    : otherwise — we don't know
    """
    abs_delta = abs(delta_pct)
    if abs_delta < rope_pct:
        return "EQUIVALENT"
    if p_holm < 0.01 and abs_delta > rope_pct:
        return "DIFFERENT"
    return "UNCLEAR"


def _fmt_classification(cls: str) -> str:
    return {
        "EQUIVALENT": "≈ (within ROPE)",
        "DIFFERENT": "**DIFFERENT**",
        "UNCLEAR": "unclear",
    }[cls]


def _precision_for(name: str) -> int:
    if "fps" in name:
        return 2
    if "ms" in name:
        return 1
    return 2


def _fmt_value(value: float, precision: int) -> str:
    return f"{value:.{precision}f}"


# Metrics to compare in throughput. Ordered by importance.
THROUGHPUT_METRICS = [
    ("steady_state_fps", "steady_state_fps"),
    ("gpu_kernel_ms_per_frame", "gpu_kernel_ms_per_frame"),
    ("gpu_thread_wall_ms_per_frame", "gpu_thread_wall_ms_per_frame"),
    ("wall_ms_per_frame", "wall_ms_per_frame"),
    ("decode_thread_wall_ms_per_frame", "decode_thread_wall_ms_per_frame"),
    ("encode_thread_wall_ms_per_frame", "encode_thread_wall_ms_per_frame"),
]


def compare_results(
    a: BenchResult,
    b: BenchResult,
    label_a: str | None = None,
    label_b: str | None = None,
    force: bool = False,
    rope_pct: float = DEFAULT_ROPE_PCT,
) -> str:
    """Return a markdown report comparing two BenchResults.

    Uses BenchResult.config.label for report headers when label_a/label_b
    are not provided.
    """
    if label_a is None:
        label_a = a.config.label
    if label_b is None:
        label_b = b.config.label

    fatal, advisory = _check_hardware(a, b)

    lines = [f"# Benchmark: {label_a} vs {label_b}", ""]

    if fatal:
        if not force:
            raise HardwareMismatchError(
                "Fatal hardware mismatch:\n  " + "\n  ".join(fatal) +
                "\nUse force=True to override (results will be misleading)."
            )
        lines.append("**ERROR: Fatal hardware mismatch — results are meaningless**")
        for m in fatal:
            lines.append(f"  - {m}")
        lines.append("")

    if advisory:
        lines.append("**Warning: advisory hardware differences (not blocking):**")
        for m in advisory:
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
    if ca.n_samples != cb.n_samples:
        lines.append(f"- samples: {ca.n_samples} vs {cb.n_samples} (different N)")
    else:
        lines.append(f"- samples: {ca.n_samples} each")
    lines.append("")

    # Compute raw p-values for all metrics first so we can apply Holm correction
    deltas = []
    raw_ps = []
    for attr, _name in THROUGHPUT_METRICS:
        metric_a = getattr(a.throughput, attr)
        metric_b = getattr(b.throughput, attr)
        delta, p_raw = _permutation_test(metric_a, metric_b)
        deltas.append(delta)
        raw_ps.append(p_raw)

    holm_ps = _holm_correct(raw_ps)

    lines.append(f"## Throughput (means ± 95% CI, ROPE=±{rope_pct}%)")
    lines.append("")
    lines.append(f"| Metric | {label_a} | {label_b} | Δ | p_raw | p_holm | verdict |")
    lines.append("|---|---:|---:|---:|---:|---:|:---|")
    for (attr, name), delta, p_raw, p_holm in zip(THROUGHPUT_METRICS, deltas, raw_ps, holm_ps):
        metric_a = getattr(a.throughput, attr)
        metric_b = getattr(b.throughput, attr)
        prec = _precision_for(name)
        ci_a = metric_a.ci95_hi - metric_a.mean
        ci_b = metric_b.ci95_hi - metric_b.mean
        cls = _classify(delta, p_holm, rope_pct)
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| {name} "
            f"| {_fmt_value(metric_a.mean, prec)} ± {_fmt_value(ci_a, prec)} "
            f"| {_fmt_value(metric_b.mean, prec)} ± {_fmt_value(ci_b, prec)} "
            f"| {sign}{delta:.1f}% "
            f"| {p_raw:.3g} "
            f"| {p_holm:.3g} "
            f"| {_fmt_classification(cls)} |"
        )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(f"- **ROPE:** ±{rope_pct}% — the region of practical equivalence. Deltas "
                 "smaller than this are classified as EQUIVALENT regardless of p-value. "
                 "Set from calibration on this workstation.")
    lines.append("- **p_raw:** uncorrected two-sided permutation test p-value.")
    lines.append("- **p_holm:** Holm-Bonferroni corrected p-value across the "
                 f"{len(THROUGHPUT_METRICS)} metrics tested. Controls family-wise "
                 "error rate at α=0.05.")
    lines.append("- **DIFFERENT:** p_holm<0.01 AND |Δ|>ROPE — confident real change.")
    lines.append("- **≈ (EQUIVALENT):** |Δ|<ROPE — change is within noise budget.")
    lines.append("- **unclear:** neither condition — retry with more samples or investigate.")
    lines.append("")
    lines.append("**Why permutation test and not Welch's t?** Benchmark samples within a "
                 "single run are autocorrelated (shared thermal state, warm caches) and "
                 "timing distributions are not normal. The permutation test makes no "
                 "distributional assumptions.")
    lines.append("")
    lines.append("**The ROPE threshold is workstation-specific.** It comes from the "
                 "calibration run documented in "
                 "`docs/learnings/2026-04-13-bench-harness-noise-floor.md`. Re-calibrate "
                 "if hardware or environment changes.")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two bench results")
    parser.add_argument("baseline", help="Path to baseline JSON")
    parser.add_argument("candidate", help="Path to candidate JSON")
    parser.add_argument("--output", help="Write report to file (default: stdout)")
    parser.add_argument("--force", action="store_true",
                        help="Allow comparison across different hardware (not recommended)")
    parser.add_argument("--rope", type=float, default=DEFAULT_ROPE_PCT,
                        help=f"Region of practical equivalence in percent "
                             f"(default: {DEFAULT_ROPE_PCT})")
    args = parser.parse_args()

    a = BenchResult.load(Path(args.baseline))
    b = BenchResult.load(Path(args.candidate))

    try:
        report = compare_results(a, b, force=args.force, rope_pct=args.rope)
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
