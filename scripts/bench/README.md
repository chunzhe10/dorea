# raune_filter Bench Harness (Phase 1, v2)

Statistically rigorous profiling for `raune_filter.py` at production config.

## Quick Start

```bash
# Set workstation paths (once per shell session)
export DOREA_RAUNE_WEIGHTS=/path/to/weights_95.pth
export DOREA_MODELS_DIR=/path/to/models/raune_net
export DOREA_TEST_CLIP=/path/to/bench_clip.mp4

# Run a benchmark (5 samples, 8 warmup frames, 1440p proxy)
python scripts/bench/run.py \
  --proxy 1440x810 \
  --label baseline \
  --tensorrt

# Compare two runs
python scripts/bench/compare.py \
  scripts/bench/results/<baseline>.json \
  scripts/bench/results/<candidate>.json
```

## What It Measures

**Phase 1 (Tier 1) — always captured:**
- `steady_state_fps` (headline metric; excludes warmup frames)
- `gpu_kernel_ms_per_frame` — **real** GPU kernel time via CUDA events
  (only recorded when `DOREA_BENCH=1` is set in the child process env;
  `run.py` sets this automatically)
- `gpu_thread_wall_ms_per_frame` — GPU thread wall clock
- `decode_thread_wall_ms_per_frame`, `encode_thread_wall_ms_per_frame`
- `wall_ms_per_frame`
- System state: GPU clocks, PCIe link, CPU governor, driver version,
  measured PCIe health (pinned bandwidth probe)

Every metric is captured across N samples (default 5) and reported as
`mean ± 95% CI`. Comparisons use **permutation testing** with
**Holm-Bonferroni correction** for multiple comparisons.

## Interpreting Results

`compare.py` classifies each metric into one of three buckets:

- **≈ (EQUIVALENT)** — |Δ| < ROPE. The change is within the noise
  budget and should not be treated as an improvement or regression.
- **DIFFERENT** — p_holm < 0.01 AND |Δ| > ROPE. Confident real change.
  Worth acting on.
- **unclear** — large Δ but weak p, or vice versa. Retry with more
  samples (`--samples 10`) or investigate the variance source.

**Never trust a single run's headline number.** Always compare against
a baseline from the same config on the same hardware.

## ROPE (Region of Practical Equivalence)

Default ROPE is **±1.5%**, chosen from the calibration run documented
in `docs/learnings/2026-04-13-bench-harness-noise-floor.md`. Override
with `--rope <pct>` on `compare.py`.

Why a ROPE? Because noise alone can produce real-looking deltas even
from identical code — the v1 harness demonstrated this and was
subsequently reworked. The ROPE is a threshold below which we refuse to
treat a change as meaningful, regardless of how "statistically
significant" it looks.

## Why Permutation Test and Not t-test?

Benchmark samples in a single run are **not independent**:
- Shared thermal state across back-to-back samples
- Warm TRT engine cache, warm CUDA context
- Variable GPU scheduling that's correlated sample-to-sample

Welch's t-test assumes iid normal samples. Neither holds for bench
data. Permutation testing on the raw samples makes **no distributional
assumption** and is honest about what we can infer from N=5 dependent
samples.

## Calibrating on a New Workstation

If you're running this on a new machine (different GPU, driver,
cooling, OS), the ROPE may need adjustment:

```bash
# Step 1: run the harness twice on the same commit
python scripts/bench/run.py --proxy 960x540 --label cal-a --tensorrt
python scripts/bench/run.py --proxy 960x540 --label cal-b --tensorrt

# Step 2: compare them
python scripts/bench/compare.py \
  scripts/bench/results/<cal-a>.json \
  scripts/bench/results/<cal-b>.json
```

Expected outcome: all headline metrics classify as **EQUIVALENT** since
they're the same code. If any classify as `DIFFERENT` or `unclear`,
something in your workstation is injecting noise — common causes:
thermal throttling, background processes on the GPU, driver
auto-updates mid-run.

Record the observed noise in `docs/learnings/<date>-noise-floor.md`
and adjust `--rope` accordingly.

## Troubleshooting

**"schema_version mismatch"** — re-run both benches with the current
harness. v2 dropped `time_to_first_frame_ms` (computation was wrong)
and `VramStats` (was shipped with zeros).

**"Fatal hardware mismatch"** — results are from incompatible hardware
(different GPU model or compute capability). `--force` is available
but strongly discouraged.

**"missing required arguments"** — set `DOREA_TEST_CLIP`,
`DOREA_RAUNE_WEIGHTS`, and `DOREA_MODELS_DIR` or pass them as CLI
flags.

**"raune_filter subprocess timed out"** — default timeout is 300s per
sample. Your first run may need longer if the TRT engine isn't cached
yet. Pre-build the engine by running raune_filter once directly.

**decode_thread noise is high** — expected. The decode thread is 93%
idle on average, so small absolute drifts become large relative
noise. Don't use decode metrics for regression decisions.

## Roadmap (Phase 2, on demand)

- NVML 50Hz sampling (util, PCIe throughput, power, clocks)
- torch.profiler CUPTI kernel traces (populate `nvtx_spans` with real
  per-span p50/p95 data; Phase 1 leaves this list empty)
- Top CUDA ops by time
- py-spy flamegraphs
- ncu DRAM/SM throughput (memory-bound vs compute-bound signal)
- `--pin` for GPU clock locking + CPU governor + cache dropping
- First-frame timing (requires raune_filter to emit a timestamp;
  v2 dropped the misleading computation it had in v1)
- VRAM metrics (requires raune_filter to emit `peak_vram_mb`)
