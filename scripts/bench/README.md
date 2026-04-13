# raune_filter Bench Harness (Phase 1)

Statistically rigorous profiling for `raune_filter.py` at production config.

## Quick Start

```bash
# Run a benchmark (5 samples, 8 warmup frames, 1440p proxy)
/opt/dorea-venv/bin/python scripts/bench/run.py \
  --clip /tmp/test_clip_30f.mp4 \
  --proxy 1440x810 \
  --label baseline \
  --tensorrt

# Compare two runs
/opt/dorea-venv/bin/python scripts/bench/compare.py \
  scripts/bench/results/<baseline>.json \
  scripts/bench/results/<candidate>.json
```

## What It Measures

**Tier 1 (Phase 1 — always captured):**
- `steady_state_fps` (headline metric, excludes warmup)
- `time_to_first_frame_ms` (cold start cost)
- `gpu_kernel_ms_per_frame` — **real** GPU time via CUDA events
- `gpu_thread_wall_ms_per_frame` — GPU thread wall clock (includes CPU overhead)
- `decode_thread_wall_ms_per_frame`, `encode_thread_wall_ms_per_frame`
- `wall_ms_per_frame`
- System state: GPU clocks, PCIe link, CPU governor, driver version, etc.
- Measured PCIe health (pinned bandwidth)

Every metric is captured across N samples (default 5) and reported as
`mean ± 95% CI`. Comparisons use Welch's t-test.

## Interpreting Results

- `***` — p<0.01 and |Δ|>2% — high confidence real improvement/regression
- `**` — p<0.05 and |Δ|>1% — likely real
- `*` — p<0.05 but |Δ|<1% — statistically significant but small
- blank — within noise floor

## Noise Floor

See `docs/learnings/2026-04-13-bench-harness-noise-floor.md` for the
documented noise floor on this workstation. **Never trust a delta smaller
than the noise floor**, even if flagged as statistically significant.

## Troubleshooting

**"schema_version mismatch"** — re-run both benches with the current harness.

**"Hardware/state mismatch"** — results are from different drivers/GPUs.
Use `--force` only if you understand the implications.

**"PCIe bandwidth below 80% of expected"** — your PCIe link may be
degraded (gen3→gen2 or x16→x8). Check the slot seating and BIOS settings.
Note: on some workstations, NVML reports gen1 x8 when idle (power saving),
even though the measured bandwidth is healthy — the measured value is
more reliable than the reported link gen.

## Roadmap (Phase 2, on demand)

- NVML 50Hz sampling (util, PCIe throughput, power, clocks)
- torch.profiler CUPTI kernel traces
- Top CUDA ops by time
- py-spy flamegraphs
- ncu DRAM/SM throughput (memory-bound vs compute-bound signal)
- `--pin` for clock locking + cache dropping
