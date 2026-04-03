//! Benchmark: GPU grading pipeline throughput.
//!
//! Run with:
//!   cargo bench -p dorea-gpu
//!
//! Measures three code paths at 720p and 1080p:
//!   - `with_grader`   — grade_frame_with_grader (PTX loaded once, reused)
//!   - `per_call_init` — grade_frame (creates new CudaGrader each call)
//!   - `cpu`           — CPU-only grade_frame_cpu baseline

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use dorea_cal::Calibration;
use dorea_gpu::GradeParams;
use dorea_hsl::derive::{HslCorrections, QualifierCorrection};
use dorea_lut::types::{DepthLuts, LutGrid};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn identity_lut(size: usize) -> LutGrid {
    let mut lut = LutGrid::new(size);
    for ri in 0..size {
        for gi in 0..size {
            for bi in 0..size {
                let r = ri as f32 / (size - 1) as f32;
                let g = gi as f32 / (size - 1) as f32;
                let b = bi as f32 / (size - 1) as f32;
                lut.set(ri, gi, bi, [r, g, b]);
            }
        }
    }
    lut
}

fn make_calibration() -> Calibration {
    let n_zones = 5;
    let luts: Vec<LutGrid> = (0..n_zones).map(|_| identity_lut(17)).collect();
    let boundaries: Vec<f32> = (0..=n_zones).map(|i| i as f32 / n_zones as f32).collect();
    let depth_luts = DepthLuts::new(luts, boundaries);
    // Passthrough HSL (weight=0 on all qualifiers)
    let hsl = HslCorrections(vec![QualifierCorrection {
        h_center: 0.0,
        h_width: 1.0,
        h_offset: 0.0,
        s_ratio: 1.0,
        v_offset: 0.0,
        weight: 0.0,
    }]);
    Calibration::new(depth_luts, hsl, 1)
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

fn bench_grade(c: &mut Criterion) {
    let params = GradeParams::default();
    let calibration = make_calibration();

    let mut group = c.benchmark_group("grading");
    // Fewer samples for GPU paths (each call is ~10-50 ms)
    group.sample_size(20);

    for (label, w, h) in [("720p", 1280usize, 720usize), ("1080p", 1920usize, 1080usize)] {
        let n = w * h;
        // Synthetic mid-range pixels and depth
        let pixels: Vec<u8> = (0..n * 3).map(|i| ((i * 7 + 128) % 256) as u8).collect();
        let depth: Vec<f32> = (0..n).map(|i| (i as f32) / n as f32 * 0.8 + 0.1).collect();

        group.throughput(Throughput::Elements(n as u64));

        // --- Path A: grade_frame_with_grader (PTX reuse) ---
        #[cfg(feature = "cuda")]
        {
            use dorea_gpu::{cuda::CudaGrader, grade_frame_with_grader};
            match CudaGrader::new() {
                Ok(grader) => {
                    group.bench_with_input(
                        BenchmarkId::new("with_grader", label),
                        &(w, h),
                        |b, _| {
                            b.iter(|| {
                                grade_frame_with_grader(
                                    &grader, &pixels, &depth, w, h, &calibration, &params,
                                )
                                .expect("grade_frame_with_grader failed")
                            });
                        },
                    );
                }
                Err(e) => {
                    eprintln!("SKIP with_grader/{label}: CudaGrader::new failed: {e}");
                }
            }
        }

        // --- Path B: grade_frame (new CudaGrader per call — shows PTX overhead) ---
        #[cfg(feature = "cuda")]
        {
            use dorea_gpu::grade_frame;
            group.bench_with_input(
                BenchmarkId::new("per_call_init", label),
                &(w, h),
                |b, _| {
                    b.iter(|| {
                        grade_frame(&pixels, &depth, w, h, &calibration, &params)
                            .expect("grade_frame failed")
                    });
                },
            );
        }

        // --- Path C: CPU baseline ---
        group.bench_with_input(
            BenchmarkId::new("cpu", label),
            &(w, h),
            |b, _| {
                b.iter(|| {
                    dorea_gpu::cpu::grade_frame_cpu(
                        &pixels, &depth, w, h, &calibration, &params,
                    )
                    .expect("cpu grade failed")
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_grade);
criterion_main!(benches);
