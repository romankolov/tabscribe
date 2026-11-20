//! Benchmarks for the classical DSP analysis stages.
//!
//! These will be populated with the real stages as they are implemented; see
//! `docs/ARCHITECTURE.md` for the pipeline breakdown.

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_placeholder(c: &mut Criterion) {
    c.bench_function("dsp/smoke", |b| b.iter(|| std::hint::black_box(1u32)));
}

criterion_group!(benches, bench_placeholder);
criterion_main!(benches);
