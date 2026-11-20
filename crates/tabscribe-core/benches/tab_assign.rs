//! Benchmarks for the tablature assignment optimizer.

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_placeholder(c: &mut Criterion) {
    c.bench_function("tab_assign/smoke", |b| {
        b.iter(|| std::hint::black_box(1u32))
    });
}

criterion_group!(benches, bench_placeholder);
criterion_main!(benches);
