//! P1 性能门禁：1024 请求 × 1 step 的 schedule + update_from_output < 200 µs。
//! P0 先搭骨架，数字在 P1 完整调度器实现后才有意义。

use criterion::{criterion_group, criterion_main, Criterion};
use dinfer_core::config::SchedulerConfig;
use dinfer_core::sample::SamplingParams;
use dinfer_core::sched::Scheduler;
use dinfer_core::Request;

fn bench_decode_step(c: &mut Criterion) {
    let n = 1024u64;
    c.bench_function("schedule+update 1024 decode", |b| {
        b.iter_batched(
            || {
                let cfg = SchedulerConfig {
                    max_num_seqs: 1024,
                    max_num_batched_tokens: 8192,
                    ..Default::default()
                };
                let mut s = Scheduler::new(cfg);
                for i in 0..n {
                    let sp = SamplingParams { max_tokens: 1_000_000, ..Default::default() };
                    s.add_request(Request::new(i, vec![1; 4], sp, 0));
                }
                let o = s.schedule();
                let sampled: Vec<(u64, u32)> = (0..n).map(|i| (i, 1)).collect();
                s.update_from_output(&o, &sampled);
                (s, sampled)
            },
            |(mut s, sampled)| {
                let o = s.schedule();
                s.update_from_output(&o, &sampled);
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_decode_step);
criterion_main!(benches);
