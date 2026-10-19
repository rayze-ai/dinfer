//! P1 性能门禁：1024 请求 × 1 decode step 的 schedule + update_from_output < 200 µs。
//!
//! 用持久状态循环（`iter`），不用 `iter_batched`：后者每次迭代重建 1024 请求，
//! setup 成本会被 criterion 计入（实测把 60 µs 的 step 抬到 200 µs）。
//! max_tokens 设为极大，1024 请求在整个测量期间都处于 decode 稳态。

use criterion::{criterion_group, criterion_main, Criterion};
use dinfer_core::config::{CacheConfig, SchedulerConfig};
use dinfer_core::kv::{GroupInfo, KvCacheSpec};
use dinfer_core::sample::SamplingParams;
use dinfer_core::sched::{ModelRunnerOutput, Scheduler, SchedulerParams};
use dinfer_core::Request;

fn make(n: u64, prefix_cache: bool) -> (Scheduler, ModelRunnerOutput) {
    let p = SchedulerParams {
        sched: SchedulerConfig { max_num_seqs: 1024, max_num_batched_tokens: 8192, ..Default::default() },
        cache: CacheConfig { enable_prefix_caching: prefix_cache, ..Default::default() },
        // 足够大：criterion 可能跑几十万步，不能触发 max_model_len 截断。
        max_model_len: 1 << 20,
        num_spec_tokens: 0,
        async_scheduling: true,
        eos_token_ids: vec![2],
    };
    let g = vec![GroupInfo {
        spec: KvCacheSpec::Mla { block_size: 16, head_size: 576, dtype_bytes: 2 },
        block_size: 16,
    }];
    // 1024 请求 × 1M token / 16 = 64M 块：BlockPool 用 u32 索引，这里给 32M 足够测量期间不抢占。
    let mut s = Scheduler::new(p, g, 32 << 20);
    for i in 0..n {
        let sp = SamplingParams { max_tokens: u32::MAX, ..Default::default() };
        s.add_request(Request::new(i, (0..64).map(|t| (t + i as u32) % 1000).collect(), sp, 0));
    }
    let o = s.schedule();
    let mro = ModelRunnerOutput {
        req_ids: (0..n).collect(),
        sampled: (0..n).map(|_| vec![7]).collect(),
        drafts: vec![Vec::new(); n as usize],
        logprobs: vec![],
        mode: None,
    };
    s.update_from_output(&o, &mro);
    (s, mro)
}

fn bench_decode_step(c: &mut Criterion) {
    for &pc in &[false, true] {
        let (mut s, mro) = make(1024, pc);
        c.bench_function(&format!("schedule+update 1024 decode prefix_cache={pc}"), |b| {
            b.iter(|| {
                let o = s.schedule();
                s.update_from_output(&o, &mro);
            })
        });
    }
}

criterion_group!(benches, bench_decode_step);
criterion_main!(benches);
