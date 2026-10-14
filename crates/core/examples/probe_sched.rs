//! 手工计时：1024 decode 请求一步，分 schedule / update 两段。配合 perf 用。
//! `cargo run --release -p dinfer-core --example probe_sched [iters]`

use dinfer_core::config::{CacheConfig, SchedulerConfig};
use dinfer_core::kv::{GroupInfo, KvCacheSpec};
use dinfer_core::sample::SamplingParams;
use dinfer_core::sched::{ModelRunnerOutput, Scheduler, SchedulerParams};
use dinfer_core::Request;
use std::time::Instant;

fn make(n: u64, pc: bool) -> (Scheduler, ModelRunnerOutput) {
    let p = SchedulerParams {
        sched: SchedulerConfig { max_num_seqs: 1024, max_num_batched_tokens: 8192, ..Default::default() },
        cache: CacheConfig { enable_prefix_caching: pc, ..Default::default() },
        max_model_len: 4096,
        num_spec_tokens: 0,
        async_scheduling: true,
        eos_token_ids: vec![2],
    };
    let g = vec![GroupInfo {
        spec: KvCacheSpec::Mla { block_size: 16, head_size: 576, dtype_bytes: 2 },
        block_size: 16,
    }];
    let mut s = Scheduler::new(p, g, 1024 * 80);
    for i in 0..n {
        let sp = SamplingParams { max_tokens: 1_000_000, ..Default::default() };
        s.add_request(Request::new(i, (0..64).map(|t| (t + i as u32) % 1000).collect(), sp, 0));
    }
    let o = s.schedule();
    let mro = ModelRunnerOutput { req_ids: (0..n).collect(), sampled: (0..n).map(|_| vec![7]).collect() };
    s.update_from_output(&o, &mro);
    (s, mro)
}

fn main() {
    let iters: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2000);
    for &pc in &[false, true] {
        let (mut s, mro) = make(1024, pc);
        let (mut t_sched, mut t_upd) = (0u128, 0u128);
        for _ in 0..iters {
            let t0 = Instant::now();
            let o = s.schedule();
            let t1 = Instant::now();
            s.update_from_output(&o, &mro);
            let t2 = Instant::now();
            t_sched += (t1 - t0).as_nanos();
            t_upd += (t2 - t1).as_nanos();
        }
        let f = iters as f64 * 1e3;
        println!(
            "prefix_cache={pc}: schedule {:.1} µs, update {:.1} µs, total {:.1} µs",
            t_sched as f64 / f,
            t_upd as f64 / f,
            (t_sched + t_upd) as f64 / f
        );
    }
}
