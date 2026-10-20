//! 引擎指标（Prometheus 文本）。全部原子量：engine 线程每 step 写，HTTP 线程读，无锁。
//! 指标名与 vLLM v1 `vllm:*` 对齐（前缀由 server 决定），直方图桶也取 vLLM 默认。

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};

/// 固定桶直方图；`sum` 以微秒整数累加（避免 f64 原子）。
pub struct Histogram {
    buckets: &'static [f64],
    counts: Vec<AtomicU64>,
    sum_us: AtomicU64,
    count: AtomicU64,
}

impl Histogram {
    pub fn new(buckets: &'static [f64]) -> Self {
        Self {
            buckets,
            counts: (0..=buckets.len()).map(|_| AtomicU64::new(0)).collect(),
            sum_us: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }
    pub fn observe(&self, secs: f64) {
        let i = self.buckets.iter().position(|&b| secs <= b).unwrap_or(self.buckets.len());
        self.counts[i].fetch_add(1, Relaxed);
        self.sum_us.fetch_add((secs * 1e6) as u64, Relaxed);
        self.count.fetch_add(1, Relaxed);
    }
    pub fn count(&self) -> u64 {
        self.count.load(Relaxed)
    }
    pub fn sum(&self) -> f64 {
        self.sum_us.load(Relaxed) as f64 / 1e6
    }
    fn render(&self, out: &mut String, name: &str, help: &str) {
        use std::fmt::Write;
        let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} histogram");
        let mut cum = 0u64;
        for (i, b) in self.buckets.iter().enumerate() {
            cum += self.counts[i].load(Relaxed);
            let _ = writeln!(out, "{name}_bucket{{le=\"{b}\"}} {cum}");
        }
        cum += self.counts[self.buckets.len()].load(Relaxed);
        let _ = writeln!(out, "{name}_bucket{{le=\"+Inf\"}} {cum}");
        let _ = writeln!(out, "{name}_sum {}", self.sum());
        let _ = writeln!(out, "{name}_count {}", self.count());
    }
}

const LATENCY_BUCKETS: &[f64] = &[
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0,
    80.0,
];
const TPOT_BUCKETS: &[f64] = &[
    0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3,
    0.5, 1.0,
];
const E2E_BUCKETS: &[f64] =
    &[0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0];
const STEP_BUCKETS: &[f64] = &[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0];

#[derive(Default)]
pub struct Metrics {
    // gauges（每 step 覆盖）
    pub num_requests_running: AtomicU64,
    pub num_requests_waiting: AtomicU64,
    /// f32 bits
    kv_cache_usage: AtomicU32,
    // counters（引擎累加）
    pub prompt_tokens_total: AtomicU64,
    pub generation_tokens_total: AtomicU64,
    pub num_preemptions_total: AtomicU64,
    pub prefix_cache_queries_total: AtomicU64,
    pub prefix_cache_hits_total: AtomicU64,
    pub host_tier_hits_total: AtomicU64,
    pub host_tier_fetch_blocks_total: AtomicU64,
    pub host_tier_offload_blocks_total: AtomicU64,
    pub pd_export_total: AtomicU64,
    /// 其中经共享槽区（shm / cuda-ipc）直写对端槽的导出数（design §10.3 b）。
    pub pd_export_slot_total: AtomicU64,
    pub pd_import_total: AtomicU64,
    pub pd_fallback_total: AtomicU64,
    /// 投机解码：提案的 draft token 总数 / 被接受的 draft token 总数（接受率 = accepted / draft）。
    pub spec_draft_total: AtomicU64,
    pub spec_accepted_total: AtomicU64,
    pub request_success_stop: AtomicU64,
    pub request_success_length: AtomicU64,
    pub request_abort: AtomicU64,
    pub engine_steps_total: AtomicU64,
    // histograms
    pub time_to_first_token: Histogram,
    pub time_per_output_token: Histogram,
    pub e2e_request_latency: Histogram,
    pub step_time: Histogram,
    /// step 内分段：调度（schedule）/ 执行（runner.execute：H2D + GPU + D2H 等待）/ 回填与下发。
    pub step_schedule: Histogram,
    pub step_execute: Histogram,
    /// `step_execute` 按 plan 模式分桶（qwen35-perf 第 5B 轮：量化 c32 下混批步比例与耗时）。
    pub step_execute_prefill: Histogram,
    pub step_execute_decode: Histogram,
    pub step_execute_mixed: Histogram,
    pub step_update: Histogram,
    /// executor host 侧分段（engine-gap 第 7 轮）：`[mode][phase]`，mode = prefill / decode / mixed，
    /// phase = upload / launch / wait（见 `runner::StepPhases`）。只有 executor 实现 `take_step_phases` 才有数据。
    pub step_phase: [[Histogram; 3]; 3],
}

pub const STEP_PHASE_MODES: [&str; 3] = ["prefill", "decode", "mixed"];
pub const STEP_PHASE_NAMES: [&str; 3] = ["upload", "launch", "wait"];

impl Default for Histogram {
    fn default() -> Self {
        Self::new(LATENCY_BUCKETS)
    }
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            time_to_first_token: Histogram::new(LATENCY_BUCKETS),
            time_per_output_token: Histogram::new(TPOT_BUCKETS),
            e2e_request_latency: Histogram::new(E2E_BUCKETS),
            step_time: Histogram::new(STEP_BUCKETS),
            step_schedule: Histogram::new(STEP_BUCKETS),
            step_execute: Histogram::new(STEP_BUCKETS),
            step_execute_prefill: Histogram::new(STEP_BUCKETS),
            step_execute_decode: Histogram::new(STEP_BUCKETS),
            step_execute_mixed: Histogram::new(STEP_BUCKETS),
            step_update: Histogram::new(STEP_BUCKETS),
            step_phase: std::array::from_fn(|_| std::array::from_fn(|_| Histogram::new(STEP_BUCKETS))),
            ..Default::default()
        }
    }
    pub fn set_kv_cache_usage(&self, frac: f32) {
        self.kv_cache_usage.store(frac.to_bits(), Relaxed);
    }
    pub fn kv_cache_usage(&self) -> f32 {
        f32::from_bits(self.kv_cache_usage.load(Relaxed))
    }

    /// Prometheus 文本（0.0.4）。`prefix` 形如 `"dinfer:"`。
    pub fn render(&self, prefix: &str) -> String {
        use std::fmt::Write;
        let mut o = String::with_capacity(4096);
        let g = |o: &mut String, name: &str, help: &str, v: String| {
            let _ =
                writeln!(o, "# HELP {prefix}{name} {help}\n# TYPE {prefix}{name} gauge\n{prefix}{name} {v}");
        };
        let c = |o: &mut String, name: &str, help: &str, v: u64| {
            let _ = writeln!(
                o,
                "# HELP {prefix}{name} {help}\n# TYPE {prefix}{name} counter\n{prefix}{name} {v}"
            );
        };
        g(
            &mut o,
            "num_requests_running",
            "Number of requests in model execution batches.",
            self.num_requests_running.load(Relaxed).to_string(),
        );
        g(
            &mut o,
            "num_requests_waiting",
            "Number of requests waiting to be processed.",
            self.num_requests_waiting.load(Relaxed).to_string(),
        );
        g(
            &mut o,
            "kv_cache_usage_perc",
            "KV-cache usage. 1 means 100 percent usage.",
            format!("{}", self.kv_cache_usage()),
        );
        c(
            &mut o,
            "prompt_tokens_total",
            "Number of prefill tokens processed.",
            self.prompt_tokens_total.load(Relaxed),
        );
        c(
            &mut o,
            "generation_tokens_total",
            "Number of generation tokens processed.",
            self.generation_tokens_total.load(Relaxed),
        );
        c(
            &mut o,
            "num_preemptions_total",
            "Cumulative number of preemption from the engine.",
            self.num_preemptions_total.load(Relaxed),
        );
        c(
            &mut o,
            "prefix_cache_queries_total",
            "Prefix cache queries, in terms of number of queried tokens.",
            self.prefix_cache_queries_total.load(Relaxed),
        );
        c(
            &mut o,
            "prefix_cache_hits_total",
            "Prefix cache hits, in terms of number of cached tokens.",
            self.prefix_cache_hits_total.load(Relaxed),
        );
        c(&mut o, "host_tier_hits_total", "KV host tier prefix hits, in blocks.", self.host_tier_hits_total.load(Relaxed));
        c(&mut o, "host_tier_fetch_blocks_total", "Blocks fetched host->GPU.", self.host_tier_fetch_blocks_total.load(Relaxed));
        c(&mut o, "host_tier_offload_blocks_total", "Blocks offloaded GPU->host.", self.host_tier_offload_blocks_total.load(Relaxed));
        c(&mut o, "pd_export_total", "PD: requests whose KV was exported (prefill side).", self.pd_export_total.load(Relaxed));
        c(&mut o, "pd_export_slot_total", "PD: exports written directly into the peer shared slot region (shm / cuda-ipc).", self.pd_export_slot_total.load(Relaxed));
        c(&mut o, "pd_import_total", "PD: requests whose KV was imported (decode side).", self.pd_import_total.load(Relaxed));
        c(&mut o, "pd_fallback_total", "PD: decode-side requests that fell back to local prefill.", self.pd_fallback_total.load(Relaxed));
        c(&mut o, "spec_draft_total", "Speculative decoding: draft tokens proposed.", self.spec_draft_total.load(Relaxed));
        c(&mut o, "spec_accepted_total", "Speculative decoding: draft tokens accepted.", self.spec_accepted_total.load(Relaxed));
        c(
            &mut o,
            "engine_steps_total",
            "Number of engine steps executed.",
            self.engine_steps_total.load(Relaxed),
        );
        let _ = writeln!(
            o,
            "# HELP {prefix}request_success_total Count of successfully processed requests.\n# TYPE {prefix}request_success_total counter\n\
             {prefix}request_success_total{{finished_reason=\"stop\"}} {}\n{prefix}request_success_total{{finished_reason=\"length\"}} {}\n\
             {prefix}request_success_total{{finished_reason=\"abort\"}} {}",
            self.request_success_stop.load(Relaxed),
            self.request_success_length.load(Relaxed),
            self.request_abort.load(Relaxed)
        );
        self.time_to_first_token.render(
            &mut o,
            &format!("{prefix}time_to_first_token_seconds"),
            "Histogram of time to first token in seconds.",
        );
        self.time_per_output_token.render(
            &mut o,
            &format!("{prefix}time_per_output_token_seconds"),
            "Histogram of time per output token in seconds.",
        );
        self.e2e_request_latency.render(
            &mut o,
            &format!("{prefix}e2e_request_latency_seconds"),
            "Histogram of end to end request latency in seconds.",
        );
        self.step_time.render(
            &mut o,
            &format!("{prefix}engine_step_seconds"),
            "Histogram of engine step wall time in seconds.",
        );
        self.step_schedule.render(
            &mut o,
            &format!("{prefix}engine_step_schedule_seconds"),
            "Scheduler time per step.",
        );
        self.step_execute.render(
            &mut o,
            &format!("{prefix}engine_step_execute_seconds"),
            "Runner execute (H2D + GPU + D2H) time per step.",
        );
        for (h, mode) in [
            (&self.step_execute_prefill, "prefill"),
            (&self.step_execute_decode, "decode"),
            (&self.step_execute_mixed, "mixed"),
        ] {
            h.render(
                &mut o,
                &format!("{prefix}engine_step_execute_{mode}_seconds"),
                &format!("Runner execute time per step, plan mode = {mode} (mixed = chunked prefill + decode)."),
            );
        }
        self.step_update.render(
            &mut o,
            &format!("{prefix}engine_step_update_seconds"),
            "Output update + dispatch time per step.",
        );
        for (mi, mode) in STEP_PHASE_MODES.iter().enumerate() {
            for (pi, phase) in STEP_PHASE_NAMES.iter().enumerate() {
                self.step_phase[mi][pi].render(
                    &mut o,
                    &format!("{prefix}engine_step_phase_{phase}_{mode}_seconds"),
                    &format!("Executor host-side {phase} time per step, plan mode = {mode} (upload = plan/sampling H2D, launch = forward+sample enqueue, wait = GPU result wait)."),
                );
            }
        }
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_buckets_cumulative_and_sum() {
        let h = Histogram::new(&[0.01, 0.1, 1.0]);
        for v in [0.005, 0.05, 0.5, 5.0] {
            h.observe(v);
        }
        let mut s = String::new();
        h.render(&mut s, "x", "h");
        assert!(s.contains("x_bucket{le=\"0.01\"} 1\n"));
        assert!(s.contains("x_bucket{le=\"0.1\"} 2\n"));
        assert!(s.contains("x_bucket{le=\"1\"} 3\n"));
        assert!(s.contains("x_bucket{le=\"+Inf\"} 4\n"));
        assert!(s.contains("x_count 4\n"));
        assert!((h.sum() - 5.555).abs() < 1e-3);
    }

    #[test]
    fn render_has_required_names() {
        let m = Metrics::new();
        m.set_kv_cache_usage(0.25);
        m.prefix_cache_hits_total.fetch_add(3, Relaxed);
        let s = m.render("dinfer:");
        for n in [
            "dinfer:num_requests_running 0",
            "dinfer:prompt_tokens_total 0",
            "dinfer:time_to_first_token_seconds_bucket{le=\"0.001\"} 0",
            "dinfer:kv_cache_usage_perc 0.25",
            "dinfer:prefix_cache_hits_total 3",
            "dinfer:request_success_total{finished_reason=\"stop\"} 0",
        ] {
            assert!(s.contains(n), "missing {n}\n{s}");
        }
    }
}
