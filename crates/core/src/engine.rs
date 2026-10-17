//! 引擎循环：请求队列 → Scheduler → Runner → 逐请求输出流。对应 vLLM `EngineCore.step()` + `OutputProcessor`
//! 的 token 层（detokenize / 解析在模型 crate 与 server 层做，这里只搬 token）。
//!
//! 线程模型：一个专用线程跑 `run()`（持有 GPU 资源，不跨线程）；外部通过 `EngineHandle` 提交请求，
//! 每个请求拿到一个 `tokio::sync::mpsc` 接收端收 `EngineEvent`。空闲时阻塞在请求通道上，不空转。

use crate::metrics::Metrics;
use crate::request::{FinishReason, Request, RequestId};
use crate::runner::{ModelExecutor, Runner};
use crate::sample::SamplingParams;
use crate::sched::{Scheduler, SchedulerParams};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// 引擎推给某个请求的事件。
#[derive(Debug, Clone)]
pub enum EngineEvent {
    /// 本 step 新产出的 token（投机解码时可能 > 1 个）。
    Tokens(Vec<u32>),
    /// 紧跟 `Tokens` 之后（仅请求了 logprobs 时）：与该批 token 平行的 logprobs。
    Logprobs(Vec<crate::sampler::TokenLogprobs>),
    Finished(FinishReason),
    /// 引擎内部错误（该请求终止）。
    Error(String),
}

pub struct Submission {
    pub prompt_ids: Vec<u32>,
    pub sampling: SamplingParams,
    pub tx: tokio::sync::mpsc::UnboundedSender<EngineEvent>,
}

enum Cmd {
    Submit(RequestId, Submission),
    Abort(RequestId),
    Shutdown,
}

/// 跨线程句柄：提交 / 取消请求 / 读指标。
#[derive(Clone)]
pub struct EngineHandle {
    tx: crossbeam_channel::Sender<Cmd>,
    next_id: Arc<AtomicU64>,
    metrics: Arc<Metrics>,
}

impl EngineHandle {
    pub fn submit(&self, sub: Submission) -> RequestId {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let _ = self.tx.send(Cmd::Submit(id, sub));
        id
    }
    pub fn abort(&self, id: RequestId) {
        let _ = self.tx.send(Cmd::Abort(id));
    }
    pub fn shutdown(&self) {
        let _ = self.tx.send(Cmd::Shutdown);
    }
    pub fn metrics(&self) -> Arc<Metrics> {
        self.metrics.clone()
    }
}

/// 每请求计时（TTFT / TPOT / E2E）。
struct Timing {
    arrival: std::time::Instant,
    prompt_len: u64,
    first_token: Option<std::time::Instant>,
    last_token: std::time::Instant,
}

pub struct Engine<E: ModelExecutor> {
    sched: Scheduler,
    /// 异步调度开关（ASYNC_SCHED=0 关；scheduler 的 async_scheduling 必须同步设置）。
    async_sched: bool,
    /// 在飞的一步：(SchedulerOutput, runner Pending, t_step, t_sched)。
    pending: Option<(crate::sched::output::SchedulerOutput, crate::runner::Pending, std::time::Instant, std::time::Instant)>,
    /// 插队 prefill 开关（CUT_IN=0 关）与计数。
    cut_in: bool,
    cut_in_max_rows: usize,
    cut_in_steps: u64,
    /// 已同步完成、等着 update 的一步（异步步之后紧跟的 prefill 步）。
    sync_done: std::collections::VecDeque<(crate::sched::output::SchedulerOutput, crate::sched::output::ModelRunnerOutput, std::time::Instant, std::time::Instant)>,
    runner: Runner<E>,
    rx: crossbeam_channel::Receiver<Cmd>,
    outs: HashMap<RequestId, tokio::sync::mpsc::UnboundedSender<EngineEvent>>,
    timing: HashMap<RequestId, Timing>,
    metrics: Arc<Metrics>,
}

impl<E: ModelExecutor> Engine<E> {
    pub fn new(sched: Scheduler, runner: Runner<E>) -> (Self, EngineHandle) {
        let (tx, rx) = crossbeam_channel::unbounded();
        let metrics = Arc::new(Metrics::new());
        let h = EngineHandle { tx, next_id: Arc::new(AtomicU64::new(1)), metrics: metrics.clone() };
        let async_sched = std::env::var("ASYNC_SCHED").map(|v| v != "0").unwrap_or(true);
        (
            Self {
                sched,
                async_sched,
                cut_in: async_sched && std::env::var("CUT_IN").map(|v| v != "0").unwrap_or(true),
                cut_in_max_rows: std::env::var("CUT_IN_MAX_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(8),
                cut_in_steps: 0,
                pending: None,
                sync_done: std::collections::VecDeque::new(),
                runner,
                rx,
                outs: HashMap::new(),
                timing: HashMap::new(),
                metrics,
            },
            h,
        )
    }

    /// 一个 step 结束后的指标回填。
    /// 一步的回填 + 指标 + 推送（schedule() 之前必须对所有已完成的步调用过它——否则 scheduler 不知道该请求已有新
    /// token，会把它再排一次；见 pitfalls 2026-10-17 "插队步结果排队晚发"）。
    fn publish_step(
        &mut self,
        so: &crate::sched::SchedulerOutput,
        mro: &crate::sched::output::ModelRunnerOutput,
        t_step: std::time::Instant,
        t_sched: std::time::Instant,
        t_exec: std::time::Instant,
    ) {
        let results = self.sched.update_from_output(so, mro);
        self.metrics.step_schedule.observe((t_sched - t_step).as_secs_f64());
        self.metrics.step_execute.observe((t_exec - t_sched).as_secs_f64());
        self.record_step(so, &results, t_step);
        self.metrics.step_update.observe(t_exec.elapsed().as_secs_f64());
        for (i, (rid, toks, finish)) in results.iter().enumerate() {
            let Some(tx) = self.outs.get(&rid) else { continue };
            if !toks.is_empty() {
                let _ = tx.send(EngineEvent::Tokens(toks.to_vec()));
                let lps = results.logprobs_of(i);
                if !lps.is_empty() {
                    let _ = tx.send(EngineEvent::Logprobs(lps.iter().flatten().cloned().collect()));
                }
            }
            if let Some(f) = finish {
                let _ = tx.send(EngineEvent::Finished(f));
            }
        }
        for rid in &so.finished_req_ids {
            self.outs.remove(rid);
        }
        for (rid, _, finish) in results.iter() {
            if finish.is_some() {
                self.outs.remove(&rid);
            }
        }
    }

    /// 执行失败：所有在跑的请求报错并清掉（GPU 状态不可信），返回 Err 结束引擎线程。
    fn fail_step(&mut self, e: anyhow::Error) -> Result<()> {
        let msg = format!("{e:#}");
        tracing::error!("engine step failed: {msg}");
        for (_, tx) in self.outs.drain() {
            let _ = tx.send(EngineEvent::Error(msg.clone()));
        }
        Err(e)
    }

    fn record_step(
        &mut self,
        so: &crate::sched::SchedulerOutput,
        results: &crate::sched::StepResults,
        t_step: std::time::Instant,
    ) {
        use std::sync::atomic::Ordering::Relaxed;
        let m = &self.metrics;
        let now = std::time::Instant::now();
        m.engine_steps_total.fetch_add(1, Relaxed);
        m.step_time.observe(t_step.elapsed().as_secs_f64());
        m.num_preemptions_total.fetch_add(so.preempted_req_ids.len() as u64, Relaxed);
        for (rid, toks, finish) in results.iter() {
            let Some(t) = self.timing.get_mut(&rid) else { continue };
            if !toks.is_empty() {
                m.generation_tokens_total.fetch_add(toks.len() as u64, Relaxed);
                match t.first_token {
                    None => {
                        t.first_token = Some(now);
                        m.time_to_first_token.observe((now - t.arrival).as_secs_f64());
                        m.prompt_tokens_total.fetch_add(t.prompt_len, Relaxed);
                    }
                    Some(_) => {
                        m.time_per_output_token
                            .observe((now - t.last_token).as_secs_f64() / toks.len() as f64);
                    }
                }
                t.last_token = now;
            }
            if let Some(f) = finish {
                m.e2e_request_latency.observe((now - t.arrival).as_secs_f64());
                match f {
                    FinishReason::Stop => m.request_success_stop.fetch_add(1, Relaxed),
                    FinishReason::Length => m.request_success_length.fetch_add(1, Relaxed),
                    FinishReason::Abort => m.request_abort.fetch_add(1, Relaxed),
                };
                self.timing.remove(&rid);
            }
        }
        for rid in &so.finished_req_ids {
            self.timing.remove(rid);
        }
        self.refresh_gauges();
    }

    fn refresh_gauges(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        let m = &self.metrics;
        m.num_requests_running.store(self.sched.num_running() as u64, Relaxed);
        m.num_requests_waiting.store(self.sched.num_waiting() as u64, Relaxed);
        m.set_kv_cache_usage(self.sched.kv_usage());
        let (q, h) = self.sched.prefix_cache_stats();
        m.prefix_cache_queries_total.store(q, Relaxed);
        m.prefix_cache_hits_total.store(h, Relaxed);
        let (hh, hf, ho, _) = self.sched.host_tier_stats();
        m.host_tier_hits_total.store(hh, Relaxed);
        m.host_tier_fetch_blocks_total.store(hf, Relaxed);
        m.host_tier_offload_blocks_total.store(ho, Relaxed);
    }

    /// 阻塞运行直到 `Shutdown`。每轮：吸收所有待处理命令（空闲时阻塞等第一条）→ schedule → execute → 分发。
    pub fn run(mut self) -> Result<()> {
        loop {
            // 空闲：阻塞等命令；忙：非阻塞吸收
            if !self.sched.has_requests() && self.pending.is_none() && self.sync_done.is_empty() {
                match self.rx.recv() {
                    Ok(cmd) => {
                        if !self.handle(cmd) {
                            return Ok(());
                        }
                    }
                    Err(_) => return Ok(()),
                }
            }
            while let Ok(cmd) = self.rx.try_recv() {
                if !self.handle(cmd) {
                    return Ok(());
                }
            }
            if !self.sched.has_requests() && self.pending.is_none() && self.sync_done.is_empty() {
                continue;
            }
            // 异步调度（Q3-l）：步 n 入队后，不等 GPU 就调度并入队步 n+1，再取 n 的结果。
            // pending 最多 1 个：(so, Pending, t_step, t_sched)。
            let (so, mro, t_step, t_sched, t_exec) = match self.pending.take() {
                Some((so_n, pend_n, t_step_n, t_sched_n)) => {
                    // 插队 prefill（Q3-o）：步 n 在飞、有新请求在等 → 立刻调度一次（running 行都有 placeholder，
                    // 产出的 step 只含新请求），同步跑完，结果排进 sync_done。新请求首 token 提前一步。
                    // 负载门槛：在飞步行数 ≤ CUT_IN_MAX_ROWS（默认 8）才插队——大 batch 下把 prefill 打散成多个独占步
                    // 反而伤 c32 TTFT / 长序列 TPOT（bench27）；小 batch（低负载）时省一步最值
                    if self.cut_in && self.sched.num_waiting() > 0 && so_n.num_scheduled_tokens.len() <= self.cut_in_max_rows {
                        let t_step = std::time::Instant::now();
                        let so_c = self.sched.schedule_waiting_only();
                        let t_sched = std::time::Instant::now();
                        if !so_c.num_scheduled_tokens.is_empty() {
                            let pend_c = match self.runner.begin_cut_in(&so_c) {
                                Ok(p) => p,
                                Err(e) => return self.fail_step(e),
                            };
                            let mro_c = match self.runner.finish(pend_c) {
                                Ok(m) => m,
                                Err(e) => return self.fail_step(e),
                            };
                            self.cut_in_steps += 1;
                            // 立刻回填 + 推送：下面 schedule() n+1 前 scheduler 必须已知道这些请求的首 token
                            self.publish_step(&so_c, &mro_c, t_step, t_sched, std::time::Instant::now());
                        }
                    }
                    // 先把 n+1 排进去（若还有请求且 n 是异步的），再等 n
                    if self.async_sched && self.sched.has_requests() {
                        let t_step = std::time::Instant::now();
                        let so_next = self.sched.schedule();
                        let t_sched = std::time::Instant::now();
                        let pend_next = match self.runner.begin(&so_next) {
                            Ok(p) => p,
                            Err(e) => return self.fail_step(e),
                        };
                        if self.runner.last_begin_was_async() {
                            self.pending = Some((so_next, pend_next, t_step, t_sched));
                        } else {
                            // n+1 是同步步（prefill 等）：begin 已经跑完，直接 finish 排队到 pending 位
                            let mro_next = match self.runner.finish(pend_next) {
                                Ok(m) => m,
                                Err(e) => return self.fail_step(e),
                            };
                            self.sync_done.push_back((so_next, mro_next, t_step, t_sched));
                        }
                    }
                    let mro_n = match self.runner.finish(pend_n) {
                        Ok(m) => m,
                        Err(e) => return self.fail_step(e),
                    };
                    (so_n, mro_n, t_step_n, t_sched_n, std::time::Instant::now())
                }
                None => match self.sync_done.pop_front() {
                    Some((so, mro, t_step, t_sched)) => (so, mro, t_step, t_sched, std::time::Instant::now()),
                    None => {
                        let t_step = std::time::Instant::now();
                        let so = self.sched.schedule();
                        let t_sched = std::time::Instant::now();
                        let pend = match self.runner.begin(&so) {
                            Ok(p) => p,
                            Err(e) => return self.fail_step(e),
                        };
                        if self.async_sched && self.runner.last_begin_was_async() {
                            self.pending = Some((so, pend, t_step, t_sched));
                            continue; // 下一轮循环：先排 n+1 再等 n
                        }
                        let mro = match self.runner.finish(pend) {
                            Ok(m) => m,
                            Err(e) => return self.fail_step(e),
                        };
                        (so, mro, t_step, t_sched, std::time::Instant::now())
                    }
                },
            };
            self.publish_step(&so, &mro, t_step, t_sched, t_exec);
        }
    }

    /// 返回 false = shutdown。
    fn handle(&mut self, cmd: Cmd) -> bool {
        match cmd {
            Cmd::Submit(id, sub) => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0);
                self.outs.insert(id, sub.tx);
                let inst = std::time::Instant::now();
                self.timing.insert(
                    id,
                    Timing {
                        arrival: inst,
                        prompt_len: sub.prompt_ids.len() as u64,
                        first_token: None,
                        last_token: inst,
                    },
                );
                self.sched.add_request(Request::new(id, sub.prompt_ids, sub.sampling, now));
                self.refresh_gauges();
                true
            }
            Cmd::Abort(id) => {
                self.sched.abort(id);
                if let Some(tx) = self.outs.remove(&id) {
                    let _ = tx.send(EngineEvent::Finished(FinishReason::Abort));
                }
                if let Some(t) = self.timing.remove(&id) {
                    self.metrics.e2e_request_latency.observe(t.arrival.elapsed().as_secs_f64());
                    self.metrics.request_abort.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                self.refresh_gauges();
                true
            }
            Cmd::Shutdown => false,
        }
    }
}

/// 便捷：在新线程上跑引擎，返回句柄与 join handle。
pub fn spawn<E: ModelExecutor + Send + 'static>(
    sched: Scheduler,
    runner: Runner<E>,
) -> (EngineHandle, std::thread::JoinHandle<Result<()>>) {
    let (eng, h) = Engine::new(sched, runner);
    let jh = std::thread::Builder::new()
        .name("dinfer-engine".into())
        .spawn(move || eng.run())
        .expect("spawn engine");
    (h, jh)
}

/// 服务端需要的、与模型无关的 scheduler 参数构造。
pub fn scheduler_params(
    sched: crate::config::SchedulerConfig,
    cache: crate::config::CacheConfig,
    max_model_len: u32,
    eos_token_ids: Vec<u32>,
    num_spec_tokens: u32,
) -> SchedulerParams {
    let async_scheduling = std::env::var("ASYNC_SCHED").map(|v| v != "0").unwrap_or(true);
    SchedulerParams { sched, cache, max_model_len, num_spec_tokens, async_scheduling, eos_token_ids }
}
