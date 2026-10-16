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
        (Self { sched, runner, rx, outs: HashMap::new(), timing: HashMap::new(), metrics }, h)
    }

    /// 一个 step 结束后的指标回填。
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
    }

    /// 阻塞运行直到 `Shutdown`。每轮：吸收所有待处理命令（空闲时阻塞等第一条）→ schedule → execute → 分发。
    pub fn run(mut self) -> Result<()> {
        loop {
            // 空闲：阻塞等命令；忙：非阻塞吸收
            if !self.sched.has_requests() {
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
            if !self.sched.has_requests() {
                continue;
            }
            let t_step = std::time::Instant::now();
            let so = self.sched.schedule();
            let mro = match self.runner.execute(&so) {
                Ok(m) => m,
                Err(e) => {
                    // 执行失败：所有在跑的请求报错并清掉（GPU 状态不可信）
                    let msg = format!("{e:#}");
                    tracing::error!("engine step failed: {msg}");
                    for (_, tx) in self.outs.drain() {
                        let _ = tx.send(EngineEvent::Error(msg.clone()));
                    }
                    return Err(e);
                }
            };
            let results = self.sched.update_from_output(&so, &mro);
            self.record_step(&so, &results, t_step);
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
    SchedulerParams { sched, cache, max_model_len, num_spec_tokens, async_scheduling: false, eos_token_ids }
}
