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
use anyhow::{Context, Result};
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
    /// 随首批 `Tokens` 之前发一次：首次调度时 prefix cache 命中的 prompt token 数（`usage.prompt_tokens_details.cached_tokens`）。
    PrefixCached(u32),
    /// 引擎内部错误（该请求终止）。
    Error(String),
}

pub struct Submission {
    pub prompt_ids: Vec<u32>,
    pub sampling: SamplingParams,
    pub tx: tokio::sync::mpsc::UnboundedSender<EngineEvent>,
    /// PD 分离（design §10.3）：`None` 普通；`Some(PdSubmit::Export)` P 侧；
    /// `Some(PdSubmit::Import { remote_id })` D 侧，`remote_id` = P 侧 request id，用来在 `IncomingKv` 里配对 payload。
    pub pd: Option<PdSubmit>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PdSubmit {
    Export,
    Import { remote_id: RequestId },
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
    /// 连续空步计数（退避 + 诊断）。
    empty_steps: u64,
    /// 已同步完成、等着 update 的一步（异步步之后紧跟的 prefill 步）。
    sync_done: std::collections::VecDeque<(crate::sched::output::SchedulerOutput, crate::sched::output::ModelRunnerOutput, std::time::Instant, std::time::Instant)>,
    runner: Runner<E>,
    rx: crossbeam_channel::Receiver<Cmd>,
    outs: HashMap<RequestId, tokio::sync::mpsc::UnboundedSender<EngineEvent>>,
    timing: HashMap<RequestId, Timing>,
    metrics: Arc<Metrics>,
    /// PD 分离状态（design §10.3）；`None` = 单机。
    pd: Option<PdState>,
}

/// engine 侧 PD 状态。
pub struct PdState {
    /// D 侧：接收线程写入的迁移（P 的 request_id → payload）。P 侧为空表。
    pub incoming: Arc<crate::pd::IncomingKv>,
    /// P 侧：导出后往哪发。D 侧 None。
    pub transport: Option<Arc<dyn crate::pd::KvTransport>>,
    pub model_tag: String,
    /// D 侧：等 KV 最长多久，超时转普通请求（本地重算 prefill）。
    pub wait_timeout: std::time::Duration,
    /// D 侧：已配对但尚未 begin 的 payload（本地 id → payload）。
    payloads: HashMap<RequestId, Box<dyn crate::pd::HostBuf>>,
    /// D 侧：KV 未到的 Import 提交（本地 id, 提交, 到达时刻）。
    pending: Vec<(RequestId, Vec<u32>, SamplingParams, RequestId, std::time::Instant)>,
    /// P 侧：Export 请求的头信息（导出时需要 prompt / sampling / 首 token）。
    export_meta: HashMap<RequestId, (Vec<u32>, SamplingParams, Option<u32>)>,
    /// P 侧共享槽区（design §10.3 b）：已映射到 executor 的对端区代数；`Some(Err)` = 该代数映射失败（走 TCP，不再重试）。
    region_gen: Option<(u64, bool)>,
}

impl PdState {
    pub fn new(incoming: Arc<crate::pd::IncomingKv>, transport: Option<Arc<dyn crate::pd::KvTransport>>, model_tag: String, wait_timeout: std::time::Duration) -> Self {
        Self { incoming, transport, model_tag, wait_timeout, payloads: HashMap::new(), pending: Vec::new(), export_meta: HashMap::new(), region_gen: None }
    }
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
                empty_steps: 0,
                pending: None,
                sync_done: std::collections::VecDeque::new(),
                runner,
                rx,
                outs: HashMap::new(),
                timing: HashMap::new(),
                metrics,
                pd: None,
            },
            h,
        )
    }

    /// 一个 step 结束后的指标回填。
    /// 一步的回填 + 指标 + 推送（schedule() 之前必须对所有已完成的步调用过它——否则 scheduler 不知道该请求已有新
    /// token，会把它再排一次；见 pitfalls 2026-10-17 "插队步结果排队晚发"）。
    /// 启用 PD（在 `run` 之前调用一次）。
    pub fn set_pd(&mut self, st: PdState) {
        self.pd = Some(st);
    }
    /// Import 提交：KV 已到 → 立即登记为 Import 请求；未到 → 挂 pending。
    fn pd_submit_import(&mut self, id: RequestId, prompt_ids: Vec<u32>, sampling: SamplingParams, remote_id: RequestId, now_ns: u64) {
        let Some(pd) = self.pd.as_mut() else { return self.add_plain(id, prompt_ids, sampling, now_ns) };
        match pd.incoming.take(remote_id) {
            Some(m) => {
                if m.header.prompt_ids.len() != prompt_ids.len() {
                    tracing::warn!(id, remote_id, "pd: prompt length mismatch, falling back to local prefill");
                    self.metrics.pd_fallback_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return self.add_plain(id, prompt_ids, sampling, now_ns);
                }
                pd.payloads.insert(id, m.payload);
                let mut r = Request::new(id, prompt_ids, sampling, now_ns);
                r.pd = Some(crate::request::PdRole::Import { first_token: m.header.first_token });
                self.sched.add_request(r);
            }
            None => pd.pending.push((id, prompt_ids, sampling, remote_id, std::time::Instant::now())),
        }
    }
    fn add_plain(&mut self, id: RequestId, prompt_ids: Vec<u32>, sampling: SamplingParams, now_ns: u64) {
        self.sched.add_request(Request::new(id, prompt_ids, sampling, now_ns));
    }
    /// 每步开头：pending 里 KV 到了的转 Import，超时的转普通（fallback）。
    fn pd_poll_pending(&mut self) {
        let Some(pd) = self.pd.as_mut() else { return };
        if pd.pending.is_empty() {
            return;
        }
        let timeout = pd.wait_timeout;
        let items = std::mem::take(&mut pd.pending);
        let now_ns = unix_ns();
        for (id, prompt_ids, sampling, remote, t0) in items {
            if self.pd.as_ref().unwrap().incoming.contains(remote) {
                self.pd_submit_import(id, prompt_ids, sampling, remote, now_ns);
            } else if t0.elapsed() >= timeout {
                tracing::warn!(id, remote, "pd: KV wait timeout, falling back to local prefill");
                self.metrics.pd_fallback_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                self.add_plain(id, prompt_ids, sampling, now_ns);
            } else {
                self.pd.as_mut().unwrap().pending.push((id, prompt_ids, sampling, remote, t0));
            }
        }
        self.pd.as_mut().unwrap().incoming.gc(timeout * 4);
    }
    /// begin 之前、任何 GPU 写之前：先导出（P），再导入（D）。
    fn pd_pre_begin(&mut self, so: &crate::sched::output::SchedulerOutput) -> Result<()> {
        if so.kv_export.is_empty() && so.kv_import.is_empty() {
            return Ok(());
        }
        let Some(pd) = self.pd.as_mut() else {
            anyhow::bail!("pd: kv_export/kv_import scheduled but PD not enabled");
        };
        let unit_bytes = self.runner.exec.kv_unit_bytes();
        anyhow::ensure!(unit_bytes > 0, "pd: executor does not support kv export/import");
        // 共享槽区：对端握手到新代数的区 → 映射到 executor（一次）；失败记下，该代数内走 TCP payload。
        let mut slots_ok = false;
        if let Some(t) = pd.transport.as_ref() {
            if let Some((info, gen)) = t.region() {
                if pd.region_gen.map_or(true, |(g, _)| g != gen) {
                    let ok = match self.runner.exec.pd_map_region(&info) {
                        Ok(()) => {
                            tracing::info!(kind = %info.kind, slots = info.slots, gen, "pd: peer shared region mapped; exports go through slots");
                            true
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, gen, "pd: cannot map peer shared region, exports fall back to tcp payload");
                            false
                        }
                    };
                    pd.region_gen = Some((gen, ok));
                }
                slots_ok = pd.region_gen.is_some_and(|(g, ok)| g == gen && ok);
            }
        }
        for (rid, blocks) in &so.kv_export {
            let mut slot = if slots_ok { pd.transport.as_ref().and_then(|t| t.acquire_slot()) } else { None };
            let mut handle = match slot {
                Some(sl) => match self.runner.exec.kv_export_slot(blocks, sl) {
                    Ok(h) => h,
                    Err(e) => {
                        tracing::warn!(id = rid, slot = sl, error = %e, "pd: slot export failed, using tcp payload");
                        if let Some(t) = pd.transport.as_ref() {
                            t.release_slot(sl);
                        }
                        slot = None;
                        self.runner.exec.kv_export(blocks)?
                    }
                },
                None => self.runner.exec.kv_export(blocks)?,
            };
            let (prompt_ids, sampling, first) = pd.export_meta.remove(rid).with_context(|| format!("pd: export meta for {rid}"))?;
            let first_token = first.with_context(|| format!("pd: req {rid} exported without a first token"))?;
            let header = crate::pd::MigrationHeader {
                request_id: *rid,
                model_tag: pd.model_tag.clone(),
                prompt_ids,
                sampling,
                first_token,
                num_units: blocks.groups.first().map_or(0, |g| g.len()) as u32,
                unit_bytes: unit_bytes as u32,
                slot,
            };
            self.metrics.pd_export_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if slot.is_some() {
                self.metrics.pd_export_slot_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            if let Some(t) = pd.transport.clone() {
                std::thread::Builder::new().name("pd-send".into()).spawn(move || {
                    // 只阻塞本线程：等 GPU→host 拷贝完成，再发；句柄 drop 归还缓冲。NIXL 后端 poll 到 Done 为止。
                    let buf = match handle.wait() {
                        Ok(b) => b,
                        Err(e) => {
                            tracing::warn!(id = header.request_id, error = %e, "pd: export wait failed");
                            return;
                        }
                    };
                    match t.send(&header, buf.desc()) {
                        Ok(mut x) => {
                            while matches!(x.poll(), Ok(crate::pd::TransferState::InFlight)) {
                                std::thread::sleep(std::time::Duration::from_micros(50));
                            }
                        }
                        Err(e) => tracing::warn!(id = header.request_id, error = %e, "pd: send failed"),
                    }
                })?;
            }
        }
        for (rid, blocks) in &so.kv_import {
            let payload = pd.payloads.remove(rid).with_context(|| format!("pd: payload for {rid}"))?;
            let want = blocks.groups.first().map_or(0, |g| g.len()) * unit_bytes;
            anyhow::ensure!(payload.len() == want, "pd: payload {} != {} blocks × {unit_bytes}", payload.len(), blocks.groups[0].len());
            self.runner.exec.kv_import(blocks, payload)?;
            self.metrics.pd_import_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }
    fn publish_step(
        &mut self,
        so: &crate::sched::SchedulerOutput,
        mro: &crate::sched::output::ModelRunnerOutput,
        t_step: std::time::Instant,
        t_sched: std::time::Instant,
        t_exec: std::time::Instant,
    ) {
        let results = self.sched.update_from_output(so, mro);
        // 投机解码：接受率计数 + 把下一步的 draft 交还 scheduler（MTP / n-gram 都走这里）。
        if !so.scheduled_spec_tokens.is_empty() {
            let (mut nd, mut na) = (0u64, 0u64);
            for (rid, spec) in &so.scheduled_spec_tokens {
                if let Some(i) = mro.req_ids.iter().position(|r| r == rid) {
                    nd += spec.len() as u64;
                    na += mro.sampled[i].len().saturating_sub(1) as u64;
                }
            }
            self.metrics.spec_draft_total.fetch_add(nd, std::sync::atomic::Ordering::Relaxed);
            self.metrics.spec_accepted_total.fetch_add(na, std::sync::atomic::Ordering::Relaxed);
        }
        if mro.drafts.iter().any(|d| !d.is_empty()) {
            let dr: Vec<(RequestId, Vec<u32>)> = mro
                .req_ids
                .iter()
                .zip(mro.drafts.iter())
                .filter(|(_, d)| !d.is_empty())
                .map(|(r, d)| (*r, d.clone()))
                .collect();
            self.sched.update_draft_tokens(&dr);
        }
        // scheduler 自行结束的请求（分不到块且无人可抢占）：推 Finished 并清理
        for (rid, f) in self.sched.take_force_finished() {
            if let Some(tx) = self.outs.remove(&rid) {
                let _ = tx.send(EngineEvent::Finished(f));
            }
        }
        self.metrics.step_schedule.observe((t_sched - t_step).as_secs_f64());
        self.metrics.step_execute.observe((t_exec - t_sched).as_secs_f64());
        if let Some(mode) = mro.mode {
            use crate::runner::BatchMode;
            let h = match mode {
                BatchMode::Prefill => &self.metrics.step_execute_prefill,
                BatchMode::Decode => &self.metrics.step_execute_decode,
                BatchMode::Paged => &self.metrics.step_execute_mixed,
            };
            h.observe((t_exec - t_sched).as_secs_f64());
            if let Some(ph) = mro.phases {
                let mi = match mode {
                    BatchMode::Prefill => 0,
                    BatchMode::Decode => 1,
                    BatchMode::Paged => 2,
                };
                let hs = &self.metrics.step_phase[mi];
                hs[0].observe(ph.upload);
                hs[1].observe(ph.launch);
                hs[2].observe(ph.wait);
            }
        }
        self.record_step(so, &results, t_step);
        self.metrics.step_update.observe(t_exec.elapsed().as_secs_f64());
        for (i, (rid, toks, finish)) in results.iter().enumerate() {
            let Some(tx) = self.outs.get(&rid) else { continue };
            if let Some(n) = results.items[i].num_cached_tokens {
                let _ = tx.send(EngineEvent::PrefixCached(n));
            }
            if !toks.is_empty() {
                let _ = tx.send(EngineEvent::Tokens(toks.to_vec()));
                if let Some(pd) = self.pd.as_mut() {
                    if let Some(m) = pd.export_meta.get_mut(&rid) {
                        m.2.get_or_insert(toks[0]);
                    }
                }
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
            if !self.sched.has_requests() && !self.sched.has_pending_export() && self.pending.is_none() && self.sync_done.is_empty() {
                // PD：D 侧有等 KV 的提交时不能死等命令，要定期醒来查到达 / 超时回退。
                let pd_wait = self.pd.as_ref().filter(|p| !p.pending.is_empty()).map(|_| std::time::Duration::from_millis(5));
                let cmd = match pd_wait {
                    Some(d) => match self.rx.recv_timeout(d) {
                        Ok(c) => Some(c),
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => None,
                        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => return Ok(()),
                    },
                    None => match self.rx.recv() {
                        Ok(c) => Some(c),
                        Err(_) => return Ok(()),
                    },
                };
                if let Some(cmd) = cmd {
                    if !self.handle(cmd) {
                        return Ok(());
                    }
                }
            }
            while let Ok(cmd) = self.rx.try_recv() {
                if !self.handle(cmd) {
                    return Ok(());
                }
            }
            self.pd_poll_pending();
            if self.pd.is_some() {
                self.runner.exec.pd_poll();
            }
            if !self.sched.has_requests() && !self.sched.has_pending_export() && self.pending.is_none() && self.sync_done.is_empty() {
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
                    // 池里没空闲块时不插队（分不到块的 schedule 只会产出空步；livelock 2026-10-17）
                    if self.cut_in
                        && self.sched.num_waiting() > 0
                        && self.sched.num_free_blocks() > 0
                        && so_n.num_scheduled_tokens.len() <= self.cut_in_max_rows
                    {
                        let t_step = std::time::Instant::now();
                        let so_c = self.sched.schedule_waiting_only();
                        let t_sched = std::time::Instant::now();
                        // PD Import 首步没有 token 要算（首 token 已由 P 出），但 new_reqs / kv_import 必须走到
                        // runner（登记 ReqState、导入 KV），否则下一步 decode 报 "cached req unknown"。
                        if !so_c.num_scheduled_tokens.is_empty() || !so_c.new_reqs.is_empty() || !so_c.kv_import.is_empty() {
                            if let Err(e) = self.pd_pre_begin(&so_c) {
                                self.fail_step(e)?;
                                continue;
                            }
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
                        if let Err(e) = self.pd_pre_begin(&so_next) {
                            self.fail_step(e)?;
                            continue;
                        }
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
                        if let Err(e) = self.pd_pre_begin(&so) {
                            self.fail_step(e)?;
                            continue;
                        }
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
            // 空步退避：有请求却一个都排不进（块不够、全部在飞）时不要满转 CPU；短暂让出等 GPU / 释放
            if so.num_scheduled_tokens.is_empty() && so.kv_import.is_empty() && so.kv_export.is_empty() && self.sched.has_requests() {
                self.empty_steps += 1;
                if self.empty_steps % 1000 == 1 {
                    tracing::warn!(
                        empty_steps = self.empty_steps,
                        running = self.sched.num_running(),
                        waiting = self.sched.num_waiting(),
                        free_blocks = self.sched.num_free_blocks(),
                        "engine: empty step with requests present (backing off 200 µs)"
                    );
                }
                std::thread::sleep(std::time::Duration::from_micros(200));
            } else {
                self.empty_steps = 0;
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
                let Submission { prompt_ids, sampling, pd: pd_kind, .. } = sub;
                match pd_kind {
                    None => self.add_plain(id, prompt_ids, sampling, now),
                    Some(PdSubmit::Export) => {
                        if let Some(pd) = self.pd.as_mut() {
                            pd.export_meta.insert(id, (prompt_ids.clone(), sampling.clone(), None));
                        }
                        let mut r = Request::new(id, prompt_ids, sampling, now);
                        r.pd = Some(crate::request::PdRole::Export);
                        self.sched.add_request(r);
                    }
                    Some(PdSubmit::Import { remote_id }) => self.pd_submit_import(id, prompt_ids, sampling, remote_id, now),
                }
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
fn unix_ns() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
}

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
/// 带 PD 状态启动（design §10.3）。
pub fn spawn_with_pd<E: ModelExecutor + Send + 'static>(
    sched: Scheduler,
    runner: Runner<E>,
    pd: PdState,
) -> (EngineHandle, std::thread::JoinHandle<Result<()>>) {
    let (mut eng, h) = Engine::new(sched, runner);
    eng.set_pd(pd);
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
    SchedulerParams {
        sched,
        cache,
        max_model_len,
        num_spec_tokens,
        async_scheduling,
        eos_token_ids,
        recurrent_snapshot_tokens: 0,
        recurrent_snapshot_slots: 0,
        recurrent_snapshot_chunk: 0,
    }
}

#[cfg(test)]
mod pd_tests {
    use super::*;
    use crate::config::{CacheConfig, SchedulerConfig};
    use crate::kv::{GroupInfo, KvBlocks, KvCacheSpec};
    use crate::pd::{IncomingKv, TcpTransport};
    use crate::runner::BatchPlan;
    use std::sync::Mutex;
    use std::time::Duration;

    const UNIT: usize = 8;
    /// 假 executor：KV = 每块 8 字节的 host 影子存储；forward 输出 = 2000 + seq_len。
    struct PdEcho {
        kv: Arc<Mutex<HashMap<u32, [u8; UNIT]>>>,
    }
    impl ModelExecutor for PdEcho {
        fn forward_sample(&mut self, plan: &BatchPlan, generated: &mut Vec<Vec<u32>>, drafts: &mut Vec<Vec<u32>>) -> Result<()> {
            for &s in &plan.seq_lens {
                generated.push(vec![2000 + s as u32]);
                drafts.push(Vec::new());
            }
            Ok(())
        }
        fn kv_unit_bytes(&self) -> usize {
            UNIT
        }
        fn kv_export(&mut self, blocks: &KvBlocks) -> Result<Box<dyn crate::pd::KvExportHandle>> {
            let kv = self.kv.lock().unwrap();
            let mut out = Vec::new();
            for &b in blocks.group(0) {
                out.extend_from_slice(&kv.get(&b).copied().unwrap_or([b as u8; UNIT]));
            }
            Ok(Box::new(crate::pd::ReadyExport(Box::new(out))))
        }
        fn kv_import(&mut self, blocks: &KvBlocks, payload: Box<dyn crate::pd::HostBuf>) -> Result<()> {
            let payload = payload.as_slice();
            let mut kv = self.kv.lock().unwrap();
            for (i, &b) in blocks.group(0).iter().enumerate() {
                let mut u = [0u8; UNIT];
                u.copy_from_slice(&payload[i * UNIT..(i + 1) * UNIT]);
                kv.insert(b, u);
            }
            Ok(())
        }
    }
    fn sched() -> Scheduler {
        let p = SchedulerParams {
            sched: SchedulerConfig { max_num_seqs: 8, max_num_batched_tokens: 1024, enable_chunked_prefill: true, long_prefill_token_threshold: 0, decode_exempt_from_budget: false },
            cache: CacheConfig { block_size: 16, hash_block_size: 16, ..Default::default() },
            max_model_len: 256,
            num_spec_tokens: 0,
            async_scheduling: false,
            eos_token_ids: vec![],
            recurrent_snapshot_tokens: 0,
            recurrent_snapshot_slots: 0,
            recurrent_snapshot_chunk: 0,
        };
        let g = vec![GroupInfo { spec: KvCacheSpec::Mla { block_size: 16, head_size: 64, dtype_bytes: 2 }, block_size: 16 }];
        Scheduler::new(p, g, 64)
    }
    fn collect(rx: &mut tokio::sync::mpsc::UnboundedReceiver<EngineEvent>) -> (Vec<u32>, Option<FinishReason>) {
        let (mut toks, mut fin) = (Vec::new(), None);
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while fin.is_none() && std::time::Instant::now() < deadline {
            match rx.try_recv() {
                Ok(EngineEvent::Tokens(t)) => toks.extend(t),
                Ok(EngineEvent::Finished(f)) => fin = Some(f),
                Ok(EngineEvent::Error(e)) => panic!("engine error: {e}"),
                Ok(_) => {}
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
                Err(_) => std::thread::sleep(Duration::from_millis(2)),
            }
        }
        (toks, fin)
    }

    /// 基线：PdEcho + 普通提交能正常结束（排除 mock/engine 装配问题）。
    #[test]
    fn plain_submit_on_pdecho_finishes() {
        std::env::set_var("ASYNC_SCHED", "0");
        let (h, _j) = spawn(sched(), Runner::new(PdEcho { kv: Arc::new(Mutex::new(HashMap::new())) }, 16));
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        h.submit(Submission { prompt_ids: (0..40).collect(), sampling: SamplingParams { max_tokens: 2, ..Default::default() }, tx, pd: None });
        let (toks, fin) = collect(&mut rx);
        assert_eq!(fin, Some(FinishReason::Length), "toks={toks:?}");
        assert_eq!(toks, vec![2040, 2041]);
        h.shutdown();
    }

    /// P（Export, max_tokens=1）→ TCP → D（Import）：D 从首 token 续 decode；D 的 KV 影子等于 P 导出的字节；超时回退计数。
    #[test]
    fn prefill_export_then_decode_import_over_tcp() {
        pd_roundtrip(false);
    }
    /// 异步调度 + 插队 prefill 开（4090 默认配置）：D 的 Import 首步走 cut-in 路径（q4 第 10 轮抓到的 "cached req unknown"）。
    #[test]
    fn prefill_export_then_decode_import_over_tcp_async_cut_in() {
        pd_roundtrip(true);
    }
    fn pd_roundtrip(async_sched: bool) {
        std::env::set_var("ASYNC_SCHED", if async_sched { "1" } else { "0" });
        std::env::set_var("CUT_IN", if async_sched { "1" } else { "0" });
        let d_kv = Arc::new(Mutex::new(HashMap::new()));
        let d_incoming = Arc::new(IncomingKv::default());
        let d_port = crate::pd::spawn_receiver("127.0.0.1:0", "t".into(), d_incoming.clone()).unwrap();
        let (dh, _dj) = spawn_with_pd(
            sched(),
            Runner::new(PdEcho { kv: d_kv.clone() }, 16),
            PdState::new(d_incoming.clone(), None, "t".into(), Duration::from_millis(300)),
        );
        let p_kv = Arc::new(Mutex::new(HashMap::new()));
        let (ph, _pj) = spawn_with_pd(
            sched(),
            Runner::new(PdEcho { kv: p_kv.clone() }, 16),
            PdState::new(
                Arc::new(IncomingKv::default()),
                Some(Arc::new(TcpTransport { addr: format!("127.0.0.1:{d_port}"), connect_timeout: Duration::from_secs(2) })),
                "t".into(),
                Duration::from_secs(1),
            ),
        );
        let prompt: Vec<u32> = (0..40).collect(); // 3 块（含尾块）
        // 1. P：max_tokens=1 Export
        let (ptx, mut prx) = tokio::sync::mpsc::unbounded_channel();
        let pid = ph.submit(Submission { prompt_ids: prompt.clone(), sampling: SamplingParams { max_tokens: 1, ..Default::default() }, tx: ptx, pd: Some(PdSubmit::Export) });
        let (ptoks, pfin) = collect(&mut prx);
        assert_eq!(ptoks.len(), 1);
        assert_eq!(pfin, Some(FinishReason::Length));
        // P 的下一次 schedule 才导出 → 再提交一个普通请求驱动一步
        let (xtx, mut xrx) = tokio::sync::mpsc::unbounded_channel();
        ph.submit(Submission { prompt_ids: vec![9, 9, 9], sampling: SamplingParams { max_tokens: 1, ..Default::default() }, tx: xtx, pd: None });
        let _ = collect(&mut xrx);
        let t0 = std::time::Instant::now();
        while ph.metrics().pd_export_total.load(Ordering::Relaxed) == 0 && t0.elapsed() < Duration::from_secs(3) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(ph.metrics().pd_export_total.load(Ordering::Relaxed), 1);
        while !d_incoming.contains(pid) && t0.elapsed() < Duration::from_secs(3) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(d_incoming.contains(pid), "migration must reach D");
        // 2. D：Import，续 3 token
        let (dtx, mut drx) = tokio::sync::mpsc::unbounded_channel();
        dh.submit(Submission { prompt_ids: prompt.clone(), sampling: SamplingParams { max_tokens: 4, ..Default::default() }, tx: dtx, pd: Some(PdSubmit::Import { remote_id: pid }) });
        let (dtoks, dfin) = collect(&mut drx);
        assert_eq!(dfin, Some(FinishReason::Length));
        // 首 token 由 P 出（不重复推送），D 再出 3 个：seq_len 41, 42, 43
        assert_eq!(dtoks, vec![2041, 2042, 2043], "D must continue from prompt+first_token");
        assert_eq!(dh.metrics().pd_import_total.load(Ordering::Relaxed), 1);
        assert_eq!(dh.metrics().pd_fallback_total.load(Ordering::Relaxed), 0);
        // D 的 KV 影子 = P 导出的 3 块字节（P 侧没写过 → 默认 [block_id; 8]）
        let got = d_kv.lock().unwrap().len();
        assert_eq!(got, 3, "3 blocks imported into D");
        // 3. 回退：没有对应迁移的 Import，300 ms 后本地重算
        let (ftx, mut frx) = tokio::sync::mpsc::unbounded_channel();
        dh.submit(Submission { prompt_ids: (100..120).collect(), sampling: SamplingParams { max_tokens: 2, ..Default::default() }, tx: ftx, pd: Some(PdSubmit::Import { remote_id: 999_999 }) });
        let (ftoks, ffin) = collect(&mut frx);
        assert_eq!(ffin, Some(FinishReason::Length));
        assert_eq!(ftoks.len(), 2);
        assert_eq!(dh.metrics().pd_fallback_total.load(Ordering::Relaxed), 1);
        ph.shutdown();
        dh.shutdown();
    }
}
