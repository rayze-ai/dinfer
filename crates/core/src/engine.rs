//! 引擎循环：请求队列 → Scheduler → Runner → 逐请求输出流。对应 vLLM `EngineCore.step()` + `OutputProcessor`
//! 的 token 层（detokenize / 解析在模型 crate 与 server 层做，这里只搬 token）。
//!
//! 线程模型：一个专用线程跑 `run()`（持有 GPU 资源，不跨线程）；外部通过 `EngineHandle` 提交请求，
//! 每个请求拿到一个 `tokio::sync::mpsc` 接收端收 `EngineEvent`。空闲时阻塞在请求通道上，不空转。

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

/// 跨线程句柄：提交 / 取消请求。
#[derive(Clone)]
pub struct EngineHandle {
    tx: crossbeam_channel::Sender<Cmd>,
    next_id: Arc<AtomicU64>,
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
}

pub struct Engine<E: ModelExecutor> {
    sched: Scheduler,
    runner: Runner<E>,
    rx: crossbeam_channel::Receiver<Cmd>,
    outs: HashMap<RequestId, tokio::sync::mpsc::UnboundedSender<EngineEvent>>,
}

impl<E: ModelExecutor> Engine<E> {
    pub fn new(sched: Scheduler, runner: Runner<E>) -> (Self, EngineHandle) {
        let (tx, rx) = crossbeam_channel::unbounded();
        let h = EngineHandle { tx, next_id: Arc::new(AtomicU64::new(1)) };
        (Self { sched, runner, rx, outs: HashMap::new() }, h)
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
            for (rid, toks, finish) in results.iter() {
                let Some(tx) = self.outs.get(&rid) else { continue };
                if !toks.is_empty() {
                    let _ = tx.send(EngineEvent::Tokens(toks.to_vec()));
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
                self.sched.add_request(Request::new(id, sub.prompt_ids, sub.sampling, now));
                true
            }
            Cmd::Abort(id) => {
                self.sched.abort(id);
                if let Some(tx) = self.outs.remove(&id) {
                    let _ = tx.send(EngineEvent::Finished(FinishReason::Abort));
                }
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
) -> SchedulerParams {
    SchedulerParams {
        sched,
        cache,
        max_model_len,
        num_spec_tokens: 0,
        async_scheduling: false,
        eos_token_ids,
    }
}
