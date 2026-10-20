//! K3 的 TP/EP 多卡 executor：一进程多线程 lockstep（design.md §13.1；与 `models/qwen3/src/tp.rs` 同构，去掉 PD 部分）。
//!
//! rank 0 的 `KimiK3Executor` 在 engine 线程上跑；其它 rank 各一个线程持自己的 executor（自己的 Device / Stream / NCCL Comm，
//! `KimiK3Executor::new_with_tp`）。每步把 `BatchPlan` + 当前 generated / drafts clone 发给各 rank，所有 rank 各自 forward + 采样
//!（NCCL all-reduce 逐位一致 + 同 seed ⇒ 同 token），只采用 rank 0 的结果。不支持：异步调度（lockstep 同步）、PD、host tier。

use anyhow::{Context, Result};
use dinfer_core::runner::{BatchPlan, ModelExecutor};
use dinfer_core::sampler::TokenLogprobs;
use std::sync::mpsc;

use crate::executor::KimiK3Executor;

struct Step {
    plan: BatchPlan,
    generated: Vec<Vec<u32>>,
    drafts: Vec<Vec<u32>>,
}

enum Msg {
    Step(Step),
    Shutdown,
}

struct Worker {
    tx: mpsc::Sender<Msg>,
    done: mpsc::Receiver<Result<()>>,
    jh: Option<std::thread::JoinHandle<()>>,
}

pub struct TpExecutor {
    rank0: KimiK3Executor,
    workers: Vec<Worker>,
}

impl TpExecutor {
    /// `others[i]` 是 rank i+1 的 executor（已带 comm）。各 executor 在各自线程上创建；这里 move 进各自的工作线程
    ///（cudarc 按 Arc<CudaContext> 重新 bind，允许）。
    pub fn new(rank0: KimiK3Executor, others: Vec<KimiK3Executor>) -> Self {
        let workers = others
            .into_iter()
            .enumerate()
            .map(|(i, mut ex)| {
                let (tx, rx) = mpsc::channel::<Msg>();
                let (dtx, done) = mpsc::channel::<Result<()>>();
                let jh = std::thread::Builder::new()
                    .name(format!("tp-rank{}", i + 1))
                    .spawn(move || {
                        if let Err(e) = ex.stream.bind_ctx() {
                            let _ = dtx.send(Err(e));
                            return;
                        }
                        loop {
                            match rx.recv() {
                                Ok(Msg::Step(mut st)) => {
                                    let r = ex.forward_sample(&st.plan, &mut st.generated, &mut st.drafts);
                                    if dtx.send(r).is_err() {
                                        break;
                                    }
                                }
                                Ok(Msg::Shutdown) | Err(_) => break,
                            }
                        }
                    })
                    .expect("spawn tp worker");
                Worker { tx, done, jh: Some(jh) }
            })
            .collect();
        Self { rank0, workers }
    }

    pub fn world(&self) -> usize {
        self.workers.len() + 1
    }

    fn broadcast(&self, plan: &BatchPlan, generated: &[Vec<u32>], drafts: &[Vec<u32>]) -> Result<()> {
        for w in &self.workers {
            w.tx.send(Msg::Step(Step { plan: plan.clone(), generated: generated.to_vec(), drafts: drafts.to_vec() }))
                .map_err(|_| anyhow::anyhow!("tp worker gone"))?;
        }
        Ok(())
    }

    fn join(&self) -> Result<()> {
        for (i, w) in self.workers.iter().enumerate() {
            w.done.recv().map_err(|_| anyhow::anyhow!("tp rank {} died", i + 1))?.with_context(|| format!("tp rank {}", i + 1))?;
        }
        Ok(())
    }
}

impl Drop for TpExecutor {
    fn drop(&mut self) {
        for w in &mut self.workers {
            let _ = w.tx.send(Msg::Shutdown);
            if let Some(jh) = w.jh.take() {
                let _ = jh.join();
            }
        }
    }
}

impl ModelExecutor for TpExecutor {
    fn forward_sample(&mut self, plan: &BatchPlan, generated: &mut Vec<Vec<u32>>, drafts: &mut Vec<Vec<u32>>) -> Result<()> {
        self.broadcast(plan, generated, drafts)?;
        let r0 = self.rank0.forward_sample(plan, generated, drafts);
        self.join()?;
        r0
    }
    fn supports_pair(&self) -> bool {
        false
    }
    fn supports_async(&self) -> bool {
        false
    }
    fn take_logprobs(&mut self, num_reqs: usize) -> Vec<Vec<TokenLogprobs>> {
        self.rank0.take_logprobs(num_reqs)
    }
}
