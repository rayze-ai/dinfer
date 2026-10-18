//! TP 多卡 executor：一进程多线程 lockstep（design §13.1）。
//!
//! rank 0 的 `Qwen3Executor` 在 engine 线程上跑；其它 rank 各一个线程持自己的 executor（自己的 Device / Stream /
//! NCCL Comm）。每步把 `BatchPlan` + 当前 generated/drafts clone 发给各 rank，所有 rank 各自 forward + 采样
//! （NCCL ring all-reduce 逐位一致 + Philox 同 seed ⇒ 同 token），只采用 rank 0 的结果。
//! 不支持：PD / host tier / 跨流 pair（v1 均 bail）。

use anyhow::{bail, Context, Result};
use dinfer_core::runner::{BatchPlan, ModelExecutor};
use std::sync::mpsc;

use crate::executor::Qwen3Executor;

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
    rank0: Qwen3Executor,
    workers: Vec<Worker>,
}

impl TpExecutor {
    /// `others[i]` 是 rank i+1 的 executor（已 `init_tp`）。调用方保证各 executor 是在各自线程上创建的；
    /// 这里把它们 move 进各自的工作线程。注意：executor 持有的 CUDA 资源只在创建线程的 ctx 下用过，
    /// move 到新线程后 cudarc 会按 Arc<CudaContext> 重新 bind，是允许的。
    pub fn new(rank0: Qwen3Executor, others: Vec<Qwen3Executor>) -> Self {
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
                        while let Ok(Msg::Step(mut st)) = rx.recv() {
                            let r = ex.forward_sample(&st.plan, &mut st.generated, &mut st.drafts);
                            if dtx.send(r).is_err() {
                                break;
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
    fn take_logprobs(&mut self, num_reqs: usize) -> Vec<Vec<dinfer_core::sampler::TokenLogprobs>> {
        self.rank0.take_logprobs(num_reqs)
    }
    fn kv_unit_bytes(&self) -> usize {
        self.rank0.kv_unit_bytes()
    }
    fn kv_fetch(&mut self, _o: &[(dinfer_core::kv::BlockId, u32)]) -> Result<()> {
        bail!("tp: host tier unsupported")
    }
    fn kv_offload(&mut self, _o: &[(dinfer_core::kv::BlockId, u32)]) -> Result<()> {
        bail!("tp: host tier unsupported")
    }
}
