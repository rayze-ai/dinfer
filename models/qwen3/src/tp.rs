//! TP 多卡 executor：一进程多线程 lockstep（design §13.1）。
//!
//! rank 0 的 `Qwen3Executor` 在 engine 线程上跑；其它 rank 各一个线程持自己的 executor（自己的 Device / Stream /
//! NCCL Comm）。每步把 `BatchPlan` + 当前 generated/drafts clone 发给各 rank，所有 rank 各自 forward + 采样
//! （NCCL ring all-reduce 逐位一致 + Philox 同 seed ⇒ 同 token），只采用 rank 0 的结果。
//!
//! PD（design §13.7）：P / D 两个角色用的是同一套方法——每 rank 只持自己 KV 头分片，导出 / 导入时各 rank 对同一段
//! host 缓冲（rank 0 的 pinned 池）或同一个对端槽按 `[rank][block][layer]` 偏移各写 / 各读自己那份；payload 单元
//! = world × 单 rank 单元，P 与 D 必须同 tp（model_tag 带 tp，不等直接拒收）。cuda-ipc 区是 rank 0 那张卡的设备内存，
//! 其它 rank 无 P2P 读不到 → TP 下只支持 tcp / shm。
//! 不支持：host tier / 跨流 pair。

use anyhow::{bail, Context, Result};
use dinfer_core::kv::{BlockId, KvBlocks};
use dinfer_core::runner::{BatchPlan, ModelExecutor};
use dinfer_layers::{Event, PinnedBuf};
use std::sync::{mpsc, Arc};

use crate::executor::{PdBuf, Qwen3Executor};

struct Step {
    plan: BatchPlan,
    generated: Vec<Vec<u32>>,
    drafts: Vec<Vec<u32>>,
}

/// 在 rank 线程上对该 rank 的 executor 执行的任意调用（PD 导出 / 导入 / 回收等）；结果由闭包自己通过通道送回。
type Call = Box<dyn FnOnce(&mut Qwen3Executor) + Send>;

enum Msg {
    Step(Step),
    Call(Call),
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
                        loop {
                            match rx.recv() {
                                Ok(Msg::Step(mut st)) => {
                                    let r = ex.forward_sample(&st.plan, &mut st.generated, &mut st.drafts);
                                    if dtx.send(r).is_err() {
                                        break;
                                    }
                                }
                                Ok(Msg::Call(f)) => f(&mut ex),
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

    /// 给每个非 0 rank 发一个调用（`f(rank, exec)`），立刻返回各自的结果通道；`collect` 按序等。
    /// 与 `Step` 走同一条消息队列，所以 rank 线程上的执行顺序与 engine 调用顺序一致（导出排在上一步 forward 之后、下一步之前）。
    fn dispatch<T: Send + 'static>(
        &self,
        f: impl Fn(usize, &mut Qwen3Executor) -> Result<T> + Send + Sync + 'static,
    ) -> Result<Vec<mpsc::Receiver<Result<T>>>> {
        let f = Arc::new(f);
        let mut rxs = Vec::with_capacity(self.workers.len());
        for (i, w) in self.workers.iter().enumerate() {
            let (tx, rx) = mpsc::channel::<Result<T>>();
            let f = f.clone();
            let rank = i + 1;
            w.tx.send(Msg::Call(Box::new(move |ex| {
                let _ = tx.send(f(rank, ex));
            })))
            .map_err(|_| anyhow::anyhow!("tp rank {rank} gone"))?;
            rxs.push(rx);
        }
        Ok(rxs)
    }
    fn collect<T>(rxs: Vec<mpsc::Receiver<Result<T>>>) -> Result<Vec<T>> {
        rxs.into_iter()
            .enumerate()
            .map(|(i, rx)| rx.recv().map_err(|_| anyhow::anyhow!("tp rank {} died", i + 1))?.with_context(|| format!("tp rank {}", i + 1)))
            .collect()
    }
    /// 不等结果的广播（每步回收等）。
    fn cast(&self, f: impl Fn(usize, &mut Qwen3Executor) + Send + Sync + 'static) {
        let f = Arc::new(f);
        for (i, w) in self.workers.iter().enumerate() {
            let f = f.clone();
            let rank = i + 1;
            let _ = w.tx.send(Msg::Call(Box::new(move |ex| f(rank, ex))));
        }
    }

    /// PD：把 rank 0 当前的区（D 侧自己建的 / P 侧映射的对端区）交给其它 rank 共用。service 在 D 侧建区后调一次；
    /// P 侧 `pd_map_region` 内部调。
    pub fn share_region(&self) -> Result<()> {
        let r = self.rank0.pd_region();
        let rxs = self.dispatch(move |_, ex| ex.pd_set_region(r.clone()))?;
        Self::collect(rxs).map(|_| ())
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

/// TP 导出句柄：等所有 rank 的 copy stream 写完；payload 是 rank 0 的 pinned 缓冲（槽模式无字节）。
struct TpExport {
    events: Vec<Event>,
    buf: Option<PdBuf>,
    _region: Option<Arc<dinfer_kernels::SharedRegion>>,
}
impl dinfer_core::pd::KvExportHandle for TpExport {
    fn wait(&mut self) -> Result<&dyn dinfer_core::pd::HostBuf> {
        for e in &self.events {
            e.synchronize()?;
        }
        Ok(match &self.buf {
            Some(b) => b,
            None => &EMPTY,
        })
    }
}
static EMPTY: Vec<u8> = Vec::new();

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
    fn kv_fetch(&mut self, _o: &[(BlockId, u32)]) -> Result<()> {
        bail!("tp: host tier unsupported")
    }
    fn kv_offload(&mut self, _o: &[(BlockId, u32)]) -> Result<()> {
        bail!("tp: host tier unsupported")
    }

    // ---------------------------------------------------------------- PD（P / D 同一套，按 rank 分片）
    /// 一个块（全层、全 rank）= world × 单 rank 单元；0 = 未 enable_pd。
    fn kv_unit_bytes(&self) -> usize {
        if !self.rank0.pd_enabled() {
            return 0;
        }
        self.rank0.pd_unit_bytes_local() * self.world()
    }
    fn kv_host_pool(&self) -> Option<Arc<dyn dinfer_core::pd::HostBufPool>> {
        ModelExecutor::kv_host_pool(&self.rank0)
    }
    /// P 侧：rank r 把自己的分片写到 rank 0 pinned 缓冲的 `r × stride` 处（pinned 是 PORTABLE 的，别的 ctx 也能异步 D2H）。
    fn kv_export(&mut self, blocks: &KvBlocks) -> Result<Box<dyn dinfer_core::pd::KvExportHandle>> {
        let ids: Arc<Vec<BlockId>> = Arc::new(blocks.group(0).to_vec());
        let stride = ids.len() * self.rank0.pd_unit_bytes_local();
        let total = stride * self.world();
        let mut buf = self.rank0.pd_take_buf().context("pd: no pinned buffer (enable_pd not called, or pool exhausted → raise --pd-pool-bufs)")?;
        anyhow::ensure!(buf.capacity() >= total, "pd: export {total} B > pinned buf {}", buf.capacity());
        let base = buf.ptr() as u64;
        let ids2 = ids.clone();
        let rxs = self.dispatch(move |rank, ex| ex.pd_export_addr(&ids2, base + (rank * stride) as u64, true))?;
        let e0 = self.rank0.pd_export_addr(&ids, base, true)?;
        let mut events = vec![e0];
        events.extend(Self::collect(rxs)?);
        buf.set_len(total)?;
        Ok(Box::new(TpExport { events, buf: Some(PdBuf(buf)), _region: None }))
    }
    /// P 侧槽路径：各 rank 写对端槽的 `r × stride` 处（shm：host 地址对所有 ctx 可见）。
    fn kv_export_slot(&mut self, blocks: &KvBlocks, slot: u32) -> Result<Box<dyn dinfer_core::pd::KvExportHandle>> {
        let r = self.rank0.pd_region().context("pd: kv_export_slot without mapped region")?;
        let ids: Arc<Vec<BlockId>> = Arc::new(blocks.group(0).to_vec());
        let stride = ids.len() * self.rank0.pd_unit_bytes_local();
        let total = stride * self.world();
        anyhow::ensure!(total <= r.slot_bytes(), "pd: export {total} B > slot {}", r.slot_bytes());
        let dst = r.slot_addr(slot as usize)?;
        let is_host = r.is_host();
        let ids2 = ids.clone();
        let rxs = self.dispatch(move |rank, ex| ex.pd_export_addr(&ids2, dst + (rank * stride) as u64, is_host))?;
        let e0 = self.rank0.pd_export_addr(&ids, dst, is_host)?;
        let mut events = vec![e0];
        events.extend(Self::collect(rxs)?);
        Ok(Box::new(TpExport { events, buf: None, _region: Some(r) }))
    }
    /// D 侧：payload（pinned 缓冲 / 本地 shm 槽）按 rank 切片，各 rank 读自己那份；源由 Arc 保活到最后一个 rank 读完。
    fn kv_import(&mut self, blocks: &KvBlocks, payload: Box<dyn dinfer_core::pd::HostBuf>) -> Result<()> {
        anyhow::ensure!(self.rank0.pd_enabled(), "pd: kv_import without resources (call enable_pd)");
        self.rank0.pd_gc_inflight();
        let ids: Arc<Vec<BlockId>> = Arc::new(blocks.group(0).to_vec());
        let stride = ids.len() * self.rank0.pd_unit_bytes_local();
        let total = stride * self.world();
        anyhow::ensure!(payload.len() == total, "pd: payload {} != {} blocks × {} ranks × {}", payload.len(), ids.len(), self.world(), stride / ids.len().max(1));
        let (src, is_host, keep): (u64, bool, Arc<dyn std::any::Any + Send + Sync>) = if let Some(slot) = payload.slot() {
            let r = self.rank0.pd_region().context("pd: slot payload but no local region")?;
            anyhow::ensure!(total <= r.slot_bytes(), "pd: import {total} B > slot {}", r.slot_bytes());
            (r.slot_addr(slot as usize)?, r.is_host(), Arc::from(payload.into_any()))
        } else {
            let pb: PinnedBuf = if payload.is_pinned() {
                match payload.into_any().downcast::<PdBuf>() {
                    Ok(b) => b.0,
                    Err(_) => bail!("pd: pinned payload is not this executor's PinnedBuf"),
                }
            } else {
                let mut b = self.rank0.pd_take_buf().context("pd: pinned pool exhausted for import staging")?;
                anyhow::ensure!(b.capacity() >= total, "pd: import {total} B > pinned buf {}", b.capacity());
                b.as_mut_cap()[..total].copy_from_slice(payload.as_slice());
                b.set_len(total)?;
                b
            };
            (pb.ptr() as u64, true, Arc::new(pb) as Arc<dyn std::any::Any + Send + Sync>)
        };
        let (ids2, keep2) = (ids.clone(), keep.clone());
        let rxs = self.dispatch(move |rank, ex| {
            ex.pd_gc_inflight();
            let ev = ex.pd_import_addr(&ids2, src + (rank * stride) as u64, is_host)?;
            ex.pd_push_inflight(ev, Box::new(keep2.clone()));
            Ok(())
        })?;
        let e0 = self.rank0.pd_import_addr(&ids, src, is_host)?;
        self.rank0.pd_push_inflight(e0, Box::new(keep));
        Self::collect(rxs).map(|_| ())
    }
    fn pd_map_region(&mut self, info: &dinfer_core::pd::RegionInfo) -> Result<()> {
        anyhow::ensure!(info.kind != "cuda-ipc", "pd: cuda-ipc region is one GPU's device memory; TP ranks cannot read it (use --kv-transport shm / tcp)");
        ModelExecutor::pd_map_region(&mut self.rank0, info)?;
        self.share_region()
    }
    fn pd_poll(&mut self) {
        self.rank0.pd_gc_inflight();
        self.cast(|_, ex| ex.pd_gc_inflight());
    }
}
