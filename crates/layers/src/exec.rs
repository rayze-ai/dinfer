//! executor 公共件（design.md "executor 公共件抽取"，2026-10-20）：调度相关的胶水不再每个模型 executor 实现一遍。
//!
//! - [`AsyncResultRing`]：异步调度的两段式结果回传——`enqueue` 在流上排异步 D2H 到 pinned 槽并记事件（不等 GPU），
//!   `wait` 等事件后读；N 槽轮转 + 在飞队列（FIFO）。
//! - [`DecodeGraphs`]：decode 步 CUDA graph 的桶表（`graph_buckets`）、懒捕获（`ensure`）、启动全捕获（`capture_all`，
//!   `GRAPH_CAPTURE=eager`）。捕获体（上传假 plan + forward + 采样）由模型 executor 以闭包传入，这里只管
//!   "预热一次 → 同步 → begin_capture → 再跑一次 → end_capture"。
//! - [`StateSlotTable`]：线性注意力 / 递推状态（GDN / KDA）按 `RequestId` 分 slot，显式 `release` + LRU 兜底。
//! - [`PlanBuffers`]：三个 executor 共用的 `BatchPlan` 固定地址缓冲（ids / pos / slot_mapping / logits_rows / cu_seqlens /
//!   seq_lens / block_table）+ 异步调度的 `gather_idx` / `prev_chosen`，`upload` 把 plan 补到桶大小写进去；模型特有列
//!   （qwen35 的 state_idx / has_init）由调用方在 `upload` 之后、`flush` 之前自己 push。
use crate::tensor::Tensor;
use anyhow::{ensure, Context, Result};
use dinfer_core::runner::{BatchMode, BatchPlan};
use dinfer_kernels as k;
use dinfer_kernels::cudarc::driver::CudaSlice;
use dinfer_kernels::{Graph, PinnedSlot, PinnedUploader, Stream};
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------- AsyncResultRing

/// `AsyncResultRing::enqueue` 的凭据：`wait` 时按 FIFO 顺序归还。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ticket {
    slot: usize,
    n: usize,
}

/// N 个 pinned 槽轮转（步 i 的 D2H 在槽 i % N）：`wait(step i)` 必在 `enqueue(step i + N)` 之前（engine 先 begin n+1
/// 再 finish n，所以 N = 2 即可）。
pub struct AsyncResultRing {
    slots: Vec<PinnedSlot>,
    next: usize,
    inflight: VecDeque<Ticket>,
}

impl AsyncResultRing {
    /// `depth` 个槽，每槽最多 `rows` 个 i32。
    pub fn new(stream: &Stream, depth: usize, rows: usize) -> Result<Self> {
        ensure!(depth >= 1, "AsyncResultRing: depth 0");
        let slots = (0..depth).map(|_| PinnedSlot::new(stream, rows.max(1))).collect::<Result<Vec<_>>>()?;
        Ok(Self { slots, next: 0, inflight: VecDeque::with_capacity(depth) })
    }

    pub fn depth(&self) -> usize {
        self.slots.len()
    }

    /// 在飞的步数。
    pub fn inflight(&self) -> usize {
        self.inflight.len()
    }

    /// 在 `stream` 上排 `chosen[..n]` 的异步 D2H（不同步），返回凭据。在飞满（= depth）时报错。
    pub fn enqueue(&mut self, stream: &Stream, chosen: &CudaSlice<i32>, n: usize) -> Result<Ticket> {
        ensure!(self.inflight.len() < self.slots.len(), "AsyncResultRing: {} async steps already in flight", self.slots.len());
        let slot = self.next;
        self.slots[slot].enqueue_d2h(stream, chosen, n)?;
        self.next = (self.next + 1) % self.slots.len();
        let t = Ticket { slot, n };
        self.inflight.push_back(t);
        Ok(t)
    }

    /// 等最早在飞的一步（必须等于 `ticket`）完成，返回其 n 个值。
    pub fn wait(&mut self, ticket: Ticket) -> Result<Vec<i32>> {
        let head = self.inflight.pop_front().context("AsyncResultRing::wait without enqueue")?;
        ensure!(head == ticket, "AsyncResultRing: wait out of order ({ticket:?} vs head {head:?})");
        let v = self.slots[ticket.slot].wait_read()?;
        ensure!(v.len() == ticket.n, "AsyncResultRing: {} values for {} rows", v.len(), ticket.n);
        Ok(v)
    }
}

/// 异步调度总开关：`ASYNC_SCHED=0` 关，默认开。
pub fn async_sched_enabled() -> bool {
    std::env::var("ASYNC_SCHED").map(|v| v != "0").unwrap_or(true)
}

// ---------------------------------------------------------------- DecodeGraphs

/// CUDA graph 的 batch 桶（vLLM 默认 1,2,4,8 再按 8 递增到上限；这里 2 的幂 + 上限，够用且捕获快）。
pub fn graph_buckets(max_rows: usize) -> Vec<usize> {
    let mut v: Vec<usize> = [1usize, 2, 4, 8, 16, 32, 64, 128, 256].into_iter().filter(|&b| b < max_rows).collect();
    v.push(max_rows);
    v
}

/// `GRAPH_CAPTURE=eager`：启动时一次捕获全部桶（默认懒捕获：首次用到某桶时捕获，~30 ms/桶）。
pub fn graph_capture_eager() -> bool {
    std::env::var("GRAPH_CAPTURE").is_ok_and(|v| v == "eager")
}

struct DecodeGraph {
    batch: usize,
    graph: Graph,
}

/// 捕获闭包的两个阶段：`Prepare` 在 capture 外（上传假 plan / 采样参数——H2D 不能进图：图会记下 host 指针，
/// 重放时读已释放的 Vec），`Body` 进图（forward + 采样），预热一次再在 capture 内跑一次。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapturePhase {
    Prepare,
    Body,
}

/// decode 图表：按桶懒捕获，`get(n)` 取 ≥ n 行的最小桶图。
pub struct DecodeGraphs {
    graphs: Vec<DecodeGraph>,
    enabled: bool,
    max_rows: usize,
    /// 捕获用的流（图在哪条流上捕获就只能在哪条流上 launch；`Stream` 是 Arc 句柄，clone 便宜）。
    stream: Stream,
    /// 累计捕获耗时（秒；eager 全捕获 + lazy 首捕获都计），给 `StartupTimer::done` 的 graph_capture 段。
    capture_secs: f64,
    /// `capture_all` 是否已跑过（`report_startup` 据此判 eager / lazy）。
    captured_all: bool,
}

impl DecodeGraphs {
    pub fn new(stream: &Stream, enabled: bool, max_rows: usize) -> Self {
        Self { graphs: Vec::new(), enabled, max_rows, stream: stream.clone(), capture_secs: 0.0, captured_all: false }
    }

    /// 迄今捕获总耗时（秒）。
    pub fn capture_secs(&self) -> f64 {
        self.capture_secs
    }

    /// 启动期一行汇总（四模型统一格式，在 `Executor::new` 末尾调一次）：
    /// - 关：`decode graphs disabled reason=…`（`reason` 由模型 executor 给，None = 命令行 / NO_GRAPH 关）；
    /// - eager（已 `capture_all`）：`decode graphs captured buckets=[…] n=… secs=… mode=eager`；
    /// - lazy：`decode graphs captured buckets=[…] n=0 secs=0 mode=lazy`，之后每桶首捕获各打一行 `decode graph captured`。
    pub fn report_startup(&self, disabled_reason: Option<&str>) {
        if !self.enabled {
            tracing::info!(reason = disabled_reason.unwrap_or("--no-cuda-graph / NO_GRAPH"), "decode graphs disabled");
            return;
        }
        tracing::info!(
            buckets = ?self.buckets(),
            n = self.graphs.len(),
            secs = format!("{:.2}", self.capture_secs),
            mode = if self.captured_all { "eager" } else { "lazy" },
            "decode graphs captured"
        );
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }
    pub fn set_enabled(&mut self, on: bool) {
        self.enabled = on;
    }
    pub fn len(&self) -> usize {
        self.graphs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.graphs.is_empty()
    }
    pub fn buckets(&self) -> Vec<usize> {
        graph_buckets(self.max_rows)
    }

    /// n 行所属的桶（≥ n 的最小桶）。
    pub fn bucket_for(&self, n: usize) -> Option<usize> {
        self.buckets().into_iter().find(|&b| b >= n)
    }

    /// 已捕获、能放下 n 行的最小桶图：(桶大小, 图)。
    pub fn get(&self, n: usize) -> Option<(usize, &Graph)> {
        self.graphs.iter().find(|g| g.batch >= n).map(|g| (g.batch, &g.graph))
    }

    /// 懒捕获：启用且 n 行没有可用图时捕获其桶。`body(nb, phase)`：`Prepare` 一次（capture 外），`Body` 两次
    ///（预热：cuBLASLt 启发式 / 模块加载在 capture 外完成；再一次在 capture 内）。
    pub fn ensure(&mut self, n: usize, body: impl FnMut(usize, CapturePhase) -> Result<()>) -> Result<()> {
        if !self.enabled || self.get(n).is_some() {
            return Ok(());
        }
        let Some(nb) = self.bucket_for(n) else { return Ok(()) };
        self.capture(nb, body)
    }

    /// 启动时一次捕获全部桶。
    pub fn capture_all(&mut self, mut body: impl FnMut(usize, CapturePhase) -> Result<()>) -> Result<()> {
        for nb in self.buckets() {
            if self.graphs.iter().any(|g| g.batch == nb) {
                continue;
            }
            self.capture_inner(nb, &mut body, false)?;
        }
        self.captured_all = true;
        Ok(())
    }

    /// 捕获一个桶（lazy 首捕获入口，打一行 `decode graph captured bucket=… secs=…`）：Prepare → Body 预热 → 同步 → capture(Body)。
    pub fn capture(&mut self, nb: usize, body: impl FnMut(usize, CapturePhase) -> Result<()>) -> Result<()> {
        self.capture_inner(nb, body, true)
    }

    fn capture_inner(&mut self, nb: usize, mut body: impl FnMut(usize, CapturePhase) -> Result<()>, log: bool) -> Result<()> {
        let t0 = std::time::Instant::now();
        let stream = self.stream.clone();
        body(nb, CapturePhase::Prepare).context("graph capture: prepare")?;
        body(nb, CapturePhase::Body).context("graph capture: warmup")?;
        stream.synchronize()?;
        stream.begin_capture()?;
        let r = body(nb, CapturePhase::Body).context("graph capture: body");
        let g = stream.end_capture();
        r?;
        let graph = g.with_context(|| format!("capture decode graph for batch {nb}"))?;
        self.graphs.push(DecodeGraph { batch: nb, graph });
        self.graphs.sort_by_key(|g| g.batch);
        stream.synchronize()?;
        let secs = t0.elapsed().as_secs_f64();
        self.capture_secs += secs;
        if log {
            tracing::info!(bucket = nb, secs = format!("{secs:.3}"), "decode graph captured");
        }
        Ok(())
    }
}

/// 捕获用的假 decode plan（nb 行：token 0、pos 0、slot=-1 不写 cache、seq_len=1、block 0、温度 0）。
#[allow(clippy::field_reassign_with_default)]
pub fn decode_capture_plan(nb: usize) -> BatchPlan {
    let mut p = BatchPlan::default();
    p.req_ids = (0..nb as u64).collect();
    p.num_tokens = vec![1; nb];
    p.input_ids = vec![0; nb];
    p.positions = vec![0; nb];
    p.slot_mapping = vec![-1; nb];
    p.cu_seqlens = (0..=nb as i32).collect();
    p.seq_lens = vec![1; nb];
    p.block_table = vec![0; nb];
    p.max_blocks = 1;
    p.logits_rows = (0..nb as i32).collect();
    p.logits_row_start = (0..=nb).collect();
    p.spec_tokens = vec![Vec::new(); nb];
    p.next_known = vec![-1; nb];
    p.mode = BatchMode::Decode;
    p.sample.clear();
    for _ in 0..nb {
        p.sample.push(&dinfer_core::sample::SamplingParams { temperature: 0.0, ..Default::default() }, 0, &[0], 1);
    }
    p
}

// ---------------------------------------------------------------- StateSlotTable

/// 请求 → 状态 slot 分配表（GDN / KDA 递推状态）。`RequestId` 首次出现分一个 slot（调用方负责清零该 slot）；
/// slot 不够时按"本步不在场且最久未见"的顺序回收。
pub struct StateSlotTable {
    slots: HashMap<u64, usize>,
    last_seen: HashMap<u64, u64>,
    free: Vec<usize>,
    cap: usize,
    step: u64,
}

impl StateSlotTable {
    pub fn new(cap: usize) -> Self {
        Self { slots: HashMap::new(), last_seen: HashMap::new(), free: (0..cap).rev().collect(), cap, step: 0 }
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    /// 请求结束（finish / abort）：归还 slot。不调用它也能工作（LRU 回收兜底），但兜底只能淘汰"本步不在场"的请求——
    /// 分块 prefill / 被抢占的活请求也可能不在场，会被误当受害者（2026-10-19 B300 c8-long bench 87 条 slot 复用告警）。
    pub fn release(&mut self, req_ids: &[u64]) {
        for r in req_ids {
            if let Some(s) = self.slots.remove(r) {
                self.last_seen.remove(r);
                self.free.push(s);
            }
        }
    }

    /// 开始一步：返回本步新分配的 slot（调用方需清零）。
    pub fn begin_step(&mut self, req_ids: &[u64]) -> Result<Vec<usize>> {
        self.step += 1;
        let mut fresh = Vec::new();
        for &r in req_ids {
            if self.slots.contains_key(&r) {
                self.last_seen.insert(r, self.step);
                continue;
            }
            let s = match self.free.pop() {
                Some(s) => s,
                None => {
                    // 回收：本步不在场、且最久未见的请求
                    let victim = self
                        .last_seen
                        .iter()
                        .filter(|(id, _)| !req_ids.contains(id))
                        .min_by_key(|(_, &st)| st)
                        .map(|(id, _)| *id);
                    match victim {
                        Some(v) => {
                            let s = self.slots.remove(&v).expect("victim slot");
                            self.last_seen.remove(&v);
                            tracing::debug!(req = v, slot = s, "state slot 回收");
                            s
                        }
                        None => anyhow::bail!(
                            "state slot 用尽（cap {}，本步 {} 请求）；提高 max_num_seqs 预算",
                            self.cap,
                            req_ids.len()
                        ),
                    }
                }
            };
            self.slots.insert(r, s);
            self.last_seen.insert(r, self.step);
            fresh.push(s);
        }
        Ok(fresh)
    }

    /// 本步每请求的 slot（按 `req_ids` 顺序）。
    pub fn indices(&self, req_ids: &[u64]) -> Vec<i32> {
        req_ids.iter().map(|r| *self.slots.get(r).expect("slot assigned in begin_step") as i32).collect()
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

// ---------------------------------------------------------------- PlanBuffers

/// `BatchPlan` 的固定地址设备缓冲（图内 kernel 只读这些地址）。
pub struct PlanBuffers {
    pub ids: Tensor<i32>,   // [max_tokens]
    pub pos: Tensor<i32>,   // [max_tokens]
    pub slots: Tensor<i64>, // [max_tokens]
    pub rows: Tensor<i32>,  // [max_rows]
    pub cu: Tensor<i32>,    // [max_rows + 1]
    pub seq: Tensor<i32>,   // [max_rows]
    pub bt: Tensor<i32>,    // [max_rows * max_blocks]
    pub max_blocks: usize,
    /// 异步调度：`ids[i] = prev_chosen[gather_idx[i]]`（gather_idx < 0 的行用 host 上传的值）。
    pub gather_idx: Tensor<i32>,  // [max_tokens]
    pub prev_chosen: Tensor<i32>, // [max_rows]
}

impl PlanBuffers {
    pub fn alloc(s: &Stream, max_tokens: usize, max_rows: usize, max_blocks: usize) -> Result<Self> {
        let mt = max_tokens.max(1);
        let rows = max_rows.max(1);
        let z32 = |n: usize| Tensor::<i32>::zeros(s, &[n.max(1)]);
        Ok(Self {
            ids: z32(mt)?,
            pos: z32(mt)?,
            slots: Tensor::<i64>::zeros(s, &[mt])?,
            rows: z32(rows)?,
            cu: z32(rows + 1)?,
            seq: z32(rows)?,
            bt: z32(rows * max_blocks.max(1))?,
            max_blocks: max_blocks.max(1),
            gather_idx: Tensor::<i32>::from_vec(s, vec![-1; mt], &[mt])?,
            prev_chosen: z32(rows)?,
        })
    }

    /// 每 step 小输入打包上传器的建议容量（字节）。
    pub fn uploader_bytes(max_tokens: usize, max_rows: usize, max_blocks: usize, extra: usize) -> usize {
        (max_tokens * 16 + max_rows * 128 + max_rows * max_blocks * 4 + extra + (1 << 20)).max(1 << 20)
    }

    /// 把 plan 写进缓冲（通过 `up` 打包；调用方之后 push 自己的列再 `up.flush`）。batch 维补到 `pad_to`：
    /// padding 行 token 0、pos 0、slot=-1（不写 cache）、seq_len=1、block 0、各占 1 行 logits；非 decode 时 `pad_to == n`。
    /// 返回 (token 数含 padding, 请求数含 padding, logits 行数含 padding)。
    pub fn upload(&mut self, s: &Stream, up: &mut PinnedUploader, plan: &BatchPlan, pad_to: usize) -> Result<(usize, usize, usize)> {
        up.reset_step(); // 每 step 的第一次打包：host 区回卷
        let n = plan.num_reqs();
        let t = plan.total_tokens();
        let nb = pad_to.max(n);
        let pad = nb - n;
        let tt = t + pad; // decode 时每请求 1 token；非 decode 时 pad == 0
        let mut ids = plan.input_ids.clone();
        let mut pos = plan.positions.clone();
        let mut slots = plan.slot_mapping.clone();
        ids.resize(tt, 0);
        pos.resize(tt, 0);
        slots.resize(tt, -1);
        up.push(s, &ids, &mut self.ids.data)?;
        // 异步调度：在飞行的 token 行由设备 gather；非异步 plan 的 gather_idx 全 -1（图内 kernel 不改 ids）
        let mut gidx = plan.gather_idx.clone();
        gidx.resize(tt, -1);
        up.push(s, &gidx, &mut self.gather_idx.data)?;
        up.push(s, &pos, &mut self.pos.data)?;
        up.push(s, &slots, &mut self.slots.data)?;
        // logits 行维（带 draft 的请求多于 1 行；padding 请求各 1 行）
        let mut rows = plan.logits_rows.clone();
        for i in 0..pad {
            rows.push((t + i) as i32);
        }
        let nrows = rows.len();
        up.push(s, &rows, &mut self.rows.data)?;
        let mut cu = plan.cu_seqlens.clone();
        for i in n..nb {
            cu.push(cu[i] + 1);
        }
        up.push(s, &cu, &mut self.cu.data)?;
        let mut seq = plan.seq_lens.clone();
        seq.resize(nb, 1);
        up.push(s, &seq, &mut self.seq.data)?;
        // block_table：图里 max_blocks 固定为缓冲宽度（按每行 stride = max_blocks 布局）
        ensure!(
            plan.max_blocks <= self.max_blocks,
            "block_table wider than buffer: {} > {} (max_model_len / block_size)",
            plan.max_blocks,
            self.max_blocks
        );
        let mb = self.max_blocks;
        let mut bt = vec![0i32; nb * mb];
        for r in 0..n {
            bt[r * mb..r * mb + plan.max_blocks].copy_from_slice(&plan.block_table[r * plan.max_blocks..(r + 1) * plan.max_blocks]);
        }
        up.push(s, &bt, &mut self.bt.data)?;
        Ok((tt, nb, nrows))
    }

    /// 异步调度：forward 开头把在飞行的 token 行从上一步采样结果 gather 进 ids（图内；非异步步全 -1 等于空操作）。
    pub fn gather_inflight(&mut self, s: &Stream, t: usize) -> Result<()> {
        k::row_gather_i32(s, &self.prev_chosen.data, &self.gather_idx.data, &mut self.ids.data, t)
    }

    /// 采样后把 `chosen[..n]` 存到 prev_chosen 供下一步 gather（图内；非异步时无害）。
    pub fn save_chosen(&mut self, s: &Stream, chosen: &CudaSlice<i32>, n: usize) -> Result<()> {
        s.dtod_copy_n(chosen, &mut self.prev_chosen.data, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_table_allocates_and_recycles() {
        let mut t = StateSlotTable::new(2);
        assert_eq!(t.begin_step(&[1, 2]).unwrap(), vec![0, 1]);
        assert_eq!(t.indices(&[2, 1]), vec![1, 0]);
        // 2 号不在场 → 被回收给 3（复用 2 的 slot 1）
        assert_eq!(t.begin_step(&[1, 3]).unwrap(), vec![1]);
        assert_eq!(t.indices(&[3, 1]), vec![1, 0]);
        // 两个都在场且要新 slot → 报错
        assert!(t.begin_step(&[1, 3, 4]).is_err());
        t.release(&[1]);
        assert_eq!(t.begin_step(&[3, 4]).unwrap(), vec![0]);
    }

    #[test]
    fn buckets_cover_max_rows() {
        assert_eq!(graph_buckets(64), vec![1, 2, 4, 8, 16, 32, 64]);
        assert_eq!(graph_buckets(100), vec![1, 2, 4, 8, 16, 32, 64, 100]);
        assert_eq!(graph_buckets(1), vec![1]);
        let bucket_for = |n: usize| graph_buckets(100).into_iter().find(|&b| b >= n);
        assert_eq!(bucket_for(5), Some(8));
        assert_eq!(bucket_for(100), Some(100));
        assert_eq!(bucket_for(101), None);
    }

    #[test]
    fn capture_plan_shape() {
        let p = decode_capture_plan(4);
        assert_eq!(p.num_reqs(), 4);
        assert_eq!(p.total_tokens(), 4);
        assert_eq!(p.sample.num_rows(), 4);
        assert!(p.async_eligible());
    }
}
