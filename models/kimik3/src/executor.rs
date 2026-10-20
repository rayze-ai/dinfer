//! `KimiK3Model` 的 `ModelExecutor`：上传 BatchPlan → forward → 采样 → D2H（里程碑 C-4b）。
//!
//! 全部用 `dinfer_layers::exec` 的四个公共件（不复制 qwen3 / qwen35 的内联实现）：
//! - `PlanBuffers`：plan 列固定地址上传 + 按桶 pad；KDA 特有的 `state_idx` / `has_init` 列在公共列之后追加；
//! - `StateSlotTable`：KDA 递推状态槽（请求 → slot），`finished_req_ids` 归还、LRU 兜底；
//! - `DecodeGraphs`：纯 decode 步按 batch 桶懒捕获 CUDA graph（`Prepare` 在图外做 H2D，`Body` 只 launch）；
//! - `AsyncResultRing`：`begin` / `finish` 两段式，让 runner 的异步调度真正重叠（`ASYNC_SCHED=0` 关）。
//!
//! K3 特有：
//! - 两种 attention 的 KV：KDA 层 = `StateSlotTable` 槽（conv + 递推状态，模型内 `KdaState`），MLA 层 = paged latent
//!   （576 维 / token，`slot_mapping` / `block_table` 走公共 plan 列，与 GLM 的 MLA 同口径）。
//! - prefill：KDA 走 chunk（`KdaChunkWorkspace`，全模型共用一份，按步 `prepare`）、MLA 走展开 K/V 的 causal attention；
//!   decode：KDA 递归、MLA 吸收路径读 paged latent。两条路径都在 `kda.rs` / `mla.rs`，executor 只搭 `StepMeta`。
//! - AttnRes 块残差在 `model.rs` 内按层顺序维护，executor 只保证一步一次 `forward_rows`。
//! - 图捕获用的假 plan 与 padding 行：KDA slot 指**废纸篓 slot**（`num_slots - 1`，装载时多分的那一个），MLA 的
//!   `slot_mapping = -1` 不写 cache、block 0 / seq_len 1；不会碰活请求。
//! - 采样流水同 Qwen3 / Qwen3.5（penalties → temperature → min_p → top_k/top_p → 采样 → logprobs）；guided mask 未接。
use crate::kda::KdaChunkWorkspace;
use crate::kda::KdaMeta;
use crate::layer::StepMeta;
use crate::mla::MlaMeta;
use crate::model::KimiK3Model;
use anyhow::{ensure, Context, Result};
use dinfer_core::runner::{BatchMode, BatchPlan, ModelExecutor};
use dinfer_core::sampler::{SampleBatch, TokenLogprobs};
use dinfer_kernels::ops as k;
use dinfer_layers::exec::{
    async_sched_enabled, decode_capture_plan, graph_capture_eager, AsyncResultRing, CapturePhase, DecodeGraphs,
    PlanBuffers, StateSlotTable, Ticket,
};
use dinfer_layers::{Gemm, PinnedUploader, Stream, Tensor};
use std::cell::RefCell;

pub use dinfer_layers::exec::graph_buckets;

/// runner 同步兜底时暂存的 (generated, drafts)。
type SyncResult = (Vec<Vec<u32>>, Vec<Vec<u32>>);

/// 固定地址的每 step 输入缓冲。
struct StepBuffers {
    /// plan 列（ids / pos / slot_mapping / rows / cu / seq / block_table + 异步 gather）：公共件。
    plan: PlanBuffers,
    /// KDA：每请求的 slot 与"是否有历史状态"（模型特有列，公共列之后追加）。
    state_idx: Tensor<i32>, // [max_rows]
    has_init: Tensor<i32>,  // [max_rows]
    // 采样列
    temperature: Tensor<f32>,
    temp01: Tensor<f32>,
    top_k: Tensor<i32>,
    top_p: Tensor<f32>,
    min_p: Tensor<f32>,
    rep: Tensor<f32>,
    freq: Tensor<f32>,
    pres: Tensor<f32>,
    seeds: Tensor<u64>,
    offsets: Tensor<u64>,
    hist_offsets: Tensor<i32>,
    hist_prompt_len: Tensor<i32>,
    hist_ids: Tensor<i32>,
    chosen: Tensor<i32>, // [max_rows]
    lp_ids: Tensor<i32>,
    lp_val: Tensor<f32>,
    lp_rank: Tensor<i32>,
    up: PinnedUploader,
}

impl StepBuffers {
    fn alloc(s: &Stream, mt: usize, rows: usize, max_blocks: usize) -> Result<Self> {
        let z32 = |n: usize| Tensor::<i32>::zeros(s, &[n.max(1)]);
        let zf = |n: usize| Tensor::<f32>::zeros(s, &[n.max(1)]);
        Ok(Self {
            plan: PlanBuffers::alloc(s, mt, rows, max_blocks)?,
            state_idx: z32(rows)?,
            has_init: z32(rows)?,
            temperature: zf(rows)?,
            temp01: zf(rows)?,
            top_k: z32(rows)?,
            top_p: zf(rows)?,
            min_p: zf(rows)?,
            rep: zf(rows)?,
            freq: zf(rows)?,
            pres: zf(rows)?,
            seeds: Tensor::<u64>::zeros(s, &[rows.max(1)])?,
            offsets: Tensor::<u64>::zeros(s, &[rows.max(1)])?,
            hist_offsets: z32(rows + 1)?,
            hist_prompt_len: z32(rows)?,
            hist_ids: z32(1024)?,
            chosen: z32(rows)?,
            lp_ids: z32(rows * 33)?,
            lp_val: zf(rows * 33)?,
            lp_rank: z32(rows)?,
            up: PinnedUploader::new(s, PlanBuffers::uploader_bytes(mt, rows, max_blocks, rows * 8 + (1 << 20)))?,
        })
    }
}

pub struct KimiK3Executor {
    pub stream: Stream,
    pub gemm: Gemm,
    pub model: KimiK3Model,
    /// KDA 状态槽表：容量 = `model.num_slots() - 1`（最后一个是废纸篓）。
    pub slots: StateSlotTable,
    logits: Tensor<f32>,
    work: Tensor<f32>,
    raw: Tensor<f32>,
    buf: StepBuffers,
    last_logprobs: Vec<Vec<TokenLogprobs>>,
    pub steps: u64,
    /// KDA chunk prefill 的共享工作区（全部 KDA 层共用；按步 `prepare`，按需增长）。`KDA_PREFILL=recurrent` 关。
    kda_chunk: Option<KdaChunkWorkspace>,
    kda_chunk_enabled: bool,
    graphs: DecodeGraphs,
    pub graph_steps: u64,
    pub eager_steps: u64,
    async_ring: AsyncResultRing,
    async_inflight: std::collections::VecDeque<Ticket>,
    skip_prev_chosen: bool,
    sync_stash: Option<SyncResult>,
    /// 日志用
    pub trace: bool,
}

impl KimiK3Executor {
    /// `max_blocks`：每请求 block_table 的最大列数（`max_model_len / block_size`）。`enable_graphs`：decode 步走 CUDA graph
    ///（`NO_GRAPH` 环境变量强制关；`GRAPH_CAPTURE=eager` 启动时全捕获）。
    pub fn new(stream: Stream, gemm: Gemm, model: KimiK3Model, max_blocks: usize, enable_graphs: bool) -> Result<Self> {
        ensure!(model.num_slots() >= 2, "kimik3 executor: num_slots 至少 2（1 个活请求 + 1 个废纸篓）");
        // 同 Qwen3 / Qwen3.5：capture 要求非默认流；关掉 cudarc 的跨流事件跟踪；cuBLASLt 句柄在新流上重建。
        let _ = gemm;
        stream.synchronize()?;
        let dev = stream.device();
        dev.disable_event_tracking();
        let stream = dev.new_stream()?;
        let gemm = Gemm::new(&stream).context("cublasLt on executor stream")?;
        tracing::info!("gemm autotune: off（kimik3 尚未提供 gemm_tune_entries；C-5 接）");
        let (mt, rows, v) = (model.max_tokens(), model.max_rows(), model.vocab());
        let slots = StateSlotTable::new(model.num_slots() - 1);
        let buf = StepBuffers::alloc(&stream, mt, rows, max_blocks)?;
        // decode 图：K3 的 LatentMoe（每步 D2H 读 offsets 展平 tile 计划）与 AttnRes（host softmax）体内有同步 D2H，
        // 不能进 capture（B300 冒烟第 1 轮：`graph capture: body: LatentMoe: pairs 0 ≠ t·K`）。默认关，`K3_GRAPH=1` 强制开
        //（接 GPU 侧 tile 计划 + AttnRes 融合 kernel 后翻回默认开；见 docs/loops/k3-serve.md TODO）。`NO_GRAPH` 仍强制关。
        let force_on = std::env::var("K3_GRAPH").is_ok_and(|v| v == "1");
        let graphs_on = enable_graphs && force_on && std::env::var("NO_GRAPH").is_err();
        if enable_graphs && !force_on {
            tracing::info!("kimik3: decode CUDA graph 默认关（LatentMoe / AttnRes 体内有 D2H，待 GPU 侧 tile 计划）；K3_GRAPH=1 强制开");
        }
        let graphs = DecodeGraphs::new(&stream, graphs_on, rows);
        let async_ring = AsyncResultRing::new(&stream, 2, rows)?;
        let logits = Tensor::zeros(&stream, &[rows * v])?;
        let work = Tensor::zeros(&stream, &[rows * v])?;
        let raw = Tensor::zeros(&stream, &[rows * v])?;
        let mut ex = Self {
            stream,
            gemm,
            model,
            slots,
            logits,
            work,
            raw,
            buf,
            last_logprobs: Vec::new(),
            steps: 0,
            kda_chunk: None,
            kda_chunk_enabled: !matches!(std::env::var("KDA_PREFILL").as_deref(), Ok("recurrent")),
            graphs,
            graph_steps: 0,
            eager_steps: 0,
            async_ring,
            async_inflight: std::collections::VecDeque::with_capacity(2),
            skip_prev_chosen: false,
            sync_stash: None,
            trace: std::env::var("KIMIK3_TRACE").is_ok(),
        };
        if ex.graphs.enabled() && graph_capture_eager() {
            ex.capture_decode_graphs()?;
        }
        Ok(ex)
    }

    /// KDA 状态的废纸篓 slot（装载时比 `max_num_seqs` 多分的最后一个）。
    fn scratch_slot(&self) -> i32 {
        (self.model.num_slots() - 1) as i32
    }

    /// 上传 plan 并执行一步（不取结果）。返回采样行数。
    fn enqueue(&mut self, plan: &BatchPlan) -> Result<usize> {
        self.skip_prev_chosen = plan.skip_prev_chosen;
        let n = plan.num_reqs();
        let t = plan.total_tokens();
        let nrows = plan.logits_rows.len();
        ensure!(t <= self.model.max_tokens(), "batch of {t} tokens > max_tokens {}", self.model.max_tokens());
        ensure!(nrows <= self.model.max_rows(), "rows {nrows} > max_rows {}", self.model.max_rows());
        ensure!(nrows >= n && plan.sample.num_rows() == nrows, "rows {nrows} / sample {}", plan.sample.num_rows());
        ensure!(
            plan.max_blocks <= self.buf.plan.max_blocks,
            "block_table {} > buffer {}",
            plan.max_blocks,
            self.buf.plan.max_blocks
        );
        ensure!(!plan.sample.any_mask, "kimik3 executor: guided decoding 未接（C-4b 不做）");
        if self.trace {
            tracing::info!(?plan.mode, ?plan.req_ids, ?plan.num_tokens, ?plan.cu_seqlens, ?plan.seq_lens, "kimik3 plan");
        }

        // KDA slot：归还已结束的、分配本步的、判定"从头开始"的请求（同 qwen35：pos0 == 0 或分到全新 slot 都清零）。
        let req_ids: Vec<u64> = plan.req_ids.clone();
        self.slots.release(&plan.finished_req_ids);
        let fresh_slots = self.slots.begin_step(&req_ids)?;
        let state_idx = self.slots.indices(&req_ids);
        let mut has_init = vec![1i32; n];
        let mut restarts = Vec::new();
        for i in 0..n {
            let p0 = plan.cu_seqlens[i] as usize;
            let pos0 = plan.positions.get(p0).copied().unwrap_or(0);
            let fresh = fresh_slots.contains(&(state_idx[i] as usize));
            if pos0 == 0 || fresh {
                if fresh && pos0 != 0 {
                    tracing::warn!(
                        req = req_ids[i],
                        pos = pos0,
                        slot = state_idx[i],
                        "kimik3: KDA 状态 slot 被复用给带历史的请求，状态已丢失（提高 max_num_seqs）"
                    );
                }
                has_init[i] = 0;
                restarts.push(state_idx[i] as usize);
            }
        }
        if !restarts.is_empty() {
            self.model.reset_slots(&self.stream, &restarts)?;
        }

        let sb = &plan.sample;
        let graph_eligible = plan.mode == BatchMode::Decode
            && t == n
            && nrows == n
            && !sb.any_penalty
            && sb.max_logprobs.is_none()
            && !sb.any_mask;
        if graph_eligible {
            self.ensure_graph(n)?;
        }
        let use_graph = graph_eligible && self.graphs.get(n).is_some();
        if use_graph {
            let nb = self.graphs.get(n).unwrap().0;
            self.upload_plan(plan, &state_idx, &has_init, nb)?;
            self.upload_sampling(sb, nb)?;
            self.graphs.get(n).unwrap().1.launch().context("launch decode graph")?;
            self.last_logprobs = vec![Vec::new(); n];
            if self.graph_steps == 0 || self.trace {
                tracing::info!(n, bucket = nb, "kimik3: decode step via cuda graph");
            }
            self.graph_steps += 1;
        } else {
            self.upload_plan(plan, &state_idx, &has_init, n)?;
            self.forward_device(plan, t, n, nrows)?;
            self.sample_eager(sb, nrows)?;
            if self.trace {
                tracing::info!(?plan.mode, t, n, nrows, "kimik3: step eager");
            }
            self.eager_steps += 1;
        }
        self.steps += 1;
        Ok(nrows)
    }

    /// plan → 固定地址缓冲，batch 维补到 `nb`（padding 行：公共列由 `PlanBuffers` 处理，KDA slot = 废纸篓、has_init = 1）。
    fn upload_plan(&mut self, plan: &BatchPlan, state_idx: &[i32], has_init: &[i32], nb: usize) -> Result<()> {
        let s = &self.stream;
        let (_t, nb, _nrows) = self.buf.plan.upload(s, &mut self.buf.up, plan, nb)?;
        let mut si = state_idx.to_vec();
        si.resize(nb, self.scratch_slot());
        self.buf.up.push(s, &si, &mut self.buf.state_idx.data)?;
        let mut hi = has_init.to_vec();
        hi.resize(nb, 1);
        self.buf.up.push(s, &hi, &mut self.buf.has_init.data)?;
        self.buf.up.flush(s)?;
        Ok(())
    }

    /// forward（固定缓冲、显式形状）。纯 decode 时可被 capture：期间不分配 / 不 D2H / 不同步。
    fn forward_device(&mut self, plan: &BatchPlan, t: usize, n: usize, nrows: usize) -> Result<()> {
        let s = &self.stream;
        // 异步调度：ids[i] = prev_chosen[gather_idx[i]]（进图；非异步步全 -1 等于空操作）
        self.buf.plan.gather_inflight(s, t)?;
        // KDA chunk 工作区：只有 prefill 步（有请求本步 > 1 token）才建表（图外：decode 步 t == n）。
        let chunk = if t > n && self.kda_chunk_enabled {
            let mut ws = self.kda_chunk.take().unwrap_or_default();
            ws.prepare(s, &plan.cu_seqlens, self.model.kda_shapes())?;
            Some(RefCell::new(ws))
        } else {
            None
        };
        let meta = StepMeta {
            kda: KdaMeta {
                cu_seqlens: &self.buf.plan.cu,
                state_indices: &self.buf.state_idx,
                has_initial: &self.buf.has_init,
                num_reqs: n,
                mode: plan.mode,
                chunk: chunk.as_ref(),
            },
            mla: MlaMeta {
                mode: plan.mode,
                cu_seqlens: &self.buf.plan.cu,
                block_table: &self.buf.plan.bt,
                seq_lens: &self.buf.plan.seq,
                num_reqs: n,
                max_blocks: self.buf.plan.max_blocks,
            },
            slot_mapping: &self.buf.plan.slots,
        };
        self.model.forward_rows(
            s,
            &self.gemm,
            t,
            &self.buf.plan.ids,
            &meta,
            &self.buf.plan.rows,
            nrows,
            &mut self.logits,
        )?;
        if let Some(ws) = chunk {
            self.kda_chunk = Some(ws.into_inner());
        }
        Ok(())
    }

    /// 图内采样：融合 kernel（温度 / min_p / top-k / top-p / 多项式；温度 0 行 = argmax），参数来自设备缓冲。
    fn sample_device_fixed(&mut self, nb: usize) -> Result<()> {
        let s = &self.stream;
        let v = self.model.vocab();
        s.dtod_copy_n(&self.logits.data, &mut self.work.data, nb * v)?;
        let StepBuffers { chosen, temperature, min_p, top_k, top_p, temp01, seeds, offsets, plan, .. } = &mut self.buf;
        let mut wv = self.work.data.slice_mut(..nb * v);
        let mut ch = chosen.data.slice_mut(..nb);
        k::sample_fused_f32(
            s,
            &mut wv,
            &temperature.data,
            &min_p.data,
            &top_k.data,
            &top_p.data,
            &temp01.data,
            &seeds.data,
            &offsets.data,
            &mut ch,
            nb,
            v,
        )?;
        plan.save_chosen(s, &chosen.data, nb)
    }

    fn save_prev_chosen(&mut self, nrows: usize) -> Result<()> {
        if self.skip_prev_chosen {
            return Ok(());
        }
        let StepBuffers { chosen, plan, .. } = &mut self.buf;
        plan.save_chosen(&self.stream, &chosen.data, nrows)
    }

    // ---------------------------------------------------------------- CUDA graph

    /// 捕获体：`Prepare` 上传假 plan（KDA slot 全部指废纸篓、has_init = 1；MLA slot_mapping = -1）+ 采样参数（图外）；
    /// `Body` = forward + 采样。
    fn capture_body(&mut self, nb: usize, phase: CapturePhase) -> Result<()> {
        let p = decode_capture_plan(nb);
        match phase {
            CapturePhase::Prepare => {
                let si = vec![self.scratch_slot(); nb];
                let hi = vec![1i32; nb];
                self.upload_plan(&p, &si, &hi, nb)?;
                self.upload_sampling(&p.sample, nb)
            }
            CapturePhase::Body => {
                let trace = std::mem::replace(&mut self.trace, false);
                let r = self.forward_device(&p, nb, nb, nb).and_then(|_| self.sample_device_fixed(nb));
                self.trace = trace;
                r
            }
        }
    }

    fn capture_decode_graphs(&mut self) -> Result<()> {
        let mut graphs = std::mem::replace(&mut self.graphs, DecodeGraphs::new(&self.stream, false, 0));
        let r = graphs.capture_all(|nb, ph| self.capture_body(nb, ph));
        self.graphs = graphs;
        r
    }

    fn ensure_graph(&mut self, n: usize) -> Result<()> {
        let mut graphs = std::mem::replace(&mut self.graphs, DecodeGraphs::new(&self.stream, false, 0));
        let r = graphs.ensure(n, |nb, ph| self.capture_body(nb, ph));
        self.graphs = graphs;
        r
    }

    fn collect(&mut self, n: usize) -> Result<Vec<i32>> {
        self.stream.dtoh_sync_copy_view(&self.buf.chosen.data.slice(..n))
    }

    // ---------------------------------------------------------------- 采样（与 Qwen3 / Qwen3.5 同序）

    fn upload_sampling(&mut self, sb: &SampleBatch, nb: usize) -> Result<()> {
        let s = &self.stream;
        self.buf.up.begin();
        let n = sb.num_rows();
        let padf = |v: &[f32], fill: f32| {
            let mut x = v.to_vec();
            x.resize(nb, fill);
            x
        };
        let padi = |v: &[i32], fill: i32| {
            let mut x = v.to_vec();
            x.resize(nb, fill);
            x
        };
        let padu = |v: &[u64], fill: u64| {
            let mut x = v.to_vec();
            x.resize(nb, fill);
            x
        };
        let temp = padf(&sb.temperature, 0.0);
        self.buf.up.push(s, &temp, &mut self.buf.temperature.data)?;
        let t01: Vec<f32> = temp.iter().map(|&t| if t > 0.0 { 1.0 } else { 0.0 }).collect();
        self.buf.up.push(s, &t01, &mut self.buf.temp01.data)?;
        self.buf.up.push(s, &padf(&sb.min_p, 0.0), &mut self.buf.min_p.data)?;
        self.buf.up.push(s, &padi(&sb.top_k, 0), &mut self.buf.top_k.data)?;
        self.buf.up.push(s, &padf(&sb.top_p, 1.0), &mut self.buf.top_p.data)?;
        self.buf.up.push(s, &padu(&sb.seed, 0), &mut self.buf.seeds.data)?;
        self.buf.up.push(s, &padu(&sb.offset, 0), &mut self.buf.offsets.data)?;
        let _ = n;
        if sb.any_penalty {
            self.buf.up.push(s, &padf(&sb.repetition_penalty, 1.0), &mut self.buf.rep.data)?;
            self.buf.up.push(s, &padf(&sb.frequency_penalty, 0.0), &mut self.buf.freq.data)?;
            self.buf.up.push(s, &padf(&sb.presence_penalty, 0.0), &mut self.buf.pres.data)?;
            let mut ho = sb.hist_offsets.clone();
            while ho.len() < nb + 1 {
                ho.push(*ho.last().unwrap_or(&0));
            }
            self.buf.up.push(s, &ho, &mut self.buf.hist_offsets.data)?;
            let mut hp = sb.hist_prompt_len.clone();
            hp.resize(nb, 0);
            self.buf.up.push(s, &hp, &mut self.buf.hist_prompt_len.data)?;
            if self.buf.hist_ids.numel() < sb.hist_ids.len().max(1) {
                self.buf.hist_ids = Tensor::<i32>::zeros(s, &[sb.hist_ids.len().next_power_of_two()])?;
            }
            if !sb.hist_ids.is_empty() {
                s.htod_into(&sb.hist_ids, &mut self.buf.hist_ids.data)?;
            }
        }
        self.buf.up.flush(s)?;
        Ok(())
    }

    fn sample_eager(&mut self, sb: &SampleBatch, nrows: usize) -> Result<()> {
        let v = self.model.vocab();
        let fast = !sb.any_random && !sb.any_penalty && sb.max_logprobs.is_none();
        if fast {
            let lg = self.logits.data.slice(..nrows * v);
            let mut ch = self.buf.chosen.data.slice_mut(..nrows);
            k::argmax_f32_n(&self.stream, &lg, &mut ch, nrows, v)?;
            self.last_logprobs = vec![Vec::new(); nrows];
            self.save_prev_chosen(nrows)?;
            return Ok(());
        }
        self.upload_sampling(sb, nrows)?;
        let s = self.stream.clone();
        let s = &s;
        s.dtod_copy_n(&self.logits.data, &mut self.work.data, nrows * v)?;
        if sb.max_logprobs.is_some() {
            s.dtod_copy_n(&self.logits.data, &mut self.raw.data, nrows * v)?;
        }
        {
            let b = &self.buf;
            let mut wv = self.work.data.slice_mut(..nrows * v);
            if sb.any_penalty {
                k::penalties_f32(
                    s,
                    &mut wv,
                    &b.hist_ids.data,
                    &b.hist_offsets.data.slice(..nrows + 1),
                    &b.hist_prompt_len.data.slice(..nrows),
                    &b.rep.data.slice(..nrows),
                    &b.freq.data.slice(..nrows),
                    &b.pres.data.slice(..nrows),
                    v,
                )?;
            }
            if sb.any_temperature {
                k::temperature_f32(s, &mut wv, &b.temperature.data.slice(..nrows), v)?;
            }
            if sb.any_min_p {
                k::min_p_mask_f32(s, &mut wv, &b.min_p.data.slice(..nrows), v)?;
            }
            if sb.any_topk_topp {
                k::topk_topp_mask_f32(s, &mut wv, &b.top_k.data.slice(..nrows), &b.top_p.data.slice(..nrows), v)?;
            }
        }
        {
            let work = &self.work;
            let b = &mut self.buf;
            let wv = work.data.slice(..nrows * v);
            let mut ch = b.chosen.data.slice_mut(..nrows);
            if sb.any_random {
                k::sample_f32(
                    s,
                    &wv,
                    &b.temp01.data.slice(..nrows),
                    &b.seeds.data.slice(..nrows),
                    &b.offsets.data.slice(..nrows),
                    &mut ch,
                    v,
                )?;
            } else {
                k::argmax_f32_n(s, &wv, &mut ch, nrows, v)?;
            }
        }
        self.save_prev_chosen(nrows)?;
        self.last_logprobs = vec![Vec::new(); nrows];
        if let Some(kmax) = sb.max_logprobs {
            let kmax = kmax as usize;
            let raw = self.raw.data.slice(..nrows * v);
            let ch = self.buf.chosen.data.slice(..nrows);
            let (ids_v, lp_v, rank_v) = {
                let mut ids = self.buf.lp_ids.data.slice_mut(..nrows * (kmax + 1));
                let mut lp = self.buf.lp_val.data.slice_mut(..nrows * (kmax + 1));
                let mut rank = self.buf.lp_rank.data.slice_mut(..nrows);
                k::logprobs_f32(s, &raw, &ch, kmax, &mut ids, &mut lp, &mut rank, v)?;
                (
                    s.dtoh_sync_copy_view(&self.buf.lp_ids.data.slice(..nrows * (kmax + 1)))?,
                    s.dtoh_sync_copy_view(&self.buf.lp_val.data.slice(..nrows * (kmax + 1)))?,
                    s.dtoh_sync_copy_view(&self.buf.lp_rank.data.slice(..nrows))?,
                )
            };
            for (r, want) in sb.logprobs.iter().enumerate() {
                let Some(kk) = want else { continue };
                let base = r * (kmax + 1);
                let top = (1..=*kk as usize).map(|j| (ids_v[base + j] as u32, lp_v[base + j])).collect();
                self.last_logprobs[r] = vec![TokenLogprobs {
                    token_id: ids_v[base] as u32,
                    logprob: lp_v[base],
                    rank: rank_v[r] as u32,
                    top,
                    prompt_pos: None,
                    prompt: Vec::new(),
                }];
            }
        }
        Ok(())
    }
}

impl ModelExecutor for KimiK3Executor {
    fn supports_async(&self) -> bool {
        async_sched_enabled()
    }

    fn begin(&mut self, plan: &BatchPlan) -> Result<()> {
        self.stream.bind_ctx()?;
        let n = self.enqueue(plan).context("kimik3 enqueue (async)")?;
        let ticket = self.async_ring.enqueue(&self.stream, &self.buf.chosen.data, n)?;
        self.async_inflight.push_back(ticket);
        Ok(())
    }

    fn finish(&mut self, generated: &mut Vec<Vec<u32>>, drafts: &mut Vec<Vec<u32>>) -> Result<()> {
        let ticket = self.async_inflight.pop_front().context("executor finish without begin")?;
        let chosen = self.async_ring.wait(ticket)?;
        push_tokens(generated, drafts, &chosen);
        Ok(())
    }

    fn stash_sync_result(&mut self, g: Vec<Vec<u32>>, d: Vec<Vec<u32>>) {
        self.sync_stash = Some((g, d));
    }
    fn take_sync_result(&mut self) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
        self.sync_stash.take().unwrap_or_default()
    }

    fn forward_sample(
        &mut self,
        plan: &BatchPlan,
        generated: &mut Vec<Vec<u32>>,
        drafts: &mut Vec<Vec<u32>>,
    ) -> Result<()> {
        let t0 = std::time::Instant::now();
        // 引擎线程 != 构建线程：首个 kernel 会报 CUDA_ERROR_INVALID_CONTEXT（Qwen3.5 真权重首请求复现）。每步绑一次很便宜。
        self.stream.bind_ctx()?;
        let nrows = self.enqueue(plan).context("kimik3 enqueue")?;
        let chosen = self.collect(nrows)?;
        push_tokens(generated, drafts, &chosen);
        tracing::trace!(rows = nrows, ms = t0.elapsed().as_millis() as u64, "kimik3 step");
        Ok(())
    }

    fn take_logprobs(&mut self, num_reqs: usize) -> Vec<Vec<TokenLogprobs>> {
        let mut v = std::mem::take(&mut self.last_logprobs);
        v.resize(num_reqs, Vec::new());
        v
    }
}

/// 采样结果 → 每请求一个 `generated`（无 draft：长度 1）。
fn push_tokens(generated: &mut Vec<Vec<u32>>, drafts: &mut Vec<Vec<u32>>, chosen: &[i32]) {
    generated.clear();
    drafts.clear();
    for &c in chosen {
        generated.push(vec![c as u32]);
        drafts.push(Vec::new());
    }
}
