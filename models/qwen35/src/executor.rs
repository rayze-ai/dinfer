//! `Qwen35Model` 的 `ModelExecutor`：上传 BatchPlan → forward → 采样 → D2H。
//!
//! **里程碑 4 第一版刻意做小**：eager 单流，没有 CUDA graph / 跨流 pair / KV host tier / PD 分离 / TP 通信
//! （Qwen3 的 executor 里这些都在，接过来是一轮独立工作）。采样流水与 Qwen3 同序：
//! penalties → temperature → min_p → top_k/top_p → 多项式采样（greedy 行 argmax）→ logprobs。
//!
//! GDN 状态：每步按 `RequestId` 分 slot（`SlotTable`）；**本步首个 token 位置为 0 的请求**当作从头开始
//! （新请求 / 被抢占后重算），清零它的 conv + 递推状态，`has_initial = 0`；其余 `has_initial = 1`。
use crate::gdn::SlotTable;
use crate::model::{Forward, ModelScratch, Qwen35Model};
use crate::layer::StepMeta;
use anyhow::{ensure, Context, Result};
use dinfer_core::runner::{BatchPlan, ModelExecutor};
use dinfer_core::sampler::{SampleBatch, TokenLogprobs};
use dinfer_kernels as k;
use dinfer_layers::{Gemm, PinnedUploader, Stream, Tensor};

/// 固定地址的每 step 输入缓冲。
struct StepBuffers {
    ids: Tensor<i32>,   // [max_tokens]
    pos: Tensor<i32>,   // [max_tokens]
    slots: Tensor<i64>, // [max_tokens]
    rows: Tensor<i32>,  // [max_rows]
    cu: Tensor<i32>,    // [max_rows + 1]
    seq: Tensor<i32>,   // [max_rows]
    bt: Tensor<i32>,    // [max_rows * max_blocks]
    /// GDN：每请求的 slot 与"是否有历史状态"。
    state_idx: Tensor<i32>, // [max_rows]
    has_init: Tensor<i32>,  // [max_rows]
    max_blocks: usize,
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
            ids: z32(mt)?,
            pos: z32(mt)?,
            slots: Tensor::<i64>::zeros(s, &[mt.max(1)])?,
            rows: z32(rows)?,
            cu: z32(rows + 1)?,
            seq: z32(rows)?,
            bt: z32(rows * max_blocks)?,
            state_idx: z32(rows)?,
            has_init: z32(rows)?,
            max_blocks,
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
            up: PinnedUploader::new(s, (mt * 16 + rows * 128 + rows * max_blocks * 4 + (1 << 20)).max(1 << 20))?,
        })
    }
}

pub struct Qwen35Executor {
    pub stream: Stream,
    pub gemm: Gemm,
    pub model: Qwen35Model,
    pub slots: SlotTable,
    logits: Tensor<f32>,
    work: Tensor<f32>,
    raw: Tensor<f32>,
    buf: StepBuffers,
    last_logprobs: Vec<Vec<TokenLogprobs>>,
    last_rows: usize,
    pub steps: u64,
    /// 跨流并行用的第二套激活缓冲（第二版接 `forward_sample_pair`）。
    pub alt: Option<ModelScratch>,
}

impl Qwen35Executor {
    pub fn new(stream: Stream, gemm: Gemm, model: Qwen35Model, max_blocks: usize) -> Result<Self> {
        let (mt, rows, v) = (model.max_tokens, model.max_logits_rows, model.cfg.vocab_size as usize);
        let slots = SlotTable::new(model.num_slots);
        let buf = StepBuffers::alloc(&stream, mt, rows, max_blocks)?;
        let logits = Tensor::zeros(&stream, &[rows * v])?;
        let work = Tensor::zeros(&stream, &[rows * v])?;
        let raw = Tensor::zeros(&stream, &[rows * v])?;
        Ok(Self {
            stream,
            gemm,
            model,
            slots,
            logits,
            work,
            raw,
            buf,
            last_logprobs: Vec::new(),
            last_rows: 0,
            steps: 0,
            alt: None,
        })
    }

    /// 上传 plan 并执行一步（不取结果）。
    fn enqueue(&mut self, plan: &BatchPlan) -> Result<usize> {
        let s = &self.stream;
        let n = plan.num_reqs();
        let t = plan.total_tokens();
        let nrows = plan.logits_rows.len();
        ensure!(t <= self.model.max_tokens, "batch of {t} tokens > max_tokens {}", self.model.max_tokens);
        ensure!(nrows <= self.model.max_logits_rows, "rows {nrows} > max_logits_rows");
        ensure!(nrows >= n && plan.sample.num_rows() == nrows, "rows {nrows} / sample {}", plan.sample.num_rows());
        ensure!(plan.max_blocks <= self.buf.max_blocks, "block_table {} > buffer {}", plan.max_blocks, self.buf.max_blocks);
        ensure!(!plan.sample.any_mask, "qwen35 executor: guided decoding 未接（里程碑 4 不做）");

        if std::env::var("QWEN35_TRACE").is_ok() {
            tracing::info!(?plan.mode, ?plan.req_ids, ?plan.num_tokens, ?plan.cu_seqlens, ?plan.positions, ?plan.seq_lens, ?plan.logits_rows, "qwen35 plan");
        }
        // GDN slot：分配 + 判定"从头开始"的请求
        let req_ids: Vec<u64> = plan.req_ids.clone();
        let fresh_slots = self.slots.begin_step(&req_ids)?;
        let state_idx = self.slots.indices(&req_ids);
        // 首个 token 位置 == 0 → 从头开始（新请求或被抢占后的重算），状态清零
        let mut has_init = vec![1i32; n];
        let mut restarts = Vec::new();
        for i in 0..n {
            let p0 = plan.cu_seqlens[i] as usize;
            if plan.positions.get(p0).copied().unwrap_or(0) == 0 {
                has_init[i] = 0;
                restarts.push(state_idx[i] as usize);
            }
        }
        if !restarts.is_empty() {
            self.model.reset_slots(s, &restarts)?;
        }
        let _ = fresh_slots;

        self.buf.up.reset_step();
        self.buf.up.push(s, &plan.input_ids, &mut self.buf.ids.data)?;
        self.buf.up.push(s, &plan.positions, &mut self.buf.pos.data)?;
        self.buf.up.push(s, &plan.slot_mapping, &mut self.buf.slots.data)?;
        self.buf.up.push(s, &plan.logits_rows, &mut self.buf.rows.data)?;
        self.buf.up.push(s, &plan.cu_seqlens, &mut self.buf.cu.data)?;
        self.buf.up.push(s, &plan.seq_lens, &mut self.buf.seq.data)?;
        let mut bt = vec![0i32; n * self.buf.max_blocks];
        for r in 0..n {
            for j in 0..plan.max_blocks {
                bt[r * self.buf.max_blocks + j] = plan.block_table[r * plan.max_blocks + j];
            }
        }
        self.buf.up.push(s, &bt, &mut self.buf.bt.data)?;
        self.buf.up.push(s, &state_idx, &mut self.buf.state_idx.data)?;
        self.buf.up.push(s, &has_init, &mut self.buf.has_init.data)?;
        self.buf.up.flush(s)?;

        let max_seqlen = plan.num_tokens.iter().copied().max().unwrap_or(0).max(1) as usize;
        let meta = StepMeta {
            mode: plan.mode,
            cu_seqlens: &self.buf.cu,
            seq_lens: &self.buf.seq,
            block_table: &self.buf.bt,
            state_indices: &self.buf.state_idx,
            has_initial: &self.buf.has_init,
            num_reqs: n,
            max_blocks: self.buf.max_blocks,
            max_seqlen,
        };
        let f = Forward {
            input_ids: &self.buf.ids,
            positions: &self.buf.pos,
            slot_mapping: &self.buf.slots,
            meta,
            logits_rows: &self.buf.rows,
            t,
            rows: nrows,
            req_ids: &req_ids,
        };
        self.model.forward(s, &self.gemm, &f, &mut self.logits)?;
        self.sample_eager(&plan.sample, nrows)?;
        self.last_rows = nrows;
        self.steps += 1;
        Ok(nrows)
    }

    fn collect(&mut self, n: usize) -> Result<Vec<i32>> {
        self.stream.dtoh_sync_copy_view(&self.buf.chosen.data.slice(..n))
    }

    fn upload_sampling(&mut self, sb: &SampleBatch, nb: usize) -> Result<()> {
        let s = &self.stream;
        self.buf.up.begin();
        let n = sb.num_rows();
        let padf = |v: &Vec<f32>, fill: f32| {
            let mut x = v.clone();
            x.resize(nb, fill);
            x
        };
        self.buf.up.push(s, &padf(&sb.temperature, 0.0), &mut self.buf.temperature.data)?;
        let t01: Vec<f32> = (0..nb).map(|i| if i < n && sb.temperature[i] > 1e-5 { 1.0 } else { 0.0 }).collect();
        self.buf.up.push(s, &t01, &mut self.buf.temp01.data)?;
        let mut tk = sb.top_k.clone();
        tk.resize(nb, -1);
        self.buf.up.push(s, &tk, &mut self.buf.top_k.data)?;
        self.buf.up.push(s, &padf(&sb.top_p, 1.0), &mut self.buf.top_p.data)?;
        self.buf.up.push(s, &padf(&sb.min_p, 0.0), &mut self.buf.min_p.data)?;
        let mut seeds = sb.seed.clone();
        seeds.resize(nb, 0);
        self.buf.up.push(s, &seeds, &mut self.buf.seeds.data)?;
        let mut offs = sb.offset.clone();
        offs.resize(nb, 0);
        self.buf.up.push(s, &offs, &mut self.buf.offsets.data)?;
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

    /// 采样流水（与 Qwen3 `sample_eager` 同序；不含 guided mask / prompt_logprobs）。
    fn sample_eager(&mut self, sb: &SampleBatch, nrows: usize) -> Result<()> {
        let v = self.model.cfg.vocab_size as usize;
        let fast = !sb.any_random && !sb.any_penalty && sb.max_logprobs.is_none();
        if fast {
            let lg = self.logits.data.slice(..nrows * v);
            let mut ch = self.buf.chosen.data.slice_mut(..nrows);
            k::argmax_f32_n(&self.stream, &lg, &mut ch, nrows, v)?;
            self.last_logprobs = vec![Vec::new(); nrows];
            return Ok(());
        }
        self.upload_sampling(sb, nrows)?;
        let s = &self.stream;
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

impl ModelExecutor for Qwen35Executor {
    fn forward_sample(
        &mut self,
        plan: &BatchPlan,
        generated: &mut Vec<Vec<u32>>,
        drafts: &mut Vec<Vec<u32>>,
    ) -> Result<()> {
        let t0 = std::time::Instant::now();
        let nrows = self.enqueue(plan).context("qwen35 enqueue")?;
        let chosen = self.collect(nrows)?;
        push_tokens(generated, drafts, &chosen);
        tracing::trace!(rows = nrows, ms = t0.elapsed().as_millis() as u64, "qwen35 step");
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
