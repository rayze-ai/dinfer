//! `Qwen3Model` 的 `ModelExecutor`：上传 BatchPlan → forward → 采样 → D2H。Q1 无投机解码（drafts 为空）。
//!
//! 采样（Q2-a）与 vLLM `Sampler.forward` 同序：penalties（prompt+output 历史）→ temperature → min_p → top_k/top_p →
//! Philox 多项式采样（greedy 行 argmax）；logprobs 用 forward 的 **原始** logits（惩罚 / 温度之前）做 log_softmax。
//! 全 greedy 且无惩罚 / 无 logprobs 的 batch 走 Q1 的 argmax 快路径（回归一致）。

use crate::model::{AttnMeta, Forward, Qwen3Model};
use anyhow::{ensure, Result};
use dinfer_core::runner::{BatchPlan, ModelExecutor};
use dinfer_core::sampler::{SampleBatch, TokenLogprobs};
use dinfer_kernels as k;
use dinfer_layers::{Gemm, Stream, Tensor};

pub struct Qwen3Executor {
    pub stream: Stream,
    pub gemm: Gemm,
    pub model: Qwen3Model,
    logits: Tensor<f32>,
    argmax: Tensor<i32>,
    /// 采样工作区 [rows, vocab]（原地改）；按本步 rows 分配，rows 变化时重分配。
    work: Option<Tensor<f32>>,
    /// logprobs 用的原始 logits 副本 [rows, vocab]（只在有请求要 logprobs 时分配）。
    raw: Option<Tensor<f32>>,
    last_logprobs: Vec<Vec<TokenLogprobs>>,
}

fn ensure_rows(stream: &Stream, slot: &mut Option<Tensor<f32>>, rows: usize, v: usize) -> Result<()> {
    if slot.as_ref().is_none_or(|t| t.numel() != rows * v) {
        *slot = Some(Tensor::zeros(stream, &[rows, v])?);
    }
    Ok(())
}

impl Qwen3Executor {
    pub fn new(stream: Stream, gemm: Gemm, model: Qwen3Model) -> Result<Self> {
        let v = model.cfg.vocab_size as usize;
        let rows = model.max_logits_rows;
        Ok(Self {
            logits: Tensor::zeros(&stream, &[rows, v])?,
            argmax: Tensor::zeros(&stream, &[rows])?,
            stream,
            gemm,
            model,
            work: None,
            raw: None,
            last_logprobs: Vec::new(),
        })
    }

    /// 上一 step 的 **原始** logits（fp32，[num_rows, vocab]）；对拍用。
    pub fn last_logits(&self, num_rows: usize) -> Result<Vec<f32>> {
        let v = self.model.cfg.vocab_size as usize;
        let all = self.logits.to_vec(&self.stream)?;
        Ok(all[..num_rows * v].to_vec())
    }

    /// 采样管线；返回每行 chosen token。`nrows == num_reqs`（无投机）。
    fn sample(&mut self, sb: &SampleBatch, nrows: usize) -> Result<Vec<i32>> {
        let s = &self.stream;
        let v = self.model.cfg.vocab_size as usize;
        let fast = !sb.any_random && !sb.any_penalty && sb.max_logprobs.is_none();
        if fast {
            k::argmax_f32_n(s, &self.logits.data, &mut self.argmax.data, nrows, v)?;
            self.last_logprobs = vec![Vec::new(); nrows];
            return Ok(s.dtoh_sync_copy(&self.argmax.data)?[..nrows].to_vec());
        }
        ensure_rows(s, &mut self.work, nrows, v)?;
        let work = self.work.as_mut().unwrap();
        s.dtod_copy_n(&self.logits.data, &mut work.data, nrows * v)?;
        if sb.max_logprobs.is_some() {
            ensure_rows(s, &mut self.raw, nrows, v)?;
            s.dtod_copy_n(&self.logits.data, &mut self.raw.as_mut().unwrap().data, nrows * v)?;
        }
        let col_f32 = |x: &Vec<f32>| Tensor::from_vec(s, x.clone(), &[nrows]);
        if sb.any_penalty {
            let ids = Tensor::from_vec(
                s,
                if sb.hist_ids.is_empty() { vec![0] } else { sb.hist_ids.clone() },
                &[sb.hist_ids.len().max(1)],
            )?;
            let offs = Tensor::from_vec(s, sb.hist_offsets.clone(), &[nrows + 1])?;
            let plen = Tensor::from_vec(s, sb.hist_prompt_len.clone(), &[nrows])?;
            let (rep, freq, pres) = (
                col_f32(&sb.repetition_penalty)?,
                col_f32(&sb.frequency_penalty)?,
                col_f32(&sb.presence_penalty)?,
            );
            k::penalties_f32(
                s,
                &mut work.data,
                &ids.data,
                &offs.data,
                &plen.data,
                &rep.data,
                &freq.data,
                &pres.data,
                v,
            )?;
        }
        if sb.any_temperature {
            let t = col_f32(&sb.temperature)?;
            k::temperature_f32(s, &mut work.data, &t.data, v)?;
        }
        if sb.any_min_p {
            let mp = col_f32(&sb.min_p)?;
            k::min_p_mask_f32(s, &mut work.data, &mp.data, v)?;
        }
        if sb.any_topk_topp {
            let tk = Tensor::from_vec(s, sb.top_k.clone(), &[nrows])?;
            let tp = col_f32(&sb.top_p)?;
            k::topk_topp_mask_f32(s, &mut work.data, &tk.data, &tp.data, v)?;
        }
        let mut chosen = Tensor::<i32>::zeros(s, &[nrows])?;
        if sb.any_random {
            // 温度已在上面缩放：随机行传 1.0，greedy 行传 0（kernel 内 argmax）
            let t1: Vec<f32> = sb.temperature.iter().map(|&t| if t <= 1e-5 { 0.0 } else { 1.0 }).collect();
            let t1 = Tensor::from_vec(s, t1, &[nrows])?;
            let seeds = Tensor::from_vec(s, sb.seed.clone(), &[nrows])?;
            let offs = Tensor::from_vec(s, sb.offset.clone(), &[nrows])?;
            k::sample_f32(s, &work.data, &t1.data, &seeds.data, &offs.data, &mut chosen.data, v)?;
        } else {
            k::argmax_f32_n(s, &work.data, &mut chosen.data, nrows, v)?;
        }
        let out = s.dtoh_sync_copy(&chosen.data)?;

        self.last_logprobs = vec![Vec::new(); nrows];
        if let Some(kmax) = sb.max_logprobs {
            let kmax = kmax as usize;
            let raw = self.raw.as_ref().unwrap();
            let mut ids = Tensor::<i32>::zeros(s, &[nrows, kmax + 1])?;
            let mut lp = Tensor::<f32>::zeros(s, &[nrows, kmax + 1])?;
            let mut rank = Tensor::<i32>::zeros(s, &[nrows])?;
            k::logprobs_f32(
                s,
                &raw.data,
                &chosen.data,
                kmax,
                &mut ids.data,
                &mut lp.data,
                &mut rank.data,
                v,
            )?;
            let (ids, lp, rank) = (ids.to_vec(s)?, lp.to_vec(s)?, rank.to_vec(s)?);
            for (r, want) in sb.logprobs.iter().enumerate() {
                let Some(kk) = want else { continue };
                let base = r * (kmax + 1);
                let top = (1..=*kk as usize).map(|j| (ids[base + j] as u32, lp[base + j])).collect();
                self.last_logprobs[r] = vec![TokenLogprobs {
                    token_id: ids[base] as u32,
                    logprob: lp[base],
                    rank: rank[r] as u32,
                    top,
                }];
            }
        }
        Ok(out)
    }
}

impl ModelExecutor for Qwen3Executor {
    fn forward_sample(
        &mut self,
        plan: &BatchPlan,
        generated: &mut Vec<Vec<u32>>,
        drafts: &mut Vec<Vec<u32>>,
    ) -> Result<()> {
        let n = plan.num_reqs();
        let t = plan.total_tokens();
        ensure!(t <= self.model.max_tokens, "batch of {t} tokens > max_tokens {}", self.model.max_tokens);
        ensure!(
            plan.spec_tokens.iter().all(|d| d.is_empty()),
            "qwen3 executor: speculative decoding not supported yet"
        );
        let nrows = plan.logits_rows.len();
        ensure!(
            nrows <= self.model.max_logits_rows,
            "{nrows} logits rows > max_logits_rows {}",
            self.model.max_logits_rows
        );
        ensure!(
            nrows == n && plan.sample.num_rows() == n,
            "qwen3 executor: rows {nrows} / sample {} vs reqs {n}",
            plan.sample.num_rows()
        );
        {
            let s = &self.stream;
            let ids = Tensor::from_vec(s, plan.input_ids.clone(), &[t])?;
            let pos = Tensor::from_vec(s, plan.positions.clone(), &[t])?;
            let slots = Tensor::from_vec(s, plan.slot_mapping.clone(), &[t])?;
            let rows = Tensor::from_vec(s, plan.logits_rows.clone(), &[nrows])?;
            let cu = Tensor::from_vec(s, plan.cu_seqlens.clone(), &[n + 1])?;
            let seq = Tensor::from_vec(s, plan.seq_lens.clone(), &[n])?;
            let bt = Tensor::from_vec(s, plan.block_table.clone(), &[n, plan.max_blocks.max(1)])?;
            let meta = AttnMeta {
                mode: plan.mode,
                cu_seqlens: &cu,
                seq_lens: &seq,
                block_table: &bt,
                num_reqs: n,
                max_blocks: plan.max_blocks,
                max_seqlen: plan.num_tokens.iter().copied().max().unwrap_or(0) as usize,
            };
            let f =
                Forward { input_ids: &ids, positions: &pos, slot_mapping: &slots, meta, logits_rows: &rows };
            self.model.forward(s, &self.gemm, &f, &mut self.logits)?;
        }
        let chosen = self.sample(&plan.sample, nrows)?;
        for &c in chosen.iter().take(n) {
            generated.push(vec![c as u32]);
            drafts.push(Vec::new());
        }
        Ok(())
    }

    fn take_logprobs(&mut self, num_reqs: usize) -> Vec<Vec<TokenLogprobs>> {
        let mut v = std::mem::take(&mut self.last_logprobs);
        v.resize(num_reqs, Vec::new());
        v
    }
}
