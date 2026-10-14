//! `Glm53Model` 的 `ModelExecutor` 实现：把 `BatchPlan` 上传为设备张量 → forward → greedy argmax → D2H。
//! P4 最小版：每 step 重新 H2D 全部输入（无 pinned / 双缓冲），采样只支持 greedy。

use crate::model::{Forward, Glm53Model};
use anyhow::{ensure, Result};
use dinfer_core::runner::{BatchPlan, ModelExecutor};
use dinfer_kernels as k;
use dinfer_layers::mla::AttnMeta;
use dinfer_layers::{Gemm, Stream, Tensor};

pub struct Glm53Executor {
    pub stream: Stream,
    pub gemm: Gemm,
    pub model: Glm53Model,
    logits: Tensor<f32>,
    argmax: Tensor<i32>,
}

impl Glm53Executor {
    pub fn new(stream: Stream, gemm: Gemm, model: Glm53Model) -> Result<Self> {
        let v = model.cfg.vocab_size as usize;
        let max_tokens = model.max_tokens;
        Ok(Self {
            logits: Tensor::zeros(&stream, &[max_tokens, v])?,
            argmax: Tensor::zeros(&stream, &[max_tokens])?,
            stream,
            gemm,
            model,
        })
    }

    /// 上一 step 的 logits（fp32，[num_reqs, vocab]）；测试 / 对拍用。
    pub fn last_logits(&self, num_reqs: usize) -> Result<Vec<f32>> {
        let v = self.model.cfg.vocab_size as usize;
        let all = self.logits.to_vec(&self.stream)?;
        Ok(all[..num_reqs * v].to_vec())
    }
}

impl ModelExecutor for Glm53Executor {
    fn forward_sample(&mut self, plan: &BatchPlan, sampled: &mut Vec<u32>) -> Result<()> {
        let s = &self.stream;
        let n = plan.num_reqs();
        let t = plan.total_tokens();
        ensure!(t <= self.model.max_tokens, "batch of {t} tokens > max_tokens {}", self.model.max_tokens);
        let v = self.model.cfg.vocab_size as usize;

        let ids = Tensor::from_vec(s, plan.input_ids.clone(), &[t])?;
        let pos = Tensor::from_vec(s, plan.positions.clone(), &[t])?;
        let slots = Tensor::from_vec(s, plan.slot_mapping.clone(), &[t])?;
        let rows = Tensor::from_vec(s, plan.logits_rows.clone(), &[n])?;
        let cu = Tensor::from_vec(s, plan.cu_seqlens.clone(), &[n + 1])?;
        let seq = Tensor::from_vec(s, plan.seq_lens.clone(), &[n])?;
        let bt = Tensor::from_vec(s, plan.block_table.clone(), &[n, plan.max_blocks.max(1)])?;
        let meta = if plan.is_prefill {
            AttnMeta::Prefill { cu_seqlens: &cu, num_reqs: n }
        } else {
            AttnMeta::Decode { block_table: &bt, seq_lens: &seq, max_blocks: plan.max_blocks }
        };
        let f = Forward { input_ids: &ids, positions: &pos, slot_mapping: &slots, meta, logits_rows: &rows };
        self.model.forward(s, &self.gemm, &f, &mut self.logits)?;
        k::argmax_f32_n(s, &self.logits.data, &mut self.argmax.data, n, v)?;
        let out = s.dtoh_sync_copy(&self.argmax.data)?;
        sampled.extend(out[..n].iter().map(|&x| x as u32));
        Ok(())
    }
}
