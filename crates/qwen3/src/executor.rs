//! `Qwen3Model` 的 `ModelExecutor`：上传 BatchPlan → forward → greedy argmax → D2H。Q1 无投机解码（drafts 为空），
//! 采样 greedy（Q2 接 topk/topp/sample kernel）。

use crate::model::{AttnMeta, Forward, Qwen3Model};
use anyhow::{ensure, Result};
use dinfer_core::runner::{BatchPlan, ModelExecutor};
use dinfer_kernels as k;
use dinfer_layers::{Gemm, Stream, Tensor};

pub struct Qwen3Executor {
    pub stream: Stream,
    pub gemm: Gemm,
    pub model: Qwen3Model,
    logits: Tensor<f32>,
    argmax: Tensor<i32>,
}

impl Qwen3Executor {
    pub fn new(stream: Stream, gemm: Gemm, model: Qwen3Model) -> Result<Self> {
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

    /// 上一 step 的 logits（fp32，[num_rows, vocab]）；对拍用。
    pub fn last_logits(&self, num_rows: usize) -> Result<Vec<f32>> {
        let v = self.model.cfg.vocab_size as usize;
        let all = self.logits.to_vec(&self.stream)?;
        Ok(all[..num_rows * v].to_vec())
    }
}

impl ModelExecutor for Qwen3Executor {
    fn forward_sample(
        &mut self,
        plan: &BatchPlan,
        generated: &mut Vec<Vec<u32>>,
        drafts: &mut Vec<Vec<u32>>,
    ) -> Result<()> {
        let s = &self.stream;
        let n = plan.num_reqs();
        let t = plan.total_tokens();
        ensure!(t <= self.model.max_tokens, "batch of {t} tokens > max_tokens {}", self.model.max_tokens);
        ensure!(
            plan.spec_tokens.iter().all(|d| d.is_empty()),
            "qwen3 executor: speculative decoding not supported yet"
        );
        let v = self.model.cfg.vocab_size as usize;
        let nrows = plan.logits_rows.len();

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
        };
        let f = Forward { input_ids: &ids, positions: &pos, slot_mapping: &slots, meta, logits_rows: &rows };
        self.model.forward(s, &self.gemm, &f, &mut self.logits)?;
        k::argmax_f32_n(s, &self.logits.data, &mut self.argmax.data, nrows, v)?;
        let am = s.dtoh_sync_copy(&self.argmax.data)?;
        for i in 0..n {
            let r = plan.logits_row_start[i + 1] - 1;
            generated.push(vec![am[r] as u32]);
            drafts.push(Vec::new());
        }
        Ok(())
    }
}
