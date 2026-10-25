//! `KaonModel` 的 `ModelExecutor` 实现（第一版，单卡、greedy、无 CUDA graph / 投机 / 异步）。
//!
//! 照 glm53 的最小 executor：每步把 `BatchPlan` 上传成设备张量 → `KaonModel::forward`（全模型前向，
//! 输出 `[tokens, vocab]` bf16 logits，softcap 已施加）→ gather 要取的 logits 行 → bf16→f32 → argmax → D2H。
//! kaon3 的 KV cache 由模型内部逐层按 `head_dim` 分配（`alloc_kv`），scheduler 只管逻辑块号；所有层共用同一套
//! block_size / num_blocks，block_table 的物理块号直接喂进每层的 paged attention。
//!
//! Gemma4 无递归状态（不像 qwen35 的 GDN），所以不需要 state slot / 快照；KV 按位置寻址，被抢占重算直接重写。

use crate::layer::{ActBuf, LayerScratch};
use crate::model::{KaonModel, StepMeta};
use anyhow::{ensure, Result};
use dinfer_core::runner::{BatchPlan, ModelExecutor};
use dinfer_kernels::{self as k, bf16};
use dinfer_layers::{Gemm, Stream, Tensor};

pub struct KaonExecutor {
    pub stream: Stream,
    pub gemm: Gemm,
    pub model: KaonModel,
    scratch: LayerScratch,
    acts: ActBuf,
    /// 全 batch logits（模型 forward 对每个 token 都算 lm_head）：[max_tokens, vocab] bf16。
    logits: Tensor<bf16>,
    /// forward 的 hidden dump 缓冲（本版不 dump，但 forward 签名要求）。
    hidden: Tensor<bf16>,
    /// 采样行 gather 后的 logits：[max_rows, vocab] bf16 → f32。
    gathered: Tensor<bf16>,
    gathered_f32: Tensor<f32>,
    argmax: Tensor<i32>,
    max_tokens: usize,
    max_rows: usize,
}

impl KaonExecutor {
    pub fn new(stream: Stream, gemm: Gemm, model: KaonModel, max_tokens: usize, max_rows: usize) -> Result<Self> {
        let v = model.cfg.vocab_size;
        let h = model.cfg.hidden;
        let scratch = LayerScratch::new(&stream, &model.cfg, max_tokens)?;
        let acts = ActBuf::new(&stream, &model.cfg)?;
        Ok(Self {
            logits: Tensor::uninit(&stream, &[max_tokens, v])?,
            hidden: Tensor::uninit(&stream, &[max_tokens, h])?,
            gathered: Tensor::uninit(&stream, &[max_rows, v])?,
            gathered_f32: Tensor::zeros(&stream, &[max_rows, v])?,
            argmax: Tensor::zeros(&stream, &[max_rows])?,
            scratch,
            acts,
            max_tokens,
            max_rows,
            stream,
            gemm,
            model,
        })
    }
}

impl ModelExecutor for KaonExecutor {
    fn forward_sample(
        &mut self,
        plan: &BatchPlan,
        generated: &mut Vec<Vec<u32>>,
        drafts: &mut Vec<Vec<u32>>,
    ) -> Result<()> {
        // 引擎线程 != 构建线程：绑一次 ctx，否则首个 kernel 报 CUDA_ERROR_INVALID_CONTEXT。
        self.stream.bind_ctx()?;
        let s = &self.stream;
        let n = plan.num_reqs();
        let t = plan.total_tokens();
        let nrows = plan.logits_rows.len();
        let v = self.model.cfg.vocab_size;
        ensure!(t <= self.max_tokens, "batch of {t} tokens > max_tokens {}", self.max_tokens);
        ensure!(nrows <= self.max_rows, "rows {nrows} > max_rows {}", self.max_rows);
        ensure!(!plan.sample.any_mask, "kaon3 executor: guided decoding 未接（第一版不做）");

        let ids = Tensor::from_vec(s, plan.input_ids.clone(), &[t])?;
        let positions = Tensor::from_vec(s, plan.positions.clone(), &[t])?;
        let slot_mapping = Tensor::from_vec(s, plan.slot_mapping.clone(), &[t])?;
        let cu_seqlens = Tensor::from_vec(s, plan.cu_seqlens.clone(), &[n + 1])?;
        let seq_lens = Tensor::from_vec(s, plan.seq_lens.clone(), &[n])?;
        let block_table = Tensor::from_vec(s, plan.block_table.clone(), &[n, plan.max_blocks.max(1)])?;
        let rows = Tensor::from_vec(s, plan.logits_rows.clone(), &[nrows])?;
        // prefill / decode / paged 的 attention 都用 max 本步每请求 token 数做 max query 长度。
        let max_seqlen = plan.num_tokens.iter().copied().max().unwrap_or(0).max(1) as usize;

        let meta = StepMeta {
            mode: plan.mode,
            cu_seqlens: &cu_seqlens,
            seq_lens: &seq_lens,
            block_table: &block_table,
            positions: &positions,
            slot_mapping: &slot_mapping,
            num_reqs: n,
            max_blocks: plan.max_blocks,
            max_seqlen,
        };

        self.model.forward(
            s,
            &self.gemm,
            &dinfer_layers::DefaultOps,
            &ids,
            &meta,
            &mut self.scratch,
            &mut self.acts,
            &mut self.logits,
            &mut self.hidden,
            t,
            None,
            None,
            None,
        )?;

        // 取采样行 → f32 → greedy argmax。
        k::ops::gather_rows_bf16(s, &self.logits.data, &mut self.gathered.data, &rows.data, nrows, v)?;
        k::ops::bf16_to_f32_n(s, &self.gathered.data, &mut self.gathered_f32.data, nrows * v)?;
        k::argmax_f32_n(s, &self.gathered_f32.data, &mut self.argmax.data, nrows, v)?;
        let am = s.dtoh_sync_copy(&self.argmax.data)?;

        // 每请求本步的 logits 行区间（无投机 → 恰 1 行）：取该行 argmax 为下一个 token。
        generated.clear();
        drafts.clear();
        for i in 0..n {
            let r0 = plan.logits_row_start[i];
            generated.push(vec![am[r0] as u32]);
            drafts.push(Vec::new());
        }
        Ok(())
    }
}
