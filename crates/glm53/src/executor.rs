//! `Glm53Model` 的 `ModelExecutor` 实现：把 `BatchPlan` 上传为设备张量 → forward → greedy argmax → D2H。
//! 最小版：每 step 重新 H2D 全部输入（无 pinned / 双缓冲），采样只支持 greedy。
//!
//! 投机解码（P6-d，MTP nextn=1，greedy）：
//! - 带 draft 的请求本步输入 `[…base tokens, draft]`，logits 取最后 `1 + 草稿数` 行；逐行 argmax：第 j 行的 argmax 是
//!   "看完第 j 个输入后的预测"；若它等于下一个 draft 则接受继续，否则停止。生成 = 接受的 draft + 1 个 bonus（第一个
//!   不匹配行 / 最后一行的 argmax）。与 vLLM `rejection_sampler` 的 greedy 路径同义。
//! - MTP draft：主干 forward 后，MTP 层以 "该请求本步最后一个被接受的 token 的下一个 token"（= bonus）为输入、
//!   previous_hidden 取对应行的主干输出、位置 = 该行位置 + 1，跑 1 行 → argmax 为下一步 draft。
//! - KV 回滚：主干 KV 写了 draft 行；被拒绝的行所在 slot 下一步会被同位置的真 token 重写（slot = block·bs + pos），
//!   paged cache 按位置寻址，不需要显式回滚。MTP 层 KV：它的输入序列比主干 "右移一位"，第 pos 行的 K 来自 token[pos+1]；
//!   同理按位置重写即可。拒绝后主干最后一步的 MTP 行（位置 pos+1）用的是 bonus token，和下一步真序列一致。
//! - 中间 chunk（prefill 未到末尾）：MTP 用 `next_known`（序列里已有的下一 token）做输入，让 MTP KV 与主干同步推进；
//!   draft 不产出。

use crate::model::{Forward, Glm53Model};
use crate::mtp::MtpLayer;
use anyhow::{ensure, Result};
use dinfer_core::runner::{BatchMode, BatchPlan, ModelExecutor};
use dinfer_kernels as k;
use dinfer_layers::mla::AttnMeta;
use dinfer_layers::{Gemm, Stream, Tensor};

pub struct Glm53Executor {
    pub stream: Stream,
    pub gemm: Gemm,
    pub model: Glm53Model,
    pub mtp: Option<MtpLayer>,
    logits: Tensor<f32>,
    argmax: Tensor<i32>,
    mtp_logits: Tensor<f32>,
    mtp_argmax: Tensor<i32>,
    /// 诊断：上一步每请求 (草稿数, 接受数)。
    pub last_accept: Vec<(usize, usize)>,
}

impl Glm53Executor {
    pub fn new(stream: Stream, gemm: Gemm, model: Glm53Model, mtp: Option<MtpLayer>) -> Result<Self> {
        let v = model.cfg.vocab_size as usize;
        let max_tokens = model.max_tokens;
        Ok(Self {
            logits: Tensor::zeros(&stream, &[max_tokens, v])?,
            argmax: Tensor::zeros(&stream, &[max_tokens])?,
            mtp_logits: Tensor::zeros(&stream, &[max_tokens, v])?,
            mtp_argmax: Tensor::zeros(&stream, &[max_tokens])?,
            last_accept: Vec::new(),
            stream,
            gemm,
            model,
            mtp,
        })
    }

    /// 上一 step 的主干 logits（fp32，[num_logits_rows, vocab]）；测试 / 对拍用。
    pub fn last_logits(&self, num_rows: usize) -> Result<Vec<f32>> {
        let v = self.model.cfg.vocab_size as usize;
        let all = self.logits.to_vec(&self.stream)?;
        Ok(all[..num_rows * v].to_vec())
    }

    /// 上一 step 的 MTP draft logits（fp32，[num_reqs, vocab]）；测试 / 对拍用。
    pub fn last_mtp_logits(&self, num_rows: usize) -> Result<Vec<f32>> {
        let v = self.model.cfg.vocab_size as usize;
        let all = self.mtp_logits.to_vec(&self.stream)?;
        Ok(all[..num_rows * v].to_vec())
    }

    fn attn_meta<'a>(
        plan: &BatchPlan,
        cu: &'a Tensor<i32>,
        bt: &'a Tensor<i32>,
        seq: &'a Tensor<i32>,
        n: usize,
    ) -> AttnMeta<'a> {
        match plan.mode {
            BatchMode::Prefill => AttnMeta::Prefill {
                cu_seqlens: cu,
                num_reqs: n,
                block_table: bt,
                seq_lens: seq,
                max_blocks: plan.max_blocks,
            },
            BatchMode::Decode => {
                AttnMeta::Decode { block_table: bt, seq_lens: seq, max_blocks: plan.max_blocks }
            }
            BatchMode::Paged => AttnMeta::Paged {
                cu_seqlens_q: cu,
                block_table: bt,
                seq_lens: seq,
                num_reqs: n,
                max_blocks: plan.max_blocks,
            },
        }
    }
}

impl ModelExecutor for Glm53Executor {
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
        let v = self.model.cfg.vocab_size as usize;
        let nrows = plan.logits_rows.len();

        let ids = Tensor::from_vec(s, plan.input_ids.clone(), &[t])?;
        let pos = Tensor::from_vec(s, plan.positions.clone(), &[t])?;
        let slots = Tensor::from_vec(s, plan.slot_mapping.clone(), &[t])?;
        let rows = Tensor::from_vec(s, plan.logits_rows.clone(), &[nrows])?;
        let cu = Tensor::from_vec(s, plan.cu_seqlens.clone(), &[n + 1])?;
        let seq = Tensor::from_vec(s, plan.seq_lens.clone(), &[n])?;
        let bt = Tensor::from_vec(s, plan.block_table.clone(), &[n, plan.max_blocks.max(1)])?;
        let meta = Self::attn_meta(plan, &cu, &bt, &seq, n);
        let f = Forward { input_ids: &ids, positions: &pos, slot_mapping: &slots, meta, logits_rows: &rows };
        self.model.forward(s, &self.gemm, &f, &mut self.logits)?;
        k::argmax_f32_n(s, &self.logits.data, &mut self.argmax.data, nrows, v)?;
        let am = s.dtoh_sync_copy(&self.argmax.data)?;

        // verify（greedy）：每请求行区间 [r0, r1)；行 j 的 argmax 是 "看完该行输入后" 的预测
        self.last_accept.clear();
        let mut bonus_pos: Vec<i32> = Vec::with_capacity(n); // MTP 输入所在位置（= 最后被接受行的位置 + 1）
        let mut bonus_tok: Vec<i32> = Vec::with_capacity(n);
        let mut hidden_row: Vec<i32> = Vec::with_capacity(n); // previous_hidden 取主干哪一行
        for i in 0..n {
            let (r0, r1) = (plan.logits_row_start[i], plan.logits_row_start[i + 1]);
            let spec = &plan.spec_tokens[i];
            let mut gen = Vec::with_capacity(spec.len() + 1);
            let mut accepted = 0usize;
            // 第 r0 行对应 base 的最后一个 token；预测与 spec[0] 比
            let mut last_row = r0;
            for (j, &d) in spec.iter().enumerate() {
                let pred = am[r0 + j] as u32;
                if pred == d {
                    accepted += 1;
                    gen.push(d);
                    last_row = r0 + j + 1;
                } else {
                    break;
                }
            }
            // bonus：第一个不匹配行（或全接受时最后一行）的 argmax
            let bonus = am[last_row] as u32;
            gen.push(bonus);
            debug_assert!(last_row < r1);
            self.last_accept.push((spec.len(), accepted));
            // MTP 输入：主干该行的位置 + 1，token = bonus（中间 chunk 则用 next_known）
            let row_in_batch = plan.logits_rows[last_row] as usize;
            let p_row = plan.positions[row_in_batch];
            let nk = plan.next_known[i];
            bonus_pos.push(p_row + 1);
            bonus_tok.push(if nk >= 0 { nk } else { bonus as i32 });
            hidden_row.push(row_in_batch as i32);
            generated.push(gen);
        }

        // MTP draft（每请求 1 行）
        if let Some(mtp) = self.mtp.as_mut() {
            // slot：位置 p_row+1 的 slot（block 表里一定有：scheduler 为 draft 预留了 lookahead 块）
            let bs = mtp.kv.block_size as i64;
            let mut mslots = Vec::with_capacity(n);
            for (i, &bp) in bonus_pos.iter().enumerate() {
                let p = bp as i64;
                let b = plan.block_table[i * plan.max_blocks + (p / bs) as usize];
                mslots.push(if b >= 0 { b as i64 * bs + p % bs } else { -1 });
            }
            let next_ids = Tensor::from_vec(s, bonus_tok.clone(), &[n])?;
            let mpos = Tensor::from_vec(s, bonus_pos.clone(), &[n])?;
            let mslot = Tensor::from_vec(s, mslots, &[n])?;
            // previous_hidden：主干最后一层输出里对应行 → 连续 [n, h]
            let h = self.model.cfg.hidden_size as usize;
            let hrows = Tensor::from_vec(s, hidden_row.clone(), &[n])?;
            let mut prev = Tensor::<dinfer_layers::bf16>::zeros(s, &[n, h])?;
            k::gather_rows_bf16(s, &self.model.last_hidden().data, &mut prev.data, &hrows.data, n, h)?;
            // MTP 层 attention：每请求 1 行 decode，seq_len = 该行位置 + 1
            let mseq: Vec<i32> = bonus_pos.iter().map(|p| p + 1).collect();
            let mseq_t = Tensor::from_vec(s, mseq, &[n])?;
            let mrows = Tensor::from_vec(s, (0..n as i32).collect(), &[n])?;
            let mmeta = AttnMeta::Decode { block_table: &bt, seq_lens: &mseq_t, max_blocks: plan.max_blocks };
            mtp.forward(
                s,
                &self.gemm,
                &mut self.model,
                &next_ids,
                &prev,
                &mpos,
                &mslot,
                &mmeta,
                &mrows,
                &mut self.mtp_logits,
            )?;
            k::argmax_f32_n(s, &self.mtp_logits.data, &mut self.mtp_argmax.data, n, v)?;
            let dm = s.dtoh_sync_copy(&self.mtp_argmax.data)?;
            // 中间 chunk 不产 draft（序列还没到末尾）
            for (&d, &nk) in dm.iter().take(n).zip(&plan.next_known) {
                drafts.push(if nk >= 0 { Vec::new() } else { vec![d as u32] });
            }
        } else {
            drafts.extend((0..n).map(|_| Vec::new()));
        }
        Ok(())
    }
}
