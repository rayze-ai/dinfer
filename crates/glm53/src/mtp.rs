//! GLM-5.3 MTP（nextn=1，权重 `model.layers.{N}.*`，N = num_hidden_layers）：
//!
//!   x      = eh_proj( [ enorm(embed(next_token)) | hnorm(prev_hidden) ] )     // prev_hidden = 主干最后一层输出（norm 前）
//!   y      = DecoderLayer_N(x)                                                  // 带自己的 MLA + MoE + （full/shared）索引器
//!   logits = lm_head( shared_head.norm(y) )                                     // lm_head / embed 与主干共享
//!
//! 对照 vLLM `deepseek_mtp.py::DeepSeekMultiTokenPredictorLayer`（`index_share_for_mtp_iteration=True`：MTP 层若是 shared
//! 索引器层，复用主干最后一个 full 层写入共享缓冲的 top-k；若是 full 层则自己算）。
//! `positions == 0` 的 embedding 置零（vLLM `torch.where(positions == 0, 0, inputs_embeds)`）。
//!
//! 投机解码用法（P6-d，每请求 draft 1 个）：主干 forward 后拿到 (prev_hidden, 本步采样 token)，MTP 以采样 token 为
//! next_token、在位置 pos+1 跑一行 → argmax 为 draft；下一步主干对 [sampled, draft] 两行做 verify。
//! MTP 层有自己的 KV cache（paged，同 block 表 / slot）；draft 行写进的 KV 若被拒绝需要回滚 —— 这里采用 "重算" 策略：
//! MTP 的 KV 永远只保存已确认 token（verify 后再写），draft 行用临时 slot。P6-d 先实现 **单步 draft + verify 的数学路径**
//! 与对拍，runner 侧接线见 `executor.rs`。

use crate::config::GlmMoeDsaConfig;
use crate::model::{FeedForward, Glm53Model};
use anyhow::{ensure, Context, Result};
use dinfer_kernels as k;
use dinfer_layers::mla::{AttnMeta, SparseCtx};
use dinfer_layers::{
    bf16, Gemm, Indexer, IndexerKv, Linear, MlaAttention, Mlp, Moe, PagedKv, RmsNorm, Stream, Tensor,
};

pub struct MtpLayer {
    pub enorm: RmsNorm,
    pub hnorm: RmsNorm,
    pub eh_proj: Linear, // [h, 2h]
    pub input_norm: RmsNorm,
    pub attn: MlaAttention,
    pub indexer: Option<(Indexer, IndexerKv)>,
    pub post_norm: RmsNorm,
    pub mlp: FeedForward,
    pub head_norm: RmsNorm,
    pub kv: PagedKv,
    pub layer_idx: u32,
    // scratch
    emb: Tensor<bf16>,
    cat: Tensor<bf16>,
    x: Tensor<bf16>,
    normed: Tensor<bf16>,
    attn_out: Tensor<bf16>,
    mlp_out: Tensor<bf16>,
    rows: Tensor<bf16>,
    final_normed: Tensor<bf16>,
    logits_bf16: Tensor<bf16>,
    max_tokens: usize,
}

impl MtpLayer {
    pub fn load(
        stream: &Stream,
        src: &dinfer_layers::WeightSource,
        cfg: &GlmMoeDsaConfig,
        max_tokens: usize,
        num_blocks: usize,
        block_size: usize,
    ) -> Result<Self> {
        ensure!(cfg.num_nextn_predict_layers == 1, "MTP: only num_nextn_predict_layers == 1 supported");
        let l = cfg.num_hidden_layers;
        let p = format!("model.layers.{l}");
        let h = cfg.hidden_size as usize;
        let mla = Glm53Model::mla_config(cfg);
        let eps = cfg.rms_norm_eps;
        let enorm = RmsNorm::load(stream, src, &format!("{p}.enorm"), eps)?;
        let hnorm = RmsNorm::load(stream, src, &format!("{p}.hnorm"), eps)?;
        let eh_proj = Linear::load(stream, src, &format!("{p}.eh_proj"))?;
        ensure!(eh_proj.out_features == h && eh_proj.in_features == 2 * h, "MTP eh_proj shape");
        let attn = MlaAttention::load(stream, src, &format!("{p}.self_attn"), mla.clone(), max_tokens)
            .context("MTP attn")?;
        let indexer = if cfg.dsa && cfg.has_indexer(l) {
            let idx_cfg = Glm53Model::indexer_config(cfg, num_blocks * block_size);
            Some((
                Indexer::load(stream, src, &format!("{p}.self_attn.indexer"), idx_cfg.clone(), max_tokens)
                    .context("MTP indexer")?,
                IndexerKv::new(stream, num_blocks, block_size, idx_cfg.head_dim)?,
            ))
        } else {
            None
        };
        let mlp = if cfg.is_moe_layer(l) || cfg.first_k_dense_replace < cfg.num_hidden_layers {
            // 真模型 MTP 层是 MoE；tiny 跟主干一致（first_k_dense_replace=1 → MoE）
            FeedForward::Moe(Box::new(
                Moe::load(stream, src, &format!("{p}.mlp"), Glm53Model::moe_config(cfg)?, max_tokens)
                    .context("MTP moe")?,
            ))
        } else {
            FeedForward::Dense(Mlp::load(stream, src, &format!("{p}.mlp"), max_tokens).context("MTP mlp")?)
        };
        let v = cfg.vocab_size as usize;
        Ok(Self {
            enorm,
            hnorm,
            eh_proj,
            input_norm: RmsNorm::load(stream, src, &format!("{p}.input_layernorm"), eps)?,
            attn,
            indexer,
            post_norm: RmsNorm::load(stream, src, &format!("{p}.post_attention_layernorm"), eps)?,
            mlp,
            head_norm: RmsNorm::load(stream, src, &format!("{p}.shared_head.norm"), eps)?,
            kv: PagedKv::new(stream, num_blocks, block_size, mla.entry())?,
            layer_idx: l,
            emb: Tensor::zeros(stream, &[max_tokens, h])?,
            cat: Tensor::zeros(stream, &[max_tokens, 2 * h])?,
            x: Tensor::zeros(stream, &[max_tokens, h])?,
            normed: Tensor::zeros(stream, &[max_tokens, h])?,
            attn_out: Tensor::zeros(stream, &[max_tokens, h])?,
            mlp_out: Tensor::zeros(stream, &[max_tokens, h])?,
            rows: Tensor::zeros(stream, &[max_tokens, h])?,
            final_normed: Tensor::zeros(stream, &[max_tokens, h])?,
            logits_bf16: Tensor::zeros(stream, &[max_tokens, v])?,
            max_tokens,
        })
    }

    /// 一次 MTP 前向：`next_ids[t]` 是第 t 行的输入 token（= 主干该行位置的下一个 token），`prev_hidden[t]` 是主干
    /// 最后一层在该行的输出（norm 前，bf16 [t, h]），`positions` 与主干同（该行的位置）。写本层 KV（`slot_mapping`），
    /// 对 `logits_rows` 行输出 draft logits（fp32）。`main` 用来借 embed / lm_head / 共享 top-k 缓冲。
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        stream: &Stream,
        gemm: &Gemm,
        main: &mut Glm53Model,
        next_ids: &Tensor<i32>,
        prev_hidden: &Tensor<bf16>,
        positions: &Tensor<i32>,
        slot_mapping: &Tensor<i64>,
        meta: &AttnMeta<'_>,
        logits_rows: &Tensor<i32>,
        logits: &mut Tensor<f32>,
    ) -> Result<()> {
        let t = next_ids.numel();
        ensure!(t <= self.max_tokens, "mtp: {t} tokens > max {}", self.max_tokens);
        let h = main.cfg.hidden_size as usize;
        let v = main.cfg.vocab_size as usize;
        let eps = main.cfg.rms_norm_eps;
        // 1. emb = embed(next); positions == 0 行置零；enorm
        k::embedding_bf16_n(stream, &next_ids.data, &main.embed.table.data, &mut self.emb.data, t, h)?;
        k::zero_rows_where_pos0_bf16(stream, &mut self.emb.data, &positions.data, t, h)?;
        k::rmsnorm_bf16_n(stream, &self.emb.data, &self.enorm.w.data, &mut self.normed.data, t, h, eps)?;
        k::copy_cols_bf16(stream, &self.normed.data, &mut self.cat.data, t, h, 0, 2 * h, 0, h)?;
        // 2. hnorm(prev_hidden) → cat[:, h:]
        k::rmsnorm_bf16_n(stream, &prev_hidden.data, &self.hnorm.w.data, &mut self.normed.data, t, h, eps)?;
        k::copy_cols_bf16(stream, &self.normed.data, &mut self.cat.data, t, h, 0, 2 * h, h, h)?;
        // 3. x = eh_proj(cat)
        gemm.forward_bf16(&self.cat.data, &self.eh_proj.w.data, None, &mut self.x.data, t, h, 2 * h)?;
        // 4. decoder 层（与主干同构）：normed = norm(x) → attn → x += attn → normed = norm(x) → mlp → x += mlp
        k::rmsnorm_bf16_n(stream, &self.x.data, &self.input_norm.w.data, &mut self.normed.data, t, h, eps)?;
        let (topk_buf, topk_k, decode_cu) = main.sparse_parts();
        let sparse = topk_buf.map(|tk| SparseCtx {
            indexer: self.indexer.as_mut().map(|(i, c)| (i, c)),
            topk: tk,
            topk_k,
            decode_cu,
            topk_override: None,
            topk_self: None,
        });
        self.attn.forward(
            stream,
            gemm,
            t,
            &self.normed,
            positions,
            slot_mapping,
            &mut self.kv,
            meta,
            sparse,
            &mut self.attn_out,
        )?;
        k::fused_add_rmsnorm_bf16_n(
            stream,
            &self.attn_out.data,
            &mut self.x.data,
            &self.post_norm.w.data,
            &mut self.normed.data,
            t,
            h,
            eps,
        )?;
        self.mlp.forward(stream, gemm, t, &self.normed, &mut self.mlp_out)?;
        k::add_inplace_bf16_n(stream, &self.mlp_out.data, &mut self.x.data, t * h)?;
        // 5. shared head：norm → lm_head（共享）→ fp32
        let rows = logits_rows.numel();
        k::gather_rows_bf16(stream, &self.x.data, &mut self.rows.data, &logits_rows.data, rows, h)?;
        k::rmsnorm_bf16_n(
            stream,
            &self.rows.data,
            &self.head_norm.w.data,
            &mut self.final_normed.data,
            rows,
            h,
            eps,
        )?;
        gemm.forward_bf16(
            &self.final_normed.data,
            &main.lm_head.w.data,
            None,
            &mut self.logits_bf16.data,
            rows,
            v,
            h,
        )?;
        k::bf16_to_f32_n(stream, &self.logits_bf16.data, &mut logits.data, rows * v)?;
        Ok(())
    }
}
