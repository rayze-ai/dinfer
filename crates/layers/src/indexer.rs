//! DSA（DeepSeek Sparse Attention）索引器参考实现。对照 transformers 5.19 `GlmMoeDsaIndexer.forward`
//! （`/data/ref/hf_tf519/.../modeling_glm_moe_dsa.py:184-242`）与 vLLM `deepseek_v32/attention.py`：
//! ```text
//! q_idx = wq_b(q_c).view(T, H_idx, D_idx)        // q_c = q_a_layernorm(q_a_proj(x))，与主 MLA 共用
//! k_idx = k_norm(wk(x))                          // [T, D_idx]，LayerNorm(带 bias, eps 1e-6, fp32 计算)
//! rope(q_idx[..., :rope], k_idx[:rope])          // interleave；注意旋转的是 **前** rope 维（主 MLA 是尾部）
//! k_idx → paged 索引 K cache（与主 KV 共用 block_table / slot_mapping）
//! w = weights_proj(x).float() · H_idx^-0.5       // [T, H_idx]
//! score[t, j] = Σ_h w[t,h] · relu(q_idx[t,h]·k_idx[j] · D_idx^-0.5)，j ≤ 可见位置
//! topk_indices[t] = top-k(score[t])（相等取小下标；可见数 < k 时补 -1）
//! ```
//! 真机版（fp8 索引 cache + 4 字节 scale、DeepGEMM fp8_mqa_logits、radix top-k）放 P10。

use crate::nn::{Linear, Rope};
use crate::tensor::Tensor;
use crate::weights::WeightSource;
use anyhow::{ensure, Context, Result};
use dinfer_kernels::{self as k, Gemm, RopeStyle, Stream};
use half::bf16;

#[derive(Debug, Clone)]
pub struct IndexerConfig {
    pub hidden: usize,
    pub q_lora: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub rope: usize,
    pub topk: usize,
    pub rope_style: RopeStyle,
    pub rope_theta: f64,
    pub max_pos: usize,
    /// 索引器 K cache 能放的最大序列长（= num_blocks * block_size 的上界，用于 score 缓冲）。
    pub max_seq: usize,
}

/// 一层的 paged 索引 K cache：[num_blocks, block_size, head_dim] bf16。
pub struct IndexerKv {
    pub cache: Tensor<bf16>,
    pub num_blocks: usize,
    pub block_size: usize,
}

impl IndexerKv {
    pub fn new(stream: &Stream, num_blocks: usize, block_size: usize, head_dim: usize) -> Result<Self> {
        Ok(Self {
            cache: Tensor::zeros(stream, &[num_blocks, block_size, head_dim])?,
            num_blocks,
            block_size,
        })
    }
}

/// 索引器需要的 batch 描述：每请求 query 行划分 + 本步结束后总长 + block 表。
pub struct IndexerMeta<'a> {
    pub cu_seqlens_q: &'a Tensor<i32>,
    pub seq_lens: &'a Tensor<i32>,
    pub block_table: &'a Tensor<i32>,
    pub num_reqs: usize,
    pub max_blocks: usize,
}

pub struct Indexer {
    pub cfg: IndexerConfig,
    pub wq_b: Linear, // [H_idx*D_idx, q_lora]
    pub wk: Linear,   // [D_idx, hidden]
    pub k_norm_w: Tensor<bf16>,
    pub k_norm_b: Tensor<bf16>,
    pub weights_proj: Linear, // [H_idx, hidden]
    pub rope: Rope,
    // scratch
    q: Tensor<bf16>,
    k_raw: Tensor<bf16>,
    k: Tensor<bf16>,
    w_bf16: Tensor<bf16>,
    w: Tensor<f32>,
    score: Tensor<f32>,
    vis: Tensor<i32>,
    max_tokens: usize,
}

impl Indexer {
    /// `prefix` 形如 `model.layers.0.self_attn.indexer`。
    pub fn load(
        stream: &Stream,
        src: &WeightSource,
        prefix: &str,
        cfg: IndexerConfig,
        max_tokens: usize,
    ) -> Result<Self> {
        let wq_b = Linear::load(stream, src, &format!("{prefix}.wq_b"))?;
        ensure!(
            wq_b.out_features == cfg.heads * cfg.head_dim && wq_b.in_features == cfg.q_lora,
            "indexer wq_b shape [{}, {}]",
            wq_b.out_features,
            wq_b.in_features
        );
        let wk = Linear::load(stream, src, &format!("{prefix}.wk"))?;
        ensure!(wk.out_features == cfg.head_dim && wk.in_features == cfg.hidden, "indexer wk shape");
        let k_norm_w = src.load_bf16(stream, &format!("{prefix}.k_norm.weight"))?;
        let k_norm_b = src.load_bf16(stream, &format!("{prefix}.k_norm.bias"))?;
        ensure!(k_norm_w.numel() == cfg.head_dim && k_norm_b.numel() == cfg.head_dim, "indexer k_norm shape");
        let weights_proj = Linear::load(stream, src, &format!("{prefix}.weights_proj"))?;
        ensure!(
            weights_proj.out_features == cfg.heads && weights_proj.in_features == cfg.hidden,
            "indexer weights_proj shape"
        );
        ensure!(cfg.rope <= cfg.head_dim && cfg.rope % 2 == 0, "indexer rope dim");
        let rope = Rope::new(stream, cfg.rope, cfg.max_pos, cfg.rope_theta, cfg.rope_style)?;
        let t = max_tokens;
        Ok(Self {
            q: Tensor::zeros(stream, &[t, cfg.heads, cfg.head_dim])?,
            k_raw: Tensor::zeros(stream, &[t, cfg.head_dim])?,
            k: Tensor::zeros(stream, &[t, cfg.head_dim])?,
            w_bf16: Tensor::zeros(stream, &[t, cfg.heads])?,
            w: Tensor::zeros(stream, &[t, cfg.heads])?,
            score: Tensor::zeros(stream, &[t, cfg.max_seq])?,
            vis: Tensor::zeros(stream, &[t])?,
            wq_b,
            wk,
            k_norm_w,
            k_norm_b,
            weights_proj,
            rope,
            cfg,
            max_tokens,
        })
    }

    /// 调试 / 对拍：上一次 forward 的 index_scores（[t, stride] f32 行主序，causal 外为 -inf）。
    pub fn last_scores(&self, stream: &Stream, t: usize) -> Result<(Vec<f32>, usize)> {
        let stride = self.cfg.max_seq;
        let all = self.score.to_vec(stream)?;
        Ok((all[..t * stride].to_vec(), stride))
    }

    /// 写索引 K cache 并算本步每个 query 行的 top-k → `topk_out[:t*topk]`（i32，-1 空槽）。
    /// `x`：本层输入（已 input_layernorm）；`q_c`：主 MLA 的 q_a_layernorm 输出。
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        stream: &Stream,
        gemm: &Gemm,
        t: usize,
        x: &Tensor<bf16>,
        q_c: &Tensor<bf16>,
        positions: &Tensor<i32>,
        slot_mapping: &Tensor<i64>,
        kv: &mut IndexerKv,
        meta: &IndexerMeta<'_>,
        topk_out: &mut Tensor<i32>,
    ) -> Result<()> {
        let c = &self.cfg;
        let (h, d, kk) = (c.heads, c.head_dim, c.topk);
        ensure!(t <= self.max_tokens, "indexer: {t} tokens > max {}", self.max_tokens);
        ensure!(topk_out.numel() >= t * kk, "indexer: topk_out too small");
        // 1. q_idx = wq_b(q_c) → [t, H, D]；k_idx = k_norm(wk(x)) → [t, D]
        gemm.forward_bf16(&q_c.data, &self.wq_b.w.data, None, &mut self.q.data, t, h * d, c.q_lora)?;
        gemm.forward_bf16(&x.data, &self.wk.w.data, None, &mut self.k_raw.data, t, d, c.hidden)?;
        k::layernorm_bf16_n(
            stream,
            &self.k_raw.data,
            &self.k_norm_w.data,
            &self.k_norm_b.data,
            &mut self.k.data,
            t,
            d,
            1e-6,
        )?;
        // 2. rope（前 rope 维）：q 当 H 个 head、k 当 1 个 head，head_dim = D
        k::rope_bf16_n(
            stream,
            &mut self.q.data,
            Some(&mut self.k.data),
            &positions.data,
            &self.rope.cos_sin.data,
            t,
            h,
            1,
            d,
            c.rope,
            c.rope_style,
        )?;
        // 3. k → paged 索引 cache
        k::mla_cache_write_bf16(
            stream,
            &self.k.data,
            &mut kv.cache.data,
            &slot_mapping.data,
            t,
            d,
            kv.block_size,
        )?;
        // 4. w = weights_proj(x).f32 · H^-0.5
        gemm.forward_bf16(&x.data, &self.weights_proj.w.data, None, &mut self.w_bf16.data, t, h, c.hidden)?;
        k::bf16_to_f32_n(stream, &self.w_bf16.data, &mut self.w.data, t * h)?;
        k::scale_f32_n(stream, &mut self.w.data, t * h, (h as f32).powf(-0.5))?;
        // 5. score + top-k
        // score 行步长固定为 cfg.max_seq（= cache 容量 num_blocks·block_size，与 last_scores 一致）。
        // block_table 宽度 meta.max_blocks 只是本 batch 最长请求的块数（runner 按需给），kernel 只在 j < vis ≤ seq_len
        // 范围内读 bt[j / block_size]，所以这里只需保证每个请求 seq_len ≤ max_seq —— 由 scheduler 的 max_model_len /
        // cache 容量保证，不另做设备侧检查。
        let max_seq = c.max_seq;
        k::dsa_row_vis_i32(
            stream,
            &meta.seq_lens.data,
            &meta.cu_seqlens_q.data,
            &mut self.vis.data,
            meta.num_reqs,
            t,
        )?;
        k::dsa_index_score_f32(
            stream,
            &self.q.data,
            &self.w.data,
            &kv.cache.data,
            &meta.block_table.data,
            &meta.seq_lens.data,
            &meta.cu_seqlens_q.data,
            &mut self.score.data,
            meta.num_reqs,
            t,
            h,
            d,
            kv.block_size,
            meta.max_blocks,
            max_seq,
            (d as f32).powf(-0.5),
        )?;
        k::dsa_topk_i32(stream, &self.score.data, &self.vis.data, &mut topk_out.data, t, max_seq, kk)
            .context("indexer topk")
    }
}
