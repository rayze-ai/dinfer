//! Qwen3 主干：embed → N × (RMSNorm → GQA attention → +res → RMSNorm → SwiGLU → +res) → norm → lm_head（tied）。
//! attention：q/k/v 投影 → per-head q_norm / k_norm（RMSNorm over head_dim）→ neox RoPE（整个 head_dim）→
//! 写 paged K/V cache → GQA attention（prefill 展开 / paged）→ o_proj。对照 HF `modeling_qwen3.py` 与 vLLM `qwen3.py`。
//!
//! 权重名（真 Qwen3-0.6B safetensors）：`model.embed_tokens`、`model.layers.{l}.{input_layernorm, self_attn.{q,k,v,o}_proj,
//! self_attn.{q,k}_norm, post_attention_layernorm, mlp.{gate,up,down}_proj}`、`model.norm`、`lm_head`（tied 时文件里也有一份）。

use crate::config::Qwen3Config;
use anyhow::{ensure, Context, Result};
use dinfer_core::runner::BatchMode;
use dinfer_kernels as k;
use dinfer_layers::{
    bf16, Embedding, Gemm, Linear, Mlp, RmsNorm, Rope, RopeStyle, Stream, Tensor, WeightSource,
};

/// 一层的 paged K/V cache：[num_blocks, block_size, kv_heads, 2, head_dim] bf16。
pub struct PagedKv {
    pub cache: Tensor<bf16>,
    pub num_blocks: usize,
    pub block_size: usize,
}

pub struct Attention {
    pub qkv: Linear, // 融合 [q | k | v]：[(H + 2·Hkv)·D, hidden]
    pub o_proj: Linear,
    pub q_norm: RmsNorm, // over head_dim
    pub k_norm: RmsNorm,
    // scratch（[max_tokens, ...]）
    qkv_buf: Tensor<bf16>,
    q: Tensor<bf16>,
    kk: Tensor<bf16>,
    v: Tensor<bf16>,
    attn: Tensor<bf16>,
}

pub struct DecoderLayer {
    pub input_norm: RmsNorm,
    pub attn: Attention,
    pub post_norm: RmsNorm,
    pub mlp: Mlp,
}

/// 一步 attention 的 batch 描述（runner 的 BatchPlan 上传后的设备张量）。
pub struct AttnMeta<'a> {
    pub mode: BatchMode,
    pub cu_seqlens: &'a Tensor<i32>, // [B+1]，本步每请求 query 数前缀和
    pub seq_lens: &'a Tensor<i32>,   // [B]，本步后每请求总长
    pub block_table: &'a Tensor<i32>,
    pub num_reqs: usize,
    pub max_blocks: usize,
}

pub struct Forward<'a> {
    pub input_ids: &'a Tensor<i32>,
    pub positions: &'a Tensor<i32>,
    pub slot_mapping: &'a Tensor<i64>,
    pub meta: AttnMeta<'a>,
    pub logits_rows: &'a Tensor<i32>,
}

/// 调试捕获：(layer_idx, attn_out [t*h] f32, 本层输出 hidden [t*h] f32)。
pub type LayerCapture = (usize, Vec<f32>, Vec<f32>);

pub struct Qwen3Model {
    pub cfg: Qwen3Config,
    pub embed: Embedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: RmsNorm,
    pub lm_head: Linear,
    pub rope: Rope,
    pub kv: Vec<PagedKv>,
    pub max_tokens: usize,
    /// 一步最多取多少行 logits（logits / final_normed 缓冲按它分配）。
    pub max_logits_rows: usize,
    /// 调试：若非空，每层 forward 后记录 (layer, attn_out[:t*h] f32, hidden_after_layer[:t*h] f32)。
    pub capture_layers: Option<Vec<LayerCapture>>,
    hidden: Tensor<bf16>,
    normed: Tensor<bf16>,
    attn_out: Tensor<bf16>,
    mlp_out: Tensor<bf16>,
    final_normed: Tensor<bf16>,
    logits_bf16: Tensor<bf16>,
}

impl Attention {
    fn load(
        stream: &Stream,
        src: &WeightSource,
        p: &str,
        cfg: &Qwen3Config,
        max_tokens: usize,
    ) -> Result<Self> {
        let (h, hk, d) =
            (cfg.num_attention_heads as usize, cfg.num_key_value_heads as usize, cfg.head_dim as usize);
        let hidden = cfg.hidden_size as usize;
        let (qs, q) = src.load_f32_host(&format!("{p}.q_proj.weight"))?;
        let (ks, kw) = src.load_f32_host(&format!("{p}.k_proj.weight"))?;
        let (vs, vw) = src.load_f32_host(&format!("{p}.v_proj.weight"))?;
        ensure!(
            qs == vec![h * d, hidden] && ks == vec![hk * d, hidden] && vs == ks,
            "qkv shapes {qs:?} {ks:?} {vs:?}"
        );
        let mut cat = Vec::with_capacity(q.len() + kw.len() + vw.len());
        cat.extend_from_slice(&q);
        cat.extend_from_slice(&kw);
        cat.extend_from_slice(&vw);
        let qkv = Linear::new(Tensor::from_f32(stream, &cat, &[(h + 2 * hk) * d, hidden])?, None)?;
        let o_proj = Linear::load(stream, src, &format!("{p}.o_proj"))?;
        ensure!(o_proj.in_features == h * d && o_proj.out_features == hidden, "o_proj shape");
        let t = max_tokens;
        Ok(Self {
            qkv,
            o_proj,
            q_norm: RmsNorm::load(stream, src, &format!("{p}.q_norm"), cfg.rms_norm_eps)?,
            k_norm: RmsNorm::load(stream, src, &format!("{p}.k_norm"), cfg.rms_norm_eps)?,
            qkv_buf: Tensor::zeros(stream, &[t, (h + 2 * hk) * d])?,
            q: Tensor::zeros(stream, &[t, h, d])?,
            kk: Tensor::zeros(stream, &[t, hk, d])?,
            v: Tensor::zeros(stream, &[t, hk, d])?,
            attn: Tensor::zeros(stream, &[t, h, d])?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        stream: &Stream,
        gemm: &Gemm,
        cfg: &Qwen3Config,
        rope: &Rope,
        t: usize,
        x: &Tensor<bf16>,
        positions: &Tensor<i32>,
        slot_mapping: &Tensor<i64>,
        kv: &mut PagedKv,
        meta: &AttnMeta<'_>,
        out: &mut Tensor<bf16>,
    ) -> Result<()> {
        let (h, hk, d) =
            (cfg.num_attention_heads as usize, cfg.num_key_value_heads as usize, cfg.head_dim as usize);
        let hidden = cfg.hidden_size as usize;
        let eps = cfg.rms_norm_eps;
        let qkv_w = (h + 2 * hk) * d;
        // 1. qkv = x · W^T → 拆成 q [t,H,D] / k [t,Hkv,D] / v [t,Hkv,D]
        gemm.forward_bf16(&x.data, &self.qkv.w.data, None, &mut self.qkv_buf.data, t, qkv_w, hidden)?;
        k::copy_cols_bf16(stream, &self.qkv_buf.data, &mut self.q.data, t, qkv_w, 0, h * d, 0, h * d)?;
        k::copy_cols_bf16(stream, &self.qkv_buf.data, &mut self.kk.data, t, qkv_w, h * d, hk * d, 0, hk * d)?;
        k::copy_cols_bf16(
            stream,
            &self.qkv_buf.data,
            &mut self.v.data,
            t,
            qkv_w,
            (h + hk) * d,
            hk * d,
            0,
            hk * d,
        )?;
        // 2. per-head RMSNorm：把 [t, H, D] 看成 [t·H, D] 行
        k::rmsnorm_bf16_n(stream, &self.q.data, &self.q_norm.w.data, &mut self.attn.data, t * h, d, eps)?;
        std::mem::swap(&mut self.q, &mut self.attn);
        k::rmsnorm_bf16_n(
            stream,
            &self.kk.data,
            &self.k_norm.w.data,
            &mut self.qkv_buf.data,
            t * hk,
            d,
            eps,
        )?;
        k::copy_cols_bf16(stream, &self.qkv_buf.data, &mut self.kk.data, t, hk * d, 0, hk * d, 0, hk * d)?;
        // 3. neox RoPE，整个 head_dim
        k::rope_bf16_n(
            stream,
            &mut self.q.data,
            Some(&mut self.kk.data),
            &positions.data,
            &rope.cos_sin.data,
            t,
            h,
            hk,
            d,
            d,
            RopeStyle::Neox,
        )?;
        // 4. 写 cache
        k::gqa_cache_write_bf16(
            stream,
            &self.kk.data,
            &self.v.data,
            &mut kv.cache.data,
            &slot_mapping.data,
            t,
            hk,
            d,
            kv.block_size,
        )?;
        // 5. attention
        let scale = (d as f32).powf(-0.5);
        match meta.mode {
            BatchMode::Prefill => k::gqa_prefill_bf16(
                stream,
                &self.q.data,
                &self.kk.data,
                &self.v.data,
                &mut self.attn.data,
                &meta.cu_seqlens.data,
                meta.num_reqs,
                h,
                hk,
                d,
                scale,
            )?,
            BatchMode::Decode | BatchMode::Paged => k::gqa_paged_bf16(
                stream,
                &self.q.data,
                &kv.cache.data,
                &meta.block_table.data,
                &meta.seq_lens.data,
                &meta.cu_seqlens.data,
                &mut self.attn.data,
                meta.num_reqs,
                h,
                hk,
                d,
                kv.block_size,
                meta.max_blocks,
                scale,
            )?,
        }
        // 6. o_proj
        gemm.forward_bf16(&self.attn.data, &self.o_proj.w.data, None, &mut out.data, t, hidden, h * d)
    }
}

impl Qwen3Model {
    pub fn load(
        stream: &Stream,
        src: &WeightSource,
        cfg: Qwen3Config,
        max_tokens: usize,
        num_blocks: Option<usize>,
        block_size: usize,
    ) -> Result<Self> {
        // 测试 / 对拍：logits 行数上限 = max_tokens（允许取全部位置的 logits）
        Self::load_with_budget(
            stream,
            src,
            cfg,
            max_tokens,
            max_tokens,
            num_blocks,
            block_size,
            0.9,
            512 << 20,
        )
    }

    /// 每个 KV block 的字节数（全部层）：`layers × block_size × kv_heads × 2(K,V) × head_dim × 2(bf16)`。
    pub fn kv_bytes_per_block(cfg: &Qwen3Config, block_size: usize) -> usize {
        cfg.num_hidden_layers as usize
            * block_size
            * cfg.num_key_value_heads as usize
            * 2
            * cfg.head_dim as usize
            * 2
    }

    /// `max_tokens`：一步最多多少 token（激活 scratch 按它分配）；`max_logits_rows`：一步最多取多少行 logits
    /// （服务 = max_num_seqs：每请求只要最后一行；对拍 = max_tokens）。logits 按 vocab 151936 宽，不能按 max_tokens 分。
    /// `num_blocks = None` 时按显存算：权重 + 激活 scratch 分配完后读 `free`，
    /// `budget = min(free, total × gpu_memory_utilization − 已用) − reserve − executor 的 fp32 logits 副本`，
    /// `num_blocks = budget / bytes_per_block`。KV 最后分配，所以 free 已经扣掉了权重与 scratch
    /// （等价 vLLM profile run 的作用，但不用真跑一次 forward）。
    #[allow(clippy::too_many_arguments)]
    pub fn load_with_budget(
        stream: &Stream,
        src: &WeightSource,
        cfg: Qwen3Config,
        max_tokens: usize,
        max_logits_rows: usize,
        num_blocks: Option<usize>,
        block_size: usize,
        gpu_memory_utilization: f32,
        reserve_bytes: usize,
    ) -> Result<Self> {
        let hidden = cfg.hidden_size as usize;
        let (hk, d) = (cfg.num_key_value_heads as usize, cfg.head_dim as usize);
        let embed = Embedding::load(stream, src, "model.embed_tokens")?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for l in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{l}");
            layers.push(DecoderLayer {
                input_norm: RmsNorm::load(stream, src, &format!("{p}.input_layernorm"), cfg.rms_norm_eps)?,
                attn: Attention::load(stream, src, &format!("{p}.self_attn"), &cfg, max_tokens)
                    .with_context(|| format!("layer {l} attn"))?,
                post_norm: RmsNorm::load(
                    stream,
                    src,
                    &format!("{p}.post_attention_layernorm"),
                    cfg.rms_norm_eps,
                )?,
                mlp: Mlp::load(stream, src, &format!("{p}.mlp"), max_tokens)
                    .with_context(|| format!("layer {l} mlp"))?,
            });
        }
        let norm = RmsNorm::load(stream, src, "model.norm", cfg.rms_norm_eps)?;
        let lm_head = if src.contains("lm_head.weight") {
            Linear::load(stream, src, "lm_head")?
        } else {
            ensure!(cfg.tie_word_embeddings, "no lm_head.weight and not tied");
            Linear::load(stream, src, "model.embed_tokens")?
        };
        let rope =
            Rope::new(stream, d, cfg.max_position_embeddings as usize, cfg.rope_theta, RopeStyle::Neox)?;
        let v = cfg.vocab_size as usize;
        // 激活 scratch 先分配，KV 最后：这样 free 已含权重 + scratch 的占用
        let hidden_t = Tensor::zeros(stream, &[max_tokens, hidden])?;
        let normed = Tensor::zeros(stream, &[max_tokens, hidden])?;
        let attn_out = Tensor::zeros(stream, &[max_tokens, hidden])?;
        let mlp_out = Tensor::zeros(stream, &[max_tokens, hidden])?;
        let final_normed = Tensor::zeros(stream, &[max_logits_rows, hidden])?;
        let logits_bf16 = Tensor::zeros(stream, &[max_logits_rows, v])?;
        // executor 还会分配 fp32 logits [max_logits_rows, vocab] + 采样工作区 + logprobs 原始副本（各 ≤ 同大小）
        let exec_scratch = 3 * max_logits_rows * v * 4;

        let per_block = Self::kv_bytes_per_block(&cfg, block_size);
        let num_blocks = match num_blocks {
            Some(n) => {
                ensure!(n > 0, "num_blocks must be > 0");
                tracing::info!(
                    num_blocks = n,
                    tokens = n * block_size,
                    kv_gib = n as f64 * per_block as f64 / (1u64 << 30) as f64,
                    "kv_cache: fixed"
                );
                n
            }
            None => {
                stream.synchronize()?;
                let (free, total) = stream.mem_info()?;
                let used = total.saturating_sub(free);
                let cap = (total as f64 * gpu_memory_utilization as f64) as usize;
                let budget = free.min(cap.saturating_sub(used)).saturating_sub(reserve_bytes + exec_scratch);
                let n = budget / per_block;
                ensure!(
                    n >= 1,
                    "not enough GPU memory for KV cache: free {:.2} GiB, cap {:.0}% of {:.2} GiB, reserve {:.2} GiB",
                    free as f64 / (1u64 << 30) as f64,
                    gpu_memory_utilization * 100.0,
                    total as f64 / (1u64 << 30) as f64,
                    (reserve_bytes + exec_scratch) as f64 / (1u64 << 30) as f64
                );
                tracing::info!(
                    num_blocks = n,
                    tokens = n * block_size,
                    kv_gib = n as f64 * per_block as f64 / (1u64 << 30) as f64,
                    available_gib = free as f64 / (1u64 << 30) as f64,
                    total_gib = total as f64 / (1u64 << 30) as f64,
                    utilization = gpu_memory_utilization,
                    "kv_cache: auto"
                );
                n
            }
        };
        let mut kv = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for _ in 0..cfg.num_hidden_layers {
            kv.push(PagedKv {
                cache: Tensor::zeros(stream, &[num_blocks, block_size, hk, 2, d])?,
                num_blocks,
                block_size,
            });
        }
        Ok(Self {
            embed,
            layers,
            norm,
            lm_head,
            rope,
            kv,
            max_tokens,
            max_logits_rows,
            capture_layers: None,
            hidden: hidden_t,
            normed,
            attn_out,
            mlp_out,
            final_normed,
            logits_bf16,
            cfg,
        })
    }

    /// 实际分配的 KV block 数。
    pub fn num_blocks(&self) -> usize {
        self.kv.first().map_or(0, |k| k.num_blocks)
    }

    /// 上一次 forward 后最后一层输出（norm 前）[t, hidden]。
    pub fn last_hidden(&self) -> &Tensor<bf16> {
        &self.hidden
    }

    /// 返回 logits（fp32，[num_logits_rows, vocab]）。
    pub fn forward(
        &mut self,
        stream: &Stream,
        gemm: &Gemm,
        f: &Forward<'_>,
        logits: &mut Tensor<f32>,
    ) -> Result<()> {
        let t = f.input_ids.numel();
        ensure!(t <= self.max_tokens, "forward: {t} tokens > max {}", self.max_tokens);
        let h = self.cfg.hidden_size as usize;
        let v = self.cfg.vocab_size as usize;
        let eps = self.cfg.rms_norm_eps;
        k::embedding_bf16_n(stream, &f.input_ids.data, &self.embed.table.data, &mut self.hidden.data, t, h)?;
        for (li, layer) in self.layers.iter_mut().enumerate() {
            k::rmsnorm_bf16_n(
                stream,
                &self.hidden.data,
                &layer.input_norm.w.data,
                &mut self.normed.data,
                t,
                h,
                eps,
            )?;
            layer.attn.forward(
                stream,
                gemm,
                &self.cfg,
                &self.rope,
                t,
                &self.normed,
                f.positions,
                f.slot_mapping,
                &mut self.kv[li],
                &f.meta,
                &mut self.attn_out,
            )?;
            k::fused_add_rmsnorm_bf16_n(
                stream,
                &self.attn_out.data,
                &mut self.hidden.data,
                &layer.post_norm.w.data,
                &mut self.normed.data,
                t,
                h,
                eps,
            )?;
            layer.mlp.forward(stream, gemm, t, &self.normed, &mut self.mlp_out)?;
            k::add_inplace_bf16_n(stream, &self.mlp_out.data, &mut self.hidden.data, t * h)?;
            if let Some(cap) = self.capture_layers.as_mut() {
                let a = self.attn_out.to_vec_f32(stream)?[..t * h].to_vec();
                let hd = self.hidden.to_vec_f32(stream)?[..t * h].to_vec();
                cap.push((li, a, hd));
            }
        }
        let rows = f.logits_rows.numel();
        ensure!(
            rows <= self.max_logits_rows,
            "{rows} logits rows > max_logits_rows {}",
            self.max_logits_rows
        );
        k::gather_rows_bf16(
            stream,
            &self.hidden.data,
            &mut self.attn_out.data,
            &f.logits_rows.data,
            rows,
            h,
        )?;
        k::rmsnorm_bf16_n(
            stream,
            &self.attn_out.data,
            &self.norm.w.data,
            &mut self.final_normed.data,
            rows,
            h,
            eps,
        )?;
        gemm.forward_bf16(
            &self.final_normed.data,
            &self.lm_head.w.data,
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
