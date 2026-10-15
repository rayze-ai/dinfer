//! GLM-5.3 主干（P3 dense MLA + P4 MoE）：embed → N × (RMSNorm → MLA → +res → RMSNorm → FFN → +res) → norm → lm_head。
//! 前 `first_k_dense_replace` 层 FFN 为 dense MLP，其后为 MoE（HF `DeepseekV3MoE`）。
//! DSA 索引器 / MTP 在 P5 加。权重名按 HF：`model.layers.N.self_attn.*`、`model.layers.N.mlp.*`、
//! `model.embed_tokens`、`model.norm`、`lm_head`。

use crate::config::GlmMoeDsaConfig;
use anyhow::{ensure, Context, Result};
use dinfer_kernels as k;
use dinfer_layers::mla::AttnMeta;
use dinfer_layers::mla::SparseCtx;
use dinfer_layers::{
    bf16, Embedding, Gemm, Indexer, IndexerConfig, IndexerKv, Linear, MlaAttention, MlaConfig, Mlp, Moe,
    MoeConfig, PagedKv, RmsNorm, RopeStyle, Stream, Tensor, WeightSource,
};

/// 每层一份、堆上只建一次，变体大小差异无关紧要（Moe 已 Box，Dense 不值得再包一层）。
#[allow(clippy::large_enum_variant)]
pub enum FeedForward {
    Dense(Mlp),
    Moe(Box<Moe>),
}

impl FeedForward {
    fn forward(
        &mut self,
        stream: &Stream,
        gemm: &Gemm,
        t: usize,
        x: &Tensor<bf16>,
        out: &mut Tensor<bf16>,
    ) -> Result<()> {
        match self {
            FeedForward::Dense(m) => m.forward(stream, gemm, t, x, out),
            FeedForward::Moe(m) => m.forward(stream, gemm, t, x, out),
        }
    }
}

pub struct DecoderLayer {
    pub input_norm: RmsNorm,
    pub attn: MlaAttention,
    /// DSA `full` 层：自带索引器 + 索引 K cache；`shared` 层为 None（复用上一 full 层的 top-k）。
    pub indexer: Option<(Indexer, IndexerKv)>,
    pub post_norm: RmsNorm,
    pub mlp: FeedForward,
}

/// 对拍捕获：(layer_idx, topk [t*k] i32, 索引器分数 [t*stride] f32, stride)。
pub type TopkCapture = (usize, Vec<i32>, Vec<f32>, usize);
/// 对拍捕获：(layer_idx, 专家 id [t*K], 归一化权重 [t*K])。
pub type RoutingCapture = (usize, Vec<i32>, Vec<f32>);
/// 调试捕获：(layer_idx, attn_out [t*h] f32, 本层输出 hidden [t*h] f32)。
pub type LayerCapture = (usize, Vec<f32>, Vec<f32>);

pub struct Glm53Model {
    pub cfg: GlmMoeDsaConfig,
    pub embed: Embedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: RmsNorm,
    pub lm_head: Linear,
    pub kv: Vec<PagedKv>,
    pub max_tokens: usize,
    /// DSA：跨层共享的 top-k 索引缓冲 [max_tokens, index_topk]；None = 稠密模型（tiny dense/MoE）。
    pub topk: Option<Tensor<i32>>,
    pub topk_k: usize,
    /// 调试 / 对拍：若非空，每个 full 层 forward 后把 `topk[:t*k]` 与索引器分数拷回 host：
    /// (layer_idx, topk [t*k] i32, scores [t*stride] f32, stride)。
    pub capture_topk: Option<Vec<TopkCapture>>,
    /// 对拍隔离：若非空，每个 full 层算完自己的 top-k 后用这里的 (layer_idx, [t*k] i32) **覆盖** 共享缓冲，
    /// 让稀疏 attention 吃参考实现的索引（把 "索引器平局分叉" 与 "稀疏 attention 正确性" 分开验证）。
    pub topk_override: Option<Vec<(usize, Vec<i32>)>>,
    /// 对拍：注入覆盖时，每个 full 层自算的 top-k 落到这里（覆盖前拷出），`capture_topk` 记录的是它而不是注入值。
    topk_self: Option<Tensor<i32>>,
    /// 对拍：Some 时每个 MoE 层 push (layer, topk_ids[t*K], topk_w[t*K])。
    pub capture_routing: Option<Vec<RoutingCapture>>,
    /// 对拍：Some 时每层 push (layer, o_proj 输入 [t*H*v] f32)。
    pub capture_attn_pre: Option<Vec<(usize, Vec<f32>)>>,
    /// 调试：若非空，每层 forward 后记录 (layer, attn_out[:t*h] f32, hidden_after_layer[:t*h] f32)。
    pub capture_layers: Option<Vec<LayerCapture>>,
    decode_cu: Tensor<i32>,
    // scratch
    hidden: Tensor<bf16>,
    normed: Tensor<bf16>,
    attn_out: Tensor<bf16>,
    mlp_out: Tensor<bf16>,
    final_normed: Tensor<bf16>,
    logits_bf16: Tensor<bf16>,
}

pub struct Forward<'a> {
    pub input_ids: &'a Tensor<i32>,
    pub positions: &'a Tensor<i32>,
    pub slot_mapping: &'a Tensor<i64>,
    pub meta: AttnMeta<'a>,
    /// 要取 logits 的 token 行（prefill 取每请求最后一个；decode 取全部）。
    pub logits_rows: &'a Tensor<i32>,
}

impl Glm53Model {
    pub fn moe_config(cfg: &GlmMoeDsaConfig) -> Result<MoeConfig> {
        ensure!(cfg.scoring_func == "sigmoid", "unsupported scoring_func {}", cfg.scoring_func);
        ensure!(cfg.topk_method == "noaux_tc", "unsupported topk_method {}", cfg.topk_method);
        ensure!(cfg.n_group == 1 && cfg.topk_group == 1, "P4 MoE only supports n_group = topk_group = 1");
        Ok(MoeConfig {
            hidden: cfg.hidden_size as usize,
            num_experts: cfg.n_routed_experts as usize,
            topk: cfg.num_experts_per_tok as usize,
            num_shared: cfg.n_shared_experts as usize,
            renormalize: cfg.norm_topk_prob,
            scale: cfg.routed_scaling_factor,
        })
    }

    pub fn indexer_config(cfg: &GlmMoeDsaConfig, max_seq: usize) -> IndexerConfig {
        IndexerConfig {
            hidden: cfg.hidden_size as usize,
            q_lora: cfg.q_lora_rank as usize,
            heads: cfg.index_n_heads as usize,
            head_dim: cfg.index_head_dim as usize,
            rope: cfg.qk_rope_head_dim as usize,
            topk: cfg.index_topk as usize,
            rope_style: if cfg.indexer_rope_interleave { RopeStyle::Interleave } else { RopeStyle::Neox },
            rope_theta: cfg.rope_parameters.rope_theta,
            max_pos: cfg.max_position_embeddings as usize,
            max_seq,
        }
    }

    pub fn mla_config(cfg: &GlmMoeDsaConfig) -> MlaConfig {
        MlaConfig {
            hidden: cfg.hidden_size as usize,
            heads: cfg.num_attention_heads as usize,
            q_lora: cfg.q_lora_rank as usize,
            kv_lora: cfg.kv_lora_rank as usize,
            nope: cfg.qk_nope_head_dim as usize,
            rope: cfg.qk_rope_head_dim as usize,
            v_dim: cfg.v_head_dim as usize,
            rope_style: if cfg.rope_interleave { RopeStyle::Interleave } else { RopeStyle::Neox },
            rope_theta: cfg.rope_parameters.rope_theta,
            max_pos: cfg.max_position_embeddings as usize,
            rms_eps: cfg.rms_norm_eps,
        }
    }

    pub fn load(
        stream: &Stream,
        src: &WeightSource,
        cfg: GlmMoeDsaConfig,
        max_tokens: usize,
        num_blocks: usize,
        block_size: usize,
    ) -> Result<Self> {
        ensure!(cfg.moe_layer_freq == 1, "moe_layer_freq != 1 unsupported");
        let mla = Self::mla_config(&cfg);
        let has_moe = cfg.first_k_dense_replace < cfg.num_hidden_layers;
        let moe = if has_moe { Some(Self::moe_config(&cfg)?) } else { None };
        let embed = Embedding::load(stream, src, "model.embed_tokens")?;
        let max_seq = num_blocks * block_size;
        let idx_cfg = Self::indexer_config(&cfg, max_seq);
        if cfg.dsa {
            ensure!(cfg.has_indexer(0), "DSA: layer 0 must be a full indexer layer");
        }
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
        let mut kv = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for l in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{l}");
            layers.push(DecoderLayer {
                input_norm: RmsNorm::load(stream, src, &format!("{p}.input_layernorm"), cfg.rms_norm_eps)?,
                attn: MlaAttention::load(stream, src, &format!("{p}.self_attn"), mla.clone(), max_tokens)
                    .with_context(|| format!("layer {l} attn"))?,
                indexer: if cfg.dsa && cfg.has_indexer(l) {
                    Some((
                        Indexer::load(
                            stream,
                            src,
                            &format!("{p}.self_attn.indexer"),
                            idx_cfg.clone(),
                            max_tokens,
                        )
                        .with_context(|| format!("layer {l} indexer"))?,
                        IndexerKv::new(stream, num_blocks, block_size, idx_cfg.head_dim)?,
                    ))
                } else {
                    None
                },
                post_norm: RmsNorm::load(
                    stream,
                    src,
                    &format!("{p}.post_attention_layernorm"),
                    cfg.rms_norm_eps,
                )?,
                mlp: if l < cfg.first_k_dense_replace {
                    FeedForward::Dense(
                        Mlp::load(stream, src, &format!("{p}.mlp"), max_tokens)
                            .with_context(|| format!("layer {l} mlp"))?,
                    )
                } else {
                    FeedForward::Moe(Box::new(
                        Moe::load(stream, src, &format!("{p}.mlp"), moe.clone().unwrap(), max_tokens)
                            .with_context(|| format!("layer {l} moe"))?,
                    ))
                },
            });
            kv.push(PagedKv::new(stream, num_blocks, block_size, mla.entry())?);
        }
        let norm = RmsNorm::load(stream, src, "model.norm", cfg.rms_norm_eps)?;
        let lm_head = Linear::load(stream, src, "lm_head")?;
        let h = cfg.hidden_size as usize;
        let v = cfg.vocab_size as usize;
        Ok(Self {
            embed,
            layers,
            norm,
            lm_head,
            kv,
            max_tokens,
            topk: if cfg.dsa {
                Some(Tensor::zeros(stream, &[max_tokens, cfg.index_topk as usize])?)
            } else {
                None
            },
            topk_k: cfg.index_topk as usize,
            capture_topk: None,
            topk_override: None,
            topk_self: None,
            capture_routing: None,
            capture_attn_pre: None,
            capture_layers: None,
            decode_cu: Tensor::from_vec(stream, (0..=max_tokens as i32).collect(), &[max_tokens + 1])?,
            hidden: Tensor::zeros(stream, &[max_tokens, h])?,
            normed: Tensor::zeros(stream, &[max_tokens, h])?,
            attn_out: Tensor::zeros(stream, &[max_tokens, h])?,
            mlp_out: Tensor::zeros(stream, &[max_tokens, h])?,
            final_normed: Tensor::zeros(stream, &[max_tokens, h])?,
            logits_bf16: Tensor::zeros(stream, &[max_tokens, v])?,
            cfg,
        })
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
            // normed = rmsnorm(hidden)
            k::rmsnorm_bf16_n(
                stream,
                &self.hidden.data,
                &layer.input_norm.w.data,
                &mut self.normed.data,
                t,
                h,
                eps,
            )?;
            let is_full = layer.indexer.is_some();
            let override_rows = match (&self.topk_override, is_full) {
                (Some(ov), true) => ov.iter().find(|(l, _)| *l == li).map(|(_, v)| v.clone()),
                _ => None,
            };
            if override_rows.is_some() && self.topk_self.is_none() {
                self.topk_self = Some(Tensor::zeros(stream, &[self.max_tokens, self.topk_k])?);
            }
            let sparse = self.topk.as_mut().map(|tk| SparseCtx {
                indexer: layer.indexer.as_mut().map(|(i, c)| (i, c)),
                topk: tk,
                topk_k: self.topk_k,
                decode_cu: &self.decode_cu,
                topk_override: override_rows.as_deref(),
                topk_self: self.topk_self.as_mut(),
            });
            layer.attn.forward(
                stream,
                gemm,
                t,
                &self.normed,
                f.positions,
                f.slot_mapping,
                &mut self.kv[li],
                &f.meta,
                sparse,
                &mut self.attn_out,
            )?;
            if let (Some(cap), Some(tk), Some((idx, _))) =
                (self.capture_topk.as_mut(), self.topk.as_ref(), layer.indexer.as_ref())
            {
                // 注入模式下记录自算值（覆盖前拷出的），否则共享缓冲里就是自算值
                let src = match (&override_rows, self.topk_self.as_ref()) {
                    (Some(_), Some(own)) => own,
                    _ => tk,
                };
                let all = src.to_vec(stream)?;
                let (sc, stride) = idx.last_scores(stream, t)?;
                cap.push((li, all[..t * self.topk_k].to_vec(), sc, stride));
            }
            // hidden += attn_out; normed = rmsnorm(hidden)
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
            // hidden += mlp_out（用 fused_add_rmsnorm 的 res 更新顺带做下一层的 input norm 会更省，但这里保持清晰：下一轮循环顶部做 norm）
            k::add_inplace_bf16_n(stream, &self.mlp_out.data, &mut self.hidden.data, t * h)?;
            if let Some(cap) = self.capture_layers.as_mut() {
                let a = self.attn_out.to_vec_f32(stream)?[..t * h].to_vec();
                let hd = self.hidden.to_vec_f32(stream)?[..t * h].to_vec();
                cap.push((li, a, hd));
            }
            if let Some(cap) = self.capture_attn_pre.as_mut() {
                cap.push((li, layer.attn.last_attn_pre(stream, t)?));
            }
            if let (Some(cap), FeedForward::Moe(m)) = (self.capture_routing.as_mut(), &layer.mlp) {
                let (ids, w) = m.last_routing(stream, t)?;
                cap.push((li, ids, w));
            }
        }
        // final norm on selected rows → lm_head → fp32（gather 到 attn_out 作临时，norm 到 final_normed）
        let rows = f.logits_rows.numel();
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
