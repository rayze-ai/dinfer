//! GLM-5.3 结构的稠密子集（P3）：embed → N × (RMSNorm → MLA → +res → RMSNorm → MLP → +res) → norm → lm_head。
//! MoE / DSA 索引器 / MTP 在 P4 / P5 加。权重名按 HF：`model.layers.N.self_attn.*`、`model.layers.N.mlp.*`、
//! `model.embed_tokens`、`model.norm`、`lm_head`。

use crate::config::GlmMoeDsaConfig;
use anyhow::{ensure, Context, Result};
use dinfer_kernels as k;
use dinfer_layers::mla::AttnMeta;
use dinfer_layers::{
    bf16, Embedding, Gemm, Linear, MlaAttention, MlaConfig, Mlp, PagedKv, RmsNorm, RopeStyle, Stream, Tensor,
    WeightSource,
};

pub struct DecoderLayer {
    pub input_norm: RmsNorm,
    pub attn: MlaAttention,
    pub post_norm: RmsNorm,
    pub mlp: Mlp,
}

pub struct Glm53Dense {
    pub cfg: GlmMoeDsaConfig,
    pub embed: Embedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: RmsNorm,
    pub lm_head: Linear,
    pub kv: Vec<PagedKv>,
    pub max_tokens: usize,
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

impl Glm53Dense {
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
        ensure!(
            cfg.first_k_dense_replace >= cfg.num_hidden_layers,
            "P3 dense model requires all layers dense (first_k_dense_replace={} < layers={})",
            cfg.first_k_dense_replace,
            cfg.num_hidden_layers
        );
        let mla = Self::mla_config(&cfg);
        let embed = Embedding::load(stream, src, "model.embed_tokens")?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
        let mut kv = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for l in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{l}");
            layers.push(DecoderLayer {
                input_norm: RmsNorm::load(stream, src, &format!("{p}.input_layernorm"), cfg.rms_norm_eps)?,
                attn: MlaAttention::load(stream, src, &format!("{p}.self_attn"), mla.clone(), max_tokens)
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
            layer.attn.forward(
                stream,
                gemm,
                t,
                &self.normed,
                f.positions,
                f.slot_mapping,
                &mut self.kv[li],
                &f.meta,
                &mut self.attn_out,
            )?;
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
