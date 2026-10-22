//! Kaon-V3 模型主体：embedding → 30 层 `Gemma4Layer` → final norm → tied lm_head → softcap。
//!
//! 结构事实（`docs/loops/kaon-v3.md`「Gemma4 前向语义」）：
//! - embedding 后乘 `sqrt(hidden_size)`（vLLM 的 `normalizer`）；
//! - `tie_word_embeddings = true`：lm_head 与 embedding 共用同一张表；
//! - `final_logit_softcapping = 30`：`logits = tanh(logits/30)·30`；
//! - 每层 KV cache 的 `head_dim` 不同（滑窗 256 / 全注意力 512），必须**逐层分配**。

use crate::config::KaonConfig;
use crate::layer::{ActBuf, Gemma4Layer, LayerScratch, Ropes, StepCtx};
use crate::weights::{self, QWeight};
use anyhow::{ensure, Result};
use dinfer_kernels::{self as k, bf16, Gemm, Stream};
use dinfer_layers::{Embedding, Norm as _, OpSet, Tensor, WeightSource};

/// 一层的 paged KV cache：[num_blocks, block_size, nkv, 2, head_dim]。
pub struct LayerKv {
    pub cache: Tensor<bf16>,
    pub nkv: usize,
    pub head_dim: usize,
}

impl LayerKv {
    pub fn new(stream: &Stream, cfg: &KaonConfig, l: usize, num_blocks: usize, block_size: usize) -> Result<Self> {
        let (head_dim, nkv, _) = cfg.attn_shape(l);
        Ok(Self {
            cache: Tensor::zeros(stream, &[num_blocks, block_size, nkv, 2, head_dim])?,
            nkv,
            head_dim,
        })
    }
}

pub struct KaonModel {
    pub cfg: KaonConfig,
    pub embed: Embedding,
    pub layers: Vec<Gemma4Layer>,
    pub final_norm: Tensor<bf16>,
    pub kv: Vec<LayerKv>,
    pub ropes: Ropes,
    pub block_size: usize,
    pub num_blocks: usize,
    /// embedding 后的缩放系数（`sqrt(hidden_size)`）。
    pub embed_scale: f32,
}

impl KaonModel {
    /// 只装权重结构（不含 KV cache，KV 由 `alloc_kv` 单独按显存预算分配）。
    pub fn load(stream: &Stream, src: &WeightSource, cfg: KaonConfig) -> Result<Self> {
        // FP8 KV 时 vLLM 的 attention 是 Q/K/V/P 全 e4m3（Triton FP8 路径）；P 的量化在 kernel 内，按进程设一次
        k::ops::gqa_set_p_fp8(cfg.kv_fp8)?;
        let embed_name = weights::EMBED;
        let embed = Embedding::load(stream, src, &format!("{}{}", weights::TEXT_PREFIX, "embed_tokens"))?;
        ensure!(embed.vocab == cfg.vocab_size, "vocab {} != {}", embed.vocab, cfg.vocab_size);

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for l in 0..cfg.num_layers {
            layers.push(Gemma4Layer::load(stream, src, &cfg, l)?);
        }
        let final_norm = src.load_bf16(stream, weights::NORM)?;
        ensure!(final_norm.numel() == cfg.hidden, "final_norm 长度 {} != hidden {}", final_norm.numel(), cfg.hidden);
        let _ = embed_name;
        let ropes = Ropes::new(stream, &cfg)?;
        Ok(Self {
            // vLLM：normalizer 以模型 dtype 存（"avoid mixed-precision drift"），sqrt(2816)=53.07 → bf16 53.0
            embed_scale: half::bf16::from_f32((cfg.hidden as f32).sqrt()).to_f32(),
            cfg,
            embed,
            layers,
            final_norm,
            kv: Vec::new(),
            ropes,
            block_size: 0,
            num_blocks: 0,
        })
    }

    /// 按 `num_blocks` 分配每层 KV cache（逐层 head_dim 不同）。
    pub fn alloc_kv(&mut self, stream: &Stream, num_blocks: usize, block_size: usize) -> Result<()> {
        ensure!(block_size > 0 && num_blocks > 0, "kv: 参数非法");
        let mut kv = Vec::with_capacity(self.cfg.num_layers);
        for l in 0..self.cfg.num_layers {
            kv.push(LayerKv::new(stream, &self.cfg, l, num_blocks, block_size)?);
        }
        self.kv = kv;
        self.num_blocks = num_blocks;
        self.block_size = block_size;
        Ok(())
    }

    /// 全模型前向：返回 `[tokens, vocab]` logits（bf16，softcap 已施加）。
    ///
    /// `tokens` 为本步 token 数；`ids` 为对应 token id（长度 ≥ tokens）。
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        stream: &Stream,
        gemm: &Gemm,
        ops: &impl OpSet,
        ids: &Tensor<i32>,
        meta: &StepMeta<'_>,
        sc: &mut LayerScratch,
        acts: &mut ActBuf,
        logits: &mut Tensor<bf16>,
        hidden: &mut Tensor<bf16>,
        tokens: usize,
        // `Some(l)` 时把第 l 层之后的残差流拷进 `hidden`（逐层对拍用）。
        dump_layer: Option<usize>,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let h = cfg.hidden;

        // ---- embedding × sqrt(hidden) ----
        self.embed.forward(stream, ids, &mut sc.x)?;
        if self.embed_scale != 1.0 {
            k::ops::scale_bf16_n(stream, &mut sc.x.data, tokens * h, self.embed_scale)?;
        }

        // ---- 30 层 ----
        // KV cache 逐层不同（head_dim 256/512），写与读都要按层取。
        let mut kv_cache = std::mem::take(&mut self.kv);
        for (l, layer) in self.layers.iter().enumerate() {
            let mut ctx = StepCtx {
                mode: meta.mode,
                cu_seqlens: &meta.cu_seqlens,
                seq_lens: &meta.seq_lens,
                block_table: &meta.block_table,
                positions: &meta.positions,
                slot_mapping: &meta.slot_mapping,
                num_reqs: meta.num_reqs,
                max_blocks: meta.max_blocks,
                max_seqlen: meta.max_seqlen,
                block_size: self.block_size,
                kv_mut: &mut kv_cache[l].cache.data,
            };
            layer.forward(stream, gemm, ops, cfg, &self.ropes, sc, &mut ctx, tokens)?;
            if dump_layer == Some(l) {
                stream.dtod_copy_n(&sc.x.data, &mut hidden.data, tokens * h)?;
            }
        }

        self.kv = kv_cache;

        // ---- final norm ----
        let mut normed = Tensor::uninit(stream, &[tokens, h])?;
        ops.norm().rmsnorm(stream, &sc.x.data, &self.final_norm.data, &mut normed.data, tokens, h, cfg.eps)?;

        // ---- tied lm_head ----
        // vocab 262144、hidden 2816：一次全词表 GEMM。decode（tokens 少）时应换 split/分块策略，
        // 阶段 5 调优时再处理（`docs/loops/kaon-v3.md` 的 roofline 一节）。
        gemm.forward_bf16(
            &normed.data,
            &self.embed.table.data,
            None,
            &mut logits.data,
            tokens,
            cfg.vocab_size,
            h,
        )?;

        // ---- softcap: tanh(x/30)·30 ----
        k::ops::softcap_bf16_n(stream, &mut logits.data, tokens * cfg.vocab_size, cfg.softcap)?;
        let _ = hidden;
        Ok(())
    }
}

/// 一步的 batch 元数据（由 executor 上传）。
pub struct StepMeta<'a> {
    pub mode: dinfer_core::runner::BatchMode,
    pub cu_seqlens: &'a Tensor<i32>,
    pub seq_lens: &'a Tensor<i32>,
    pub block_table: &'a Tensor<i32>,
    pub positions: &'a Tensor<i32>,
    pub slot_mapping: &'a Tensor<i64>,
    pub num_reqs: usize,
    pub max_blocks: usize,
    pub max_seqlen: usize,
}

/// 权重清单核对：跑之前确认三件事（避免"编错 feature 空跑"这类假数据）。
pub fn check_weights(src: &WeightSource, cfg: &KaonConfig) -> Result<()> {
    ensure!(src.contains(weights::EMBED), "缺 embedding：{}", weights::EMBED);
    ensure!(src.contains(weights::NORM), "缺 final norm：{}", weights::NORM);
    let l0 = weights::layer_prefix(0);
    for n in [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "pre_feedforward_layernorm_2.weight",
        "post_feedforward_layernorm.weight",
        "post_feedforward_layernorm_1.weight",
        "post_feedforward_layernorm_2.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "router.proj.weight",
        "router.scale",
        "router.per_expert_scale",
        "layer_scalar",
    ] {
        ensure!(src.contains(&format!("{l0}.{n}")), "缺第 0 层的 {n}");
    }
    ensure!(src.contains(&format!("{l0}.experts.0.gate_proj.weight")), "缺第 0 层专家 0");
    ensure!(cfg.layer_types.len() == cfg.num_layers, "layer_types 长度不符");
    Ok(())
}

/// 权重格式统计（启动日志用）。
pub fn format_summary(src: &WeightSource, name: &str) -> &'static str {
    match name {
        _ if src.info(&format!("{name}_scale")).map(|i| i.shape.len() == 2).unwrap_or(false) => "NVFP4",
        _ => match src.info(name).map(|i| i.dtype.as_str()) {
            Some("F8_E4M3") => "FP8",
            Some("BF16") => "BF16",
            _ => "?",
        },
    }
}

/// 未使用但保留：确认 `QWeight` 的三态与加载分派一致（编译期契约）。
const _: fn(QWeight) = |_q| ();
