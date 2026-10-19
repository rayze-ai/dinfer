//! 解码层：`input_norm → attn(Gdn | Full) → +res → post_norm → SwiGLU → +res`。
//! 混合层类型由 `cfg.layer_types[l]` 决定（Qwen3.5-27B：每第 4 层全注意力，其余 GDN）。
use crate::config::{LayerType, Qwen35Config};
use crate::full_attn::{AttnMeta, FullAttention, PagedKv};
use crate::gdn::{GdnLayer, GdnMeta};
use crate::weights::Table;
use anyhow::Result;
use dinfer_core::runner::BatchMode;
use dinfer_layers::{bf16, Gemm, Mlp, OpSet, RmsNorm, Rope, Stream, Tensor, WeightSource};

/// 一层里注意力部分的两种实现。
pub enum Attn {
    Full(FullAttention),
    Gdn(GdnLayer),
}

impl Attn {
    pub fn is_full(&self) -> bool {
        matches!(self, Attn::Full(_))
    }
    /// 给全注意力层换一套激活缓冲（跨流并行用）；GDN 层不换（副流路径暂不支持 GDN）。
    pub fn swap_bufs(&mut self, qkv: &mut Tensor<bf16>, gate: &mut Tensor<bf16>, attn: &mut Tensor<bf16>) {
        if let Attn::Full(a) = self {
            a.swap_bufs(qkv, gate, attn);
        }
    }
}

pub struct DecoderLayer {
    pub input_norm: RmsNorm,
    pub attn: Attn,
    pub post_norm: RmsNorm,
    pub mlp: Mlp,
}

/// 一步 forward 的元数据（两种层共用一份上传，字段按层类型取用）。
pub struct StepMeta<'a> {
    pub mode: BatchMode,
    pub cu_seqlens: &'a Tensor<i32>,
    pub seq_lens: &'a Tensor<i32>,
    pub block_table: &'a Tensor<i32>,
    /// GDN 状态 slot（每请求一个）与"该 slot 是否已有历史"（新请求 = 0）。
    pub state_indices: &'a Tensor<i32>,
    pub has_initial: &'a Tensor<i32>,
    pub num_reqs: usize,
    pub max_blocks: usize,
    pub max_seqlen: usize,
}

impl<'a> StepMeta<'a> {
    pub fn attn_meta(&'a self) -> AttnMeta<'a> {
        AttnMeta {
            mode: self.mode,
            cu_seqlens: self.cu_seqlens,
            seq_lens: self.seq_lens,
            block_table: self.block_table,
            num_reqs: self.num_reqs,
            max_blocks: self.max_blocks,
            max_seqlen: self.max_seqlen,
        }
    }
    pub fn gdn_meta(&'a self) -> GdnMeta<'a> {
        GdnMeta {
            cu_seqlens: self.cu_seqlens,
            state_indices: self.state_indices,
            has_initial: self.has_initial,
            num_reqs: self.num_reqs,
            mode: self.mode,
        }
    }
}

impl DecoderLayer {
    pub fn load(
        stream: &Stream,
        src: &WeightSource,
        t: &Table,
        l: usize,
        cfg: &Qwen35Config,
        max_tokens: usize,
        num_slots: usize,
        tp: (usize, usize),
    ) -> Result<Self> {
        let (rank, world) = tp;
        let p = format!("layers.{l}");
        let attn = match cfg.layer_types[l] {
            LayerType::FullAttention => {
                Attn::Full(FullAttention::load(stream, src, t, l, cfg, max_tokens, tp)?)
            }
            LayerType::LinearAttention => Attn::Gdn(GdnLayer::load(stream, src, t, l, cfg, max_tokens, num_slots)?),
        };
        Ok(Self {
            input_norm: t.rmsnorm(stream, src, &format!("{p}.input_layernorm.weight"), cfg.rms_norm_eps)?,
            attn,
            post_norm: t.rmsnorm(stream, src, &format!("{p}.post_attention_layernorm.weight"), cfg.rms_norm_eps)?,
            mlp: Mlp::load_tp(stream, src, &t.prefix_of(&format!("{p}.mlp")), max_tokens, rank, world)?,
        })
    }

    /// 调试用：层类型名。
    pub fn kind(&self) -> &'static str {
        match self.attn {
            Attn::Full(_) => "full",
            Attn::Gdn(_) => "gdn",
        }
    }

    /// `normed` 是本层 attn 的输入（调用方已算好），`hidden` 是主干残差，`out` 本层 attn 输出。
    #[allow(clippy::too_many_arguments)]
    pub fn forward<O: OpSet>(
        &mut self,
        stream: &Stream,
        gemm: &Gemm,
        ops: &O,
        cfg: &Qwen35Config,
        rope: &Rope,
        t: usize,
        normed: &Tensor<bf16>,
        positions: &Tensor<i32>,
        slot_mapping: &Tensor<i64>,
        kv: &mut Option<PagedKv>,
        meta: &StepMeta<'_>,
        decode_scratch: &mut Tensor<f32>,
        out: &mut Tensor<bf16>,
    ) -> Result<()> {
        match (&mut self.attn, kv) {
            (Attn::Full(a), Some(kv)) => a.forward(
                stream,
                gemm,
                ops,
                cfg,
                rope,
                t,
                normed,
                positions,
                slot_mapping,
                kv,
                &meta.attn_meta(),
                decode_scratch,
                out,
            ),
            (Attn::Gdn(g), None) => g.forward(stream, gemm, ops.act(), cfg, t, normed, &meta.gdn_meta(), out),
            (Attn::Full(_), None) => anyhow::bail!("full attention layer without a KV cache slot"),
            (Attn::Gdn(_), Some(_)) => anyhow::bail!("GDN layer with a KV cache slot"),
        }
    }
}
