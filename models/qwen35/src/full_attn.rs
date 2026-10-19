//! 全注意力层（16 / 64 层，每第 4 层）：q/k/v → per-head zero-centered RMSNorm + **partial** RoPE（前 64 / 256 维）
//! → 写 paged KV → GQA(D=256, GROUP 6) → `out *= sigmoid(gate)` → o_proj。
//!
//! 与 Qwen3 的差异（`docs/design-qwen35.md` §1.1 "扩模板"）：
//! - q_proj 输出是 per-head 交错的 `[q(D) | gate(D)]`，装载时已拆成两条紧凑 GEMM（`weights::load_qkv` / `load_gate`）；
//!   本文件的行布局回到 `qmul = 1` 的 `[q | k | v]`。
//! - `qkv_post_ext`：`rope_dim = 64`、`norm_beta = 1.0`（`q_norm/k_norm` 是 zero-centered，`w + 1`）。
//! - attention 之后、o_proj 之前 `out *= sigmoid(gate)`（vLLM `qwen3_next.py`）。
use crate::config::Qwen35Config;
use crate::weights::Table;
use anyhow::{ensure, Result};
use dinfer_core::runner::BatchMode;
use dinfer_kernels as k;
use dinfer_layers::{bf16, Gemm, Linear, OpSet, Rope, Stream, Tensor, WeightSource};
use dinfer_layers::Attention as _;

/// 一层 paged KV：[num_blocks, block_size, kv_heads, 2, head_dim] bf16（与 Qwen3 同布局）。
pub struct PagedKv {
    pub cache: Tensor<bf16>,
    pub num_blocks: usize,
    pub block_size: usize,
}

/// 一步 attention 的 batch 描述（executor 上传后的设备张量）。
pub struct AttnMeta<'a> {
    pub mode: BatchMode,
    pub cu_seqlens: &'a Tensor<i32>,
    pub seq_lens: &'a Tensor<i32>,
    pub block_table: &'a Tensor<i32>,
    pub num_reqs: usize,
    pub max_blocks: usize,
    pub max_seqlen: usize,
}

pub struct FullAttention {
    qkv: Linear,   // [q(H·D) | k | v] → [(H + 2Hkv)·D, hidden]
    gate: Linear,  // [H·D, hidden]
    o_proj: Linear,
    q_norm: Tensor<bf16>, // [D]，zero-centered（kernel 里 +1）
    k_norm: Tensor<bf16>,
    /// TP 世界大小（本 rank 的 heads = cfg / world）。
    world: usize,
    qkv_buf: Tensor<bf16>, // [T, (H + 2Hkv)·D]
    gate_buf: Tensor<bf16>, // [T, H·D]
    attn: Tensor<bf16>,    // [T, H·D]
}

impl FullAttention {
    pub fn load(
        stream: &Stream,
        src: &WeightSource,
        t: &Table,
        layer: usize,
        cfg: &Qwen35Config,
        max_tokens: usize,
        tp: (usize, usize),
    ) -> Result<Self> {
        let (rank, world) = tp;
        // cfg 已是本 rank 的配置（TP 在 `Qwen35Model::load_with_budget` 里拒绝 world > 1，这里 world 恒为 1）。
        let (h, hk, d) =
            (cfg.num_attention_heads as usize, cfg.num_key_value_heads as usize, cfg.head_dim as usize);
        let hidden = cfg.hidden_size as usize;
        let qkv = crate::weights::load_qkv(stream, src, t, layer, cfg, tp)?;
        ensure!(qkv.out_features == (h + 2 * hk) * d, "qkv out {} != {}", qkv.out_features, (h + 2 * hk) * d);
        let gate = crate::weights::load_gate(stream, src, t, layer, cfg, tp)?;
        ensure!(gate.out_features == h * d, "gate out {} != {}", gate.out_features, h * d);
        let o_proj = t.linear(stream, src, &format!("layers.{layer}.self_attn.o_proj.weight"), Some((1, (rank, world))))?;
        ensure!(o_proj.in_features == h * d && o_proj.out_features == hidden, "o_proj shape {:?}", o_proj.w.shape);
        let q_norm = t.load_bf16(stream, src, &format!("layers.{layer}.self_attn.q_norm.weight"))?;
        let k_norm = t.load_bf16(stream, src, &format!("layers.{layer}.self_attn.k_norm.weight"))?;
        ensure!(q_norm.numel() == d && k_norm.numel() == d, "q/k_norm {} {}", q_norm.numel(), k_norm.numel());
        Ok(Self {
            qkv,
            gate,
            o_proj,
            q_norm,
            k_norm,
            world,
            qkv_buf: Tensor::zeros(stream, &[max_tokens, (h + 2 * hk) * d])?,
            gate_buf: Tensor::zeros(stream, &[max_tokens, h * d])?,
            attn: Tensor::zeros(stream, &[max_tokens, h * d])?,
        })
    }

    pub fn swap_bufs(&mut self, qkv_buf: &mut Tensor<bf16>, gate_buf: &mut Tensor<bf16>, attn: &mut Tensor<bf16>) {
        std::mem::swap(&mut self.qkv_buf, qkv_buf);
        std::mem::swap(&mut self.gate_buf, gate_buf);
        std::mem::swap(&mut self.attn, attn);
    }

    #[allow(clippy::too_many_arguments)]
    /// 本层所有 Linear（GEMM 自调优用）。
    pub fn linears(&self) -> Vec<(&'static str, &Linear)> {
        vec![("attn.qkv", &self.qkv), ("attn.gate", &self.gate), ("attn.o_proj", &self.o_proj)]
    }

    pub fn forward<O: OpSet>(
        &mut self,
        stream: &Stream,
        gemm: &Gemm,
        ops: &O,
        cfg: &Qwen35Config,
        rope: &Rope,
        t: usize,
        x: &Tensor<bf16>,
        positions: &Tensor<i32>,
        slot_mapping: &Tensor<i64>,
        kv: &mut PagedKv,
        meta: &AttnMeta<'_>,
        decode_scratch: &mut Tensor<f32>,
        out: &mut Tensor<bf16>,
    ) -> Result<()> {
        let (h, hk, d) = (
            cfg.num_attention_heads as usize,
            cfg.num_key_value_heads as usize,
            cfg.head_dim as usize,
        );
        debug_assert_eq!(self.world, 1, "qwen35: TP 未实现");
        let eps = cfg.rms_norm_eps;
        let qkv_w = (h + 2 * hk) * d;
        let rope_dim = cfg.rope_dim() as usize;
        // 1. qkv = x · W^T → [t, (H+2Hkv)·D]（q | k | v）
        gemm.forward_bf16(&x.data, &self.qkv.w.data, None, &mut self.qkv_buf.data, t, qkv_w, cfg.hidden_size as usize)?;
        // 2. per-head RMSNorm（zero-centered）+ partial RoPE（原地）+ K/V 写 paged cache
        // `qkv_post_ext` 是 ops.rs 的自由函数（QkvPost trait 只有 Qwen3 的旧签名）；Qwen3.5 需要 partial RoPE +
        // zero-centered norm，直接调。
        k::qkv_post_ext_bf16(
            stream,
            &mut self.qkv_buf.data,
            &self.q_norm.data,
            &self.k_norm.data,
            &positions.data,
            &rope.cos_sin.data,
            &mut kv.cache.data,
            &slot_mapping.data,
            t,
            h,
            hk,
            d,
            rope_dim,
            eps,
            1.0, // zero-centered
            1,   // qmul = 1（gate 已拆成独立 GEMM）
            kv.block_size,
        )?;
        // 3. GQA：q 从第 0 列起、k 从 H·D、v 从 (H+Hkv)·D，行跨度 qkv_w
        let scale = (d as f32).powf(-0.5);
        let q_view = self.qkv_buf.data.slice(..);
        let k_view = self.qkv_buf.data.slice(h * d..);
        let v_view = self.qkv_buf.data.slice((h + hk) * d..);
        match meta.mode {
            BatchMode::Prefill => ops.attn().prefill(
                stream,
                &q_view,
                &k_view,
                &v_view,
                &mut self.attn.data,
                &meta.cu_seqlens.data,
                meta.num_reqs,
                meta.max_seqlen,
                h,
                hk,
                d,
                scale,
                qkv_w,
                qkv_w,
            )?,
            BatchMode::Decode => ops.attn().decode(
                stream,
                &q_view,
                &kv.cache.data,
                &meta.block_table.data,
                &meta.seq_lens.data,
                &mut self.attn.data,
                &mut decode_scratch.data,
                meta.num_reqs,
                h,
                hk,
                d,
                kv.block_size,
                meta.max_blocks,
                scale,
                qkv_w,
                None,
            )?,
            BatchMode::Paged => ops.attn().mixed(
                stream,
                &q_view,
                &kv.cache.data,
                &meta.block_table.data,
                &meta.seq_lens.data,
                &meta.cu_seqlens.data,
                &mut self.attn.data,
                &mut decode_scratch.data,
                meta.num_reqs,
                meta.max_seqlen,
                h,
                hk,
                d,
                kv.block_size,
                meta.max_blocks,
                scale,
                qkv_w,
            )?,
        }
        // 4. 输出门：attn *= sigmoid(gate)
        self.gate.forward(gemm, x, &mut self.gate_buf)?;
        k::mul_sigmoid_bf16(stream, &mut self.attn.data, &self.gate_buf.data, t * h * d)?;
        // 5. o_proj
        self.o_proj.forward(gemm, &self.attn, out)
    }
}
