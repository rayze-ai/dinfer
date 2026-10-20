//! 全注意力层（16 / 64 层，每第 4 层）：q/k/v → per-head zero-centered RMSNorm + **partial** RoPE（前 64 / 256 维）
//! → 写 paged KV → GQA(D=256, GROUP 6) → `out *= sigmoid(gate)` → o_proj。
//!
//! 与 Qwen3 的差异（`docs/design-qwen35.md` §1.1 "扩模板"）：
//! - q_proj 输出是 per-head 交错的 `[q(D) | gate(D)]`，装载时重排成一条融合 GEMM `[q(H·D) | gate(H·D) | k | v]`
//!   （`weights::load_qkv`，`qmul = 2`，第 9 轮；attention 后 `mul_sigmoid_rows` 按行距读 gate 段）。
//! - `qkv_post_ext`：`rope_dim = 64`、`norm_beta = 1.0`（`q_norm/k_norm` 是 zero-centered，`w + 1`）。
//! - attention 之后、o_proj 之前 `out *= sigmoid(gate)`（vLLM `qwen3_next.py`）。
use crate::config::Qwen35Config;
use crate::scratch::ActScratch;
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
    qkv: Linear, // [q(H·D) | gate(H·D) | k | v] → [(2H + 2Hkv)·D, hidden]（无输出门时无 gate 段）
    qmul: usize, // 2 = 有 gate 段
    o_proj: Linear,
    q_norm: Tensor<bf16>, // [D]，zero-centered（kernel 里 +1）
    k_norm: Tensor<bf16>,
    /// TP 世界大小（本 rank 的 heads = cfg / world）。
    world: usize,
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
        let _ = max_tokens; // 激活缓冲全模型共享（`crate::scratch::ActScratch`）
        // cfg 已是本 rank 的配置（TP 在 `Qwen35Model::load_with_budget` 里拒绝 world > 1，这里 world 恒为 1）。
        let (h, hk, d) =
            (cfg.num_attention_heads as usize, cfg.num_key_value_heads as usize, cfg.head_dim as usize);
        let hidden = cfg.hidden_size as usize;
        let qmul = if cfg.attn_output_gate { 2 } else { 1 };
        let qkv = crate::weights::load_qkv(stream, src, t, layer, cfg, tp)?;
        ensure!(qkv.out_features == (h * qmul + 2 * hk) * d, "qkv out {} != {}", qkv.out_features, (h * qmul + 2 * hk) * d);
        let o_proj = t.linear(stream, src, &format!("layers.{layer}.self_attn.o_proj.weight"), Some((1, (rank, world))))?;
        ensure!(o_proj.in_features == h * d && o_proj.out_features == hidden, "o_proj shape {:?}", o_proj.w.shape);
        let q_norm = t.load_bf16(stream, src, &format!("layers.{layer}.self_attn.q_norm.weight"))?;
        let k_norm = t.load_bf16(stream, src, &format!("layers.{layer}.self_attn.k_norm.weight"))?;
        ensure!(q_norm.numel() == d && k_norm.numel() == d, "q/k_norm {} {}", q_norm.numel(), k_norm.numel());
        Ok(Self {
            qkv,
            qmul,
            o_proj,
            q_norm,
            k_norm,
            world,
        })
    }

    #[allow(clippy::too_many_arguments)]
    /// 本层所有 Linear（GEMM 自调优用）。
    pub fn linears(&self) -> Vec<(&'static str, &Linear)> {
        vec![("attn.qkv", &self.qkv), ("attn.o_proj", &self.o_proj)]
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
        sc: &mut ActScratch,
        out: &mut Tensor<bf16>,
    ) -> Result<()> {
        ensure!(t <= sc.max_tokens, "attn: {t} tokens > max {}", sc.max_tokens);
        let (h, hk, d) = (
            cfg.num_attention_heads as usize,
            cfg.num_key_value_heads as usize,
            cfg.head_dim as usize,
        );
        debug_assert_eq!(self.world, 1, "qwen35: TP 未实现");
        let eps = cfg.rms_norm_eps;
        let qmul = self.qmul;
        let qkv_w = (h * qmul + 2 * hk) * d;
        let rope_dim = cfg.rope_dim() as usize;
        // 1. qkv = x · W^T → [t, (H·qmul+2Hkv)·D]（q | gate | k | v；gate 段 qkv_post 不碰）
        ensure!(sc.qkv.numel() >= t * qkv_w, "attn: qkv 缓冲 < t × {qkv_w}");
        gemm.forward_bf16(&x.data, &self.qkv.w.data, None, &mut sc.qkv.data, t, qkv_w, cfg.hidden_size as usize)?;
        // 2. per-head RMSNorm（zero-centered）+ partial RoPE（原地）+ K/V 写 paged cache
        // `qkv_post_ext` 是 ops.rs 的自由函数（QkvPost trait 只有 Qwen3 的旧签名）；Qwen3.5 需要 partial RoPE +
        // zero-centered norm，直接调。
        k::qkv_post_ext_bf16(
            stream,
            &mut sc.qkv.data,
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
            qmul,
            kv.block_size,
        )?;
        // 3. GQA：q 从第 0 列起、k 从 H·qmul·D、v 从 (H·qmul+Hkv)·D，行跨度 qkv_w
        let scale = (d as f32).powf(-0.5);
        let q_view = sc.qkv.data.slice(..);
        let k_view = sc.qkv.data.slice(h * qmul * d..);
        let v_view = sc.qkv.data.slice((h * qmul + hk) * d..);
        match meta.mode {
            BatchMode::Prefill => ops.attn().prefill(
                stream,
                &q_view,
                &k_view,
                &v_view,
                &mut sc.o.data,
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
                &mut sc.o.data,
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
                &mut sc.o.data,
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
        // 4. 输出门：attn *= sigmoid(gate)，gate 段在 qkv 行内 [H·D, 2H·D)，按行距 qkv_w 读
        if qmul == 2 {
            let gate_view = sc.qkv.data.slice(h * d..);
            k::mul_sigmoid_rows_bf16(stream, &mut sc.o.data, &gate_view, t, h * d, qkv_w)?;
        }
        // 5. o_proj
        self.o_proj.forward_rows(gemm, &sc.o, out, t)
    }
}
