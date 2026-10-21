//! 算子插拔（design §10.1 方案 A）：热路径算子按组抽成 trait，模型 crate 对 `O: OpSet` 泛型，
//! 默认实现 `DefaultOps` 是 `ops.rs` 函数的零成本包装（单态化后与直调相同）。
//! 新实现（FlashInfer `.so`、自研 GEMM、Triton AOT）只需 impl 对应 trait 并在启动期选中；
//! 所有实现跑同一套 ref 测试（`kernels/tests/ref_tests.rs` 泛型化，待做）。
//!
//! 本轮只覆盖模型 forward 里的四组：Attention / Norm / QkvPost / Act。Gemm 是独立类型（`gemm.rs`），
//! Sampler 在 executor；两者下一轮并入。
use crate::device::Stream;
use crate::ops;
use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr};
use half::bf16;

/// 多头注意力（GQA，bf16，paged KV）。四条路径对应 `BatchMode`：纯 prefill / 带历史 paged / 纯 decode / 混批。
pub trait Attention {
    #[allow(clippy::too_many_arguments)]
    fn prefill<Q: DevicePtr<bf16> + ?Sized, K: DevicePtr<bf16> + ?Sized, V: DevicePtr<bf16> + ?Sized>(
        &self, stream: &Stream, q: &Q, k: &K, v: &V, out: &mut CudaSlice<bf16>, cu_seqlens: &CudaSlice<i32>,
        num_reqs: usize, max_seqlen: usize, heads: usize, kv_heads: usize, head_dim: usize, scale: f32,
        q_stride: usize, kv_stride: usize,
    ) -> Result<()>;
    #[allow(clippy::too_many_arguments)]
    fn paged<Q: DevicePtr<bf16> + ?Sized>(
        &self, stream: &Stream, q: &Q, cache: &CudaSlice<bf16>, block_table: &CudaSlice<i32>,
        seq_lens: &CudaSlice<i32>, cu_seqlens_q: &CudaSlice<i32>, out: &mut CudaSlice<bf16>, num_reqs: usize,
        max_q_len: usize, heads: usize, kv_heads: usize, head_dim: usize, block_size: usize, max_blocks: usize,
        scale: f32, q_stride: usize, skip_single: bool,
    ) -> Result<()>;
    #[allow(clippy::too_many_arguments)]
    fn decode<Q: DevicePtr<bf16> + ?Sized>(
        &self, stream: &Stream, q: &Q, cache: &CudaSlice<bf16>, block_table: &CudaSlice<i32>,
        seq_lens: &CudaSlice<i32>, out: &mut CudaSlice<bf16>, scratch: &mut CudaSlice<f32>, num_reqs: usize,
        heads: usize, kv_heads: usize, head_dim: usize, block_size: usize, max_blocks: usize, scale: f32,
        q_stride: usize, cu_seqlens_q: Option<&CudaSlice<i32>>,
    ) -> Result<()>;
    #[allow(clippy::too_many_arguments)]
    fn mixed<Q: DevicePtr<bf16> + ?Sized>(
        &self, stream: &Stream, q: &Q, cache: &CudaSlice<bf16>, block_table: &CudaSlice<i32>,
        seq_lens: &CudaSlice<i32>, cu_seqlens_q: &CudaSlice<i32>, out: &mut CudaSlice<bf16>,
        scratch: &mut CudaSlice<f32>, num_reqs: usize, max_q_len: usize, heads: usize, kv_heads: usize,
        head_dim: usize, block_size: usize, max_blocks: usize, scale: f32, q_stride: usize,
    ) -> Result<()>;
    /// 带滑窗的 prefill（Gemma4 sliding 层）：可见 key ∈ `[pos-window+1, pos]`。
    /// 默认实现报错——后端必须显式支持，否则窗口会被静默忽略（对拍时表现为"只有滑窗层错"）。
    #[allow(clippy::too_many_arguments)]
    fn prefill_win<Q: DevicePtr<bf16> + ?Sized, K: DevicePtr<bf16> + ?Sized, V: DevicePtr<bf16> + ?Sized>(
        &self, _stream: &Stream, _q: &Q, _k: &K, _v: &V, _out: &mut CudaSlice<bf16>, _cu_seqlens: &CudaSlice<i32>,
        _num_reqs: usize, _max_seqlen: usize, _heads: usize, _kv_heads: usize, _head_dim: usize, _scale: f32,
        _q_stride: usize, _kv_stride: usize, _window: usize,
    ) -> Result<()> {
        anyhow::bail!("当前后端不支持滑窗 prefill（prefill_win）")
    }
    /// 带滑窗的 paged attention；默认同样报错。
    #[allow(clippy::too_many_arguments)]
    fn paged_win<Q: DevicePtr<bf16> + ?Sized>(
        &self, _stream: &Stream, _q: &Q, _cache: &CudaSlice<bf16>, _block_table: &CudaSlice<i32>,
        _seq_lens: &CudaSlice<i32>, _cu_seqlens_q: &CudaSlice<i32>, _out: &mut CudaSlice<bf16>, _num_reqs: usize,
        _max_q_len: usize, _heads: usize, _kv_heads: usize, _head_dim: usize, _block_size: usize, _max_blocks: usize,
        _scale: f32, _q_stride: usize, _skip_single: bool, _window: usize,
    ) -> Result<()> {
        anyhow::bail!("当前后端不支持滑窗 paged attention（paged_win）")
    }
    fn decode_scratch_len(&self, batch: usize, heads: usize, head_dim: usize) -> usize;
    /// 启动期能力校验（不支持直接退出，不进运行期）。
    fn supports(&self, head_dim: usize, group: usize, block_size: usize) -> bool;
}

/// RMSNorm 及其融合变体。
pub trait Norm {
    fn rmsnorm(&self, stream: &Stream, x: &CudaSlice<bf16>, w: &CudaSlice<bf16>, out: &mut CudaSlice<bf16>, tokens: usize, hidden: usize, eps: f32) -> Result<()>;
    /// 无权重 RMSNorm：`out = x / rms(x)`（Gemma4 的 v_norm / router 预处理）。
    fn rmsnorm_noweight(&self, stream: &Stream, x: &CudaSlice<bf16>, out: &mut CudaSlice<bf16>, tokens: usize, hidden: usize, eps: f32) -> Result<()> {
        let _ = (stream, x, out, tokens, hidden, eps);
        anyhow::bail!("当前后端不支持无权重 RMSNorm（rmsnorm_noweight）")
    }
    /// res += x; out = rmsnorm(res) · w
    #[allow(clippy::too_many_arguments)]
    fn fused_add_rmsnorm(&self, stream: &Stream, x: &CudaSlice<bf16>, res: &mut CudaSlice<bf16>, w: &CudaSlice<bf16>, out: &mut CudaSlice<bf16>, tokens: usize, hidden: usize, eps: f32) -> Result<()>;
    /// out[r] = rmsnorm(x[rows[r]]) · w
    #[allow(clippy::too_many_arguments)]
    fn gather_rmsnorm(&self, stream: &Stream, x: &CudaSlice<bf16>, rows: &CudaSlice<i32>, w: &CudaSlice<bf16>, out: &mut CudaSlice<bf16>, num_rows: usize, hidden: usize, eps: f32) -> Result<()>;
    /// y += x
    fn add_inplace(&self, stream: &Stream, x: &CudaSlice<bf16>, y: &mut CudaSlice<bf16>, n: usize) -> Result<()>;
}

/// qkv GEMM 输出的后处理：q/k per-head RMSNorm + RoPE（原地）+ K/V 写 paged cache。
pub trait QkvPost {
    #[allow(clippy::too_many_arguments)]
    fn apply(&self, stream: &Stream, qkv: &mut CudaSlice<bf16>, wq: &CudaSlice<bf16>, wk: &CudaSlice<bf16>, positions: &CudaSlice<i32>, cos_sin: &CudaSlice<f32>, cache: &mut CudaSlice<bf16>, slot_mapping: &CudaSlice<i64>, tokens: usize, heads: usize, kv_heads: usize, head_dim: usize, eps: f32, block_size: usize) -> Result<()>;
}

/// 激活：out = silu(x[:, :inter]) * x[:, inter:]
pub trait Act {
    fn silu_and_mul(&self, stream: &Stream, x: &CudaSlice<bf16>, out: &mut CudaSlice<bf16>, tokens: usize, inter: usize) -> Result<()>;
}

/// 一套算子实现。模型对 `O: OpSet` 泛型，启动期选择一次。
pub trait OpSet: Send + Sync + 'static {
    type Attn: Attention;
    type Norm: Norm;
    type QkvPost: QkvPost;
    type Act: Act;
    fn attn(&self) -> &Self::Attn;
    fn norm(&self) -> &Self::Norm;
    fn qkv_post(&self) -> &Self::QkvPost;
    fn act(&self) -> &Self::Act;
    fn name(&self) -> &'static str;
}

/// 默认实现：`ops.rs`（本仓库 CUDA kernel）。零大小类型，所有方法内联转发。
#[derive(Default, Clone, Copy, Debug)]
pub struct DefaultOps;

impl Attention for DefaultOps {
    #[inline]
    fn prefill<Q: DevicePtr<bf16> + ?Sized, K: DevicePtr<bf16> + ?Sized, V: DevicePtr<bf16> + ?Sized>(
        &self, stream: &Stream, q: &Q, k: &K, v: &V, out: &mut CudaSlice<bf16>, cu_seqlens: &CudaSlice<i32>,
        num_reqs: usize, max_seqlen: usize, heads: usize, kv_heads: usize, head_dim: usize, scale: f32,
        q_stride: usize, kv_stride: usize,
    ) -> Result<()> {
        ops::gqa_prefill_bf16(stream, q, k, v, out, cu_seqlens, num_reqs, max_seqlen, heads, kv_heads, head_dim, scale, q_stride, kv_stride)
    }
    #[inline]
    fn paged<Q: DevicePtr<bf16> + ?Sized>(
        &self, stream: &Stream, q: &Q, cache: &CudaSlice<bf16>, block_table: &CudaSlice<i32>,
        seq_lens: &CudaSlice<i32>, cu_seqlens_q: &CudaSlice<i32>, out: &mut CudaSlice<bf16>, num_reqs: usize,
        max_q_len: usize, heads: usize, kv_heads: usize, head_dim: usize, block_size: usize, max_blocks: usize,
        scale: f32, q_stride: usize, skip_single: bool,
    ) -> Result<()> {
        ops::gqa_paged_bf16(stream, q, cache, block_table, seq_lens, cu_seqlens_q, out, num_reqs, max_q_len, heads, kv_heads, head_dim, block_size, max_blocks, scale, q_stride, skip_single)
    }
    #[inline]
    fn decode<Q: DevicePtr<bf16> + ?Sized>(
        &self, stream: &Stream, q: &Q, cache: &CudaSlice<bf16>, block_table: &CudaSlice<i32>,
        seq_lens: &CudaSlice<i32>, out: &mut CudaSlice<bf16>, scratch: &mut CudaSlice<f32>, num_reqs: usize,
        heads: usize, kv_heads: usize, head_dim: usize, block_size: usize, max_blocks: usize, scale: f32,
        q_stride: usize, cu_seqlens_q: Option<&CudaSlice<i32>>,
    ) -> Result<()> {
        ops::gqa_decode_bf16(stream, q, cache, block_table, seq_lens, out, scratch, num_reqs, heads, kv_heads, head_dim, block_size, max_blocks, scale, q_stride, cu_seqlens_q)
    }
    #[inline]
    fn mixed<Q: DevicePtr<bf16> + ?Sized>(
        &self, stream: &Stream, q: &Q, cache: &CudaSlice<bf16>, block_table: &CudaSlice<i32>,
        seq_lens: &CudaSlice<i32>, cu_seqlens_q: &CudaSlice<i32>, out: &mut CudaSlice<bf16>,
        scratch: &mut CudaSlice<f32>, num_reqs: usize, max_q_len: usize, heads: usize, kv_heads: usize,
        head_dim: usize, block_size: usize, max_blocks: usize, scale: f32, q_stride: usize,
    ) -> Result<()> {
        ops::gqa_mixed_bf16(stream, q, cache, block_table, seq_lens, cu_seqlens_q, out, scratch, num_reqs, max_q_len, heads, kv_heads, head_dim, block_size, max_blocks, scale, q_stride)
    }
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn prefill_win<Q: DevicePtr<bf16> + ?Sized, K: DevicePtr<bf16> + ?Sized, V: DevicePtr<bf16> + ?Sized>(
        &self, stream: &Stream, q: &Q, k: &K, v: &V, out: &mut CudaSlice<bf16>, cu_seqlens: &CudaSlice<i32>,
        num_reqs: usize, max_seqlen: usize, heads: usize, kv_heads: usize, head_dim: usize, scale: f32,
        q_stride: usize, kv_stride: usize, window: usize,
    ) -> Result<()> {
        ops::gqa_prefill_win_bf16(stream, q, k, v, out, cu_seqlens, num_reqs, max_seqlen, heads, kv_heads, head_dim, scale, q_stride, kv_stride, window)
    }
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn paged_win<Q: DevicePtr<bf16> + ?Sized>(
        &self, stream: &Stream, q: &Q, cache: &CudaSlice<bf16>, block_table: &CudaSlice<i32>,
        seq_lens: &CudaSlice<i32>, cu_seqlens_q: &CudaSlice<i32>, out: &mut CudaSlice<bf16>, num_reqs: usize,
        max_q_len: usize, heads: usize, kv_heads: usize, head_dim: usize, block_size: usize, max_blocks: usize,
        scale: f32, q_stride: usize, skip_single: bool, window: usize,
    ) -> Result<()> {
        ops::gqa_paged_win_bf16(stream, q, cache, block_table, seq_lens, cu_seqlens_q, out, num_reqs, max_q_len, heads, kv_heads, head_dim, block_size, max_blocks, scale, q_stride, skip_single, window)
    }
    #[inline]
    fn decode_scratch_len(&self, batch: usize, heads: usize, head_dim: usize) -> usize {
        ops::gqa_decode_scratch_len(batch, heads, head_dim)
    }
    fn supports(&self, head_dim: usize, group: usize, block_size: usize) -> bool {
        // tensor-core 路径要求 D=128；其他 D 走参考 kernel（慢但正确）；decode v2 要求 group ≤ 2、block_size 16 的倍数
        let _ = head_dim;
        group <= 2 && block_size % 16 == 0
    }
}
impl Norm for DefaultOps {
    #[inline]
    #[inline]
    fn rmsnorm_noweight(&self, stream: &Stream, x: &CudaSlice<bf16>, out: &mut CudaSlice<bf16>, _tokens: usize, hidden: usize, eps: f32) -> Result<()> {
        ops::rmsnorm_bf16_noweight(stream, x, out, hidden, eps)
    }
    fn rmsnorm(&self, stream: &Stream, x: &CudaSlice<bf16>, w: &CudaSlice<bf16>, out: &mut CudaSlice<bf16>, tokens: usize, hidden: usize, eps: f32) -> Result<()> {
        ops::rmsnorm_bf16_n(stream, x, w, out, tokens, hidden, eps)
    }
    #[inline]
    fn fused_add_rmsnorm(&self, stream: &Stream, x: &CudaSlice<bf16>, res: &mut CudaSlice<bf16>, w: &CudaSlice<bf16>, out: &mut CudaSlice<bf16>, tokens: usize, hidden: usize, eps: f32) -> Result<()> {
        ops::fused_add_rmsnorm_bf16_n(stream, x, res, w, out, tokens, hidden, eps)
    }
    #[inline]
    fn gather_rmsnorm(&self, stream: &Stream, x: &CudaSlice<bf16>, rows: &CudaSlice<i32>, w: &CudaSlice<bf16>, out: &mut CudaSlice<bf16>, num_rows: usize, hidden: usize, eps: f32) -> Result<()> {
        ops::gather_rmsnorm_bf16(stream, x, rows, w, out, num_rows, hidden, eps)
    }
    #[inline]
    fn add_inplace(&self, stream: &Stream, x: &CudaSlice<bf16>, y: &mut CudaSlice<bf16>, n: usize) -> Result<()> {
        ops::add_inplace_bf16_n(stream, x, y, n)
    }
}
impl QkvPost for DefaultOps {
    #[inline]
    fn apply(&self, stream: &Stream, qkv: &mut CudaSlice<bf16>, wq: &CudaSlice<bf16>, wk: &CudaSlice<bf16>, positions: &CudaSlice<i32>, cos_sin: &CudaSlice<f32>, cache: &mut CudaSlice<bf16>, slot_mapping: &CudaSlice<i64>, tokens: usize, heads: usize, kv_heads: usize, head_dim: usize, eps: f32, block_size: usize) -> Result<()> {
        ops::qkv_post_bf16(stream, qkv, wq, wk, positions, cos_sin, cache, slot_mapping, tokens, heads, kv_heads, head_dim, eps, block_size)
    }
}
impl Act for DefaultOps {
    #[inline]
    fn silu_and_mul(&self, stream: &Stream, x: &CudaSlice<bf16>, out: &mut CudaSlice<bf16>, tokens: usize, inter: usize) -> Result<()> {
        ops::silu_and_mul_bf16_n(stream, x, out, tokens, inter)
    }
}
impl OpSet for DefaultOps {
    type Attn = DefaultOps;
    type Norm = DefaultOps;
    type QkvPost = DefaultOps;
    type Act = DefaultOps;
    #[inline] fn attn(&self) -> &Self::Attn { self }
    #[inline] fn norm(&self) -> &Self::Norm { self }
    #[inline] fn qkv_post(&self) -> &Self::QkvPost { self }
    #[inline] fn act(&self) -> &Self::Act { self }
    fn name(&self) -> &'static str { "default" }
}

/// 启动期选择：`OPS` 环境变量（默认 "default"）。目前只有一个实现；未知名字报错，不静默回落。
pub fn selected_ops_name() -> String {
    std::env::var("OPS").unwrap_or_else(|_| "default".to_string())
}
