//! 类型安全的 kernel 包装。所有 `unsafe` 集中在这里；上层只见到 `CudaSlice` 与维度。
//!
//! bf16 数据在 Rust 侧用 `half::bf16`（与 `__nv_bfloat16` 位布局一致）。

use crate::device::Stream;
use crate::ffi;
use anyhow::{bail, ensure, Result};
use cudarc::driver::{CudaSlice, CudaView, DevicePtr, DevicePtrMut};
use half::bf16;
use std::ffi::c_void;

pub(crate) fn check(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("{what} failed: cudaError {rc}");
    }
    Ok(())
}

/// 小 M（≤ 8）bf16 GEMM：`out[m,n] = a[m,k]·w[n,k]^T (+bias)`。`splits` 为 split-K 份数（host 给，1 = 不切）；
/// `partial` 在 splits > 1 时必须 ≥ splits·m·n 个 f32。
#[allow(clippy::too_many_arguments)]
pub fn gemm_small_m_bf16(
    stream: &Stream,
    a: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    bias: Option<&CudaSlice<bf16>>,
    out: &mut CudaSlice<bf16>,
    partial: Option<&mut CudaSlice<f32>>,
    m: usize,
    n: usize,
    k: usize,
    splits: usize,
) -> Result<()> {
    ensure!(m <= 32 && k % 8 == 0, "gemm_small_m: m={m} (≤32) k={k} (8 的倍数)");
    ensure!(a.len() >= m * k && w.len() == n * k && out.len() >= m * n, "gemm_small_m: 形状");
    if let Some(b) = bias {
        ensure!(b.len() == n, "gemm_small_m: bias.len {} != n {n}", b.len());
    }
    let pp = match partial {
        Some(p) if splits > 1 => {
            ensure!(p.len() >= splits * m * n, "gemm_small_m: partial.len {} < {}", p.len(), splits * m * n);
            stream.ptr_mut(p) as *mut f32
        }
        _ => {
            ensure!(splits <= 1, "gemm_small_m: splits > 1 需要 partial");
            std::ptr::null_mut()
        }
    };
    let pa = stream.ptr(a) as *const c_void;
    let pw = stream.ptr(w) as *const c_void;
    let pb = bias.map_or(std::ptr::null(), |b| stream.ptr(b) as *const c_void);
    let po = stream.ptr_mut(out) as *mut c_void;
    // SAFETY: 指针来自有效 CudaSlice，长度已校验；stream 同 context。
    check(
        unsafe {
            ffi::gemm_small_m_bf16(
                pa,
                pw,
                pb,
                po,
                pp,
                m as i64,
                n as i64,
                k as i64,
                splits as i64,
                stream.raw() as ffi::cudaStream_t,
            )
        },
        "gemm_small_m_bf16",
    )
}

/// tensor-core 小 M GEMM（r11b）：同 `gemm_small_m_bf16` 语义，K % 128 == 0，`warps` ∈ {4, 8}，`stripes`（每 warp 8 行 W 条数）∈ {1,2,4}。
/// splits > 1 时 `partial` ≥ splits·m·n 个 f32、`counters` ≥ ceil(n/8) 个 i32（初值 0，kernel 自复位）。
#[allow(clippy::too_many_arguments)]
pub fn gemm_small_m_mma_bf16(
    stream: &Stream,
    a: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    bias: Option<&CudaSlice<bf16>>,
    out: &mut CudaSlice<bf16>,
    partial: Option<&mut CudaSlice<f32>>,
    counters: Option<&mut CudaSlice<i32>>,
    m: usize,
    n: usize,
    k: usize,
    splits: usize,
    warps: usize,
    stripes: usize,
) -> Result<()> {
    ensure!(m >= 1 && m <= 32 && k % 128 == 0, "gemm_small_m_mma: m={m} (1..=32) k={k} (128 的倍数)");
    ensure!(warps == 4 || warps == 8, "gemm_small_m_mma: warps={warps}");
    ensure!(matches!(stripes, 1 | 2 | 4), "gemm_small_m_mma: stripes={stripes}");
    ensure!(a.len() >= m * k && w.len() == n * k && out.len() >= m * n, "gemm_small_m_mma: 形状");
    if let Some(b) = bias {
        ensure!(b.len() == n, "gemm_small_m_mma: bias.len {} != n {n}", b.len());
    }
    let (pp, pc) = if splits > 1 {
        let (Some(p), Some(c)) = (partial, counters) else { bail!("gemm_small_m_mma: splits > 1 需要 partial + counters") };
        ensure!(p.len() >= splits * m * n, "gemm_small_m_mma: partial.len {} < {}", p.len(), splits * m * n);
        ensure!(c.len() >= n.div_ceil(8), "gemm_small_m_mma: counters.len {} < {}", c.len(), n.div_ceil(8));
        (stream.ptr_mut(p) as *mut f32, stream.ptr_mut(c) as *mut i32)
    } else {
        (std::ptr::null_mut(), std::ptr::null_mut())
    };
    let pa = stream.ptr(a) as *const c_void;
    let pw = stream.ptr(w) as *const c_void;
    let pb = bias.map_or(std::ptr::null(), |b| stream.ptr(b) as *const c_void);
    let po = stream.ptr_mut(out) as *mut c_void;
    // SAFETY: 指针来自有效 CudaSlice，长度已校验；stream 同 context。
    check(
        unsafe {
            ffi::gemm_small_m_mma_bf16(pa, pw, pb, po, pp, pc, m as i64, n as i64, k as i64, splits as i64, warps as i64, stripes as i64, stream.raw() as ffi::cudaStream_t)
        },
        "gemm_small_m_mma_bf16",
    )
}

/// out = a + b（fp32）。P0 自检。
pub fn add_f32(
    stream: &Stream,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
) -> Result<()> {
    let n = a.len();
    ensure!(
        b.len() == n && out.len() == n,
        "add_f32: length mismatch a={} b={} out={}",
        n,
        b.len(),
        out.len()
    );
    let pa = stream.ptr(a) as *const f32;
    let pb = stream.ptr(b) as *const f32;
    let po = stream.ptr_mut(out) as *mut f32;
    // SAFETY: 三个指针来自有效的 CudaSlice，长度已校验；stream 来自同一 context。
    check(unsafe { ffi::add_f32(pa, pb, po, n as i64, stream.raw() as ffi::cudaStream_t) }, "add_f32")
}

/// out[t] = rmsnorm(x[t]) * w。x/out: [tokens*hidden]，w: [hidden]。
pub fn rmsnorm_bf16(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    ensure!(
        hidden > 0 && x.len() % hidden == 0,
        "rmsnorm: x.len {} not multiple of hidden {hidden}",
        x.len()
    );
    ensure!(w.len() == hidden && out.len() == x.len(), "rmsnorm: shape mismatch");
    let tokens = (x.len() / hidden) as i64;
    let (px, pw) = (stream.ptr(x) as *const c_void, stream.ptr(w) as *const c_void);
    let po = stream.ptr_mut(out) as *mut c_void;
    // SAFETY: 指针与长度已校验。
    check(
        unsafe {
            ffi::rmsnorm_bf16(
                px,
                pw,
                po,
                tokens,
                hidden as i64,
                eps,
                rmsnorm_layout(stream, tokens as usize, hidden, None),
                norm_mode(),
                stream.raw() as _,
            )
        },
        "rmsnorm_bf16",
    )
}

/// 无权重 RMSNorm：`out = x / rms(x)`（Gemma4 的 `v_norm`、router 预处理）。
///
/// kernel 侧把空权重指针当作 1.0（`norm.cu`），所以不需要额外的 C 符号，传 `nullptr` 即可。
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_bf16_noweight(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    ensure!(hidden > 0 && x.len() % hidden == 0, "rmsnorm_noweight: x.len {} not multiple of hidden {hidden}", x.len());
    ensure!(out.len() == x.len(), "rmsnorm_noweight: out 长度不符");
    let tokens = (x.len() / hidden) as i64;
    let (px, po) = (stream.ptr(x) as *const c_void, stream.ptr_mut(out) as *mut c_void);
    // SAFETY: 指针与长度已校验；w 传空 → kernel 乘 1.0。
    check(
        unsafe {
            ffi::rmsnorm_bf16(
                px,
                std::ptr::null(),
                po,
                tokens,
                hidden as i64,
                eps,
                rmsnorm_layout(stream, tokens as usize, hidden, None),
                norm_mode(),
                stream.raw() as _,
            )
        },
        "rmsnorm_bf16_noweight",
    )
}

/// res += x; out = rmsnorm(res) * w。res 原位更新。
pub fn fused_add_rmsnorm_bf16(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    res: &mut CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    ensure!(hidden > 0 && x.len() % hidden == 0, "fused_add_rmsnorm: bad shape");
    ensure!(
        res.len() == x.len() && out.len() == x.len() && w.len() == hidden,
        "fused_add_rmsnorm: shape mismatch"
    );
    let tokens = (x.len() / hidden) as i64;
    let px = stream.ptr(x) as *const c_void;
    let pw = stream.ptr(w) as *const c_void;
    let pr = stream.ptr_mut(res) as *mut c_void;
    let po = stream.ptr_mut(out) as *mut c_void;
    // SAFETY: 指针与长度已校验；res 与 out 是不同缓冲。
    check(
        unsafe {
            ffi::fused_add_rmsnorm_bf16(
                px,
                pr,
                pw,
                po,
                tokens,
                hidden as i64,
                eps,
                rmsnorm_layout(stream, tokens as usize, hidden, None),
                norm_mode(),
                stream.raw() as _,
            )
        },
        "fused_add_rmsnorm_bf16",
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeStyle {
    /// 成对 (2i, 2i+1)。DeepSeek / GLM-5.3（`rope_interleave=True`）。
    Interleave,
    /// 成对 (i, i+rope_dim/2)。Llama 系。
    Neox,
}

/// 就地对 q / k 的前 `rope_dim` 维做 RoPE。q: [tokens, q_heads, head_dim]，k 可选。
#[allow(clippy::too_many_arguments)]
pub fn rope_bf16(
    stream: &Stream,
    q: &mut CudaSlice<bf16>,
    k: Option<&mut CudaSlice<bf16>>,
    positions: &CudaSlice<i32>,
    cos_sin: &CudaSlice<f32>,
    q_heads: usize,
    k_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    style: RopeStyle,
) -> Result<()> {
    rope_span_bf16(stream, q, k, positions, cos_sin, q_heads, k_heads, head_dim, rope_dim, rope_dim / 2, style)
}

/// 同 [`rope_bf16`]，neox 配对改为 `(i, i + span)`（`i < rope_dim/2`）。`span = rope_dim/2` 即常规 partial RoPE；
/// Gemma4 proportional（零频补齐到整头再按 neox 旋转）等价于 `span = head_dim/2`。
#[allow(clippy::too_many_arguments)]
pub fn rope_span_bf16(
    stream: &Stream,
    q: &mut CudaSlice<bf16>,
    k: Option<&mut CudaSlice<bf16>>,
    positions: &CudaSlice<i32>,
    cos_sin: &CudaSlice<f32>,
    q_heads: usize,
    k_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    span: usize,
    style: RopeStyle,
) -> Result<()> {
    let tokens = positions.len();
    ensure!(
        span >= rope_dim / 2 && span + rope_dim / 2 <= head_dim,
        "rope: span {span} 越界（rope_dim {rope_dim} / head_dim {head_dim}）"
    );
    ensure!(q.len() == tokens * q_heads * head_dim, "rope: q shape mismatch");
    ensure!(rope_dim > 0 && rope_dim % 2 == 0 && rope_dim <= head_dim, "rope: bad rope_dim {rope_dim}");
    ensure!(cos_sin.len() % rope_dim == 0, "rope: cos_sin not multiple of rope_dim");
    let pq = stream.ptr_mut(q) as *mut c_void;
    let pk = match k {
        Some(k) => {
            ensure!(k.len() == tokens * k_heads * head_dim, "rope: k shape mismatch");
            stream.ptr_mut(k) as *mut c_void
        }
        None => {
            ensure!(k_heads == 0, "rope: k_heads > 0 but k is None");
            std::ptr::null_mut()
        }
    };
    let pp = stream.ptr(positions) as *const i32;
    let pc = stream.ptr(cos_sin) as *const f32;
    // SAFETY: 指针与长度已校验；positions 越界由调用方保证 < cos_sin.len()/rope_dim。
    check(
        unsafe {
            ffi::rope_span_bf16(
                pq,
                pk,
                pp,
                pc,
                tokens as i64,
                q_heads as i64,
                k_heads as i64,
                head_dim as i64,
                rope_dim as i64,
                span as i64,
                matches!(style, RopeStyle::Neox) as i32,
                stream.raw() as _,
            )
        },
        "rope_span_bf16",
    )
}

/// out = gelu_tanh(gate) * up（Gemma 系 `gelu_pytorch_tanh`，双缓冲）。
pub fn gelu_tanh_mul_bf16(
    stream: &Stream,
    gate: &CudaSlice<bf16>,
    up: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    tokens: usize,
    inter: usize,
) -> Result<()> {
    gelu_tanh_mul_mode(stream, gate, up, out, tokens, inter, 0)
}

/// 同 [`gelu_tanh_mul_bf16`]，但按 vLLM `act_and_mul_kernel` 口径：gelu 先舍入 bf16 再乘 up（两次舍入）。
pub fn gelu_tanh_mul_round2_bf16(
    stream: &Stream,
    gate: &CudaSlice<bf16>,
    up: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    tokens: usize,
    inter: usize,
) -> Result<()> {
    gelu_tanh_mul_mode(stream, gate, up, out, tokens, inter, 1)
}

fn gelu_tanh_mul_mode(
    stream: &Stream,
    gate: &CudaSlice<bf16>,
    up: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    tokens: usize,
    inter: usize,
    mode: i32,
) -> Result<()> {
    ensure!(gate.len() >= tokens * inter && up.len() >= tokens * inter && out.len() >= tokens * inter,
            "gelu_tanh_mul: shape");
    // SAFETY: 长度已校验。
    check(
        unsafe {
            ffi::gelu_tanh_mul_bf16(
                stream.ptr(gate) as *const c_void,
                stream.ptr(up) as *const c_void,
                stream.ptr_mut(out) as *mut c_void,
                tokens as i64,
                inter as i64,
                mode,
                stream.raw() as _,
            )
        },
        "gelu_tanh_mul_bf16",
    )
}

/// logit softcapping（Gemma 系）：`x = tanh(x/c)·c`，bf16 原地。
pub fn softcap_bf16_n(stream: &Stream, x: &mut CudaSlice<bf16>, n: usize, c: f32) -> Result<()> {
    ensure!(x.len() >= n, "softcap: buffer");
    // SAFETY: 长度已校验。
    check(unsafe { ffi::softcap_bf16(stream.ptr_mut(x) as *mut c_void, n as i64, c, stream.raw() as _) }, "softcap_bf16")
}


/// 逐行乘向量：`x[r, j] *= w[j]`（router.scale 这种逐 hidden 缩放）。
pub fn mul_bf16_n(stream: &Stream, x: &mut CudaSlice<bf16>, w: &CudaSlice<bf16>, rows: usize, cols: usize) -> Result<()> {
    ensure!(x.len() >= rows * cols && w.len() >= cols, "mul_row: shape");
    // SAFETY: 长度已校验。
    check(
        unsafe {
            ffi::mul_row_bf16(
                stream.ptr_mut(x) as *mut c_void,
                stream.ptr(w) as *const c_void,
                rows as i64,
                cols as i64,
                stream.raw() as _,
            )
        },
        "mul_row_bf16",
    )
}

/// out = silu(x[:, :inter]) * x[:, inter:]。x: [tokens*2*inter]，out: [tokens*inter]。
pub fn silu_and_mul_bf16(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    inter: usize,
) -> Result<()> {
    ensure!(inter > 0 && x.len() % (2 * inter) == 0, "silu_and_mul: bad shape");
    let tokens = x.len() / (2 * inter);
    ensure!(out.len() == tokens * inter, "silu_and_mul: out shape mismatch");
    let px = stream.ptr(x) as *const c_void;
    let po = stream.ptr_mut(out) as *mut c_void;
    // SAFETY: 指针与长度已校验。
    check(
        unsafe { ffi::silu_and_mul_bf16(px, po, tokens as i64, inter as i64, stream.raw() as _) },
        "silu_and_mul_bf16",
    )
}

/// out[t] = table[ids[t]]。table: [vocab*hidden]。
pub fn embedding_bf16(
    stream: &Stream,
    ids: &CudaSlice<i32>,
    table: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    hidden: usize,
) -> Result<()> {
    ensure!(hidden > 0 && table.len() % hidden == 0, "embedding: bad table shape");
    let vocab = table.len() / hidden;
    ensure!(out.len() == ids.len() * hidden, "embedding: out shape mismatch");
    let pi = stream.ptr(ids) as *const i32;
    let pt = stream.ptr(table) as *const c_void;
    let po = stream.ptr_mut(out) as *mut c_void;
    // SAFETY: 指针与长度已校验；越界 id 在 kernel 内兜底。
    check(
        unsafe {
            ffi::embedding_bf16(pi, pt, po, ids.len() as i64, hidden as i64, vocab as i64, stream.raw() as _)
        },
        "embedding_bf16",
    )
}

/// out[b] = argmax(logits[b])。logits: [batch*vocab] fp32。
pub fn argmax_f32(
    stream: &Stream,
    logits: &CudaSlice<f32>,
    out: &mut CudaSlice<i32>,
    vocab: usize,
) -> Result<()> {
    ensure!(vocab > 0 && logits.len() % vocab == 0, "argmax: bad shape");
    let batch = logits.len() / vocab;
    ensure!(out.len() == batch, "argmax: out shape mismatch");
    let pl = stream.ptr(logits) as *const f32;
    let po = stream.ptr_mut(out) as *mut i32;
    // SAFETY: 指针与长度已校验。
    check(unsafe { ffi::argmax_f32(pl, po, batch as i64, vocab as i64, stream.raw() as _) }, "argmax_f32")
}

/// 就地对 logits 做 top-k / top-p 掩码。top_k / top_p: [batch]。
pub fn topk_topp_mask_f32<
    L: DevicePtrMut<f32> + ?Sized,
    K: DevicePtr<i32> + ?Sized,
    P: DevicePtr<f32> + ?Sized,
>(
    stream: &Stream,
    logits: &mut L,
    top_k: &K,
    top_p: &P,
    vocab: usize,
) -> Result<()> {
    ensure!(vocab > 0 && logits.len() % vocab == 0, "topk_topp_mask: bad shape");
    let batch = logits.len() / vocab;
    ensure!(top_k.len() == batch && top_p.len() == batch, "topk_topp_mask: param shape mismatch");
    let pl = stream.ptr_mut(logits) as *mut f32;
    let pk = stream.ptr(top_k) as *const i32;
    let pp = stream.ptr(top_p) as *const f32;
    // SAFETY: 指针与长度已校验。
    check(
        unsafe { ffi::topk_topp_mask_f32(pl, pk, pp, batch as i64, vocab as i64, stream.raw() as _) },
        "topk_topp_mask_f32",
    )
}

/// 温度缩放 + softmax + 多项式采样。temperature<=0 的行走 argmax。seeds/offsets: [batch]。
pub fn sample_f32<
    L: DevicePtr<f32> + ?Sized,
    T: DevicePtr<f32> + ?Sized,
    S1: DevicePtr<u64> + ?Sized,
    S2: DevicePtr<u64> + ?Sized,
    O: DevicePtrMut<i32> + ?Sized,
>(
    stream: &Stream,
    logits: &L,
    temperature: &T,
    seeds: &S1,
    offsets: &S2,
    out: &mut O,
    vocab: usize,
) -> Result<()> {
    ensure!(vocab > 0 && logits.len() % vocab == 0, "sample: bad shape");
    let batch = logits.len() / vocab;
    ensure!(
        temperature.len() == batch && seeds.len() == batch && offsets.len() == batch && out.len() == batch,
        "sample: param shape mismatch"
    );
    let pl = stream.ptr(logits) as *const f32;
    let pt = stream.ptr(temperature) as *const f32;
    let ps = stream.ptr(seeds) as *const u64;
    let pof = stream.ptr(offsets) as *const u64;
    let po = stream.ptr_mut(out) as *mut i32;
    // SAFETY: 指针与长度已校验。
    check(
        unsafe { ffi::sample_f32(pl, pt, ps, pof, po, batch as i64, vocab as i64, stream.raw() as _) },
        "sample_f32",
    )
}

// ---------------------------------------------------------------------------
// 带显式 token 数的变体：缓冲按 max_tokens 预分配，只用前 `tokens` 行。
// ---------------------------------------------------------------------------

/// out[:tokens*inter] = silu_and_mul(x[:tokens*2*inter])。
pub fn silu_and_mul_bf16_n(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    tokens: usize,
    inter: usize,
) -> Result<()> {
    ensure!(x.len() >= tokens * 2 * inter && out.len() >= tokens * inter, "silu_and_mul_n: buffer too small");
    let px = stream.ptr(x) as *const c_void;
    let po = stream.ptr_mut(out) as *mut c_void;
    // SAFETY: 缓冲容量已校验 ≥ 所需。
    check(
        unsafe { ffi::silu_and_mul_bf16(px, po, tokens as i64, inter as i64, stream.raw() as _) },
        "silu_and_mul_bf16",
    )
}

/// rmsnorm 前 `tokens` 行。
pub fn rmsnorm_bf16_n(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    tokens: usize,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    rmsnorm_bf16_layout(stream, x, w, out, tokens, hidden, eps, None)
}

/// 同上，`layout` 显式给（`None` = `ArchProfile::rmsnorm_layout` / `RMSNORM_LAYOUT=`）；`rmsnorm_bench` 用。
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_bf16_layout(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    tokens: usize,
    hidden: usize,
    eps: f32,
    layout: Option<crate::arch::RmsnormLayout>,
) -> Result<()> {
    ensure!(
        x.len() >= tokens * hidden && out.len() >= tokens * hidden && w.len() == hidden,
        "rmsnorm_n: shape"
    );
    let (px, pw) = (stream.ptr(x) as *const c_void, stream.ptr(w) as *const c_void);
    let po = stream.ptr_mut(out) as *mut c_void;
    // SAFETY: 缓冲容量已校验。
    check(
        unsafe {
            ffi::rmsnorm_bf16(
                px,
                pw,
                po,
                tokens as i64,
                hidden as i64,
                eps,
                rmsnorm_layout(stream, tokens, hidden, layout),
                norm_mode(),
                stream.raw() as _,
            )
        },
        "rmsnorm_bf16",
    )
}

/// router 预处理融合：`out = bf16(norm(x, 无权重) · rs · scale)`，**全程 fp32、只舍入一次**。
///
/// vLLM compile 口径（`docs/loops/kaon3-parity.md` 第 23 轮）：FX 图上 `rms_norm(None) → ×root_size → ×scale`
/// 三步都标 bf16，但 inductor 融成一条 fp32 链、只在写回时舍入一次（两步 bf16 版差 17299/67584，fp32 版 0）。
pub fn rmsnorm_scale2_bf16(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    scale: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    tokens: usize,
    hidden: usize,
    eps: f32,
    rs: f32,
) -> Result<()> {
    ensure!(
        x.len() >= tokens * hidden && out.len() >= tokens * hidden && scale.len() == hidden,
        "rmsnorm_scale2: shape"
    );
    let (px, ps) = (stream.ptr(x) as *const c_void, stream.ptr(scale) as *const c_void);
    let po = stream.ptr_mut(out) as *mut c_void;
    // SAFETY: 缓冲容量已校验。
    check(
        unsafe {
            ffi::rmsnorm_scale2_bf16(px, ps, po, tokens as i64, hidden as i64, eps, rs, norm_mode(), stream.raw() as _)
        },
        "rmsnorm_scale2_bf16",
    )
}

/// `out[i] = fp8(norm(x)[i] * (1/a_scale))`：norm 与 fp8 量化**融合**，中间不落 bf16。
///
/// 这是 vLLM **compile 口径**（默认；inductor 把 RMSNorm 融进下游量化 kernel）。eager BI 口径是
/// `rmsnorm_bf16` + `quant_fp8` 两步、中间一次 bf16 舍入——两者实测差 668/67584 个 fp8 元素
/// （max 一个 fp8 步长），详见 `docs/loops/kaon3-parity.md` 第 20 轮。
pub fn rmsnorm_quant_fp8(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    out: &mut CudaSlice<u8>,
    tokens: usize,
    hidden: usize,
    eps: f32,
    a_scale: f32,
) -> Result<()> {
    ensure!(
        x.len() >= tokens * hidden && out.len() >= tokens * hidden && w.len() == hidden,
        "rmsnorm_quant_fp8: shape"
    );
    let (px, pw) = (stream.ptr(x) as *const c_void, stream.ptr(w) as *const c_void);
    let po = stream.ptr_mut(out) as *mut c_void;
    // SAFETY: 缓冲容量已校验。
    check(
        unsafe {
            ffi::rmsnorm_quant_fp8_bf16(
                px,
                pw,
                po,
                tokens as i64,
                hidden as i64,
                eps,
                a_scale,
                stream.raw() as _,
            )
        },
        "rmsnorm_quant_fp8_bf16",
    )
}

/// out[r] = rmsnorm(x[rows[r]]) · w（gather + rmsnorm 融合）。
pub fn gather_rmsnorm_bf16(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    rows: &CudaSlice<i32>,
    w: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    num_rows: usize,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    ensure!(
        rows.len() >= num_rows && out.len() >= num_rows * hidden && w.len() == hidden,
        "gather_rmsnorm: shape"
    );
    let (px, pw) = (stream.ptr(x) as *const c_void, stream.ptr(w) as *const c_void);
    let pr = stream.ptr(rows) as *const i32;
    let po = stream.ptr_mut(out) as *mut c_void;
    // SAFETY: 容量已校验；rows 下标范围（< x 行数）由 runner 保证。
    check(
        unsafe {
            ffi::gather_rmsnorm_bf16(
                px,
                pr,
                pw,
                po,
                num_rows as i64,
                hidden as i64,
                eps,
                rmsnorm_layout(stream, num_rows, hidden, None),
                norm_mode(),
                stream.raw() as _,
            )
        },
        "gather_rmsnorm_bf16",
    )
}

/// fused_add_rmsnorm 前 `tokens` 行。
#[allow(clippy::too_many_arguments)]
pub fn fused_add_rmsnorm_bf16_n(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    res: &mut CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    tokens: usize,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    fused_add_rmsnorm_bf16_layout(stream, x, res, w, out, tokens, hidden, eps, None)
}

/// 同上，`layout` 显式给（`None` = `ArchProfile::rmsnorm_layout` / `RMSNORM_LAYOUT=`）；`rmsnorm_bench` 用。
#[allow(clippy::too_many_arguments)]
pub fn fused_add_rmsnorm_bf16_layout(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    res: &mut CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    tokens: usize,
    hidden: usize,
    eps: f32,
    layout: Option<crate::arch::RmsnormLayout>,
) -> Result<()> {
    ensure!(
        x.len() >= tokens * hidden
            && res.len() >= tokens * hidden
            && out.len() >= tokens * hidden
            && w.len() == hidden,
        "fused_add_rmsnorm_n: shape"
    );
    let px = stream.ptr(x) as *const c_void;
    let pw = stream.ptr(w) as *const c_void;
    let pr = stream.ptr_mut(res) as *mut c_void;
    let po = stream.ptr_mut(out) as *mut c_void;
    // SAFETY: 缓冲容量已校验。
    check(
        unsafe {
            ffi::fused_add_rmsnorm_bf16(
                px,
                pr,
                pw,
                po,
                tokens as i64,
                hidden as i64,
                eps,
                rmsnorm_layout(stream, tokens, hidden, layout),
                norm_mode(),
                stream.raw() as _,
            )
        },
        "fused_add_rmsnorm_bf16",
    )
}

/// embedding 前 `tokens` 个 id。
pub fn embedding_bf16_n(
    stream: &Stream,
    ids: &CudaSlice<i32>,
    table: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    tokens: usize,
    hidden: usize,
) -> Result<()> {
    ensure!(
        hidden > 0 && table.len() % hidden == 0 && ids.len() >= tokens && out.len() >= tokens * hidden,
        "embedding_n: shape"
    );
    let vocab = table.len() / hidden;
    let pi = stream.ptr(ids) as *const i32;
    let pt = stream.ptr(table) as *const c_void;
    let po = stream.ptr_mut(out) as *mut c_void;
    // SAFETY: 缓冲容量已校验。
    check(
        unsafe {
            ffi::embedding_bf16(pi, pt, po, tokens as i64, hidden as i64, vocab as i64, stream.raw() as _)
        },
        "embedding_bf16",
    )
}

/// rope 前 `tokens` 行（q/k 缓冲可大于 tokens）。
#[allow(clippy::too_many_arguments)]
pub fn rope_bf16_n(
    stream: &Stream,
    q: &mut CudaSlice<bf16>,
    k: Option<&mut CudaSlice<bf16>>,
    positions: &CudaSlice<i32>,
    cos_sin: &CudaSlice<f32>,
    tokens: usize,
    q_heads: usize,
    k_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    style: RopeStyle,
) -> Result<()> {
    ensure!(
        q.len() >= tokens * q_heads * head_dim && positions.len() >= tokens,
        "rope_n: q.len {} < tokens {} * q_heads {} * head_dim {} = {}, or positions.len {} < tokens",
        q.len(),
        tokens,
        q_heads,
        head_dim,
        tokens * q_heads * head_dim,
        positions.len()
    );
    ensure!(rope_dim > 0 && rope_dim % 2 == 0 && rope_dim <= head_dim, "rope_n: bad rope_dim {rope_dim}");
    let pq = stream.ptr_mut(q) as *mut c_void;
    let pk = match k {
        Some(k) => {
            ensure!(k.len() >= tokens * k_heads * head_dim, "rope_n: k");
            stream.ptr_mut(k) as *mut c_void
        }
        None => {
            ensure!(k_heads == 0, "rope_n: k_heads > 0 but k is None");
            std::ptr::null_mut()
        }
    };
    let pp = stream.ptr(positions) as *const i32;
    let pc = stream.ptr(cos_sin) as *const f32;
    // SAFETY: 缓冲容量已校验。
    check(
        unsafe {
            ffi::rope_bf16(
                pq,
                pk,
                pp,
                pc,
                tokens as i64,
                q_heads as i64,
                k_heads as i64,
                head_dim as i64,
                rope_dim as i64,
                matches!(style, RopeStyle::Neox) as i32,
                stream.raw() as _,
            )
        },
        "rope_bf16",
    )
}

/// argmax 前 `batch` 行。
pub fn argmax_f32_n<L: DevicePtr<f32> + ?Sized, O: DevicePtrMut<i32> + ?Sized>(
    stream: &Stream,
    logits: &L,
    out: &mut O,
    batch: usize,
    vocab: usize,
) -> Result<()> {
    ensure!(logits.len() >= batch * vocab && out.len() >= batch, "argmax_n: shape");
    let pl = stream.ptr(logits) as *const f32;
    let po = stream.ptr_mut(out) as *mut i32;
    // SAFETY: 缓冲容量已校验。
    check(unsafe { ffi::argmax_f32(pl, po, batch as i64, vocab as i64, stream.raw() as _) }, "argmax_f32")
}

/// 融合采样：在 `work`（[batch, vocab] fp32，原地改）上依次 temperature → min_p → top-k/top-p → sample，
/// 受限解码 mask + logit_bias（原地，penalties 之前；vLLM 顺序）。`mask_row[b] < 0` = 该行无 mask，否则读
/// `mask_bits[mask_row[b] * mask_words ..]`，bit=0 的列置 -inf；bias 为 CSR（`bias_offsets` 长 batch+1），None = 无 bias。
/// 无约束行 kernel 立即返回，整批无约束时只剩一次 launch 的固定开销（CUDA graph 内常驻）。
#[allow(clippy::too_many_arguments)]
pub fn logits_mask_f32<W: DevicePtrMut<f32> + ?Sized>(
    stream: &Stream,
    logits: &mut W,
    mask_row: &CudaSlice<i32>,
    mask_bits: &CudaSlice<u32>,
    mask_words: usize,
    bias: Option<(&CudaSlice<i32>, &CudaSlice<i32>, &CudaSlice<f32>)>,
    batch: usize,
    vocab: usize,
) -> Result<()> {
    ensure!(vocab > 0 && logits.len() >= batch * vocab, "logits_mask: shape");
    ensure!(mask_row.len() >= batch, "logits_mask: mask_row");
    ensure!(mask_words == 0 || mask_bits.len() >= mask_words, "logits_mask: mask_bits");
    let (po, pi, pv) = match bias {
        Some((o, i, v)) => {
            ensure!(o.len() >= batch + 1 && i.len() == v.len(), "logits_mask: bias csr");
            (stream.ptr(o) as *const i32, stream.ptr(i) as *const i32, stream.ptr(v) as *const f32)
        }
        None => (std::ptr::null(), std::ptr::null(), std::ptr::null()),
    };
    let pl = stream.ptr_mut(logits) as *mut f32;
    // SAFETY: 长度已校验；mask_row 的值由 host 侧保证 < mask_bits.len() / mask_words。
    check(
        unsafe {
            ffi::logits_mask_f32(
                pl,
                stream.ptr(mask_row) as *const i32,
                stream.ptr(mask_bits) as *const u32,
                mask_words as i64,
                po,
                pi,
                pv,
                batch as i64,
                vocab as i64,
                stream.raw() as _,
            )
        },
        "logits_mask_f32",
    )
}

/// 一次 launch，语义 = `temperature_f32 → min_p_mask_f32 → topk_topp_mask_f32 → sample_f32(temp01)` 串行。
#[allow(clippy::too_many_arguments)]
pub fn sample_fused_f32<W: DevicePtrMut<f32> + ?Sized, O: DevicePtrMut<i32> + ?Sized>(
    stream: &Stream,
    work: &mut W,
    temperature: &CudaSlice<f32>,
    min_p: &CudaSlice<f32>,
    top_k: &CudaSlice<i32>,
    top_p: &CudaSlice<f32>,
    temp01: &CudaSlice<f32>,
    seeds: &CudaSlice<u64>,
    offsets: &CudaSlice<u64>,
    out: &mut O,
    batch: usize,
    vocab: usize,
) -> Result<()> {
    ensure!(vocab > 0 && work.len() >= batch * vocab && out.len() >= batch, "sample_fused: shape");
    ensure!(
        [temperature.len(), min_p.len(), top_p.len(), temp01.len()].iter().all(|&l| l >= batch)
            && top_k.len() >= batch
            && seeds.len() >= batch
            && offsets.len() >= batch,
        "sample_fused: param shape"
    );
    let pw = stream.ptr_mut(work) as *mut f32;
    let po = stream.ptr_mut(out) as *mut i32;
    // SAFETY: 容量已校验。
    check(
        unsafe {
            ffi::sample_fused_f32(
                pw,
                stream.ptr(temperature) as *const f32,
                stream.ptr(min_p) as *const f32,
                stream.ptr(top_k) as *const i32,
                stream.ptr(top_p) as *const f32,
                stream.ptr(temp01) as *const f32,
                stream.ptr(seeds) as *const u64,
                stream.ptr(offsets) as *const u64,
                po,
                batch as i64,
                vocab as i64,
                stream.raw() as _,
            )
        },
        "sample_fused_f32",
    )
}

/// KV 打包（host tier / PD）：把 `blocks[..n]` 的全部层打包进 `staging`（设备，[n][L][per_layer_bytes]），
/// 每层内转成 **head-major** `[kv_head][slot][K/V][D]`（cache 是 `[slot][kv_head][2][D]`），TP 分片按头连续（design §13.7）。
/// `layer_ptrs`：设备数组，每层 cache 的基址（u64）。`per_layer_bytes = block_size × kv_heads × 2 × D × 2`。
pub fn kv_pack(
    stream: &Stream,
    layer_ptrs: &CudaSlice<u64>,
    blocks: &CudaSlice<i32>,
    staging: &mut CudaSlice<u8>,
    n_blocks: usize,
    num_layers: usize,
    per_layer_bytes: usize,
    block_size: usize,
    kv_heads: usize,
) -> Result<()> {
    ensure!(
        layer_ptrs.len() >= num_layers
            && blocks.len() >= n_blocks
            && staging.len() >= n_blocks * num_layers * per_layer_bytes,
        "kv_pack: shape"
    );
    ensure!(
        block_size > 0 && kv_heads > 0 && per_layer_bytes % (block_size * kv_heads) == 0,
        "kv_pack: per_layer_bytes vs bs×hk"
    );
    let lp = stream.ptr(layer_ptrs) as *const *const c_void;
    let pb = stream.ptr(blocks) as *const i32;
    let ps = stream.ptr_mut(staging) as *mut c_void;
    // SAFETY: 容量已校验；layer_ptrs 内容是有效设备地址（由 executor 填）。
    check(
        unsafe {
            ffi::kv_pack(
                lp,
                pb,
                ps,
                n_blocks as i64,
                num_layers as i64,
                per_layer_bytes as i64,
                block_size as i64,
                kv_heads as i64,
                stream.raw() as _,
            )
        },
        "kv_pack",
    )
}
/// 反向：staging → 各层 cache 的 `blocks[..n]`。
pub fn kv_unpack(
    stream: &Stream,
    layer_ptrs: &CudaSlice<u64>,
    blocks: &CudaSlice<i32>,
    staging: &CudaSlice<u8>,
    n_blocks: usize,
    num_layers: usize,
    per_layer_bytes: usize,
    block_size: usize,
    kv_heads: usize,
) -> Result<()> {
    ensure!(
        layer_ptrs.len() >= num_layers
            && blocks.len() >= n_blocks
            && staging.len() >= n_blocks * num_layers * per_layer_bytes,
        "kv_unpack: shape"
    );
    ensure!(
        block_size > 0 && kv_heads > 0 && per_layer_bytes % (block_size * kv_heads) == 0,
        "kv_unpack: per_layer_bytes vs bs×hk"
    );
    let lp = stream.ptr(layer_ptrs) as *const *mut c_void;
    let pb = stream.ptr(blocks) as *const i32;
    let ps = stream.ptr(staging) as *const c_void;
    // SAFETY: 同上。
    check(
        unsafe {
            ffi::kv_unpack(
                lp,
                pb,
                ps,
                n_blocks as i64,
                num_layers as i64,
                per_layer_bytes as i64,
                block_size as i64,
                kv_heads as i64,
                stream.raw() as _,
            )
        },
        "kv_unpack",
    )
}

/// 异步调度：`ids[i] = prev[gather_idx[i]]`（gather_idx[i] < 0 的行保持 host 上传的值）。可进 CUDA graph。
pub fn row_gather_i32(
    stream: &Stream,
    prev: &CudaSlice<i32>,
    gather_idx: &CudaSlice<i32>,
    ids: &mut CudaSlice<i32>,
    n: usize,
) -> Result<()> {
    ensure!(gather_idx.len() >= n && ids.len() >= n, "row_gather_i32: shape");
    let pp = stream.ptr(prev) as *const i32;
    let pg = stream.ptr(gather_idx) as *const i32;
    let pi = stream.ptr_mut(ids) as *mut i32;
    // SAFETY: 容量已校验；prev 下标范围由 runner 保证（< 上一步行数）。
    check(unsafe { ffi::row_gather_i32(pp, pg, pi, n as i64, stream.raw() as _) }, "row_gather_i32")
}

// ---------------------------------------------------------------------------
// MLA
// ---------------------------------------------------------------------------

/// latent[tokens, entry] → paged cache。slot_mapping 为 -1 的 token 跳过。
pub fn mla_cache_write_bf16(
    stream: &Stream,
    latent: &CudaSlice<bf16>,
    cache: &mut CudaSlice<bf16>,
    slot_mapping: &CudaSlice<i64>,
    tokens: usize,
    entry: usize,
    block_size: usize,
) -> Result<()> {
    ensure!(latent.len() >= tokens * entry && slot_mapping.len() >= tokens, "mla_cache_write: shape");
    ensure!(cache.len() % (entry * block_size) == 0, "mla_cache_write: cache not multiple of page");
    let pl = stream.ptr(latent) as *const c_void;
    let pc = stream.ptr_mut(cache) as *mut c_void;
    let ps = stream.ptr(slot_mapping) as *const i64;
    // SAFETY: 缓冲容量已校验；slot 越界由调度器保证 < num_blocks*block_size。
    check(
        unsafe {
            ffi::mla_cache_write_bf16(
                pl,
                pc,
                ps,
                tokens as i64,
                entry as i64,
                block_size as i64,
                stream.raw() as _,
            )
        },
        "mla_cache_write_bf16",
    )
}

/// causal attention（展开 K/V）。q/k: [T, heads, dq]，v/out: [T, heads, dv]。
#[allow(clippy::too_many_arguments)]
pub fn mla_prefill_bf16(
    stream: &Stream,
    q: &CudaSlice<bf16>,
    k: &CudaSlice<bf16>,
    v: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    cu_seqlens: &CudaSlice<i32>,
    topk: Option<(&CudaSlice<i32>, usize)>,
    num_reqs: usize,
    total_tokens: usize,
    heads: usize,
    dq: usize,
    dv: usize,
    scale: f32,
) -> Result<()> {
    ensure!(cu_seqlens.len() > num_reqs, "mla_prefill: cu_seqlens");
    let (pti, tk) = topk_ptr(stream, topk, total_tokens, "mla_prefill")?;
    ensure!(q.len() >= total_tokens * heads * dq && k.len() >= total_tokens * heads * dq, "mla_prefill: q/k");
    ensure!(
        v.len() >= total_tokens * heads * dv && out.len() >= total_tokens * heads * dv,
        "mla_prefill: v/out"
    );
    let (pq, pk, pv) =
        (stream.ptr(q) as *const c_void, stream.ptr(k) as *const c_void, stream.ptr(v) as *const c_void);
    let po = stream.ptr_mut(out) as *mut c_void;
    let pc = stream.ptr(cu_seqlens) as *const i32;
    // SAFETY: 缓冲容量已校验；cu_seqlens 单调且末值 == total_tokens 由调用方保证。
    check(
        unsafe {
            ffi::mla_prefill_bf16(
                pq,
                pk,
                pv,
                po,
                pc,
                pti,
                tk,
                num_reqs as i64,
                heads as i64,
                dq as i64,
                dv as i64,
                scale,
                stream.raw() as _,
            )
        },
        "mla_prefill_bf16",
    )
}

/// 吸收后 MQA、任意 query 长、历史 KV 在 paged cache 中。q: [total_q, heads, entry]，out: [total_q, heads, kv_lora]。
/// `seq_lens[b]` = 该请求历史 + 本步 query 数；本步的 latent 必须已写入 cache。
#[allow(clippy::too_many_arguments)]
pub fn mla_prefill_paged_bf16(
    stream: &Stream,
    q: &CudaSlice<bf16>,
    cache: &CudaSlice<bf16>,
    block_table: &CudaSlice<i32>,
    seq_lens: &CudaSlice<i32>,
    cu_seqlens_q: &CudaSlice<i32>,
    topk: Option<(&CudaSlice<i32>, usize)>,
    out: &mut CudaSlice<bf16>,
    num_reqs: usize,
    total_q: usize,
    heads: usize,
    entry: usize,
    kv_lora: usize,
    block_size: usize,
    max_blocks: usize,
    scale: f32,
) -> Result<()> {
    ensure!(
        q.len() >= total_q * heads * entry && out.len() >= total_q * heads * kv_lora,
        "mla_prefill_paged: q/out"
    );
    ensure!(
        block_table.len() >= num_reqs * max_blocks
            && seq_lens.len() >= num_reqs
            && cu_seqlens_q.len() > num_reqs,
        "mla_prefill_paged: table/seq_lens/cu"
    );
    ensure!(kv_lora <= entry, "mla_prefill_paged: kv_lora > entry");
    let (pti, tk) = topk_ptr(stream, topk, total_q, "mla_prefill_paged")?;
    let pq = stream.ptr(q) as *const c_void;
    let pc = stream.ptr(cache) as *const c_void;
    let pbt = stream.ptr(block_table) as *const i32;
    let psl = stream.ptr(seq_lens) as *const i32;
    let pcu = stream.ptr(cu_seqlens_q) as *const i32;
    let po = stream.ptr_mut(out) as *mut c_void;
    // SAFETY: 缓冲容量已校验；block_table 内 id < num_blocks、cu_seqlens_q 末值 == total_q 由调用方保证。
    check(
        unsafe {
            ffi::mla_prefill_paged_bf16(
                pq,
                pc,
                pbt,
                psl,
                pcu,
                pti,
                tk,
                po,
                num_reqs as i64,
                heads as i64,
                entry as i64,
                kv_lora as i64,
                block_size as i64,
                max_blocks as i64,
                scale,
                stream.raw() as _,
            )
        },
        "mla_prefill_paged_bf16",
    )
}

/// 吸收后 MQA decode。q: [batch, heads, entry]，out: [batch, heads, kv_lora]。
#[allow(clippy::too_many_arguments)]
pub fn mla_decode_bf16(
    stream: &Stream,
    q: &CudaSlice<bf16>,
    cache: &CudaSlice<bf16>,
    block_table: &CudaSlice<i32>,
    seq_lens: &CudaSlice<i32>,
    topk: Option<(&CudaSlice<i32>, usize)>,
    out: &mut CudaSlice<bf16>,
    batch: usize,
    heads: usize,
    entry: usize,
    kv_lora: usize,
    block_size: usize,
    max_blocks: usize,
    scale: f32,
) -> Result<()> {
    ensure!(q.len() >= batch * heads * entry && out.len() >= batch * heads * kv_lora, "mla_decode: q/out");
    ensure!(block_table.len() >= batch * max_blocks && seq_lens.len() >= batch, "mla_decode: table/seq_lens");
    ensure!(kv_lora <= entry, "mla_decode: kv_lora > entry");
    let (pti, tk) = topk_ptr(stream, topk, batch, "mla_decode")?;
    let pq = stream.ptr(q) as *const c_void;
    let pc = stream.ptr(cache) as *const c_void;
    let pbt = stream.ptr(block_table) as *const i32;
    let psl = stream.ptr(seq_lens) as *const i32;
    let po = stream.ptr_mut(out) as *mut c_void;
    // SAFETY: 缓冲容量已校验；block_table 内 id < num_blocks 由调度器保证。
    check(
        unsafe {
            ffi::mla_decode_bf16(
                pq,
                pc,
                pbt,
                psl,
                pti,
                tk,
                po,
                batch as i64,
                heads as i64,
                entry as i64,
                kv_lora as i64,
                block_size as i64,
                max_blocks as i64,
                scale,
                stream.raw() as _,
            )
        },
        "mla_decode_bf16",
    )
}

// ---------------------------------------------------------------------------
// 布局搬运
// ---------------------------------------------------------------------------

/// dst[t, dcol0:dcol0+width] = src[t, col0:col0+width]，前 `tokens` 行。
#[allow(clippy::too_many_arguments)]
pub fn copy_cols_bf16(
    stream: &Stream,
    src: &CudaSlice<bf16>,
    dst: &mut CudaSlice<bf16>,
    tokens: usize,
    src_cols: usize,
    col0: usize,
    dst_cols: usize,
    dcol0: usize,
    width: usize,
) -> Result<()> {
    // 参数顺序与 C ABI 一致：(src_cols, col0, dst_cols, dcol0, width)。
    ensure!(src.len() >= tokens * src_cols && dst.len() >= tokens * dst_cols, "copy_cols: buffer");
    ensure!(
        col0 + width <= src_cols && dcol0 + width <= dst_cols,
        "copy_cols: range src_cols={src_cols} col0={col0} dst_cols={dst_cols} dcol0={dcol0} width={width}"
    );
    let ps = stream.ptr(src) as *const c_void;
    let pd = stream.ptr_mut(dst) as *mut c_void;
    // SAFETY: 范围与容量已校验。
    check(
        unsafe {
            ffi::copy_cols_bf16(
                ps,
                pd,
                tokens as i64,
                src_cols as i64,
                col0 as i64,
                dst_cols as i64,
                dcol0 as i64,
                width as i64,
                stream.raw() as _,
            )
        },
        "copy_cols_bf16",
    )
}

/// k[t, h, nope:nope+rope] = kpe[t, :]，k: [t, heads, qk]。
#[allow(clippy::too_many_arguments)]
pub fn bcast_kpe_bf16(
    stream: &Stream,
    kpe: &CudaSlice<bf16>,
    k: &mut CudaSlice<bf16>,
    tokens: usize,
    heads: usize,
    qk: usize,
    nope: usize,
    rope: usize,
) -> Result<()> {
    ensure!(
        kpe.len() >= tokens * rope && k.len() >= tokens * heads * qk && nope + rope == qk,
        "bcast_kpe: shape"
    );
    let ps = stream.ptr(kpe) as *const c_void;
    let pd = stream.ptr_mut(k) as *mut c_void;
    // SAFETY: 容量已校验。
    check(
        unsafe {
            ffi::bcast_kpe_bf16(
                ps,
                pd,
                tokens as i64,
                heads as i64,
                qk as i64,
                nope as i64,
                rope as i64,
                stream.raw() as _,
            )
        },
        "bcast_kpe_bf16",
    )
}

/// 对 [t, heads, in_w] 每 head 取列区间写到 [t, heads, out_w] 的列区间。
#[allow(clippy::too_many_arguments)]
pub fn copy_head_cols_bf16(
    stream: &Stream,
    src: &CudaSlice<bf16>,
    dst: &mut CudaSlice<bf16>,
    tokens: usize,
    heads: usize,
    in_w: usize,
    in_col0: usize,
    out_w: usize,
    out_col0: usize,
    width: usize,
) -> Result<()> {
    ensure!(
        src.len() >= tokens * heads * in_w && dst.len() >= tokens * heads * out_w,
        "copy_head_cols: buffer"
    );
    ensure!(in_col0 + width <= in_w && out_col0 + width <= out_w, "copy_head_cols: range");
    let ps = stream.ptr(src) as *const c_void;
    let pd = stream.ptr_mut(dst) as *mut c_void;
    // SAFETY: 范围与容量已校验。
    check(
        unsafe {
            ffi::copy_head_cols_bf16(
                ps,
                pd,
                tokens as i64,
                heads as i64,
                in_w as i64,
                in_col0 as i64,
                out_w as i64,
                out_col0 as i64,
                width as i64,
                stream.raw() as _,
            )
        },
        "copy_head_cols_bf16",
    )
}

/// dst[t, :width] = src[t, head, col0:col0+width]，src: [t, heads, w]。
#[allow(clippy::too_many_arguments)]
pub fn gather_head_bf16(
    stream: &Stream,
    src: &CudaSlice<bf16>,
    dst: &mut CudaSlice<bf16>,
    tokens: usize,
    heads: usize,
    w: usize,
    head: usize,
    col0: usize,
    width: usize,
) -> Result<()> {
    ensure!(src.len() >= tokens * heads * w && dst.len() >= tokens * width, "gather_head: buffer");
    ensure!(head < heads && col0 + width <= w, "gather_head: range");
    let ps = stream.ptr(src) as *const c_void;
    let pd = stream.ptr_mut(dst) as *mut c_void;
    // SAFETY: 范围与容量已校验。
    check(
        unsafe {
            ffi::gather_head_bf16(
                ps,
                pd,
                tokens as i64,
                heads as i64,
                w as i64,
                head as i64,
                col0 as i64,
                width as i64,
                stream.raw() as _,
            )
        },
        "gather_head_bf16",
    )
}

/// dst[t, head, col0:col0+width] = src[t, :width]，dst: [t, heads, w]。
#[allow(clippy::too_many_arguments)]
pub fn scatter_head_bf16(
    stream: &Stream,
    src: &CudaSlice<bf16>,
    dst: &mut CudaSlice<bf16>,
    tokens: usize,
    heads: usize,
    w: usize,
    head: usize,
    col0: usize,
    width: usize,
) -> Result<()> {
    ensure!(src.len() >= tokens * width && dst.len() >= tokens * heads * w, "scatter_head: buffer");
    ensure!(head < heads && col0 + width <= w, "scatter_head: range");
    let ps = stream.ptr(src) as *const c_void;
    let pd = stream.ptr_mut(dst) as *mut c_void;
    // SAFETY: 范围与容量已校验。
    check(
        unsafe {
            ffi::scatter_head_bf16(
                ps,
                pd,
                tokens as i64,
                heads as i64,
                w as i64,
                head as i64,
                col0 as i64,
                width as i64,
                stream.raw() as _,
            )
        },
        "scatter_head_bf16",
    )
}

// ---------------------------------------------------------------------------
// 逐元素
// ---------------------------------------------------------------------------

/// y[:n] += x[:n]
pub fn add_inplace_bf16_n(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    y: &mut CudaSlice<bf16>,
    n: usize,
) -> Result<()> {
    ensure!(x.len() >= n && y.len() >= n, "add_inplace: buffer");
    let px = stream.ptr(x) as *const c_void;
    let py = stream.ptr_mut(y) as *mut c_void;
    // SAFETY: 容量已校验。
    check(unsafe { ffi::add_inplace_bf16(px, py, n as i64, stream.raw() as _) }, "add_inplace_bf16")
}

/// bf16 原地乘标量：`x *= a`（前 `n` 个元素）。
pub fn scale_bf16_n(stream: &Stream, x: &mut CudaSlice<bf16>, n: usize, a: f32) -> Result<()> {
    ensure!(x.len() >= n, "scale_bf16: buffer");
    // SAFETY: 长度已校验。
    check(
        unsafe { ffi::scale_bf16(stream.ptr_mut(x) as *mut c_void, n as i64, a, stream.raw() as _) },
        "scale_bf16",
    )
}

/// `x[:n] = bf16(e4m3(x / scale) · scale)`（饱和 ±448、RNE）：模拟 FP8 KV cache 的存取误差，`scale` 即 vLLM 的
/// `k_scale` / `v_scale`（checkpoint 没给时为 1.0）。
pub fn fp8_e4m3_roundtrip_bf16_n(stream: &Stream, x: &mut CudaSlice<bf16>, n: usize, scale: f32) -> Result<()> {
    ensure!(x.len() >= n, "fp8_e4m3_roundtrip_bf16: buffer");
    // SAFETY: 长度已校验。
    check(
        unsafe { ffi::fp8_e4m3_roundtrip_bf16(stream.ptr_mut(x) as *mut c_void, n as i64, scale, stream.raw() as _) },
        "fp8_e4m3_roundtrip_bf16",
    )
}

/// dst[r, :] = src[rows[r], :]
pub fn gather_rows_bf16(
    stream: &Stream,
    src: &CudaSlice<bf16>,
    dst: &mut CudaSlice<bf16>,
    rows: &CudaSlice<i32>,
    num_rows: usize,
    width: usize,
) -> Result<()> {
    ensure!(
        rows.len() >= num_rows && dst.len() >= num_rows * width && src.len() % width == 0,
        "gather_rows: buffer"
    );
    let ps = stream.ptr(src) as *const c_void;
    let pd = stream.ptr_mut(dst) as *mut c_void;
    let pr = stream.ptr(rows) as *const i32;
    // SAFETY: 容量已校验；行号越界由调用方保证 < src.len()/width。
    check(
        unsafe { ffi::gather_rows_bf16(ps, pd, pr, num_rows as i64, width as i64, stream.raw() as _) },
        "gather_rows_bf16",
    )
}

/// y[:n] = f32(x[:n])
pub fn bf16_to_f32_n(stream: &Stream, x: &CudaSlice<bf16>, y: &mut CudaSlice<f32>, n: usize) -> Result<()> {
    ensure!(x.len() >= n && y.len() >= n, "bf16_to_f32: buffer");
    let px = stream.ptr(x) as *const c_void;
    let py = stream.ptr_mut(y) as *mut f32;
    // SAFETY: 容量已校验。
    check(unsafe { ffi::bf16_to_f32(px, py, n as i64, stream.raw() as _) }, "bf16_to_f32")
}

/// y[rows, nb*cols] (f32) = 拼接 x 的 nb 个 [rows, cols] bf16 块（块 b 落到列段 [b*cols, (b+1)*cols)）。
/// 用于 TP 按输出维切分的 GEMM 结果经 `all_gather` 后的 `[world][rows][cols]` 块布局：一趟修正布局 + 升 f32。
pub fn gather_blocks_bf16_to_f32(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    y: &mut CudaSlice<f32>,
    nb: usize,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let n = nb * rows * cols;
    ensure!(
        x.len() >= n && y.len() >= n,
        "gather_blocks_bf16_to_f32: buffer ({} / {} < {n})",
        x.len(),
        y.len()
    );
    let px = stream.ptr(x) as *const c_void;
    let py = stream.ptr_mut(y) as *mut f32;
    // SAFETY: 容量已校验。
    check(
        unsafe {
            ffi::gather_blocks_bf16_to_f32(px, py, nb as i64, rows as i64, cols as i64, stream.raw() as _)
        },
        "gather_blocks_bf16_to_f32",
    )
}

/// dst[:n] = src[:n]（i32，设备内拷贝）。
pub fn copy_i32_n(stream: &Stream, src: &CudaSlice<i32>, dst: &mut CudaSlice<i32>, n: usize) -> Result<()> {
    stream.dtod_copy_n(src, dst, n)
}

/// x[:n] *= a（fp32，就地）。
pub fn scale_f32_n(stream: &Stream, x: &mut CudaSlice<f32>, n: usize, a: f32) -> Result<()> {
    ensure!(x.len() >= n, "scale_f32: buffer");
    let px = stream.ptr_mut(x) as *mut f32;
    // SAFETY: 容量已校验。
    check(unsafe { ffi::scale_f32(px, n as i64, a, stream.raw() as _) }, "scale_f32")
}

// ---------------------------------------------------------------------------
// DSA 索引器
// ---------------------------------------------------------------------------

/// 可选 top-k 索引 → (裸指针或 null, topk)。`rows`：需要覆盖的 query 行数。
fn topk_ptr(
    stream: &Stream,
    topk: Option<(&CudaSlice<i32>, usize)>,
    rows: usize,
    what: &str,
) -> Result<(*const i32, i64)> {
    match topk {
        None => Ok((std::ptr::null(), 0)),
        Some((t, k)) => {
            ensure!(k > 0 && t.len() >= rows * k, "{what}: topk_indices capacity ({} < {rows}*{k})", t.len());
            Ok((stream.ptr(t) as *const i32, k as i64))
        }
    }
}

/// LayerNorm（带 bias）：x/out [tokens, hidden] bf16，w/b [hidden] bf16，fp32 计算。
#[allow(clippy::too_many_arguments)]
pub fn layernorm_bf16_n(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    b: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    tokens: usize,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    ensure!(x.len() >= tokens * hidden && out.len() >= tokens * hidden, "layernorm: buffer");
    ensure!(w.len() == hidden && b.len() == hidden, "layernorm: w/b");
    let px = stream.ptr(x) as *const c_void;
    let pw = stream.ptr(w) as *const c_void;
    let pb = stream.ptr(b) as *const c_void;
    let po = stream.ptr_mut(out) as *mut c_void;
    // SAFETY: 容量已校验。
    check(
        unsafe { ffi::layernorm_bf16(px, pw, pb, po, tokens as i64, hidden as i64, eps, stream.raw() as _) },
        "layernorm_bf16",
    )
}

/// vis[row] = 该 query 行的 causal 可见 key 数。
pub fn dsa_row_vis_i32(
    stream: &Stream,
    seq_lens: &CudaSlice<i32>,
    cu_seqlens_q: &CudaSlice<i32>,
    vis: &mut CudaSlice<i32>,
    num_reqs: usize,
    total_q: usize,
) -> Result<()> {
    ensure!(
        seq_lens.len() >= num_reqs && cu_seqlens_q.len() > num_reqs && vis.len() >= total_q,
        "dsa_row_vis"
    );
    let psl = stream.ptr(seq_lens) as *const i32;
    let pcu = stream.ptr(cu_seqlens_q) as *const i32;
    let pv = stream.ptr_mut(vis) as *mut i32;
    // SAFETY: 容量已校验；cu_seqlens_q 末值 == total_q 由调用方保证。
    check(
        unsafe { ffi::dsa_row_vis_i32(psl, pcu, pv, num_reqs as i64, stream.raw() as _) },
        "dsa_row_vis_i32",
    )
}

/// score[row, j] = Σ_h w[row,h] · relu(q[row,h]·k_cache[j] · scale)，j < vis(row)，其余 -inf。
#[allow(clippy::too_many_arguments)]
pub fn dsa_index_score_f32(
    stream: &Stream,
    q: &CudaSlice<bf16>,
    w: &CudaSlice<f32>,
    k_cache: &CudaSlice<bf16>,
    block_table: &CudaSlice<i32>,
    seq_lens: &CudaSlice<i32>,
    cu_seqlens_q: &CudaSlice<i32>,
    score: &mut CudaSlice<f32>,
    num_reqs: usize,
    total_q: usize,
    heads: usize,
    head_dim: usize,
    block_size: usize,
    max_blocks: usize,
    max_seq: usize,
    scale: f32,
) -> Result<()> {
    ensure!(q.len() >= total_q * heads * head_dim && w.len() >= total_q * heads, "dsa_index_score: q/w");
    ensure!(score.len() >= total_q * max_seq, "dsa_index_score: score [{total_q}, {max_seq}]");
    ensure!(
        block_table.len() >= num_reqs * max_blocks
            && seq_lens.len() >= num_reqs
            && cu_seqlens_q.len() > num_reqs,
        "dsa_index_score: table/seq_lens/cu"
    );
    // block_table 宽度只需覆盖每个请求自己的 seq_len（kernel 只读 bt[j / block_size], j < vis）；不要求覆盖 max_seq
    let pq = stream.ptr(q) as *const c_void;
    let pw = stream.ptr(w) as *const f32;
    let pk = stream.ptr(k_cache) as *const c_void;
    let pbt = stream.ptr(block_table) as *const i32;
    let psl = stream.ptr(seq_lens) as *const i32;
    let pcu = stream.ptr(cu_seqlens_q) as *const i32;
    let ps = stream.ptr_mut(score) as *mut f32;
    // SAFETY: 容量已校验；block_table 内 id < num_blocks 由调度器保证；seq_lens ≤ max_seq 由调用方保证。
    check(
        unsafe {
            ffi::dsa_index_score_f32(
                pq,
                pw,
                pk,
                pbt,
                psl,
                pcu,
                ps,
                num_reqs as i64,
                heads as i64,
                head_dim as i64,
                block_size as i64,
                max_blocks as i64,
                max_seq as i64,
                scale,
                stream.raw() as _,
            )
        },
        "dsa_index_score_f32",
    )
}

/// 每行 top-k 下标（相等取小下标），不足 k 填 -1。
pub fn dsa_topk_i32(
    stream: &Stream,
    score: &CudaSlice<f32>,
    vis: &CudaSlice<i32>,
    out: &mut CudaSlice<i32>,
    rows: usize,
    max_seq: usize,
    topk: usize,
) -> Result<()> {
    ensure!(
        score.len() >= rows * max_seq && vis.len() >= rows && out.len() >= rows * topk,
        "dsa_topk: buffer"
    );
    let ps = stream.ptr(score) as *const f32;
    let pv = stream.ptr(vis) as *const i32;
    let po = stream.ptr_mut(out) as *mut i32;
    // SAFETY: 容量已校验。
    check(
        unsafe { ffi::dsa_topk_i32(ps, pv, po, rows as i64, max_seq as i64, topk as i64, stream.raw() as _) },
        "dsa_topk_i32",
    )
}

// ---------------------------------------------------------------------------
// MoE
// ---------------------------------------------------------------------------

/// 路由 logits（fp32）：logits[tokens, E] = x[tokens, H] · w[E, H]^T。
pub fn moe_gate_f32(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    logits: &mut CudaSlice<f32>,
    tokens: usize,
    num_experts: usize,
    hidden: usize,
) -> Result<()> {
    ensure!(x.len() >= tokens * hidden && w.len() == num_experts * hidden, "moe_gate: input");
    ensure!(logits.len() >= tokens * num_experts, "moe_gate: logits");
    let px = stream.ptr(x) as *const c_void;
    let pw = stream.ptr(w) as *const c_void;
    let pl = stream.ptr_mut(logits) as *mut f32;
    // SAFETY: 容量已校验。
    check(
        unsafe {
            ffi::moe_gate_f32(px, pw, pl, tokens as i64, num_experts as i64, hidden as i64, stream.raw() as _)
        },
        "moe_gate_f32",
    )
}

/// router 的 **BI persistent matmul**（复刻 vLLM `matmul_persistent`）：
/// `out[T,E] f32 = x[T,H] bf16 · w[E,H]ᵀ bf16`，**K 方向严格顺序累加、末尾舍 bf16 再转 f32**。
///
/// 为什么不能换 cuBLAS：同形状（M×128×2816）实测 **43% 输出元素不同**（max 7.8e-3），
/// 而 router logits 差一个 bf16 ulp 就会在 top-8 边界换专家 → 端到端 flip 16%。
/// 见 `docs/loops/kaon3-compile.md` 第 5 轮。
///
/// `hidden` 必须是 64 的倍数、`num_experts` 必须是 8 的倍数（参照 BLOCK 配置的前提）。
pub fn router_gemm_bi_f32(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    out: &mut CudaSlice<f32>,
    tokens: usize,
    num_experts: usize,
    hidden: usize,
) -> Result<()> {
    ensure!(x.len() >= tokens * hidden && w.len() == num_experts * hidden, "router_gemm_bi: input");
    ensure!(out.len() >= tokens * num_experts, "router_gemm_bi: out");
    ensure!(hidden % 64 == 0 && num_experts % 8 == 0, "router_gemm_bi: 形状不满足 BLOCK 约束");
    let (px, pw) = (stream.ptr(x) as *const c_void, stream.ptr(w) as *const c_void);
    let po = stream.ptr_mut(out) as *mut f32;
    // SAFETY: 容量已校验。
    check(
        unsafe {
            ffi::router_gemm_bi_f32(
                px,
                pw,
                po,
                tokens as i64,
                num_experts as i64,
                hidden as i64,
                stream.raw() as _,
            )
        },
        "router_gemm_bi_f32",
    )
}

/// 路由：logits[tokens*E] f32 → (topk_ids[tokens*K], topk_w[tokens*K])。
#[allow(clippy::too_many_arguments)]
pub fn moe_route_f32(
    stream: &Stream,
    logits: &CudaSlice<f32>,
    bias: Option<&CudaSlice<f32>>,
    topk_ids: &mut CudaSlice<i32>,
    topk_w: &mut CudaSlice<f32>,
    tokens: usize,
    num_experts: usize,
    topk: usize,
    renormalize: bool,
    scale: f32,
) -> Result<()> {
    ensure!(logits.len() >= tokens * num_experts, "moe_route: logits");
    ensure!(topk_ids.len() >= tokens * topk && topk_w.len() >= tokens * topk, "moe_route: out");
    if let Some(b) = bias {
        ensure!(b.len() == num_experts, "moe_route: bias len");
    }
    let pl = stream.ptr(logits) as *const f32;
    let pb = bias.map_or(std::ptr::null(), |b| stream.ptr(b) as *const f32);
    let pi = stream.ptr_mut(topk_ids) as *mut i32;
    let pw = stream.ptr_mut(topk_w) as *mut f32;
    // SAFETY: 容量已校验；bias 可为 null（kernel 内判空）。
    check(
        unsafe {
            ffi::moe_route_f32(
                pl,
                pb,
                pi,
                pw,
                tokens as i64,
                num_experts as i64,
                topk as i64,
                renormalize as i32,
                scale,
                stream.raw() as _,
            )
        },
        "moe_route_f32",
    )
}

/// Gemma4 路由（全专家 softmax 口径）：logits[tokens*E] → topk_ids / topk_w。
///
/// 语义严格照 vLLM `model_executor/models/gemma4.py`：按**原始 logits** 取 top-k，
/// 在全专家上做 softmax，只在 top-k 上重新归一化，再乘 `per_expert_scale`。
/// 与 [`moe_route_f32`]（sigmoid，DeepSeekV3 / GLM / K3 口径）不可互换。
#[allow(clippy::too_many_arguments)]
pub fn moe_route_softmax_f32(
    stream: &Stream,
    logits: &CudaSlice<f32>,
    per_expert_scale: Option<&CudaSlice<bf16>>,
    topk_ids: &mut CudaSlice<i32>,
    topk_w: &mut CudaSlice<f32>,
    tokens: usize,
    num_experts: usize,
    topk: usize,
) -> Result<()> {
    ensure!(logits.len() >= tokens * num_experts, "moe_route_softmax: logits");
    ensure!(topk_ids.len() >= tokens * topk && topk_w.len() >= tokens * topk, "moe_route_softmax: out");
    if let Some(p) = per_expert_scale {
        ensure!(p.len() == num_experts, "moe_route_softmax: per_expert_scale len");
    }
    let pl = stream.ptr(logits) as *const f32;
    let pp = per_expert_scale.map_or(std::ptr::null(), |p| stream.ptr(p) as *const c_void);
    let pi = stream.ptr_mut(topk_ids) as *mut i32;
    let pw = stream.ptr_mut(topk_w) as *mut f32;
    // SAFETY: 容量已校验；per_expert_scale 可为 null。
    check(
        unsafe {
            ffi::moe_route_softmax_f32(
                pl,
                pi,
                pp,
                pw,
                tokens as i64,
                num_experts as i64,
                topk as i64,
                stream.raw() as _,
            )
        },
        "moe_route_softmax_f32",
    )
}

/// 分桶工作区：counts[E] / offsets[E+1] / cursor[E] / sorted_tok[T*K] / sorted_k[T*K]。
pub struct MoeBuckets {
    pub counts: CudaSlice<i32>,
    pub offsets: CudaSlice<i32>,
    pub cursor: CudaSlice<i32>,
    pub sorted_tok: CudaSlice<i32>,
    pub sorted_k: CudaSlice<i32>,
    pub num_experts: usize,
    pub max_pairs: usize,
}

impl MoeBuckets {
    pub fn new(stream: &Stream, num_experts: usize, max_tokens: usize, topk: usize) -> Result<Self> {
        let max_pairs = max_tokens * topk;
        Ok(Self {
            counts: stream.alloc_zeros::<i32>(num_experts)?,
            offsets: stream.alloc_zeros::<i32>(num_experts + 1)?,
            cursor: stream.alloc_zeros::<i32>(num_experts)?,
            sorted_tok: stream.alloc_zeros::<i32>(max_pairs)?,
            sorted_k: stream.alloc_zeros::<i32>(max_pairs)?,
            num_experts,
            max_pairs,
        })
    }
}

/// 按专家分桶。之后 `offsets` 可 D2H 读出每专家的 [start, end)。
pub fn moe_bucket(
    stream: &Stream,
    topk_ids: &CudaSlice<i32>,
    b: &mut MoeBuckets,
    tokens: usize,
    topk: usize,
) -> Result<()> {
    ensure!(tokens * topk <= b.max_pairs && topk_ids.len() >= tokens * topk, "moe_bucket: capacity");
    let pi = stream.ptr(topk_ids) as *const i32;
    let pc = stream.ptr_mut(&mut b.counts) as *mut i32;
    let po = stream.ptr_mut(&mut b.offsets) as *mut i32;
    let pcu = stream.ptr_mut(&mut b.cursor) as *mut i32;
    let pst = stream.ptr_mut(&mut b.sorted_tok) as *mut i32;
    let psk = stream.ptr_mut(&mut b.sorted_k) as *mut i32;
    // SAFETY: 容量已校验。
    check(
        unsafe {
            ffi::moe_bucket(
                pi,
                pc,
                po,
                pcu,
                pst,
                psk,
                tokens as i64,
                b.num_experts as i64,
                topk as i64,
                stream.raw() as _,
            )
        },
        "moe_bucket",
    )
}

/// xg[i] = x[sorted_tok[start + i]]，i < n（取排序后 [start, start+n) 这一段，即某个专家的全部 token 行）。
#[allow(clippy::too_many_arguments)]
pub fn moe_gather_bf16(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    xg: &mut CudaSlice<bf16>,
    sorted_tok: &CudaSlice<i32>,
    start: usize,
    n: usize,
    hidden: usize,
) -> Result<()> {
    ensure!(
        xg.len() >= n * hidden && sorted_tok.len() >= start + n && x.len() % hidden == 0,
        "moe_gather: capacity"
    );
    let px = stream.ptr(x) as *const c_void;
    let pg = stream.ptr_mut(xg) as *mut c_void;
    let ps = stream.ptr(sorted_tok) as *const i32;
    // SAFETY: 容量已校验（start+n ≤ len，i32 偏移在界内）；sorted_tok 内 token 下标 < x.len()/hidden 由 bucket 保证。
    let ps = unsafe { ps.add(start) };
    check(
        unsafe { ffi::moe_gather_bf16(px, pg, ps, n as i64, hidden as i64, stream.raw() as _) },
        "moe_gather_bf16",
    )
}

/// acc[sorted_tok[start+i]] += topk_w[sorted_tok[start+i], sorted_k[start+i]] * y[i]，i < n。acc 需先清零。
#[allow(clippy::too_many_arguments)]
pub fn moe_scatter_add_f32(
    stream: &Stream,
    y: &CudaSlice<bf16>,
    acc: &mut CudaSlice<f32>,
    sorted_tok: &CudaSlice<i32>,
    sorted_k: &CudaSlice<i32>,
    topk_w: &CudaSlice<f32>,
    start: usize,
    n: usize,
    topk: usize,
    hidden: usize,
) -> Result<()> {
    ensure!(
        y.len() >= n * hidden && sorted_tok.len() >= start + n && sorted_k.len() >= start + n,
        "moe_scatter_add: capacity"
    );
    let py = stream.ptr(y) as *const c_void;
    let pa = stream.ptr_mut(acc) as *mut f32;
    let pst = stream.ptr(sorted_tok) as *const i32;
    let psk = stream.ptr(sorted_k) as *const i32;
    let pw = stream.ptr(topk_w) as *const f32;
    // SAFETY: 容量已校验；偏移 start 在界内。
    let (pst, psk) = unsafe { (pst.add(start), psk.add(start)) };
    check(
        unsafe {
            ffi::moe_scatter_add_f32(
                py,
                pa,
                pst,
                psk,
                pw,
                std::ptr::null(),
                1.0,
                n as i64,
                topk as i64,
                hidden as i64,
                0,
                stream.raw() as _,
            )
        },
        "moe_scatter_add_f32",
    )
}

/// vLLM `cutlass_moe` 合并口径：`w = bf16(topk_w · per_expert[e])`，每项 `bf16(w · y)` 再累加（两次 bf16 舍入）。
///
/// `per_expert` 为 None 时视为全 1。`w_scale` 供探针做系数 A/B（正常传 1.0）。
#[allow(clippy::too_many_arguments)]
pub fn moe_scatter_add_bf16w_f32(
    stream: &Stream,
    y: &CudaSlice<bf16>,
    acc: &mut CudaSlice<f32>,
    sorted_tok: &CudaSlice<i32>,
    sorted_k: &CudaSlice<i32>,
    topk_w: &CudaSlice<f32>,
    per_expert: Option<&CudaSlice<bf16>>,
    start: usize,
    n: usize,
    topk: usize,
    hidden: usize,
    w_scale: f32,
) -> Result<()> {
    ensure!(
        y.len() >= n * hidden && sorted_tok.len() >= start + n && sorted_k.len() >= start + n,
        "moe_scatter_add: capacity"
    );
    if n == 0 {
        return Ok(());
    }
    let py = stream.ptr(y) as *const c_void;
    let pa = stream.ptr_mut(acc) as *mut f32;
    let pst = stream.ptr(sorted_tok) as *const i32;
    let psk = stream.ptr(sorted_k) as *const i32;
    let pw = stream.ptr(topk_w) as *const f32;
    let pe = per_expert.map(|t| stream.ptr(t) as *const c_void).unwrap_or(std::ptr::null());
    // SAFETY: 容量已校验；偏移 start 在界内。
    let (pst, psk) = unsafe { (pst.add(start), psk.add(start)) };
    check(
        unsafe {
            ffi::moe_scatter_add_f32(
                py,
                pa,
                pst,
                psk,
                pw,
                pe,
                w_scale,
                n as i64,
                topk as i64,
                hidden as i64,
                1,
                stream.raw() as _,
            )
        },
        "moe_scatter_add_bf16w_f32",
    )
}

/// out[:n] += acc[:n]（fp32 → bf16）。
pub fn f32_add_to_bf16_n(
    stream: &Stream,
    acc: &CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
    n: usize,
) -> Result<()> {
    ensure!(acc.len() >= n && out.len() >= n, "f32_add_to_bf16: capacity");
    let pa = stream.ptr(acc) as *const f32;
    let po = stream.ptr_mut(out) as *mut c_void;
    // SAFETY: 容量已校验。
    check(unsafe { ffi::f32_add_to_bf16(pa, po, n as i64, stream.raw() as _) }, "f32_add_to_bf16")
}

/// 把 D2H 读出的 offsets 变成 (expert, start, len) 列表（只含非空专家）。
pub fn moe_expert_ranges(offsets: &[i32]) -> Vec<(usize, usize, usize)> {
    let mut v = Vec::new();
    for e in 0..offsets.len() - 1 {
        let (s, t) = (offsets[e] as usize, offsets[e + 1] as usize);
        if t > s {
            v.push((e, s, t - s));
        }
    }
    v
}

/// x[t, :] = 0 where pos[t] == 0（MTP：位置 0 的 embedding 置零，vLLM `torch.where(positions == 0, 0, inputs_embeds)`）。
pub fn zero_rows_where_pos0_bf16(
    stream: &Stream,
    x: &mut CudaSlice<bf16>,
    pos: &CudaSlice<i32>,
    tokens: usize,
    cols: usize,
) -> Result<()> {
    ensure!(x.len() >= tokens * cols && pos.len() >= tokens, "zero_rows_where_pos0: buffer");
    let px = stream.ptr_mut(x) as *mut c_void;
    let pp = stream.ptr(pos) as *const i32;
    // SAFETY: 容量已校验。
    check(
        unsafe { ffi::zero_rows_where_pos0_bf16(px, pp, tokens as i64, cols as i64, stream.raw() as _) },
        "zero_rows_where_pos0",
    )
}

// ---------------------------------------------------------------------------
// GQA paged attention（gqa.cu）
// ---------------------------------------------------------------------------

/// k, v: [tokens, kv_heads, head_dim] → cache [num_blocks, block_size, kv_heads, 2, head_dim]（slot_mapping，-1 跳过）。
#[allow(clippy::too_many_arguments)]
pub fn gqa_cache_write_bf16(
    stream: &Stream,
    k: &CudaSlice<bf16>,
    v: &CudaSlice<bf16>,
    cache: &mut CudaSlice<bf16>,
    slot_mapping: &CudaSlice<i64>,
    tokens: usize,
    kv_heads: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<()> {
    ensure!(
        k.len() >= tokens * kv_heads * head_dim && v.len() >= tokens * kv_heads * head_dim,
        "gqa_cache_write: k/v"
    );
    ensure!(slot_mapping.len() >= tokens, "gqa_cache_write: slot_mapping");
    let pk = stream.ptr(k) as *const c_void;
    let pv = stream.ptr(v) as *const c_void;
    let pc = stream.ptr_mut(cache) as *mut c_void;
    let ps = stream.ptr(slot_mapping) as *const i64;
    // SAFETY: 容量已校验；slot < num_blocks*block_size 由调度器保证。
    check(
        unsafe {
            ffi::gqa_cache_write_bf16(
                pk,
                pv,
                pc,
                ps,
                tokens as i64,
                kv_heads as i64,
                head_dim as i64,
                block_size as i64,
                stream.raw() as _,
            )
        },
        "gqa_cache_write_bf16",
    )
}

/// 无历史 causal prefill。q: [T, heads, D]，k/v: [T, kv_heads, D]，out: [T, heads, D]。
#[allow(clippy::too_many_arguments)]
fn gqa_prefill_impl<
    Q: DevicePtr<bf16> + ?Sized,
    K: DevicePtr<bf16> + ?Sized,
    V: DevicePtr<bf16> + ?Sized,
>(
    stream: &Stream,
    q: &Q,
    k: &K,
    v: &V,
    out: &mut CudaSlice<bf16>,
    cu_seqlens: &CudaSlice<i32>,
    num_reqs: usize,
    max_seqlen: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
    // 每 token 行的元素跨度；0 = 紧凑（heads·D / kv_heads·D）。传 qkv GEMM 输出时为 (H+2Hkv)·D。
    q_stride: usize,
    kv_stride: usize,
    window: usize,
    p_fp8: bool,
) -> Result<()> {
    ensure!(cu_seqlens.len() > num_reqs, "gqa_prefill: cu_seqlens");
    ensure!(max_seqlen > 0, "gqa_prefill: max_seqlen");
    ensure!(heads % kv_heads == 0, "gqa_prefill: heads % kv_heads");
    let pq = stream.ptr(q) as *const c_void;
    let pk = stream.ptr(k) as *const c_void;
    let pv = stream.ptr(v) as *const c_void;
    let po = stream.ptr_mut(out) as *mut c_void;
    let pcu = stream.ptr(cu_seqlens) as *const i32;
    // SAFETY: 调用方保证 q/k/v/out 覆盖 cu_seqlens 的总 token 数。
    check(
        unsafe {
            ffi::gqa_prefill_win_bf16(
                pq,
                pk,
                pv,
                po,
                pcu,
                num_reqs as i64,
                max_seqlen as i64,
                heads as i64,
                kv_heads as i64,
                head_dim as i64,
                scale,
                q_stride as i64,
                kv_stride as i64,
                gqa_prefill_cfg(stream),
                window as i32,
                p_fp8 as i32 | (attn_mode_ab() << 1),
                stream.raw() as _,
            )
        },
        "gqa_prefill_bf16",
    )
}
/// 无滑窗 prefill（`window = 0`）。带窗口的版本见 [`gqa_prefill_win_bf16`]。
#[allow(clippy::too_many_arguments)]
pub fn gqa_prefill_bf16<
    Q: DevicePtr<bf16> + ?Sized,
    K: DevicePtr<bf16> + ?Sized,
    V: DevicePtr<bf16> + ?Sized,
>(
    stream: &Stream,
    q: &Q,
    k: &K,
    v: &V,
    out: &mut CudaSlice<bf16>,
    cu_seqlens: &CudaSlice<i32>,
    num_reqs: usize,
    max_seqlen: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
    q_stride: usize,
    kv_stride: usize,
) -> Result<()> {
    gqa_prefill_impl(stream, q, k, v, out, cu_seqlens, num_reqs, max_seqlen, heads, kv_heads, head_dim, scale, q_stride, kv_stride, 0, false)
}

/// 带滑窗的 prefill（Gemma4 的 sliding 层）：每行 query 只能看到 `[pos - window + 1, pos]`。
/// `window = 0` 等价全注意力。
#[allow(clippy::too_many_arguments)]
pub fn gqa_prefill_win_bf16<
    Q: DevicePtr<bf16> + ?Sized,
    K: DevicePtr<bf16> + ?Sized,
    V: DevicePtr<bf16> + ?Sized,
>(
    stream: &Stream,
    q: &Q,
    k: &K,
    v: &V,
    out: &mut CudaSlice<bf16>,
    cu_seqlens: &CudaSlice<i32>,
    num_reqs: usize,
    max_seqlen: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
    q_stride: usize,
    kv_stride: usize,
    window: usize,
    p_fp8: bool,
) -> Result<()> {
    gqa_prefill_impl(stream, q, k, v, out, cu_seqlens, num_reqs, max_seqlen, heads, kv_heads, head_dim, scale, q_stride, kv_stride, window, p_fp8)
}

/// v6 attention 的数值口径位（仅 v6 kernel 读），`ATTN_MODE=<bits>` 覆盖：
/// bit1 = 指数参数走自然域 `ex2((s - m)·log2e)`（= vLLM Triton `tl.exp(S - m)`），**默认开**（Kaon-V3 L4 隔离 4457 → 1，§54）；
/// bit0 = 归一化 `o / l`（vLLM `acc / L`），现有数据无区分力，默认关。`ATTN_MODE=0` 退回旧口径。
/// RMSNorm 的数值口径（`RMSNORM_MODE` 覆盖）：1 = eager BI；其余 = compile 系。
///   **0 = compile，fp32 连乘 `x·inv·w`，只在写回时舍一次**（早期按生成代码推断的口径）；
///   **2 = compile，`x·inv` 后先舍 bf16 再乘 w**（照 vLLM FX 图 + `ir/ops/layernorm.py:17` 源码，
///   `docs/loops/kaon3-compile.md` 第 3 轮）。两者是 A/B 关系，谁对由整跑 flip 决定。
/// vLLM 默认就走 torch.compile（`enforce_eager=False`；`vllm/config/vllm.py:1379` 的 `mode` 解析），
/// 两种模式下 norm 是不同实现：eager BI 是 `rcp(sqrt(fma(rcp(n), ss, eps)))`（SASS 复刻）。
fn norm_mode() -> i32 {
    static V: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("RMSNORM_MODE").ok().and_then(|s| s.trim().parse::<i32>().ok()).unwrap_or(1))
}

fn attn_mode_ab() -> i32 {
    static V: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("ATTN_MODE").ok().and_then(|s| s.trim().parse::<i32>().ok()).unwrap_or(2) & 3)
}

/// D=256 全注意力 prefill kernel 结构（`ArchProfile::gqa_prefill_cfg_for`），`GQA_PREFILL_CFG` 覆盖；进程内只算一次。
fn gqa_prefill_cfg(stream: &Stream) -> i32 {
    use std::sync::OnceLock;
    static CFG: OnceLock<i32> = OnceLock::new();
    *CFG.get_or_init(|| {
        if let Some(v) = std::env::var("GQA_PREFILL_CFG").ok().and_then(|s| s.trim().parse::<i32>().ok()) {
            tracing::info!(cfg = v, "gqa prefill cfg 由环境变量覆盖");
            return v;
        }
        let cc = stream.device().compute_capability().unwrap_or((8, 0));
        let sm = stream.device().sm_count().unwrap_or(1);
        let c = crate::arch::ArchProfile::gqa_prefill_cfg_for(crate::arch::ArchProfile::for_cc(cc, sm).arch);
        // r12a：sm100 的 tcgen05 路径尚未赢过 v6，默认仍走 v6；`GQA_PREFILL_CFG=5xxx` 可显式开启。
        c
    })
}

/// 从 paged cache 读的 attention：decode（q_len = 1）与带历史 chunked prefill 统一。q: [Tq, heads, D]，out 同形。
#[allow(clippy::too_many_arguments)]
fn gqa_paged_impl<Q: DevicePtr<bf16> + ?Sized>(
    stream: &Stream,
    q: &Q,
    cache: &CudaSlice<bf16>,
    block_table: &CudaSlice<i32>,
    seq_lens: &CudaSlice<i32>,
    cu_seqlens_q: &CudaSlice<i32>,
    out: &mut CudaSlice<bf16>,
    num_reqs: usize,
    max_q_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_blocks: usize,
    scale: f32,
    q_stride: usize,
    // 混批：跳过 q_len==1 的请求（它们由 gqa_decode_bf16 处理）。
    skip_single: bool,
    window: usize,
    p_fp8: bool,
) -> Result<()> {
    ensure!(
        block_table.len() >= num_reqs * max_blocks && seq_lens.len() >= num_reqs,
        "gqa_paged: table/seq_lens"
    );
    ensure!(max_q_len > 0, "gqa_paged: max_q_len");
    ensure!(cu_seqlens_q.len() > num_reqs, "gqa_paged: cu_seqlens_q");
    ensure!(heads % kv_heads == 0, "gqa_paged: heads % kv_heads");
    let pq = stream.ptr(q) as *const c_void;
    let pc = stream.ptr(cache) as *const c_void;
    let pbt = stream.ptr(block_table) as *const i32;
    let psl = stream.ptr(seq_lens) as *const i32;
    let pcu = stream.ptr(cu_seqlens_q) as *const i32;
    let po = stream.ptr_mut(out) as *mut c_void;
    // SAFETY: 容量已校验；block_table 内 id < num_blocks 由调度器保证。
    check(
        unsafe {
            ffi::gqa_paged_win_bf16(
                pq,
                pc,
                pbt,
                psl,
                pcu,
                po,
                num_reqs as i64,
                max_q_len as i64,
                heads as i64,
                kv_heads as i64,
                head_dim as i64,
                block_size as i64,
                max_blocks as i64,
                scale,
                q_stride as i64,
                skip_single as i32,
                gqa_prefill_cfg(stream),
                window as i32,
                p_fp8 as i32 | (attn_mode_ab() << 1),
                stream.raw() as _,
            )
        },
        "gqa_paged_bf16",
    )
}
/// 无滑窗 paged attention（`window = 0`）。带窗口的版本见 [`gqa_paged_win_bf16`]。
#[allow(clippy::too_many_arguments)]
pub fn gqa_paged_bf16<Q: DevicePtr<bf16> + ?Sized>(
    stream: &Stream,
    q: &Q,
    cache: &CudaSlice<bf16>,
    block_table: &CudaSlice<i32>,
    seq_lens: &CudaSlice<i32>,
    cu_seqlens_q: &CudaSlice<i32>,
    out: &mut CudaSlice<bf16>,
    num_reqs: usize,
    max_q_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_blocks: usize,
    scale: f32,
    q_stride: usize,
    skip_single: bool,
) -> Result<()> {
    gqa_paged_impl(stream, q, cache, block_table, seq_lens, cu_seqlens_q, out, num_reqs, max_q_len, heads, kv_heads, head_dim, block_size, max_blocks, scale, q_stride, skip_single, 0, false)
}

/// 带滑窗的 paged attention：可见 key ∈ `[pos - window + 1, pos]`；`window = 0` 等价全注意力。
#[allow(clippy::too_many_arguments)]
pub fn gqa_paged_win_bf16<Q: DevicePtr<bf16> + ?Sized>(
    stream: &Stream,
    q: &Q,
    cache: &CudaSlice<bf16>,
    block_table: &CudaSlice<i32>,
    seq_lens: &CudaSlice<i32>,
    cu_seqlens_q: &CudaSlice<i32>,
    out: &mut CudaSlice<bf16>,
    num_reqs: usize,
    max_q_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_blocks: usize,
    scale: f32,
    q_stride: usize,
    skip_single: bool,
    window: usize,
    p_fp8: bool,
) -> Result<()> {
    gqa_paged_impl(stream, q, cache, block_table, seq_lens, cu_seqlens_q, out, num_reqs, max_q_len, heads, kv_heads, head_dim, block_size, max_blocks, scale, q_stride, skip_single, window, p_fp8)
}

/// 纯 decode（每请求 1 行）split-K attention（D ∈ {128, 256}）。q: [B, heads, D]，out 同形；`scratch` ≥ `gqa_decode_scratch_len`。
#[allow(clippy::too_many_arguments)]
pub fn gqa_decode_bf16<Q: DevicePtr<bf16> + ?Sized>(
    stream: &Stream,
    q: &Q,
    cache: &CudaSlice<bf16>,
    block_table: &CudaSlice<i32>,
    seq_lens: &CudaSlice<i32>,
    out: &mut CudaSlice<bf16>,
    scratch: &mut CudaSlice<f32>,
    num_reqs: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_blocks: usize,
    scale: f32,
    q_stride: usize,
    // 混批：Some 时只算 q_len==1 的请求，q/out 行号取 cu_seqlens_q[b]。
    cu_seqlens_q: Option<&CudaSlice<i32>>,
) -> Result<()> {
    ensure!(head_dim == 128 || head_dim == 256, "gqa_decode: head_dim must be 128 or 256");
    ensure!(
        block_table.len() >= num_reqs * max_blocks && seq_lens.len() >= num_reqs,
        "gqa_decode: table/seq_lens"
    );
    ensure!(
        cu_seqlens_q.is_some()
            || (q.len() >= (num_reqs - 1) * q_stride.max(heads * head_dim) + heads * head_dim
                && out.len() >= num_reqs * heads * head_dim),
        "gqa_decode: q/out"
    );
    ensure!(scratch.len() >= gqa_decode_scratch_len(num_reqs, heads, head_dim), "gqa_decode: scratch");
    // SAFETY: 长度已校验；block_table 下标范围由 scheduler 保证。
    check(
        unsafe {
            ffi::gqa_decode_bf16(
                stream.ptr(q) as *const c_void,
                stream.ptr(cache) as *const c_void,
                stream.ptr(block_table) as *const i32,
                stream.ptr(seq_lens) as *const i32,
                stream.ptr_mut(out) as *mut c_void,
                stream.ptr_mut(scratch) as *mut f32,
                num_reqs as i64,
                heads as i64,
                kv_heads as i64,
                head_dim as i64,
                block_size as i64,
                max_blocks as i64,
                scale,
                q_stride as i64,
                cu_seqlens_q.map_or(std::ptr::null(), |c| stream.ptr(c) as *const i32),
                stream.raw() as _,
            )
        },
        "gqa_decode_bf16",
    )
}

/// 混批（prefill 行 + decode 行同一步）attention（D ∈ {128, 256}）：q_len==1 的请求走 split-K decode kernel（按 cu_seqlens_q 定位行），
/// 其余请求走 paged tensor-core kernel（跳过 q_len==1）。两个 kernel 同流串行，各自只写自己的行。
#[allow(clippy::too_many_arguments)]
pub fn gqa_mixed_bf16<Q: DevicePtr<bf16> + ?Sized>(
    stream: &Stream,
    q: &Q,
    cache: &CudaSlice<bf16>,
    block_table: &CudaSlice<i32>,
    seq_lens: &CudaSlice<i32>,
    cu_seqlens_q: &CudaSlice<i32>,
    out: &mut CudaSlice<bf16>,
    scratch: &mut CudaSlice<f32>,
    num_reqs: usize,
    max_q_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_blocks: usize,
    scale: f32,
    q_stride: usize,
) -> Result<()> {
    ensure!(head_dim == 128 || head_dim == 256, "gqa_mixed: head_dim must be 128 or 256");
    gqa_decode_bf16(
        stream,
        q,
        cache,
        block_table,
        seq_lens,
        out,
        scratch,
        num_reqs,
        heads,
        kv_heads,
        head_dim,
        block_size,
        max_blocks,
        scale,
        q_stride,
        Some(cu_seqlens_q),
    )?;
    gqa_paged_bf16(
        stream,
        q,
        cache,
        block_table,
        seq_lens,
        cu_seqlens_q,
        out,
        num_reqs,
        max_q_len,
        heads,
        kv_heads,
        head_dim,
        block_size,
        max_blocks,
        scale,
        q_stride,
        true,
    )
}

/// `gqa_decode_bf16` 的 scratch 元素数：B × heads × 8 split × (D + 2)。
pub fn gqa_decode_scratch_len(num_reqs: usize, heads: usize, head_dim: usize) -> usize {
    // 与 gqa.cu `D2_MAX_SPLITS` 同步（B4 第 1 轮 8 → 32）
    num_reqs * heads * 32 * (head_dim + 2)
}

/// Qwen3 QKV 后处理融合（D=128）：q/k per-head RMSNorm + neox RoPE 原地写回 `qkv`（[T, (H+2Hkv)·D]），K/V 写 paged cache。
#[allow(clippy::too_many_arguments)]
pub fn qkv_post_bf16(
    stream: &Stream,
    qkv: &mut CudaSlice<bf16>,
    wq: &CudaSlice<bf16>,
    wk: &CudaSlice<bf16>,
    positions: &CudaSlice<i32>,
    cos_sin: &CudaSlice<f32>,
    cache: &mut CudaSlice<bf16>,
    slot_mapping: &CudaSlice<i64>,
    tokens: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f32,
    block_size: usize,
) -> Result<()> {
    ensure!(head_dim == 128, "qkv_post: head_dim must be 128");
    ensure!(qkv.len() >= tokens * (heads + 2 * kv_heads) * head_dim, "qkv_post: qkv");
    ensure!(wq.len() >= head_dim && wk.len() >= head_dim, "qkv_post: norm weights");
    ensure!(positions.len() >= tokens && slot_mapping.len() >= tokens, "qkv_post: positions/slots");
    ensure!(cache.len() % (kv_heads * 2 * head_dim) == 0, "qkv_post: cache not a multiple of slot");
    // SAFETY: 长度已校验；slot 范围由 scheduler 保证；cos_sin 行数 ≥ max position 由 Rope 构造保证。
    check(
        unsafe {
            ffi::qkv_post_bf16(
                stream.ptr_mut(qkv) as *mut c_void,
                stream.ptr(wq) as *const c_void,
                stream.ptr(wk) as *const c_void,
                stream.ptr(positions) as *const i32,
                stream.ptr(cos_sin) as *const f32,
                stream.ptr_mut(cache) as *mut c_void,
                stream.ptr(slot_mapping) as *const i64,
                tokens as i64,
                heads as i64,
                kv_heads as i64,
                head_dim as i64,
                eps,
                block_size as i64,
                stream.raw() as _,
            )
        },
        "qkv_post_bf16",
    )
}

/// QKV 后处理（扩展）：per-head RMSNorm（权重加 `norm_beta`）+ 只旋前 `rope_dim` 维的 neox RoPE（原地）
/// + K/V 写 paged cache。行布局 `[q(H·D) | gate(H·D)? | k | v]`（`qmul = 2` 时带输出门段，本算子不碰它）。
#[allow(clippy::too_many_arguments)]
pub fn qkv_post_ext_bf16(
    stream: &Stream,
    qkv: &mut CudaSlice<bf16>,
    wq: &CudaSlice<bf16>,
    wk: &CudaSlice<bf16>,
    positions: &CudaSlice<i32>,
    cos_sin: &CudaSlice<f32>,
    cache: &mut CudaSlice<bf16>,
    slot_mapping: &CudaSlice<i64>,
    tokens: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    eps: f32,
    norm_beta: f32,
    qmul: usize,
    block_size: usize,
) -> Result<()> {
    ensure!(head_dim == 128 || head_dim == 256, "qkv_post_ext: head_dim must be 128 or 256");
    ensure!(
        rope_dim > 0 && rope_dim <= head_dim && rope_dim % 64 == 0,
        "qkv_post_ext: rope_dim must be a multiple of 64 and <= head_dim"
    );
    ensure!(qmul == 1 || qmul == 2, "qkv_post_ext: qmul must be 1 or 2");
    ensure!(qkv.len() >= tokens * (heads * qmul + 2 * kv_heads) * head_dim, "qkv_post_ext: qkv");
    ensure!(wq.len() >= head_dim && wk.len() >= head_dim, "qkv_post_ext: norm weights");
    ensure!(positions.len() >= tokens && slot_mapping.len() >= tokens, "qkv_post_ext: positions/slots");
    ensure!(cache.len() % (kv_heads * 2 * head_dim) == 0, "qkv_post_ext: cache not a multiple of slot");
    // SAFETY: 长度已校验；slot 范围由 scheduler 保证；cos_sin 行数 ≥ max position 由 Rope 构造保证。
    check(
        unsafe {
            ffi::qkv_post_ext_bf16(
                stream.ptr_mut(qkv) as *mut c_void,
                stream.ptr(wq) as *const c_void,
                stream.ptr(wk) as *const c_void,
                stream.ptr(positions) as *const i32,
                stream.ptr(cos_sin) as *const f32,
                stream.ptr_mut(cache) as *mut c_void,
                stream.ptr(slot_mapping) as *const i64,
                tokens as i64,
                heads as i64,
                kv_heads as i64,
                head_dim as i64,
                rope_dim as i64,
                eps,
                norm_beta,
                qmul as i64,
                block_size as i64,
                stream.raw() as _,
            )
        },
        "qkv_post_ext_bf16",
    )
}

/// 输出门（原地）：`x *= sigmoid(gate)`，`n` 元素；`n` 必须是 8 的倍数。
pub fn mul_sigmoid_bf16(
    stream: &Stream,
    x: &mut CudaSlice<bf16>,
    gate: &CudaSlice<bf16>,
    n: usize,
) -> Result<()> {
    ensure!(x.len() >= n && gate.len() >= n, "mul_sigmoid: buffer too small");
    ensure!(n % 8 == 0, "mul_sigmoid: n must be a multiple of 8");
    check(
        unsafe {
            ffi::mul_sigmoid_bf16(
                stream.ptr_mut(x) as *mut c_void,
                stream.ptr(gate) as *const c_void,
                n as i64,
                stream.raw() as _,
            )
        },
        "mul_sigmoid_bf16",
    )
}

/// 行式输出门：`x` 紧凑 `[rows, cols]`，`gate` 是 `[rows, gate_stride]` 缓冲的列段视图（融合 qkv+gate GEMM 输出里的
/// gate 段，第 9 轮）。`cols`、`gate_stride` 为 8 的倍数。
pub fn mul_sigmoid_rows_bf16(
    stream: &Stream,
    x: &mut CudaSlice<bf16>,
    gate: &CudaView<'_, bf16>,
    rows: usize,
    cols: usize,
    gate_stride: usize,
) -> Result<()> {
    ensure!(cols % 8 == 0 && gate_stride % 8 == 0 && gate_stride >= cols, "mul_sigmoid_rows: cols/gate_stride");
    ensure!(x.len() >= rows * cols, "mul_sigmoid_rows: x");
    ensure!(rows == 0 || gate.len() >= (rows - 1) * gate_stride + cols, "mul_sigmoid_rows: gate");
    check(
        unsafe {
            ffi::mul_sigmoid_rows_bf16(
                stream.ptr_mut(x) as *mut c_void,
                stream.ptr(gate) as *const c_void,
                rows as i64,
                cols as i64,
                gate_stride as i64,
                stream.raw() as _,
            )
        },
        "mul_sigmoid_rows_bf16",
    )
}

// ---------------------------------------------------------------------------
// 采样（Q2）：penalties / temperature / min_p / logprobs；与 topk_topp_mask_f32 / sample_f32 组成 vLLM 顺序
// ---------------------------------------------------------------------------

/// 惩罚（原地）。`ids` CSR：`offsets[b]..offsets[b+1]`，前 `prompt_len[b]` 个是 prompt，其余是 output。
#[allow(clippy::too_many_arguments)]
pub fn penalties_f32<
    L: DevicePtrMut<f32> + ?Sized,
    I: DevicePtr<i32> + ?Sized,
    O1: DevicePtr<i32> + ?Sized,
    O2: DevicePtr<i32> + ?Sized,
    R: DevicePtr<f32> + ?Sized,
    F: DevicePtr<f32> + ?Sized,
    Q: DevicePtr<f32> + ?Sized,
>(
    stream: &Stream,
    logits: &mut L,
    ids: &I,
    offsets: &O1,
    prompt_len: &O2,
    rep: &R,
    freq: &F,
    pres: &Q,
    vocab: usize,
) -> Result<()> {
    ensure!(vocab > 0 && logits.len() % vocab == 0, "penalties: bad shape");
    let batch = logits.len() / vocab;
    ensure!(offsets.len() > batch && prompt_len.len() >= batch, "penalties: offsets/prompt_len");
    ensure!(rep.len() >= batch && freq.len() >= batch && pres.len() >= batch, "penalties: params");
    let pl = stream.ptr_mut(logits) as *mut f32;
    // SAFETY: 长度已校验；ids 索引范围由 offsets 保证。
    check(
        unsafe {
            ffi::penalties_f32(
                pl,
                stream.ptr(ids) as *const i32,
                stream.ptr(offsets) as *const i32,
                stream.ptr(prompt_len) as *const i32,
                stream.ptr(rep) as *const f32,
                stream.ptr(freq) as *const f32,
                stream.ptr(pres) as *const f32,
                batch as i64,
                vocab as i64,
                stream.raw() as _,
            )
        },
        "penalties_f32",
    )
}

/// 温度缩放（原地；t ≤ 1e-5 或 == 1 跳过）。
pub fn temperature_f32<L: DevicePtrMut<f32> + ?Sized, T: DevicePtr<f32> + ?Sized>(
    stream: &Stream,
    logits: &mut L,
    temperature: &T,
    vocab: usize,
) -> Result<()> {
    ensure!(vocab > 0 && logits.len() % vocab == 0, "temperature: bad shape");
    let batch = logits.len() / vocab;
    ensure!(temperature.len() >= batch, "temperature: params");
    let pl = stream.ptr_mut(logits) as *mut f32;
    // SAFETY: 长度已校验。
    check(
        unsafe {
            ffi::temperature_f32(
                pl,
                stream.ptr(temperature) as *const f32,
                batch as i64,
                vocab as i64,
                stream.raw() as _,
            )
        },
        "temperature_f32",
    )
}

/// min_p 掩码（原地）：概率 < min_p · max_prob 的置 -inf。
pub fn min_p_mask_f32<L: DevicePtrMut<f32> + ?Sized, T: DevicePtr<f32> + ?Sized>(
    stream: &Stream,
    logits: &mut L,
    min_p: &T,
    vocab: usize,
) -> Result<()> {
    ensure!(vocab > 0 && logits.len() % vocab == 0, "min_p: bad shape");
    let batch = logits.len() / vocab;
    ensure!(min_p.len() >= batch, "min_p: params");
    let pl = stream.ptr_mut(logits) as *mut f32;
    // SAFETY: 长度已校验。
    check(
        unsafe {
            ffi::min_p_mask_f32(
                pl,
                stream.ptr(min_p) as *const f32,
                batch as i64,
                vocab as i64,
                stream.raw() as _,
            )
        },
        "min_p_mask_f32",
    )
}

/// logprobs：对 **原始** logits 做 log_softmax；输出 `[batch, k+1]`（第 0 列 chosen，后 k 列 top-k 降序）与 chosen 的 rank。
#[allow(clippy::too_many_arguments)]
pub fn logprobs_f32<
    L: DevicePtr<f32> + ?Sized,
    C: DevicePtr<i32> + ?Sized,
    O1: DevicePtrMut<i32> + ?Sized,
    O2: DevicePtrMut<f32> + ?Sized,
    O3: DevicePtrMut<i32> + ?Sized,
>(
    stream: &Stream,
    logits: &L,
    chosen: &C,
    k: usize,
    out_ids: &mut O1,
    out_lp: &mut O2,
    rank: &mut O3,
    vocab: usize,
) -> Result<()> {
    ensure!(vocab > 0 && logits.len() % vocab == 0, "logprobs: bad shape");
    let batch = logits.len() / vocab;
    ensure!(k <= 32, "logprobs: k > 32");
    ensure!(chosen.len() >= batch && rank.len() >= batch, "logprobs: chosen/rank");
    ensure!(out_ids.len() >= batch * (k + 1) && out_lp.len() >= batch * (k + 1), "logprobs: out");
    // SAFETY: 长度已校验。
    check(
        unsafe {
            ffi::logprobs_f32(
                stream.ptr(logits) as *const f32,
                stream.ptr(chosen) as *const i32,
                k as i64,
                stream.ptr_mut(out_ids) as *mut i32,
                stream.ptr_mut(out_lp) as *mut f32,
                stream.ptr_mut(rank) as *mut i32,
                batch as i64,
                vocab as i64,
                stream.raw() as _,
            )
        },
        "logprobs_f32",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    #[test]
    #[ignore = "needs a CUDA device; run with `cargo test -p dinfer-kernels -- --ignored`"]
    fn add_roundtrip() {
        let dev = Device::new(0).unwrap();
        let s = dev.default_stream();
        let n = 1 << 20;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (2 * i) as f32).collect();
        let da = s.htod_copy(a.clone()).unwrap();
        let db = s.htod_copy(b.clone()).unwrap();
        let mut dout = s.alloc_zeros::<f32>(n).unwrap();
        add_f32(&s, &da, &db, &mut dout).unwrap();
        let out = s.dtoh_sync_copy(&dout).unwrap();
        for i in (0..n).step_by(4097) {
            assert_eq!(out[i], a[i] + b[i]);
        }
    }

    #[test]
    #[ignore = "needs a CUDA device"]
    fn rmsnorm_matches_cpu() {
        let dev = Device::new(0).unwrap();
        let s = dev.default_stream();
        let (t, h) = (7usize, 1536usize);
        let x: Vec<f32> = (0..t * h).map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0).collect();
        let w: Vec<f32> = (0..h).map(|i| 0.5 + (i % 10) as f32 * 0.1).collect();
        let dx = s.htod_copy(x.iter().map(|&v| bf16::from_f32(v)).collect()).unwrap();
        let dw = s.htod_copy(w.iter().map(|&v| bf16::from_f32(v)).collect()).unwrap();
        let mut dout = s.alloc_zeros::<bf16>(t * h).unwrap();
        rmsnorm_bf16(&s, &dx, &dw, &mut dout, h, 1e-5).unwrap();
        let out = s.dtoh_sync_copy(&dout).unwrap();
        for r in 0..t {
            let row = &x[r * h..(r + 1) * h];
            let xb: Vec<f32> = row.iter().map(|&v| bf16::from_f32(v).to_f32()).collect();
            let ss: f32 = xb.iter().map(|v| v * v).sum::<f32>() / h as f32;
            let inv = 1.0 / (ss + 1e-5).sqrt();
            for i in (0..h).step_by(97) {
                let expect = xb[i] * inv * bf16::from_f32(w[i]).to_f32();
                let got = out[r * h + i].to_f32();
                assert!(
                    (got - expect).abs() <= 1e-2 + 1e-2 * expect.abs(),
                    "r={r} i={i} got={got} want={expect}"
                );
            }
        }
    }

    #[test]
    #[ignore = "needs a CUDA device"]
    fn argmax_and_greedy_sample_agree() {
        let dev = Device::new(0).unwrap();
        let s = dev.default_stream();
        let (b, v) = (5usize, 163840usize);
        let mut logits = vec![0f32; b * v];
        let want: Vec<i32> = vec![3, 163839, 0, 77777, 12345];
        for (r, &w) in want.iter().enumerate() {
            for i in 0..v {
                logits[r * v + i] = ((i * 31 + r * 7) % 997) as f32 / 1000.0;
            }
            logits[r * v + w as usize] = 10.0;
        }
        let dl = s.htod_copy(logits).unwrap();
        let mut dout = s.alloc_zeros::<i32>(b).unwrap();
        argmax_f32(&s, &dl, &mut dout, v).unwrap();
        assert_eq!(s.dtoh_sync_copy(&dout).unwrap(), want);
        // greedy sample
        let temp = s.htod_copy(vec![0f32; b]).unwrap();
        let seeds = s.htod_copy(vec![1u64; b]).unwrap();
        let offs = s.htod_copy(vec![0u64; b]).unwrap();
        let mut d2 = s.alloc_zeros::<i32>(b).unwrap();
        sample_f32(&s, &dl, &temp, &seeds, &offs, &mut d2, v).unwrap();
        assert_eq!(s.dtoh_sync_copy(&d2).unwrap(), want);
    }
}

// ---------------------------------------------------------------------------
// KDA（Kimi Delta Attention，模型无关）。数学口径见 cuda/kda.cu 顶部注释。
// ---------------------------------------------------------------------------

/// depthwise causal conv1d + siLU 的 prefill 段。`x`/`out`: varlen `[T, dim]` bf16；
/// `w`: `[dim, width]` bf16（width 维按时间从旧到新）；`conv_state`: `[slots, dim, width-1]` bf16。
/// `cu_seqlens` 长 `num_seqs + 1`；`state_indices` 可为空（则槽号 = 序列号）。
#[allow(clippy::too_many_arguments)]
pub fn kda_conv1d_prefill(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    conv_state: &mut CudaSlice<bf16>,
    cu_seqlens: &CudaSlice<i32>,
    state_indices: Option<&CudaSlice<i32>>,
    has_initial: Option<&CudaSlice<i32>>,
    num_seqs: usize,
    dim: usize,
    width: usize,
    state_len: usize,
) -> Result<()> {
    ensure!(width >= 1 && width <= 8, "kda_conv1d_prefill: width {width} 超出 [1,8]");
    ensure!(state_len >= width.saturating_sub(1), "kda_conv1d_prefill: state_len {state_len} < width-1");
    ensure!(cu_seqlens.len() >= num_seqs + 1, "kda_conv1d_prefill: cu_seqlens");
    ensure!(w.len() >= dim * width && out.len() >= x.len(), "kda_conv1d_prefill: w/out");
    ensure!(conv_state.len() >= state_len, "kda_conv1d_prefill: conv_state");
    let psi = state_indices.map_or(std::ptr::null(), |s| stream.ptr(s) as *const i32);
    let phi = has_initial.map_or(std::ptr::null(), |s| stream.ptr(s) as *const i32);
    // SAFETY: 上面已校验各缓冲容量；cu_seqlens 的单调性由调度侧保证。
    check(
        unsafe {
            ffi::kda_conv1d_prefill(
                stream.ptr(x) as *const c_void,
                stream.ptr(w) as *const c_void,
                stream.ptr_mut(out) as *mut c_void,
                stream.ptr_mut(conv_state) as *mut c_void,
                stream.ptr(cu_seqlens) as *const i32,
                psi,
                phi,
                num_seqs as i64,
                dim as i64,
                width as i64,
                state_len as i64,
                stream.raw() as _,
            )
        },
        "kda_conv1d_prefill",
    )
}

/// depthwise causal conv1d + siLU 的单 token update（decode）。`x`/`out`: `[batch, dim]`。
/// `conv_state` 原地左移并追加新 token。
pub fn kda_conv1d_update(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    conv_state: &mut CudaSlice<bf16>,
    state_indices: Option<&CudaSlice<i32>>,
    batch: usize,
    dim: usize,
    width: usize,
    state_len: usize,
) -> Result<()> {
    ensure!(width >= 1 && width <= 8, "kda_conv1d_update: width {width} 超出 [1,8]");
    ensure!(state_len >= width.saturating_sub(1), "kda_conv1d_update: state_len");
    ensure!(x.len() >= batch * dim && out.len() >= batch * dim, "kda_conv1d_update: x/out");
    ensure!(w.len() >= dim * width, "kda_conv1d_update: w");
    let psi = state_indices.map_or(std::ptr::null(), |s| stream.ptr(s) as *const i32);
    // SAFETY: 容量已校验；槽号由调度器保证 < slots。
    check(
        unsafe {
            ffi::kda_conv1d_update(
                stream.ptr(x) as *const c_void,
                stream.ptr(w) as *const c_void,
                stream.ptr_mut(out) as *mut c_void,
                stream.ptr_mut(conv_state) as *mut c_void,
                psi,
                batch as i64,
                dim as i64,
                width as i64,
                state_len as i64,
                stream.raw() as _,
            )
        },
        "kda_conv1d_update",
    )
}

/// KDA 的衰减门。`g`: `[tokens, heads, head_dim]` bf16 原始投影；`a_log`: `[heads]` f32；
/// `dt_bias`: `[heads*head_dim]` f32；`out`: 同 `g` 形状的 f32（log 域）。
pub fn kda_gate(
    stream: &Stream,
    g: &CudaSlice<bf16>,
    a_log: &CudaSlice<f32>,
    dt_bias: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    tokens: usize,
    heads: usize,
    head_dim: usize,
    lower_bound: Option<f32>,
) -> Result<()> {
    ensure!(a_log.len() >= heads, "kda_gate: a_log");
    ensure!(dt_bias.len() >= heads * head_dim, "kda_gate: dt_bias");
    ensure!(
        g.len() >= tokens * heads * head_dim && out.len() >= tokens * heads * head_dim,
        "kda_gate: g/out"
    );
    // SAFETY: 容量已校验。
    check(
        unsafe {
            ffi::kda_gate(
                stream.ptr(g) as *const c_void,
                stream.ptr(a_log) as *const f32,
                stream.ptr(dt_bias) as *const f32,
                stream.ptr_mut(out) as *mut f32,
                tokens as i64,
                heads as i64,
                head_dim as i64,
                lower_bound.unwrap_or(0.0),
                i32::from(lower_bound.is_some()),
                stream.raw() as _,
            )
        },
        "kda_gate",
    )
}

/// KDA 递归核心（prefill 与 decode 同一条路径，内部按 `cu_seqlens` 分段逐 token 递推）。
/// `q`/`k`/`v`: `[T, heads, head_dim]` bf16，`raw_g` 同形 bf16，`raw_beta`: `[T, heads]` f32，
/// `state`: `[num_slots, heads, head_dim, head_dim]` f32（原地更新），`out`: `[T, heads, head_dim]`。
#[allow(clippy::too_many_arguments)]
pub fn kda_recurrent(
    stream: &Stream,
    q: &CudaSlice<bf16>,
    k: &CudaSlice<bf16>,
    v: &CudaSlice<bf16>,
    raw_g: &CudaSlice<bf16>,
    raw_beta: &CudaSlice<f32>,
    a_log: &CudaSlice<f32>,
    dt_bias: &CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
    state: &mut CudaSlice<f32>,
    cu_seqlens: &CudaSlice<i32>,
    state_indices: Option<&CudaSlice<i32>>,
    num_seqs: usize,
    heads: usize,
    head_dim: usize,
    scale: f32,
    lower_bound: Option<f32>,
) -> Result<()> {
    ensure!(cu_seqlens.len() >= num_seqs + 1, "kda_recurrent: cu_seqlens");
    let n = q.len();
    ensure!(
        k.len() >= n && v.len() >= n && raw_g.len() >= n && out.len() >= n,
        "kda_recurrent: q/k/v/g/out 长度不一致"
    );
    ensure!(raw_beta.len() >= n / head_dim, "kda_recurrent: raw_beta");
    ensure!(a_log.len() >= heads && dt_bias.len() >= heads * head_dim, "kda_recurrent: a_log/dt_bias");
    let per_slot = heads * head_dim * head_dim;
    ensure!(state.len() >= per_slot, "kda_recurrent: state");
    let psi = state_indices.map_or(std::ptr::null(), |s| stream.ptr(s) as *const i32);
    // SAFETY: 容量已校验；slot 编号由调度器保证 < state 槽数。
    check(
        unsafe {
            ffi::kda_recurrent(
                stream.ptr(q) as *const c_void,
                stream.ptr(k) as *const c_void,
                stream.ptr(v) as *const c_void,
                stream.ptr(raw_g) as *const c_void,
                stream.ptr(raw_beta) as *const f32,
                stream.ptr(a_log) as *const f32,
                stream.ptr(dt_bias) as *const f32,
                stream.ptr_mut(out) as *mut c_void,
                stream.ptr_mut(state) as *mut c_void,
                stream.ptr(cu_seqlens) as *const i32,
                psi,
                num_seqs as i64,
                heads as i64,
                head_dim as i64,
                per_slot as i64,
                scale,
                lower_bound.unwrap_or(0.0),
                i32::from(lower_bound.is_some()),
                stream.raw() as _,
            )
        },
        "kda_recurrent",
    )
}

// --------------------------------------------------------------------------- MXFP4 grouped GEMM

/// 分组 GEMM 的展平 tile 计划：每专家按 `ceil(ntok / BM)` 个 M tile × `N / BN` 个 N tile 展开。
/// 块三元组放在三个 i32 数组里（16 的倍数、并行度足够时用 `token4` 打包会更省，见 loop 文档 TODO）。
pub struct Mxfp4GemmTiles {
    pub tile_e: CudaSlice<i32>,
    pub tile_m: CudaSlice<i32>,
    pub tile_n: CudaSlice<i32>,
    pub num_tiles: usize,
}

impl Mxfp4GemmTiles {
    /// `expert_off`：host 侧 `[E+1]`（每专家在排序后 token 段的起止）。
    /// `bm`/`bn` 必须与 kernel 的模板参数一致（当前 16 / 64）。
    pub fn new(stream: &Stream, expert_off: &[i32], n: usize, bm: usize, bn: usize) -> Result<Self> {
        ensure!(expert_off.len() >= 2 && bm > 0 && bn > 0 && n % bn == 0, "mxfp4 tiles: 参数");
        ensure!(expert_off.windows(2).all(|w| w[1] >= w[0]), "mxfp4 tiles: expert_off 必须单调");
        let nt_n = n / bn;
        let mut te = Vec::new();
        let mut tm = Vec::new();
        let mut tn = Vec::new();
        for e in 0..expert_off.len() - 1 {
            let ntok = (expert_off[e + 1] - expert_off[e]).max(0) as usize;
            if ntok == 0 {
                continue;
            }
            for mt in 0..ntok.div_ceil(bm) {
                for nt in 0..nt_n {
                    te.push(e as i32);
                    tm.push(mt as i32);
                    tn.push(nt as i32);
                }
            }
        }
        let num_tiles = te.len();
        Ok(Self {
            tile_e: stream.htod_copy(te)?,
            tile_m: stream.htod_copy(tm)?,
            tile_n: stream.htod_copy(tn)?,
            num_tiles,
        })
    }
}

/// MXFP4 grouped GEMM（sm8x 路径）。`xg`: `[pairs, K]` bf16（`moe_gather_bf16` 的结果），
/// `wp`: `[E, N, K/2]` u8，`ws`: `[E, N, K/group]` u8，`out`: `[pairs, N]` bf16。
#[allow(clippy::too_many_arguments)]
pub fn mxfp4_gemm_grouped(
    stream: &Stream,
    xg: &CudaSlice<bf16>,
    wp: &CudaSlice<u8>,
    ws: &CudaSlice<u8>,
    expert_off: &CudaSlice<i32>,
    tiles: &Mxfp4GemmTiles,
    out: &mut CudaSlice<bf16>,
    k: usize,
    n: usize,
    group: usize,
) -> Result<()> {
    ensure!(k % group == 0 && k % 2 == 0, "mxfp4 gemm: K % group / 2");
    let pairs = xg.len() / k;
    ensure!(xg.len() >= pairs * k, "mxfp4 gemm: xg");
    ensure!(out.len() >= pairs * n, "mxfp4 gemm: out");
    let experts = expert_off.len().saturating_sub(1);
    ensure!(wp.len() >= experts * n * k / 2, "mxfp4 gemm: wp 容量");
    ensure!(ws.len() >= experts * n * k / group, "mxfp4 gemm: ws 容量");
    // SAFETY: 容量已校验；tiles 由同一 expert_off 生成。
    check(
        unsafe {
            ffi::mxfp4_gemm_grouped(
                stream.ptr(xg) as *const c_void,
                stream.ptr(wp) as *const c_void,
                stream.ptr(ws) as *const c_void,
                stream.ptr(expert_off) as *const i32,
                stream.ptr(&tiles.tile_e) as *const i32,
                stream.ptr(&tiles.tile_m) as *const i32,
                stream.ptr(&tiles.tile_n) as *const i32,
                stream.ptr_mut(out) as *mut c_void,
                tiles.num_tiles as i64,
                k as i64,
                n as i64,
                group as i64,
                stream.raw() as _,
            )
        },
        "mxfp4_gemm_grouped",
    )
}

// --------------------------------------------------------------------------- KDA chunk 形式 prefill

/// KDA chunk 形式 prefill 的工作区与执行计划。
///
/// 规格见 `docs/loops/k3-kda.md`「v2 规格」。所有中间量 fp32（`qn`/`kn` 为 bf16），
/// 尺寸见 `cuda/kernels.h` 的同名注释。
pub struct KdaChunkPlan {
    /// `[chunk, 2]`：chunk → (序列下标, 该序列内的块号)。
    pub chunk_tab: CudaSlice<i32>,
    /// `[num_seqs]`：每序列的块数。
    pub cu_nt: CudaSlice<i32>,
    /// `[num_seqs + 1]`：每序列首块在 `chunk_tab` 中的下标。
    pub chunk_off: CudaSlice<i32>,
    pub num_chunks: usize,
    pub num_seqs: usize,
    pub bt: usize,
}

impl KdaChunkPlan {
    /// host 侧切块表：(chunk_tab, cu_nt, chunk_off)。
    fn tables(cu_seqlens: &[i32], bt: usize) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let num_seqs = cu_seqlens.len() - 1;
        let mut tab = Vec::new();
        let mut nt = Vec::with_capacity(num_seqs);
        let mut off = Vec::with_capacity(num_seqs + 1);
        for s in 0..num_seqs {
            off.push(tab.len() as i32 / 2);
            let len = (cu_seqlens[s + 1] - cu_seqlens[s]).max(0) as usize;
            let n = len.div_ceil(bt);
            nt.push(n as i32);
            for c in 0..n {
                tab.push(s as i32);
                tab.push(c as i32);
            }
        }
        off.push(tab.len() as i32 / 2);
        (tab, nt, off)
    }

    /// 按 `cu_seqlens` 切块。`bt` 必须是 kernel 支持的块大小（当前 64）。
    pub fn new(stream: &Stream, cu_seqlens: &[i32], bt: usize) -> Result<Self> {
        ensure!(bt > 0 && cu_seqlens.len() >= 2, "kda_chunk plan: bt/cu_seqlens");
        let (tab, nt, off) = Self::tables(cu_seqlens, bt);
        let num_chunks = tab.len() / 2;
        Ok(Self {
            chunk_tab: stream.htod_copy(tab)?,
            cu_nt: stream.htod_copy(nt)?,
            chunk_off: stream.htod_copy(off)?,
            num_chunks,
            num_seqs: cu_seqlens.len() - 1,
            bt,
        })
    }

    /// 预分配容量（`max_chunks` 块 / `max_seqs` 序列），表内容为空；之后每步 `fill`。
    pub fn with_capacity(stream: &Stream, max_chunks: usize, max_seqs: usize, bt: usize) -> Result<Self> {
        ensure!(bt > 0, "kda_chunk plan: bt");
        Ok(Self {
            chunk_tab: stream.alloc_zeros::<i32>((2 * max_chunks).max(2))?,
            cu_nt: stream.alloc_zeros::<i32>(max_seqs.max(1))?,
            chunk_off: stream.alloc_zeros::<i32>(max_seqs + 1)?,
            num_chunks: 0,
            num_seqs: 0,
            bt,
        })
    }

    /// 复用设备缓冲重填切块表（engine-gap 第 7 轮：每 prefill 步省 3 次 cuMemAlloc）；容量不够时就地重分配。
    pub fn fill(&mut self, stream: &Stream, cu_seqlens: &[i32]) -> Result<()> {
        ensure!(cu_seqlens.len() >= 2, "kda_chunk plan: cu_seqlens");
        let (tab, nt, off) = Self::tables(cu_seqlens, self.bt);
        if tab.len() > self.chunk_tab.len() {
            self.chunk_tab = stream.alloc_zeros::<i32>(tab.len())?;
        }
        if nt.len() > self.cu_nt.len() {
            self.cu_nt = stream.alloc_zeros::<i32>(nt.len())?;
        }
        if off.len() > self.chunk_off.len() {
            self.chunk_off = stream.alloc_zeros::<i32>(off.len())?;
        }
        self.num_chunks = tab.len() / 2;
        self.num_seqs = cu_seqlens.len() - 1;
        if !tab.is_empty() {
            stream.htod_into(&tab, &mut self.chunk_tab)?;
        }
        stream.htod_into(&nt, &mut self.cu_nt)?;
        stream.htod_into(&off, &mut self.chunk_off)?;
        Ok(())
    }
}

/// chunk 形式 prefill 的中间缓冲。尺寸随 `[T, heads, head_dim]` 与块数变化。
pub struct KdaChunkScratch {
    pub qn: CudaSlice<bf16>,
    pub kn: CudaSlice<bf16>,
    pub beta: CudaSlice<f32>,
    pub gc: CudaSlice<f32>,
    pub aq: CudaSlice<f32>,
    pub u: CudaSlice<f32>,
    pub w: CudaSlice<f32>,
    pub kg: CudaSlice<f32>,
}

impl KdaChunkScratch {
    pub fn new(
        stream: &Stream,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        num_chunks: usize,
        bt: usize,
    ) -> Result<Self> {
        let n = tokens * heads * head_dim;
        Ok(Self {
            qn: stream.alloc_uninit::<bf16>(n)?,
            kn: stream.alloc_uninit::<bf16>(n)?,
            beta: stream.alloc_uninit::<f32>(tokens * heads)?,
            gc: stream.alloc_uninit::<f32>(n)?,
            aq: stream.alloc_uninit::<f32>(num_chunks * heads * bt * bt)?,
            u: stream.alloc_uninit::<f32>(n)?,
            w: stream.alloc_uninit::<f32>(n)?,
            kg: stream.alloc_uninit::<f32>(n)?,
        })
    }
}

/// 三段的第 1 段：q/k 归一化（q 乘 `scale`）、beta、块内 inclusive cumsum(gate)·1/ln2。
#[allow(clippy::too_many_arguments)]
pub fn kda_chunk_prep(
    stream: &Stream,
    q: &CudaSlice<bf16>,
    k: &CudaSlice<bf16>,
    raw_g: &CudaSlice<bf16>,
    raw_beta: &CudaSlice<f32>,
    a_log: &CudaSlice<f32>,
    dt_bias: &CudaSlice<f32>,
    cu_seqlens: &CudaSlice<i32>,
    plan: &KdaChunkPlan,
    sc: &mut KdaChunkScratch,
    tokens: usize,
    heads: usize,
    head_dim: usize,
    scale: f32,
    lower_bound: Option<f32>,
) -> Result<()> {
    // 输入 / scratch 都允许是按 max_tokens 预分配的更大缓冲：只按本步 `tokens` 判容量。
    let n = tokens * heads * head_dim;
    ensure!(
        q.len() >= n && k.len() >= n && raw_g.len() >= n && raw_beta.len() >= tokens * heads,
        "kda_chunk_prep: q/k/g/beta < {tokens} 行"
    );
    ensure!(
        sc.qn.len() >= n && sc.kn.len() >= n && sc.gc.len() >= n,
        "kda_chunk_prep: scratch 容量 < {tokens} 行"
    );
    ensure!(dt_bias.len() >= heads * head_dim && a_log.len() >= heads, "kda_chunk_prep: a_log/dt_bias");
    // SAFETY: 容量已校验；plan 由同一 cu_seqlens 生成。
    check(
        unsafe {
            ffi::kda_chunk_prep(
                stream.ptr(q) as *const c_void,
                stream.ptr(k) as *const c_void,
                stream.ptr(raw_g) as *const c_void,
                stream.ptr(raw_beta) as *const f32,
                stream.ptr(a_log) as *const f32,
                stream.ptr(dt_bias) as *const f32,
                stream.ptr_mut(&mut sc.qn) as *mut c_void,
                stream.ptr_mut(&mut sc.kn) as *mut c_void,
                stream.ptr_mut(&mut sc.beta) as *mut f32,
                stream.ptr_mut(&mut sc.gc) as *mut f32,
                stream.ptr(cu_seqlens) as *const i32,
                plan.num_seqs as i64,
                heads as i64,
                head_dim as i64,
                plan.bt as i64,
                scale,
                lower_bound.unwrap_or(0.0),
                i32::from(lower_bound.is_some()),
                stream.raw() as _,
            )
        },
        "kda_chunk_prep",
    )
}

/// 第 2 段：A/Aq → M = (I+A)⁻¹ → u / w / kg。
#[allow(clippy::too_many_arguments)]
pub fn kda_chunk_wy(
    stream: &Stream,
    v: &CudaSlice<bf16>,
    cu_seqlens: &CudaSlice<i32>,
    plan: &KdaChunkPlan,
    sc: &mut KdaChunkScratch,
    tokens: usize,
    heads: usize,
    head_dim: usize,
) -> Result<()> {
    let n = tokens * heads * head_dim;
    ensure!(sc.aq.len() >= plan.num_chunks * heads * plan.bt * plan.bt, "kda_chunk_wy: aq 容量");
    ensure!(
        v.len() >= n && sc.u.len() >= n && sc.w.len() >= n && sc.kg.len() >= n,
        "kda_chunk_wy: v/scratch < {tokens} 行"
    );
    // SAFETY: 容量已校验。
    check(
        unsafe {
            ffi::kda_chunk_wy(
                stream.ptr(&sc.qn) as *const c_void,
                stream.ptr(&sc.kn) as *const c_void,
                stream.ptr(v) as *const c_void,
                stream.ptr(&sc.gc) as *const f32,
                stream.ptr(&sc.beta) as *const f32,
                stream.ptr_mut(&mut sc.aq) as *mut f32,
                stream.ptr_mut(&mut sc.u) as *mut f32,
                stream.ptr_mut(&mut sc.w) as *mut f32,
                stream.ptr_mut(&mut sc.kg) as *mut f32,
                stream.ptr(cu_seqlens) as *const i32,
                stream.ptr(&plan.chunk_tab) as *const i32,
                plan.num_chunks as i64,
                heads as i64,
                head_dim as i64,
                plan.bt as i64,
                stream.raw() as _,
            )
        },
        "kda_chunk_wy",
    )
}

/// 第 3 段：跨 chunk 扫 h（fp32）并输出 o。
#[allow(clippy::too_many_arguments)]
pub fn kda_chunk_ho(
    stream: &Stream,
    cu_seqlens: &CudaSlice<i32>,
    plan: &KdaChunkPlan,
    sc: &KdaChunkScratch,
    out: &mut CudaSlice<bf16>,
    state: &mut CudaSlice<f32>,
    state_indices: Option<&CudaSlice<i32>>,
    tokens: usize,
    heads: usize,
    head_dim: usize,
) -> Result<()> {
    let per_slot = heads * head_dim * head_dim;
    ensure!(state.len() >= per_slot, "kda_chunk_ho: state");
    ensure!(out.len() >= tokens * heads * head_dim, "kda_chunk_ho: out < {tokens} 行");
    let psi = state_indices.map_or(std::ptr::null(), |s| stream.ptr(s) as *const i32);
    // SAFETY: 容量已校验；slot 编号由调度器保证 < state 槽数。
    check(
        unsafe {
            ffi::kda_chunk_ho(
                stream.ptr(&sc.qn) as *const c_void,
                stream.ptr(&sc.gc) as *const f32,
                stream.ptr(&sc.aq) as *const f32,
                stream.ptr(&sc.u) as *const f32,
                stream.ptr(&sc.w) as *const f32,
                stream.ptr(&sc.kg) as *const f32,
                stream.ptr_mut(out) as *mut c_void,
                stream.ptr_mut(state) as *mut c_void,
                stream.ptr(cu_seqlens) as *const i32,
                stream.ptr(&plan.cu_nt) as *const i32,
                stream.ptr(&plan.chunk_off) as *const i32,
                psi,
                per_slot as i64,
                plan.num_seqs as i64,
                heads as i64,
                head_dim as i64,
                plan.bt as i64,
                stream.raw() as _,
            )
        },
        "kda_chunk_ho",
    )
}

/// 输出门 + gated RMSNorm：`y = rmsnorm(x, w, head_dim, eps) * sigmoid(gate)`。
/// `x`/`gate`/`out`: `[rows, head_dim]` bf16，`w`: `[head_dim]` bf16。
pub fn kda_gated_rmsnorm(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    gate: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    rows: usize,
    head_dim: usize,
    eps: f32,
) -> Result<()> {
    ensure!(w.len() >= head_dim, "kda_gated_rmsnorm: w");
    ensure!(
        x.len() >= rows * head_dim && gate.len() >= rows * head_dim && out.len() >= rows * head_dim,
        "kda_gated_rmsnorm: x/gate/out"
    );
    // SAFETY: 容量已校验。
    check(
        unsafe {
            ffi::kda_gated_rmsnorm(
                stream.ptr(x) as *const c_void,
                stream.ptr(w) as *const c_void,
                stream.ptr(gate) as *const c_void,
                stream.ptr_mut(out) as *mut c_void,
                rows as i64,
                head_dim as i64,
                eps,
                stream.raw() as _,
            )
        },
        "kda_gated_rmsnorm",
    )
}

// ------------------------------------------------------------------ GDN（Qwen3.5 线的 Gated DeltaNet）
// 数学口径见 `cuda/gdn.cu` 顶部与 `docs/loops/qwen35-gdn.md`。全部只收形状参数。

/// 短卷积的激活：`siLU`（Qwen3.5）或 `sigmoid`（其它 GDN 变体）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnAct {
    Silu,
    Sigmoid,
}

impl GdnAct {
    /// kernel 侧的整数编码（与 `cuda/gdn.cu` 约定一致）。
    pub fn code(self) -> i32 {
        match self {
            GdnAct::Silu => 0,
            GdnAct::Sigmoid => 1,
        }
    }
}

/// GDN 短卷积 prefill（varlen）。`x`/`out`: `[T, dim]` bf16，`w`: `[dim, width]` bf16（从旧到新），
/// `conv_state`: `[slots, dim, state_len]` bf16（`state_len = width-1`，原地写回）。
#[allow(clippy::too_many_arguments)]
pub fn gdn_conv1d_prefill(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    conv_state: &mut CudaSlice<bf16>,
    cu_seqlens: &CudaSlice<i32>,
    state_indices: Option<&CudaSlice<i32>>,
    has_initial: Option<&CudaSlice<i32>>,
    num_seqs: usize,
    tokens: usize,
    dim: usize,
    width: usize,
    state_len: usize,
    act: GdnAct,
) -> Result<()> {
    let rows = gdn_conv_rows(stream, tokens, dim, width, state_len);
    gdn_conv1d_prefill_with(
        stream,
        x,
        w,
        out,
        conv_state,
        cu_seqlens,
        state_indices,
        has_initial,
        num_seqs,
        tokens,
        dim,
        width,
        state_len,
        act,
        rows,
    )
}

/// `gdn_conv1d_prefill` 指定分块行数：`rows_per_block == 0` 走老 kernel（每 (seq, 通道) 一线程串行扫 T），
/// 否则按 token 分块（`docs/loops/qwen35-perf.md` 第 5A 轮）。对拍 / 计时用。
#[allow(clippy::too_many_arguments)]
pub fn gdn_conv1d_prefill_with(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    conv_state: &mut CudaSlice<bf16>,
    cu_seqlens: &CudaSlice<i32>,
    state_indices: Option<&CudaSlice<i32>>,
    has_initial: Option<&CudaSlice<i32>>,
    num_seqs: usize,
    tokens: usize,
    dim: usize,
    width: usize,
    state_len: usize,
    act: GdnAct,
    rows_per_block: usize,
) -> Result<()> {
    gdn_conv1d_prefill_strided(
        stream,
        &x.slice(..),
        w,
        out,
        conv_state,
        cu_seqlens,
        state_indices,
        has_initial,
        num_seqs,
        tokens,
        dim,
        dim,
        width,
        state_len,
        act,
        rows_per_block,
    )
}

/// `x` 带行距版：`x` 是 `[tokens, x_stride]` 缓冲的列段视图（融合 in_proj 输出里的 qkv 段，第 9 轮），`out` 紧凑 `[tokens, dim]`。
#[allow(clippy::too_many_arguments)]
pub fn gdn_conv1d_prefill_strided(
    stream: &Stream,
    x: &CudaView<'_, bf16>,
    w: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    conv_state: &mut CudaSlice<bf16>,
    cu_seqlens: &CudaSlice<i32>,
    state_indices: Option<&CudaSlice<i32>>,
    has_initial: Option<&CudaSlice<i32>>,
    num_seqs: usize,
    tokens: usize,
    dim: usize,
    x_stride: usize,
    width: usize,
    state_len: usize,
    act: GdnAct,
    rows_per_block: usize,
) -> Result<()> {
    ensure!(width >= 1 && width <= 8, "gdn_conv1d_prefill: width {width} 超出 [1,8]");
    ensure!(x_stride >= dim, "gdn_conv1d_prefill: x_stride {x_stride} < dim {dim}");
    ensure!(tokens == 0 || x.len() >= (tokens - 1) * x_stride + dim, "gdn_conv1d_prefill: x 容量 < tokens 行");
    ensure!(state_len >= width.saturating_sub(1), "gdn_conv1d_prefill: state_len {state_len} < width-1");
    ensure!(cu_seqlens.len() >= num_seqs + 1, "gdn_conv1d_prefill: cu_seqlens");
    ensure!(w.len() >= dim * width && out.len() >= tokens * dim, "gdn_conv1d_prefill: w/out");
    ensure!(conv_state.len() >= state_len, "gdn_conv1d_prefill: conv_state");
    let psi = state_indices.map_or(std::ptr::null(), |s| stream.ptr(s) as *const i32);
    let phi = has_initial.map_or(std::ptr::null(), |s| stream.ptr(s) as *const i32);
    // SAFETY: 容量已校验；cu_seqlens 的单调性与槽号范围由调度侧保证。
    check(
        unsafe {
            ffi::gdn_conv1d_prefill(
                stream.ptr(x) as *const c_void,
                stream.ptr(w) as *const c_void,
                stream.ptr_mut(out) as *mut c_void,
                stream.ptr_mut(conv_state) as *mut c_void,
                stream.ptr(cu_seqlens) as *const i32,
                psi,
                phi,
                num_seqs as i64,
                dim as i64,
                width as i64,
                state_len as i64,
                act.code(),
                tokens as i64,
                rows_per_block as i64,
                x_stride as i64,
                stream.raw() as ffi::cudaStream_t,
            )
        },
        "gdn_conv1d_prefill",
    )
}

/// GDN 短卷积单 token 更新（decode）：`x`/`out`: `[batch, dim]`，`conv_state` 原地左移并追加。
#[allow(clippy::too_many_arguments)]
pub fn gdn_conv1d_update(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    conv_state: &mut CudaSlice<bf16>,
    state_indices: Option<&CudaSlice<i32>>,
    batch: usize,
    dim: usize,
    width: usize,
    state_len: usize,
    act: GdnAct,
) -> Result<()> {
    gdn_conv1d_update_strided(stream, &x.slice(..), w, out, conv_state, state_indices, batch, dim, dim, width, state_len, act)
}

/// `x` 带行距版（`x` 是 `[batch, x_stride]` 的列段视图），`out` 紧凑 `[batch, dim]`。
#[allow(clippy::too_many_arguments)]
pub fn gdn_conv1d_update_strided(
    stream: &Stream,
    x: &CudaView<'_, bf16>,
    w: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    conv_state: &mut CudaSlice<bf16>,
    state_indices: Option<&CudaSlice<i32>>,
    batch: usize,
    dim: usize,
    x_stride: usize,
    width: usize,
    state_len: usize,
    act: GdnAct,
) -> Result<()> {
    ensure!(width >= 1 && width <= 8, "gdn_conv1d_update: width {width} 超出 [1,8]");
    ensure!(state_len >= width.saturating_sub(1), "gdn_conv1d_update: state_len");
    ensure!(x_stride >= dim, "gdn_conv1d_update: x_stride {x_stride} < dim {dim}");
    ensure!(batch == 0 || x.len() >= (batch - 1) * x_stride + dim, "gdn_conv1d_update: x");
    ensure!(out.len() >= batch * dim, "gdn_conv1d_update: out");
    ensure!(w.len() >= dim * width, "gdn_conv1d_update: w");
    let psi = state_indices.map_or(std::ptr::null(), |s| stream.ptr(s) as *const i32);
    // SAFETY: 容量已校验；槽号由调度器保证 < state 槽数。
    check(
        unsafe {
            ffi::gdn_conv1d_update(
                stream.ptr(x) as *const c_void,
                stream.ptr(w) as *const c_void,
                stream.ptr_mut(out) as *mut c_void,
                stream.ptr_mut(conv_state) as *mut c_void,
                psi,
                batch as i64,
                dim as i64,
                width as i64,
                state_len as i64,
                act.code(),
                x_stride as i64,
                stream.raw() as ffi::cudaStream_t,
            )
        },
        "gdn_conv1d_update",
    )
}

/// 逐 v 头标量的 gate / beta：`g = -exp(A_log[hv])·softplus(a + dt_bias[hv])`、`beta = sigmoid(b)`。
/// `a`/`b`: `[T, heads_v]` bf16，`a_log`/`dt_bias`: `[heads_v]` f32，输出 f32 同形。
pub fn gdn_gate(
    stream: &Stream,
    a: &CudaSlice<bf16>,
    b: &CudaSlice<bf16>,
    a_log: &CudaSlice<f32>,
    dt_bias: &CudaSlice<f32>,
    g_out: &mut CudaSlice<f32>,
    beta_out: &mut CudaSlice<f32>,
    tokens: usize,
    heads_v: usize,
) -> Result<()> {
    gdn_gate_strided(stream, &a.slice(..), &b.slice(..), a_log, dt_bias, g_out, beta_out, tokens, heads_v, heads_v)
}

/// `a`/`b` 带行距版：`[tokens, ab_stride]` 缓冲的列段视图（融合 in_proj 输出的 a / b 段）。
#[allow(clippy::too_many_arguments)]
pub fn gdn_gate_strided(
    stream: &Stream,
    a: &CudaView<'_, bf16>,
    b: &CudaView<'_, bf16>,
    a_log: &CudaSlice<f32>,
    dt_bias: &CudaSlice<f32>,
    g_out: &mut CudaSlice<f32>,
    beta_out: &mut CudaSlice<f32>,
    tokens: usize,
    heads_v: usize,
    ab_stride: usize,
) -> Result<()> {
    let n = tokens * heads_v;
    ensure!(a_log.len() >= heads_v && dt_bias.len() >= heads_v, "gdn_gate: a_log/dt_bias");
    ensure!(ab_stride >= heads_v, "gdn_gate: ab_stride {ab_stride} < heads_v {heads_v}");
    let need = if tokens == 0 { 0 } else { (tokens - 1) * ab_stride + heads_v };
    ensure!(a.len() >= need && b.len() >= need, "gdn_gate: a/b");
    ensure!(g_out.len() >= n && beta_out.len() >= n, "gdn_gate: outputs");
    // SAFETY: 容量已校验。
    check(
        unsafe {
            ffi::gdn_gate(
                stream.ptr(a) as *const c_void,
                stream.ptr(b) as *const c_void,
                stream.ptr(a_log) as *const f32,
                stream.ptr(dt_bias) as *const f32,
                stream.ptr_mut(g_out) as *mut f32,
                stream.ptr_mut(beta_out) as *mut f32,
                n as i64,
                heads_v as i64,
                1,
                ab_stride as i64,
                stream.raw() as ffi::cudaStream_t,
            )
        },
        "gdn_gate",
    )
}

/// gate + beta + 块内 cumsum 一次算完（prefill）。替代 `gdn_gate_strided` + `cumsum_kernel` 两次 launch，
/// 对齐 vLLM FLA 的 `fused_kda_gate_chunk_cumsum`。
///
/// `a`/`b` 是融合 in_proj 输出里 `[T, ·]` 的两个列段（行距 `ab_stride`）；
/// `g_out`/`beta_out`/`gc_out` 都是 `[T, heads_v]` f32；`gc` 已是 exp2 空间（乘了 1/ln2）。
#[allow(clippy::too_many_arguments)]
pub fn gdn_gate_cumsum_strided(
    stream: &Stream,
    a: &CudaView<'_, bf16>,
    b: &CudaView<'_, bf16>,
    a_log: &CudaSlice<f32>,
    dt_bias: &CudaSlice<f32>,
    g_out: &mut CudaSlice<f32>,
    beta_out: &mut CudaSlice<f32>,
    gc_out: &mut CudaSlice<f32>,
    cu_seqlens: &CudaSlice<i32>,
    chunk_tab: &CudaSlice<i32>,
    num_chunks: usize,
    heads_v: usize,
    bt: usize,
    ab_stride: usize,
) -> Result<()> {
    ensure!(a_log.len() >= heads_v && dt_bias.len() >= heads_v, "gdn_gate_cumsum: a_log/dt_bias");
    ensure!(bt == 64, "gdn_gate_cumsum: bt {bt} != 64（kernel 模板写死）");
    let n = num_chunks * heads_v;
    ensure!(g_out.len() >= n && beta_out.len() >= n && gc_out.len() >= n, "gdn_gate_cumsum: outputs");
    if n == 0 {
        return Ok(());
    }
    // SAFETY: 容量已校验。
    check(
        unsafe {
            ffi::gdn_gate_cumsum(
                stream.ptr(a) as *const c_void,
                stream.ptr(b) as *const c_void,
                stream.ptr(a_log) as *const f32,
                stream.ptr(dt_bias) as *const f32,
                stream.ptr_mut(g_out) as *mut f32,
                stream.ptr_mut(beta_out) as *mut f32,
                stream.ptr_mut(gc_out) as *mut f32,
                stream.ptr(cu_seqlens) as *const i32,
                stream.ptr(chunk_tab) as *const i32,
                num_chunks as i64,
                heads_v as i64,
                bt as i64,
                1,
                ab_stride as i64,
                stream.raw() as ffi::cudaStream_t,
            )
        },
        "gdn_gate_cumsum",
    )
}

/// GDN 递归核心（prefill 与 decode 同一条逐 token 路径）。
/// `mixed_qkv`: `[T, 2·heads_k·k_dim + heads_v·v_dim]` bf16（q|k|v 平铺）；
/// `out`: `[T, heads_v, v_dim]` bf16；`state`: `[num_slots, heads_v, v_dim, k_dim]` f32（原地更新）。
/// `fused_gate = true` 时 gate/beta 由 `a_raw`/`b_raw` + `a_log`/`dt_bias` 现算（decode）；
/// 否则读调用方预算好的 `g`/`beta`（prefill，形状 `[T, heads_v]` f32）。
/// decode 步（每序列 1 token，`tokens == num_seqs`）走寄存器版 `recurrent_decode_kernel`（BV 由
/// `ArchProfile::gdn_decode_bv` 给，`GDN_DECODE_BV=<n>` 覆盖，`GDN_DECODE=legacy` 回老 shared 版）；
/// prefill 尾段走老 kernel。
#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrent(
    stream: &Stream,
    mixed_qkv: &CudaSlice<bf16>,
    g: Option<&CudaSlice<f32>>,
    beta: Option<&CudaSlice<f32>>,
    a_raw: Option<&CudaSlice<bf16>>,
    b_raw: Option<&CudaSlice<bf16>>,
    a_log: &CudaSlice<f32>,
    dt_bias: &CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
    state: &mut CudaSlice<f32>,
    cu_seqlens: &CudaSlice<i32>,
    state_indices: Option<&CudaSlice<i32>>,
    num_seqs: usize,
    tokens: usize,
    heads_k: usize,
    heads_v: usize,
    k_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Result<()> {
    gdn_recurrent_with(
        stream,
        mixed_qkv,
        g,
        beta,
        a_raw,
        b_raw,
        a_log,
        dt_bias,
        out,
        state,
        cu_seqlens,
        state_indices,
        num_seqs,
        tokens,
        heads_k,
        heads_v,
        k_dim,
        v_dim,
        scale,
        None,
    )
}

/// `gdn_recurrent` 的显式版：`decode_bv = Some(0)` 强制老 kernel，`Some(n)` 强制寄存器版 BV=n，`None` 自动
/// （变体 / 持久 grid 走 `ArchProfile` 默认 + 环境变量）。
#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrent_with(
    stream: &Stream,
    mixed_qkv: &CudaSlice<bf16>,
    g: Option<&CudaSlice<f32>>,
    beta: Option<&CudaSlice<f32>>,
    a_raw: Option<&CudaSlice<bf16>>,
    b_raw: Option<&CudaSlice<bf16>>,
    a_log: &CudaSlice<f32>,
    dt_bias: &CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
    state: &mut CudaSlice<f32>,
    cu_seqlens: &CudaSlice<i32>,
    state_indices: Option<&CudaSlice<i32>>,
    num_seqs: usize,
    tokens: usize,
    heads_k: usize,
    heads_v: usize,
    k_dim: usize,
    v_dim: usize,
    scale: f32,
    decode_bv: Option<usize>,
) -> Result<()> {
    gdn_recurrent_variant(
        stream,
        mixed_qkv,
        g,
        beta,
        a_raw,
        b_raw,
        a_log,
        dt_bias,
        out,
        state,
        cu_seqlens,
        state_indices,
        num_seqs,
        tokens,
        heads_k,
        heads_v,
        k_dim,
        v_dim,
        scale,
        decode_bv,
        None,
        None,
    )
}

/// 最显式的版本：`decode_variant`（`None` = `ArchProfile::gdn_decode_variant` / `GDN_DECODE_VARIANT=`），
/// `decode_grid`（pipe 变体的持久 block 数，`None` = occupancy × SM / `GDN_DECODE_GRID=`）。对拍 / 计时用。
#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrent_variant(
    stream: &Stream,
    mixed_qkv: &CudaSlice<bf16>,
    g: Option<&CudaSlice<f32>>,
    beta: Option<&CudaSlice<f32>>,
    a_raw: Option<&CudaSlice<bf16>>,
    b_raw: Option<&CudaSlice<bf16>>,
    a_log: &CudaSlice<f32>,
    dt_bias: &CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
    state: &mut CudaSlice<f32>,
    cu_seqlens: &CudaSlice<i32>,
    state_indices: Option<&CudaSlice<i32>>,
    num_seqs: usize,
    tokens: usize,
    heads_k: usize,
    heads_v: usize,
    k_dim: usize,
    v_dim: usize,
    scale: f32,
    decode_bv: Option<usize>,
    decode_variant: Option<crate::arch::GdnDecodeVariant>,
    decode_grid: Option<usize>,
) -> Result<()> {
    let (av, bv) = (a_raw.map(|s| s.slice(..)), b_raw.map(|s| s.slice(..)));
    gdn_recurrent_strided(
        stream,
        mixed_qkv,
        g,
        beta,
        av.as_ref(),
        bv.as_ref(),
        heads_v,
        a_log,
        dt_bias,
        out,
        state,
        cu_seqlens,
        state_indices,
        num_seqs,
        tokens,
        heads_k,
        heads_v,
        k_dim,
        v_dim,
        scale,
        decode_bv,
        decode_variant,
        decode_grid,
    )
}

/// `a_raw`/`b_raw` 带行距版（fused gate 的 decode 路径读融合 in_proj 输出的 a / b 列段，行距 `ab_stride`）。
#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrent_strided(
    stream: &Stream,
    mixed_qkv: &CudaSlice<bf16>,
    g: Option<&CudaSlice<f32>>,
    beta: Option<&CudaSlice<f32>>,
    a_raw: Option<&CudaView<'_, bf16>>,
    b_raw: Option<&CudaView<'_, bf16>>,
    ab_stride: usize,
    a_log: &CudaSlice<f32>,
    dt_bias: &CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
    state: &mut CudaSlice<f32>,
    cu_seqlens: &CudaSlice<i32>,
    state_indices: Option<&CudaSlice<i32>>,
    num_seqs: usize,
    tokens: usize,
    heads_k: usize,
    heads_v: usize,
    k_dim: usize,
    v_dim: usize,
    scale: f32,
    decode_bv: Option<usize>,
    decode_variant: Option<crate::arch::GdnDecodeVariant>,
    decode_grid: Option<usize>,
) -> Result<()> {
    ensure!(ab_stride >= heads_v, "gdn_recurrent: ab_stride {ab_stride} < heads_v {heads_v}");
    ensure!(
        heads_k > 0 && heads_v % heads_k == 0,
        "gdn_recurrent: heads_v {heads_v} 不能被 heads_k {heads_k} 整除"
    );
    ensure!(cu_seqlens.len() >= num_seqs + 1, "gdn_recurrent: cu_seqlens");
    ensure!(a_log.len() >= heads_v && dt_bias.len() >= heads_v, "gdn_recurrent: a_log/dt_bias");
    let row = 2 * heads_k * k_dim + heads_v * v_dim;
    // tokens 由调用方显式给（不能从缓冲长度推：激活缓冲按 max_tokens 预分配，推出来永远 ≠ num_seqs，
    // decode 寄存器 kernel 分发条件就永远不成立——2026-10-20 bench_b300_8 踩坑）。
    ensure!(
        mixed_qkv.len() >= tokens * row,
        "gdn_recurrent: mixed_qkv {} < tokens {tokens} × row {row}",
        mixed_qkv.len()
    );
    ensure!(tokens >= num_seqs, "gdn_recurrent: tokens {tokens} < num_seqs {num_seqs}");
    ensure!(out.len() >= tokens * heads_v * v_dim, "gdn_recurrent: out");
    let per_slot = heads_v * v_dim * k_dim;
    ensure!(state.len() >= per_slot, "gdn_recurrent: state");
    let group = heads_v / heads_k;
    let fused = g.is_none();
    let vsplit = gdn_recurrent_vsplit(stream, num_seqs * heads_v, v_dim);
    let decode_bv = match decode_bv {
        Some(bv) => bv,
        None if tokens == num_seqs => gdn_decode_bv(stream, num_seqs * heads_v, k_dim, v_dim),
        None => 0,
    };
    if decode_bv > 0 {
        ensure!(
            v_dim % decode_bv == 0 && (k_dim == 64 || k_dim == 128),
            "gdn_recurrent: decode_bv {decode_bv} 不适用于 K={k_dim} V={v_dim}"
        );
    }
    if fused {
        ensure!(a_raw.is_some() && b_raw.is_some(), "gdn_recurrent: fused_gate 需要 a_raw/b_raw");
        let need = if tokens == 0 { 0 } else { (tokens - 1) * ab_stride + heads_v };
        ensure!(a_raw.unwrap().len() >= need && b_raw.unwrap().len() >= need, "gdn_recurrent: a_raw/b_raw");
    } else {
        ensure!(beta.is_some(), "gdn_recurrent: 非 fused 路径需要 beta");
        ensure!(
            g.unwrap().len() >= tokens * heads_v && beta.unwrap().len() >= tokens * heads_v,
            "gdn_recurrent: g/beta"
        );
    }
    let (variant, grid) = if decode_bv > 0 {
        (
            decode_variant.unwrap_or_else(|| gdn_decode_variant(stream)),
            decode_grid.unwrap_or_else(gdn_decode_grid),
        )
    } else {
        (crate::arch::GdnDecodeVariant::Base, 0)
    };
    let null_bf16 = std::ptr::null::<c_void>();
    let pg = g.map_or(std::ptr::null(), |s| stream.ptr(s) as *const f32);
    let pb = beta.map_or(std::ptr::null(), |s| stream.ptr(s) as *const f32);
    let par = a_raw.map_or(null_bf16, |s| stream.ptr(s) as *const c_void);
    let pbr = b_raw.map_or(null_bf16, |s| stream.ptr(s) as *const c_void);
    let psi = state_indices.map_or(std::ptr::null(), |s| stream.ptr(s) as *const i32);
    // SAFETY: 容量已校验；槽号与 cu_seqlens 由调度侧保证。
    check(
        unsafe {
            ffi::gdn_recurrent(
                stream.ptr(mixed_qkv) as *const c_void,
                pg,
                pb,
                par,
                pbr,
                stream.ptr(a_log) as *const f32,
                stream.ptr(dt_bias) as *const f32,
                stream.ptr_mut(out) as *mut c_void,
                stream.ptr_mut(state) as *mut c_void,
                stream.ptr(cu_seqlens) as *const i32,
                psi,
                num_seqs as i64,
                heads_k as i64,
                heads_v as i64,
                k_dim as i64,
                v_dim as i64,
                group as i64,
                scale,
                per_slot as i64,
                i32::from(fused),
                vsplit as i64,
                decode_bv as i64,
                variant.code(),
                grid as i64,
                ab_stride as i64,
                stream.raw() as ffi::cudaStream_t,
            )
        },
        "gdn_recurrent",
    )
}

/// decode 寄存器版 kernel 的 BV：默认 `ArchProfile::gdn_decode_bv`（按 SM 数选最大的仍能铺满的档）；
/// `GDN_DECODE=legacy` 关闭（返回 0 → 老 kernel）；`GDN_DECODE_BV=<n>` 覆盖（须是 kernel 支持的档且整除 V）。
fn gdn_decode_bv(stream: &Stream, base_blocks: usize, k_dim: usize, v_dim: usize) -> usize {
    use std::sync::OnceLock;
    static MODE: OnceLock<(bool, Option<usize>)> = OnceLock::new();
    static SM: OnceLock<usize> = OnceLock::new();
    let (legacy, over) = *MODE.get_or_init(|| {
        let legacy = matches!(std::env::var("GDN_DECODE").as_deref(), Ok("legacy"));
        let over = std::env::var("GDN_DECODE_BV").ok().and_then(|s| s.parse().ok());
        if legacy || over.is_some() {
            tracing::info!(legacy, ?over, "gdn decode kernel 由环境变量覆盖");
        }
        (legacy, over)
    });
    if legacy || (k_dim != 64 && k_dim != 128) {
        return 0;
    }
    if let Some(bv) = over {
        if matches!(bv, 16 | 32 | 64) && v_dim % bv == 0 {
            return bv;
        }
    }
    let sm = *SM.get_or_init(|| stream.device().sm_count().unwrap_or(1));
    static ARCH: OnceLock<crate::arch::Arch> = OnceLock::new();
    let arch = *ARCH.get_or_init(|| {
        stream.device().compute_capability().map(|cc| crate::arch::ArchProfile::for_cc(cc, sm).arch).unwrap_or(crate::arch::Arch::Sm8x)
    });
    crate::arch::ArchProfile::gdn_decode_bv_for(arch, sm, base_blocks, k_dim, v_dim)
}

/// rmsnorm 系的线程布局：显式 `layout` 优先；否则 `RMSNORM_LAYOUT=block256|warp|block1024` 覆盖；否则
/// `ArchProfile::rmsnorm_layout(rows, hidden)`。返回 C ABI 编码。
fn rmsnorm_layout(
    stream: &Stream,
    rows: usize,
    hidden: usize,
    layout: Option<crate::arch::RmsnormLayout>,
) -> i32 {
    use crate::arch::{ArchProfile, RmsnormLayout};
    use std::sync::OnceLock;
    if let Some(l) = layout {
        return l.code();
    }
    static OVER: OnceLock<Option<RmsnormLayout>> = OnceLock::new();
    static PROF: OnceLock<ArchProfile> = OnceLock::new();
    let over = *OVER.get_or_init(|| {
        let s = std::env::var("RMSNORM_LAYOUT").ok()?;
        let l = RmsnormLayout::parse(&s);
        match l {
            Some(_) => tracing::info!(layout = s, "rmsnorm 布局由环境变量覆盖"),
            None => tracing::warn!(layout = s, "RMSNORM_LAYOUT 不识别，用 ArchProfile 默认"),
        }
        l
    });
    if let Some(l) = over {
        return l.code();
    }
    PROF.get_or_init(|| {
        let cc = stream.device().compute_capability().unwrap_or((8, 0));
        let sm = stream.device().sm_count().unwrap_or(1);
        ArchProfile::for_cc(cc, sm)
    })
    .rmsnorm_layout(rows, hidden)
    .code()
}

/// decode 寄存器版 kernel 的变体：默认 `ArchProfile::gdn_decode_variant(arch)`；`GDN_DECODE_VARIANT=base|tr|pipe|pipe_base` 覆盖。
fn gdn_decode_variant(stream: &Stream) -> crate::arch::GdnDecodeVariant {
    use crate::arch::{ArchProfile, GdnDecodeVariant};
    use std::sync::OnceLock;
    static V: OnceLock<GdnDecodeVariant> = OnceLock::new();
    *V.get_or_init(|| {
        if let Ok(s) = std::env::var("GDN_DECODE_VARIANT") {
            if let Some(v) = GdnDecodeVariant::parse(&s) {
                tracing::info!(variant = s, "gdn decode 变体由环境变量覆盖");
                return v;
            }
            tracing::warn!(variant = s, "GDN_DECODE_VARIANT 不识别，用 ArchProfile 默认");
        }
        let cc = stream.device().compute_capability().unwrap_or((8, 0));
        let sm = stream.device().sm_count().unwrap_or(1);
        ArchProfile::for_cc(cc, sm).gdn_decode_variant()
    })
}

/// pipe 变体的持久 block 数：0 = kernel 侧按 occupancy × SM 数算；`GDN_DECODE_GRID=<n>` 覆盖（B300 对比用）。
fn gdn_decode_grid() -> usize {
    use std::sync::OnceLock;
    static G: OnceLock<usize> = OnceLock::new();
    *G.get_or_init(|| {
        let g = std::env::var("GDN_DECODE_GRID").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        if g > 0 {
            tracing::info!(grid = g, "gdn decode pipe grid 由环境变量覆盖");
        }
        g
    })
}

/// chunk prefill `ho` 的 V 分片 BV：默认 `ArchProfile::gdn_chunk_ho_bv`；`GDN_CHUNK_HO=legacy` 回老 fp32 kernel（返回 0）；
/// `GDN_CHUNK_HO_BV=<16|32>` 覆盖。
fn gdn_chunk_ho_bv(stream: &Stream, base_blocks: usize, k_dim: usize, v_dim: usize) -> usize {
    use std::sync::OnceLock;
    static MODE: OnceLock<(bool, Option<usize>)> = OnceLock::new();
    static SM: OnceLock<usize> = OnceLock::new();
    let (legacy, over) = *MODE.get_or_init(|| {
        let legacy = matches!(std::env::var("GDN_CHUNK_HO").as_deref(), Ok("legacy"));
        let over = std::env::var("GDN_CHUNK_HO_BV").ok().and_then(|s| s.parse().ok());
        if legacy || over.is_some() {
            tracing::info!(legacy, ?over, "gdn chunk ho kernel 由环境变量覆盖");
        }
        (legacy, over)
    });
    if legacy {
        return 0;
    }
    if let Some(bv) = over {
        if matches!(bv, 16 | 32 | 64) && v_dim % bv == 0 {
            return bv;
        }
    }
    let sm = *SM.get_or_init(|| stream.device().sm_count().unwrap_or(1));
    static ARCH: OnceLock<crate::arch::Arch> = OnceLock::new();
    let arch = *ARCH.get_or_init(|| {
        stream.device().compute_capability().map(|cc| crate::arch::ArchProfile::for_cc(cc, sm).arch).unwrap_or(crate::arch::Arch::Sm8x)
    });
    crate::arch::ArchProfile::gdn_chunk_ho_bv_for(arch, sm, base_blocks, k_dim, v_dim)
}

/// chunk prefill `ho` 是否寄存器预取：默认 `ArchProfile::gdn_chunk_ho_prefetch(bv)`；`GDN_CHUNK_HO_PREFETCH=0|1` 覆盖。
fn gdn_chunk_ho_prefetch(stream: &Stream, ho_bv: usize) -> bool {
    use std::sync::OnceLock;
    static OVER: OnceLock<Option<bool>> = OnceLock::new();
    static ARCH: OnceLock<crate::arch::Arch> = OnceLock::new();
    let arch = *ARCH.get_or_init(|| {
        let sm = stream.device().sm_count().unwrap_or(1);
        stream.device().compute_capability().map(|cc| crate::arch::ArchProfile::for_cc(cc, sm).arch).unwrap_or(crate::arch::Arch::Sm8x)
    });
    let over = *OVER.get_or_init(|| {
        let o = std::env::var("GDN_CHUNK_HO_PREFETCH").ok().map(|s| s != "0");
        if o.is_some() {
            tracing::info!(?o, "gdn chunk ho 预取由环境变量覆盖");
        }
        o
    });
    if ho_bv == 0 {
        return false;
    }
    over.unwrap_or_else(|| crate::arch::ArchProfile::gdn_chunk_ho_prefetch_for(arch, ho_bv))
}

/// chunk prefill `wy` 的实现：`Legacy` 老 fp32；`Mma` tensor core + 分块三角逆（r9 第 3 轮，默认）；
/// `MmaSerial` tensor core + 64 步前代（第 5A 轮，回退）。`GDN_CHUNK_WY=legacy|serial` 覆盖。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnChunkWy {
    Legacy,
    Mma,
    MmaSerial,
    /// r10 第 3 轮：分块三角逆 + 8 warp（B300 ncu：wy 延迟型，加 warp 掩 smem 依赖链）。
    Mma8,
}

impl GdnChunkWy {
    fn code(self) -> i64 {
        match self {
            Self::Legacy => 0,
            Self::Mma => 1,
            Self::Mma8 => 3,
            Self::MmaSerial => 2,
        }
    }
}

impl From<bool> for GdnChunkWy {
    fn from(mma: bool) -> Self {
        if mma { Self::Mma } else { Self::Legacy }
    }
}

/// 默认 wy 实现：K ∈ {64,128,256} 且 V % 8 == 0 才有 tensor core 版，否则老 fp32。
fn gdn_chunk_wy_default(stream: &Stream, k_dim: usize, v_dim: usize) -> GdnChunkWy {
    use std::sync::OnceLock;
    static OVER: OnceLock<Option<GdnChunkWy>> = OnceLock::new();
    static ARCH: OnceLock<crate::arch::Arch> = OnceLock::new();
    let arch = *ARCH.get_or_init(|| {
        let sm = stream.device().sm_count().unwrap_or(1);
        stream.device().compute_capability().map(|cc| crate::arch::ArchProfile::for_cc(cc, sm).arch).unwrap_or(crate::arch::Arch::Sm8x)
    });
    let over = *OVER.get_or_init(|| {
        let o = match std::env::var("GDN_CHUNK_WY").as_deref() {
            Ok("legacy") => Some(GdnChunkWy::Legacy),
            Ok("serial") => Some(GdnChunkWy::MmaSerial),
            Ok("mma4") => Some(GdnChunkWy::Mma),
            Ok("mma8") => Some(GdnChunkWy::Mma8),
            _ => None,
        };
        if o.is_some() {
            tracing::info!(?o, "gdn chunk wy kernel 由环境变量覆盖");
        }
        o
    });
    if !(matches!(k_dim, 64 | 128 | 256) && v_dim % 8 == 0) {
        return GdnChunkWy::Legacy;
    }
    over.unwrap_or(if crate::arch::ArchProfile::gdn_chunk_wy_warps_for(arch) == 8 { GdnChunkWy::Mma8 } else { GdnChunkWy::Mma })
}

/// conv1d prefill 的按 token 分块行数：默认 `ArchProfile::gdn_conv_rows`；`CONV_PREFILL=legacy` 回老 kernel（返回 0）；
/// `CONV_PREFILL_ROWS=<n>` 覆盖。
pub fn gdn_conv_rows(stream: &Stream, tokens: usize, dim: usize, width: usize, state_len: usize) -> usize {
    use std::sync::OnceLock;
    static MODE: OnceLock<(bool, Option<usize>)> = OnceLock::new();
    static SM: OnceLock<usize> = OnceLock::new();
    let (legacy, over) = *MODE.get_or_init(|| {
        let legacy = matches!(std::env::var("CONV_PREFILL").as_deref(), Ok("legacy"));
        let over = std::env::var("CONV_PREFILL_ROWS").ok().and_then(|s| s.parse().ok());
        if legacy || over.is_some() {
            tracing::info!(legacy, ?over, "conv1d prefill kernel 由环境变量覆盖");
        }
        (legacy, over)
    });
    // 新 kernel 的形状前提：width ∈ [2,4] 且 state_len == width-1（qwen35 形状；其余走老版）
    if legacy || !(2..=4).contains(&width) || state_len != width - 1 {
        return 0;
    }
    if let Some(r) = over {
        if r >= width - 1 {
            return r;
        }
    }
    let sm = *SM.get_or_init(|| stream.device().sm_count().unwrap_or(1));
    static ARCH: OnceLock<crate::arch::Arch> = OnceLock::new();
    let arch = *ARCH.get_or_init(|| {
        stream.device().compute_capability().map(|cc| crate::arch::ArchProfile::for_cc(cc, sm).arch).unwrap_or(crate::arch::Arch::Sm8x)
    });
    crate::arch::ArchProfile::gdn_conv_rows_for(arch, sm, tokens, dim, width)
}

/// `gdn_recurrent` 的 V 维切分数：默认由 `ArchProfile::gdn_recurrent_vsplit`（按 SM 数把 block 数补到 ≥ SM 数，
/// 只取 V 的约数、上限 4）；`GDN_VSPLIT=<n>` 环境变量覆盖（对拍 / 回退用，须整除 V）。SM 数按首个 stream 的设备缓存一次。
fn gdn_recurrent_vsplit(stream: &Stream, base_blocks: usize, v_dim: usize) -> usize {
    use std::sync::OnceLock;
    static OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
    static SM: OnceLock<usize> = OnceLock::new();
    if let Some(v) = OVERRIDE.get_or_init(|| std::env::var("GDN_VSPLIT").ok().and_then(|s| s.parse().ok())) {
        if *v >= 1 && v_dim % *v == 0 {
            return *v;
        }
    }
    let sm = *SM.get_or_init(|| stream.device().sm_count().unwrap_or(1));
    crate::arch::ArchProfile::gdn_recurrent_vsplit(sm, base_blocks, v_dim)
}

/// 输出门 + RMSNorm：`y = rmsnorm(x, w, eps) · act(z)`，一个 `(token, v 头)` 一行。
pub fn gdn_gated_rmsnorm(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    z: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    rows: usize,
    head_dim: usize,
    eps: f32,
    act: GdnAct,
) -> Result<()> {
    gdn_gated_rmsnorm_strided(stream, x, w, &z.slice(..), out, rows, head_dim, eps, act, 1, head_dim)
}

/// `z` 带行距版：`rows = tokens·heads`，`z` 是 `[tokens, z_stride]` 缓冲的列段视图（融合 in_proj 输出的 z 段），
/// 第 (t, hv) 行的 z 在 `z[t·z_stride + hv·head_dim ..]`。紧凑时 `heads = 1, z_stride = head_dim`（等价于老接口）。
#[allow(clippy::too_many_arguments)]
pub fn gdn_gated_rmsnorm_strided(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    z: &CudaView<'_, bf16>,
    out: &mut CudaSlice<bf16>,
    rows: usize,
    head_dim: usize,
    eps: f32,
    act: GdnAct,
    heads: usize,
    z_stride: usize,
) -> Result<()> {
    ensure!(w.len() >= head_dim, "gdn_gated_rmsnorm: w");
    ensure!(heads >= 1 && rows % heads == 0 && z_stride >= heads * head_dim, "gdn_gated_rmsnorm: heads/z_stride");
    let tokens = rows / heads;
    let need_z = if tokens == 0 { 0 } else { (tokens - 1) * z_stride + heads * head_dim };
    ensure!(
        x.len() >= rows * head_dim && z.len() >= need_z && out.len() >= rows * head_dim,
        "gdn_gated_rmsnorm: x/z/out"
    );
    // SAFETY: 容量已校验。
    check(
        unsafe {
            ffi::gdn_gated_rmsnorm(
                stream.ptr(x) as *const c_void,
                stream.ptr(w) as *const c_void,
                stream.ptr(z) as *const c_void,
                stream.ptr_mut(out) as *mut c_void,
                rows as i64,
                head_dim as i64,
                eps,
                act.code(),
                heads as i64,
                z_stride as i64,
                stream.raw() as ffi::cudaStream_t,
            )
        },
        "gdn_gated_rmsnorm",
    )
}

// ------------------------------------------------------------------ GDN chunk 形式 prefill（gdn_chunk.cu）
// 切块表与 KDA 完全同构（只看 cu_seqlens / bt），直接复用 `KdaChunkPlan`。

/// GDN chunk prefill 的中间缓冲。尺寸随 `[T, H, K]` / `[T, HV, V]` 与块数变化。
pub struct GdnChunkScratch {
    pub qn: CudaSlice<bf16>,
    pub kn: CudaSlice<bf16>,
    pub gc: CudaSlice<f32>,
    /// r10 第 1 轮：wy 的中间量 aq / u / w / kg 落 bf16（fla 同口径；ho 装进 smem 本就转 bf16），写 / 读量减半。
    pub aq: CudaSlice<bf16>,
    pub u: CudaSlice<bf16>,
    pub w: CudaSlice<bf16>,
    pub kg: CudaSlice<bf16>,
    /// r9 两段版 ho 的中间量：每 chunk 入口状态 bf16 `[num_chunks, HV, V, K]`、v' bf16 `[T, HV, V]`。
    pub h16: CudaSlice<bf16>,
    pub v16: CudaSlice<bf16>,
    pub tokens: usize,
    pub num_chunks: usize,
}

impl GdnChunkScratch {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        stream: &Stream,
        tokens: usize,
        heads_k: usize,
        heads_v: usize,
        k_dim: usize,
        v_dim: usize,
        num_chunks: usize,
        bt: usize,
    ) -> Result<Self> {
        Ok(Self {
            qn: stream.alloc_uninit::<bf16>(tokens * heads_k * k_dim)?,
            kn: stream.alloc_uninit::<bf16>(tokens * heads_k * k_dim)?,
            gc: stream.alloc_uninit::<f32>(tokens * heads_v)?,
            aq: stream.alloc_uninit::<bf16>(num_chunks * heads_v * bt * bt)?,
            u: stream.alloc_uninit::<bf16>(tokens * heads_v * v_dim)?,
            w: stream.alloc_uninit::<bf16>(tokens * heads_v * k_dim)?,
            kg: stream.alloc_uninit::<bf16>(tokens * heads_v * k_dim)?,
            h16: stream.alloc_uninit::<bf16>(num_chunks * heads_v * v_dim * k_dim)?,
            v16: stream.alloc_uninit::<bf16>(tokens * heads_v * v_dim)?,
            tokens,
            num_chunks,
        })
    }

    /// 已分配容量是否够这一步（tokens / num_chunks 都不超）。
    pub fn fits(&self, tokens: usize, num_chunks: usize) -> bool {
        self.tokens >= tokens && self.num_chunks >= num_chunks
    }
}

/// 三段的第 1 段：q/k 的 l2norm（q 乘 `scale`）→ qn/kn；块内 inclusive cumsum(g)·1/ln2 → gc。
/// `mixed_qkv`: `[T, 2·H·K + HV·V]` bf16（与 `gdn_recurrent` 同布局）；`g`: `[T, HV]` f32（`gdn_gate` 的输出）。
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_prep(
    stream: &Stream,
    mixed_qkv: &CudaSlice<bf16>,
    g: &CudaSlice<f32>,
    cu_seqlens: &CudaSlice<i32>,
    plan: &KdaChunkPlan,
    sc: &mut GdnChunkScratch,
    tokens: usize,
    heads_k: usize,
    heads_v: usize,
    k_dim: usize,
    v_dim: usize,
    scale: f32,
    do_cumsum: bool,
) -> Result<()> {
    ensure!(
        heads_k > 0 && heads_v % heads_k == 0,
        "gdn_chunk_prep: heads_v {heads_v} 不能被 heads_k {heads_k} 整除"
    );
    let row = 2 * heads_k * k_dim + heads_v * v_dim;
    ensure!(mixed_qkv.len() >= tokens * row, "gdn_chunk_prep: mixed_qkv");
    ensure!(g.len() >= tokens * heads_v, "gdn_chunk_prep: g");
    ensure!(sc.fits(tokens, plan.num_chunks), "gdn_chunk_prep: scratch 容量");
    ensure!(cu_seqlens.len() >= plan.num_seqs + 1, "gdn_chunk_prep: cu_seqlens");
    // SAFETY: 容量已校验；plan 由同一 cu_seqlens 生成。
    check(
        unsafe {
            ffi::gdn_chunk_prep(
                stream.ptr(mixed_qkv) as *const c_void,
                stream.ptr(g) as *const f32,
                stream.ptr_mut(&mut sc.qn) as *mut c_void,
                stream.ptr_mut(&mut sc.kn) as *mut c_void,
                stream.ptr_mut(&mut sc.gc) as *mut f32,
                stream.ptr(cu_seqlens) as *const i32,
                stream.ptr(&plan.chunk_tab) as *const i32,
                tokens as i64,
                plan.num_chunks as i64,
                heads_k as i64,
                heads_v as i64,
                k_dim as i64,
                v_dim as i64,
                plan.bt as i64,
                scale,
                if do_cumsum { 1 } else { 0 },
                stream.raw() as _,
            )
        },
        "gdn_chunk_prep",
    )
}

/// 第 2 段：A/Aq → M = (I+A)⁻¹ → u / w / kg。`beta`: `[T, HV]` f32（`gdn_gate` 的输出）。
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_wy(
    stream: &Stream,
    mixed_qkv: &CudaSlice<bf16>,
    beta: &CudaSlice<f32>,
    cu_seqlens: &CudaSlice<i32>,
    plan: &KdaChunkPlan,
    sc: &mut GdnChunkScratch,
    tokens: usize,
    heads_k: usize,
    heads_v: usize,
    k_dim: usize,
    v_dim: usize,
) -> Result<()> {
    let how = gdn_chunk_wy_default(stream, k_dim, v_dim);
    gdn_chunk_wy_with(
        stream, mixed_qkv, beta, cu_seqlens, plan, sc, tokens, heads_k, heads_v, k_dim, v_dim, how,
    )
}

/// `gdn_chunk_wy` 指定实现（见 `GdnChunkWy`；`bool` 可直接传：true = Mma，false = Legacy）。对拍 / 计时用。
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_wy_with(
    stream: &Stream,
    mixed_qkv: &CudaSlice<bf16>,
    beta: &CudaSlice<f32>,
    cu_seqlens: &CudaSlice<i32>,
    plan: &KdaChunkPlan,
    sc: &mut GdnChunkScratch,
    tokens: usize,
    heads_k: usize,
    heads_v: usize,
    k_dim: usize,
    v_dim: usize,
    how: impl Into<GdnChunkWy>,
) -> Result<()> {
    let how = how.into();
    ensure!(sc.fits(tokens, plan.num_chunks), "gdn_chunk_wy: scratch 容量");
    ensure!(beta.len() >= tokens * heads_v, "gdn_chunk_wy: beta");
    // SAFETY: 容量已校验。
    check(
        unsafe {
            ffi::gdn_chunk_wy(
                stream.ptr(&sc.qn) as *const c_void,
                stream.ptr(&sc.kn) as *const c_void,
                stream.ptr(mixed_qkv) as *const c_void,
                stream.ptr(&sc.gc) as *const f32,
                stream.ptr(beta) as *const f32,
                stream.ptr_mut(&mut sc.aq) as *mut c_void,
                stream.ptr_mut(&mut sc.u) as *mut c_void,
                stream.ptr_mut(&mut sc.w) as *mut c_void,
                stream.ptr_mut(&mut sc.kg) as *mut c_void,
                stream.ptr(cu_seqlens) as *const i32,
                stream.ptr(&plan.chunk_tab) as *const i32,
                plan.num_chunks as i64,
                heads_k as i64,
                heads_v as i64,
                k_dim as i64,
                v_dim as i64,
                plan.bt as i64,
                how.code(),
                stream.raw() as _,
            )
        },
        "gdn_chunk_wy",
    )
}

/// 第 3 段：跨 chunk 扫 h（fp32，`state` 原地）并输出 `out` `[T, HV, V]` bf16。
#[allow(clippy::too_many_arguments)]
/// ho 段的 tcgen05（UMMA）版（r11d）：BV = 全 V、h 主本在 state、A 补零到 M=128。
/// 只在 K=V=128 的形状可用；返回 `cudaErrorInvalidValue` 时调用方应回退 `gdn_chunk_ho`。
/// 参数口径与 [`gdn_chunk_ho`] 一致（`qn/gc/aq/u/w/kg` 都是 `GdnChunkScratch` 里的字段）。
#[allow(clippy::too_many_arguments)]
pub fn gdn_ho_umma(
    stream: &Stream,
    sc: &GdnChunkScratch,
    out: &mut CudaSlice<bf16>,
    state: &mut CudaSlice<f32>,
    cu_seqlens: &CudaSlice<i32>,
    cu_nt: &CudaSlice<i32>,
    state_indices: Option<&CudaSlice<i32>>,
    state_stride_tok: i64,
    num_seqs: usize,
    heads_k: usize,
    heads_v: usize,
    k_dim: usize,
    v_dim: usize,
) -> Result<()> {
    // SAFETY: 容量由调用方校验（`sc` 与本步 tokens / chunks 一致）；shape 由 kernel 侧再校验一次。
    check(unsafe {
        ffi::gdn_ho_umma(
            stream.ptr(&sc.qn) as *const c_void,
            stream.ptr(&sc.gc) as *const f32,
            stream.ptr(&sc.aq) as *const c_void,
            stream.ptr(&sc.u) as *const c_void,
            stream.ptr(&sc.w) as *const c_void,
            stream.ptr(&sc.kg) as *const c_void,
            stream.ptr(out) as *mut c_void,
            stream.ptr_mut(state) as *mut c_void,
            stream.ptr(cu_seqlens) as *const i32,
            stream.ptr(cu_nt) as *const i32,
            state_indices.map_or(std::ptr::null(), |x| stream.ptr(x) as *const i32),
            state_stride_tok,
            num_seqs as i64,
            heads_k as i64,
            heads_v as i64,
            k_dim as i64,
            v_dim as i64,
            stream.raw() as _,
        )
    }, "gdn_ho_umma")
}

pub fn gdn_chunk_ho(
    stream: &Stream,
    cu_seqlens: &CudaSlice<i32>,
    plan: &KdaChunkPlan,
    sc: &GdnChunkScratch,
    out: &mut CudaSlice<bf16>,
    state: &mut CudaSlice<f32>,
    state_indices: Option<&CudaSlice<i32>>,
    tokens: usize,
    heads_k: usize,
    heads_v: usize,
    k_dim: usize,
    v_dim: usize,
) -> Result<()> {
    gdn_chunk_ho_snap(stream, cu_seqlens, plan, sc, out, state, state_indices, tokens, heads_k, heads_v, k_dim, v_dim, None)
}

/// 递归状态快照表（design §15.7 r8b）：序列 i 的条目在 `pos / slot[off[i]..off[i+1])`，`pos` = 相对该序列段首的
/// token 偏移（必须落在 chunk 边界：是 BT 的倍数或 == 序列长度），kernel 在该 chunk 末把状态多写一份到 `state` 的 `slot` 槽。
/// `state` 与活槽状态同布局（每槽 stride 相同）。
pub struct GdnSnapshot<'a> {
    pub off: &'a CudaSlice<i32>,
    pub pos: &'a CudaSlice<i32>,
    pub slot: &'a CudaSlice<i32>,
    pub state: &'a mut CudaSlice<f32>,
}

/// `gdn_chunk_ho` + 可选的边界快照（r8b）。实现选择同 `gdn_chunk_ho`（ArchProfile / 环境变量）。
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_ho_snap(
    stream: &Stream,
    cu_seqlens: &CudaSlice<i32>,
    plan: &KdaChunkPlan,
    sc: &GdnChunkScratch,
    out: &mut CudaSlice<bf16>,
    state: &mut CudaSlice<f32>,
    state_indices: Option<&CudaSlice<i32>>,
    tokens: usize,
    heads_k: usize,
    heads_v: usize,
    k_dim: usize,
    v_dim: usize,
    snap: Option<GdnSnapshot<'_>>,
) -> Result<()> {
    let bv = gdn_chunk_ho_bv(stream, plan.num_seqs * heads_v, k_dim, v_dim);
    let pf = gdn_chunk_ho_prefetch(stream, bv);
    let split = gdn_chunk_ho_split(stream, bv, plan.num_seqs, plan.num_chunks, heads_v, k_dim, v_dim);
    let how = GdnChunkHo { bv, prefetch: pf, split };
    let per_slot = heads_v * v_dim * k_dim;
    ensure!(state.len() >= per_slot, "gdn_chunk_ho: state");
    ensure!(out.len() >= tokens * heads_v * v_dim, "gdn_chunk_ho: out 容量");
    ensure!(sc.fits(tokens, plan.num_chunks), "gdn_chunk_ho: scratch 容量");
    let psi = state_indices.map_or(std::ptr::null(), |s| stream.ptr(s) as *const i32);
    let (so, sp, ss, sst) = match snap {
        Some(sn) => {
            ensure!(sn.off.len() >= plan.num_seqs + 1, "gdn_chunk_ho: snap_off 长度");
            ensure!(sn.state.len() >= per_slot, "gdn_chunk_ho: snap_state");
            (
                stream.ptr(sn.off) as *const i32,
                stream.ptr(sn.pos) as *const i32,
                stream.ptr(sn.slot) as *const i32,
                stream.ptr_mut(sn.state) as *mut c_void,
            )
        }
        None => (std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null_mut()),
    };
    // SAFETY: 容量已校验；slot / 快照槽编号由调度器保证在各自槽数内。
    check(
        unsafe {
            ffi::gdn_chunk_ho_v3(
                stream.ptr(&sc.qn) as *const c_void,
                stream.ptr(&sc.gc) as *const f32,
                stream.ptr(&sc.aq) as *const c_void,
                stream.ptr(&sc.u) as *const c_void,
                stream.ptr(&sc.w) as *const c_void,
                stream.ptr(&sc.kg) as *const c_void,
                stream.ptr(&sc.h16) as *mut c_void,
                stream.ptr(&sc.v16) as *mut c_void,
                stream.ptr_mut(out) as *mut c_void,
                stream.ptr_mut(state) as *mut c_void,
                stream.ptr(cu_seqlens) as *const i32,
                stream.ptr(&plan.cu_nt) as *const i32,
                stream.ptr(&plan.chunk_off) as *const i32,
                stream.ptr(&plan.chunk_tab) as *const i32,
                psi,
                per_slot as i64,
                plan.num_seqs as i64,
                plan.num_chunks as i64,
                heads_k as i64,
                heads_v as i64,
                k_dim as i64,
                v_dim as i64,
                plan.bt as i64,
                how.bv as i64,
                (how.bv != 0 && how.prefetch) as i64,
                if how.bv != 0 && how.split { crate::arch::ArchProfile::gdn_chunk_opass_vt(v_dim) as i64 } else { 0 },
                so,
                sp,
                ss,
                sst,
                stream.raw() as _,
            )
        },
        "gdn_chunk_ho_v3",
    )
}

/// 短卷积状态的边界快照（r8b）：对每个快照条目把"算到该位置时"的 conv 状态（最后 W-1 个原始输入行，不足从入口状态补）
/// 转置写到 `snap_state` 的槽。必须在 `gdn_conv1d_prefill_strided` 覆盖活槽之前调用。`x` 行距 `x_stride`。
#[allow(clippy::too_many_arguments)]
pub fn gdn_conv_snapshot_strided(
    stream: &Stream,
    x: &CudaView<'_, bf16>,
    conv_state: &CudaSlice<bf16>,
    snap_state: &mut CudaSlice<bf16>,
    cu_seqlens: &CudaSlice<i32>,
    state_indices: Option<&CudaSlice<i32>>,
    has_initial: Option<&CudaSlice<i32>>,
    snap_off: &CudaSlice<i32>,
    snap_pos: &CudaSlice<i32>,
    snap_slot: &CudaSlice<i32>,
    num_seqs: usize,
    dim: usize,
    x_stride: usize,
    state_len: usize,
) -> Result<()> {
    ensure!(x_stride >= dim, "gdn_conv_snapshot: x_stride {x_stride} < dim {dim}");
    ensure!(cu_seqlens.len() >= num_seqs + 1 && snap_off.len() >= num_seqs + 1, "gdn_conv_snapshot: cu_seqlens / snap_off");
    ensure!(conv_state.len() >= dim * state_len && snap_state.len() >= dim * state_len, "gdn_conv_snapshot: state");
    let psi = state_indices.map_or(std::ptr::null(), |s| stream.ptr(s) as *const i32);
    let phi = has_initial.map_or(std::ptr::null(), |s| stream.ptr(s) as *const i32);
    // SAFETY: 容量已校验；位置 / 槽号由调度侧保证。
    check(
        unsafe {
            ffi::gdn_conv_snapshot(
                stream.ptr(x) as *const c_void,
                stream.ptr(conv_state) as *const c_void,
                stream.ptr_mut(snap_state) as *mut c_void,
                stream.ptr(cu_seqlens) as *const i32,
                psi,
                phi,
                stream.ptr(snap_off) as *const i32,
                stream.ptr(snap_pos) as *const i32,
                stream.ptr(snap_slot) as *const i32,
                num_seqs as i64,
                dim as i64,
                state_len as i64,
                x_stride as i64,
                stream.raw() as ffi::cudaStream_t,
            )
        },
        "gdn_conv_snapshot",
    )
}

/// chunk prefill `ho` 是否走两段版（h-pass + o-pass，r9 第 2 轮）：默认 `ArchProfile::gdn_chunk_ho_split`；
/// `GDN_CHUNK_HO_SPLIT=0|1` 覆盖。bv = 0（老 fp32 版）或 (K, V) 无 o-pass 特化时恒为 false。
fn gdn_chunk_ho_split(stream: &Stream, bv: usize, num_seqs: usize, num_chunks: usize, heads_v: usize, k_dim: usize, v_dim: usize) -> bool {
    use std::sync::OnceLock;
    static OVER: OnceLock<Option<bool>> = OnceLock::new();
    static SM: OnceLock<usize> = OnceLock::new();
    if bv == 0 || !matches!(k_dim, 64 | 128 | 256) || crate::arch::ArchProfile::gdn_chunk_opass_vt(v_dim) == 0 {
        return false;
    }
    let over = *OVER.get_or_init(|| {
        let o = std::env::var("GDN_CHUNK_HO_SPLIT").ok().map(|s| s != "0");
        if o.is_some() {
            tracing::info!(?o, "gdn chunk ho 两段版由环境变量覆盖");
        }
        o
    });
    if let Some(o) = over {
        return o;
    }
    let sm = *SM.get_or_init(|| stream.device().sm_count().unwrap_or(1));
    crate::arch::ArchProfile::gdn_chunk_ho_split(sm, num_seqs, num_chunks, heads_v)
}

/// `gdn_chunk_ho` 的实现选择（对拍 / 计时用；正常路径由 `ArchProfile` 给）。
/// `bv ∈ {16, 32}` 走 tensor core 版（第 5A 轮），`0` 走老 fp32 版；`prefetch` 寄存器预取下一 chunk（r9 第 1 轮，仅 bv ≠ 0 有效）。
/// `split`：两段版 h-pass + o-pass（r9 第 2 轮；中间量 `GdnChunkScratch::{h16, v16}`），仅 bv ≠ 0 且 (K, V) 有特化时有效。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GdnChunkHo {
    pub bv: usize,
    pub prefetch: bool,
    pub split: bool,
}

impl From<usize> for GdnChunkHo {
    fn from(bv: usize) -> Self {
        Self { bv, prefetch: false, split: false }
    }
}

/// `gdn_chunk_ho` 指定实现（见 `GdnChunkHo`；`usize` 可直接传作 bv、不预取）。
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_ho_with(
    stream: &Stream,
    cu_seqlens: &CudaSlice<i32>,
    plan: &KdaChunkPlan,
    sc: &GdnChunkScratch,
    out: &mut CudaSlice<bf16>,
    state: &mut CudaSlice<f32>,
    state_indices: Option<&CudaSlice<i32>>,
    tokens: usize,
    heads_k: usize,
    heads_v: usize,
    k_dim: usize,
    v_dim: usize,
    how: impl Into<GdnChunkHo>,
) -> Result<()> {
    let how = how.into();
    let per_slot = heads_v * v_dim * k_dim;
    ensure!(state.len() >= per_slot, "gdn_chunk_ho: state");
    ensure!(out.len() >= tokens * heads_v * v_dim, "gdn_chunk_ho: out 容量");
    ensure!(sc.fits(tokens, plan.num_chunks), "gdn_chunk_ho: scratch 容量");
    let psi = state_indices.map_or(std::ptr::null(), |s| stream.ptr(s) as *const i32);
    // SAFETY: 容量已校验；slot 编号由调度器保证 < state 槽数。
    check(
        unsafe {
            ffi::gdn_chunk_ho_v2(
                stream.ptr(&sc.qn) as *const c_void,
                stream.ptr(&sc.gc) as *const f32,
                stream.ptr(&sc.aq) as *const c_void,
                stream.ptr(&sc.u) as *const c_void,
                stream.ptr(&sc.w) as *const c_void,
                stream.ptr(&sc.kg) as *const c_void,
                // h16 / v16 是 scratch 的中间缓冲（只有两段版写）；`sc` 按共享引用传是为了保持 `gdn_chunk_ho` 的签名，
                // 调用方（models/qwen35）在这一步之外不读它们。
                stream.ptr(&sc.h16) as *mut c_void,
                stream.ptr(&sc.v16) as *mut c_void,
                stream.ptr_mut(out) as *mut c_void,
                stream.ptr_mut(state) as *mut c_void,
                stream.ptr(cu_seqlens) as *const i32,
                stream.ptr(&plan.cu_nt) as *const i32,
                stream.ptr(&plan.chunk_off) as *const i32,
                stream.ptr(&plan.chunk_tab) as *const i32,
                psi,
                per_slot as i64,
                plan.num_seqs as i64,
                plan.num_chunks as i64,
                heads_k as i64,
                heads_v as i64,
                k_dim as i64,
                v_dim as i64,
                plan.bt as i64,
                how.bv as i64,
                (how.bv != 0 && how.prefetch) as i64,
                if how.bv != 0 && how.split { crate::arch::ArchProfile::gdn_chunk_opass_vt(v_dim) as i64 } else { 0 },
                stream.raw() as _,
            )
        },
        "gdn_chunk_ho",
    )
}

/// MXFP4 解包（compressed-tensors `mxfp4-pack-quantized` 的参考回退路径）。
/// `packed`: `[rows, cols/2]` u8（低半字节 = 偶数下标）；`scale`: `[rows, cols/group]` u8，e8m0 = 2^(u8-127)；
/// `out`: `[rows, cols]` bf16。原生 MXFP4 GEMM 见 `docs/loops/k3-moe.md` 的 TODO。
pub fn mxfp4_dequant_bf16(
    stream: &Stream,
    packed: &CudaSlice<u8>,
    scale: &CudaSlice<u8>,
    out: &mut CudaSlice<bf16>,
    rows: usize,
    cols: usize,
    group: usize,
) -> Result<()> {
    ensure!(
        cols % 2 == 0 && cols % group == 0,
        "mxfp4_dequant: cols {cols} 必须能被 2 和 group {group} 整除"
    );
    ensure!(packed.len() >= rows * cols / 2, "mxfp4_dequant: packed");
    ensure!(scale.len() >= rows * cols / group, "mxfp4_dequant: scale");
    ensure!(out.len() >= rows * cols, "mxfp4_dequant: out");
    // SAFETY: 容量已校验。
    check(
        unsafe {
            ffi::mxfp4_dequant_bf16(
                stream.ptr(packed) as *const c_void,
                stream.ptr(scale) as *const c_void,
                stream.ptr_mut(out) as *mut c_void,
                rows as i64,
                cols as i64,
                group as i64,
                stream.raw() as _,
            )
        },
        "mxfp4_dequant_bf16",
    )
}

// --------------------------------------------------------------------------- SituAndMul（Kimi-K3）

/// `out[:tokens] = situ(x[:, :inter]) · up'(x[:, inter:])`：`beta·tanh(g/beta)·sigmoid(g)`，`linear_beta > 0` 时
/// `up' = linear_beta·tanh(u/linear_beta)`，否则 `up' = u`（HF `SituAndMul`）。`beta` / `linear_beta` 是模型参数。
pub fn situ_and_mul_bf16_n(
    stream: &Stream,
    x: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    tokens: usize,
    inter: usize,
    beta: f32,
    linear_beta: Option<f32>,
) -> Result<()> {
    ensure!(inter > 0 && x.len() >= tokens * 2 * inter && out.len() >= tokens * inter, "situ_and_mul: shape");
    ensure!(beta > 0.0, "situ_and_mul: beta 必须 > 0");
    let px = stream.ptr(x) as *const c_void;
    let po = stream.ptr_mut(out) as *mut c_void;
    // SAFETY: 容量已校验。
    check(
        unsafe {
            ffi::situ_and_mul_bf16(
                px,
                po,
                tokens as i64,
                inter as i64,
                beta,
                linear_beta.unwrap_or(0.0),
                stream.raw() as _,
            )
        },
        "situ_and_mul_bf16",
    )
}
