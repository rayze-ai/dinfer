//! 类型安全的 kernel 包装。所有 `unsafe` 集中在这里；上层只见到 `CudaSlice` 与维度。
//!
//! bf16 数据在 Rust 侧用 `half::bf16`（与 `__nv_bfloat16` 位布局一致）。

use crate::device::Stream;
use crate::ffi;
use anyhow::{bail, ensure, Result};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use std::ffi::c_void;

fn check(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("{what} failed: cudaError {rc}");
    }
    Ok(())
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
        unsafe { ffi::rmsnorm_bf16(px, pw, po, tokens, hidden as i64, eps, stream.raw() as _) },
        "rmsnorm_bf16",
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
        unsafe { ffi::fused_add_rmsnorm_bf16(px, pr, pw, po, tokens, hidden as i64, eps, stream.raw() as _) },
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
    let tokens = positions.len();
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
    ensure!(
        x.len() >= tokens * hidden && out.len() >= tokens * hidden && w.len() == hidden,
        "rmsnorm_n: shape"
    );
    let (px, pw) = (stream.ptr(x) as *const c_void, stream.ptr(w) as *const c_void);
    let po = stream.ptr_mut(out) as *mut c_void;
    // SAFETY: 缓冲容量已校验。
    check(
        unsafe { ffi::rmsnorm_bf16(px, pw, po, tokens as i64, hidden as i64, eps, stream.raw() as _) },
        "rmsnorm_bf16",
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
    ensure!(rows.len() >= num_rows && out.len() >= num_rows * hidden && w.len() == hidden, "gather_rmsnorm: shape");
    let (px, pw) = (stream.ptr(x) as *const c_void, stream.ptr(w) as *const c_void);
    let pr = stream.ptr(rows) as *const i32;
    let po = stream.ptr_mut(out) as *mut c_void;
    // SAFETY: 容量已校验；rows 下标范围（< x 行数）由 runner 保证。
    check(
        unsafe { ffi::gather_rmsnorm_bf16(px, pr, pw, po, num_rows as i64, hidden as i64, eps, stream.raw() as _) },
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
            ffi::fused_add_rmsnorm_bf16(px, pr, pw, po, tokens as i64, hidden as i64, eps, stream.raw() as _)
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
        layer_ptrs.len() >= num_layers && blocks.len() >= n_blocks && staging.len() >= n_blocks * num_layers * per_layer_bytes,
        "kv_pack: shape"
    );
    ensure!(block_size > 0 && kv_heads > 0 && per_layer_bytes % (block_size * kv_heads) == 0, "kv_pack: per_layer_bytes vs bs×hk");
    let lp = stream.ptr(layer_ptrs) as *const *const c_void;
    let pb = stream.ptr(blocks) as *const i32;
    let ps = stream.ptr_mut(staging) as *mut c_void;
    // SAFETY: 容量已校验；layer_ptrs 内容是有效设备地址（由 executor 填）。
    check(unsafe { ffi::kv_pack(lp, pb, ps, n_blocks as i64, num_layers as i64, per_layer_bytes as i64, block_size as i64, kv_heads as i64, stream.raw() as _) }, "kv_pack")
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
        layer_ptrs.len() >= num_layers && blocks.len() >= n_blocks && staging.len() >= n_blocks * num_layers * per_layer_bytes,
        "kv_unpack: shape"
    );
    ensure!(block_size > 0 && kv_heads > 0 && per_layer_bytes % (block_size * kv_heads) == 0, "kv_unpack: per_layer_bytes vs bs×hk");
    let lp = stream.ptr(layer_ptrs) as *const *mut c_void;
    let pb = stream.ptr(blocks) as *const i32;
    let ps = stream.ptr(staging) as *const c_void;
    // SAFETY: 同上。
    check(unsafe { ffi::kv_unpack(lp, pb, ps, n_blocks as i64, num_layers as i64, per_layer_bytes as i64, block_size as i64, kv_heads as i64, stream.raw() as _) }, "kv_unpack")
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
pub fn gather_blocks_bf16_to_f32(stream: &Stream, x: &CudaSlice<bf16>, y: &mut CudaSlice<f32>, nb: usize, rows: usize, cols: usize) -> Result<()> {
    let n = nb * rows * cols;
    ensure!(x.len() >= n && y.len() >= n, "gather_blocks_bf16_to_f32: buffer ({} / {} < {n})", x.len(), y.len());
    let px = stream.ptr(x) as *const c_void;
    let py = stream.ptr_mut(y) as *mut f32;
    // SAFETY: 容量已校验。
    check(
        unsafe { ffi::gather_blocks_bf16_to_f32(px, py, nb as i64, rows as i64, cols as i64, stream.raw() as _) },
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
                n as i64,
                topk as i64,
                hidden as i64,
                stream.raw() as _,
            )
        },
        "moe_scatter_add_f32",
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
    // 每 token 行的元素跨度；0 = 紧凑（heads·D / kv_heads·D）。传 qkv GEMM 输出时为 (H+2Hkv)·D。
    q_stride: usize,
    kv_stride: usize,
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
            ffi::gqa_prefill_bf16(
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
                stream.raw() as _,
            )
        },
        "gqa_prefill_bf16",
    )
}

/// 从 paged cache 读的 attention：decode（q_len = 1）与带历史 chunked prefill 统一。q: [Tq, heads, D]，out 同形。
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
    // 混批：跳过 q_len==1 的请求（它们由 gqa_decode_bf16 处理）。
    skip_single: bool,
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
            ffi::gqa_paged_bf16(
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
                stream.raw() as _,
            )
        },
        "gqa_paged_bf16",
    )
}

/// 纯 decode（每请求 1 行）split-K attention（D=128）。q: [B, heads, D]，out 同形；`scratch` ≥ `gqa_decode_scratch_len`。
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
    ensure!(head_dim == 128, "gqa_decode: head_dim must be 128");
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

/// 混批（prefill 行 + decode 行同一步）attention：q_len==1 的请求走 split-K decode kernel（按 cu_seqlens_q 定位行），
/// 其余请求走 paged tensor-core kernel（跳过 q_len==1）。两个 kernel 同流串行，各自只写自己的行。仅 D=128。
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
    ensure!(head_dim == 128, "gqa_mixed: head_dim must be 128");
    gqa_decode_bf16(
        stream, q, cache, block_table, seq_lens, out, scratch, num_reqs, heads, kv_heads, head_dim, block_size,
        max_blocks, scale, q_stride, Some(cu_seqlens_q),
    )?;
    gqa_paged_bf16(
        stream, q, cache, block_table, seq_lens, cu_seqlens_q, out, num_reqs, max_q_len, heads, kv_heads, head_dim,
        block_size, max_blocks, scale, q_stride, true,
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
    ensure!(g.len() >= tokens * heads * head_dim && out.len() >= tokens * heads * head_dim, "kda_gate: g/out");
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
    ensure!(x.len() >= rows * head_dim && gate.len() >= rows * head_dim && out.len() >= rows * head_dim,
            "kda_gated_rmsnorm: x/gate/out");
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
    ensure!(cols % 2 == 0 && cols % group == 0, "mxfp4_dequant: cols {cols} 必须能被 2 和 group {group} 整除");
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
