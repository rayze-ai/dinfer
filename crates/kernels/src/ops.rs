//! 类型安全的 kernel 包装。所有 `unsafe` 集中在这里；上层只见到 `CudaSlice` 与维度。
//!
//! bf16 数据在 Rust 侧用 `half::bf16`（与 `__nv_bfloat16` 位布局一致）。

use crate::device::Stream;
use crate::ffi;
use anyhow::{bail, ensure, Result};
use cudarc::driver::CudaSlice;
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
pub fn topk_topp_mask_f32(
    stream: &Stream,
    logits: &mut CudaSlice<f32>,
    top_k: &CudaSlice<i32>,
    top_p: &CudaSlice<f32>,
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
pub fn sample_f32(
    stream: &Stream,
    logits: &CudaSlice<f32>,
    temperature: &CudaSlice<f32>,
    seeds: &CudaSlice<u64>,
    offsets: &CudaSlice<u64>,
    out: &mut CudaSlice<i32>,
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
