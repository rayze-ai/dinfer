//! `cuda/dinfer_kernels.h` 的 Rust 声明。手写而非 bindgen，保持签名可审计。
//! 每个入口返回 cudaError_t 数值（0 = success）。符号名不带项目前缀（项目可能改名）。

#![allow(non_camel_case_types)]

use std::ffi::c_void;

pub type cudaStream_t = *mut c_void;

#[cfg(not(dinfer_no_nvcc))]
extern "C" {
    pub fn add_f32(a: *const f32, b: *const f32, out: *mut f32, n: i64, stream: cudaStream_t) -> i32;

    pub fn rmsnorm_bf16(
        x: *const c_void,
        w: *const c_void,
        out: *mut c_void,
        tokens: i64,
        hidden: i64,
        eps: f32,
        stream: cudaStream_t,
    ) -> i32;
    pub fn fused_add_rmsnorm_bf16(
        x: *const c_void,
        res: *mut c_void,
        w: *const c_void,
        out: *mut c_void,
        tokens: i64,
        hidden: i64,
        eps: f32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn rope_bf16(
        q: *mut c_void,
        k: *mut c_void,
        positions: *const i32,
        cos_sin: *const f32,
        tokens: i64,
        q_heads: i64,
        k_heads: i64,
        head_dim: i64,
        rope_dim: i64,
        neox: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn silu_and_mul_bf16(
        x: *const c_void,
        out: *mut c_void,
        tokens: i64,
        inter: i64,
        stream: cudaStream_t,
    ) -> i32;

    pub fn embedding_bf16(
        ids: *const i32,
        table: *const c_void,
        out: *mut c_void,
        tokens: i64,
        hidden: i64,
        vocab: i64,
        stream: cudaStream_t,
    ) -> i32;

    pub fn argmax_f32(logits: *const f32, out: *mut i32, batch: i64, vocab: i64, stream: cudaStream_t)
        -> i32;
    pub fn topk_topp_mask_f32(
        logits: *mut f32,
        top_k: *const i32,
        top_p: *const f32,
        batch: i64,
        vocab: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn sample_f32(
        logits: *const f32,
        temperature: *const f32,
        seeds: *const u64,
        offsets: *const u64,
        out: *mut i32,
        batch: i64,
        vocab: i64,
        stream: cudaStream_t,
    ) -> i32;
}

#[cfg(dinfer_no_nvcc)]
mod stubs {
    use super::*;
    macro_rules! stub {
        ($name:ident ( $($arg:ident : $ty:ty),* )) => {
            #[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
            pub unsafe fn $name($(_: $ty),*) -> i32 {
                panic!(concat!("dinfer-kernels built with `no-nvcc`: ", stringify!($name), " unavailable"))
            }
        };
    }
    stub!(add_f32(a: *const f32, b: *const f32, out: *mut f32, n: i64, s: cudaStream_t));
    stub!(rmsnorm_bf16(x: *const c_void, w: *const c_void, o: *mut c_void, t: i64, h: i64, e: f32, s: cudaStream_t));
    stub!(fused_add_rmsnorm_bf16(x: *const c_void, r: *mut c_void, w: *const c_void, o: *mut c_void, t: i64, h: i64, e: f32, s: cudaStream_t));
    stub!(rope_bf16(q: *mut c_void, k: *mut c_void, p: *const i32, cs: *const f32, t: i64, qh: i64, kh: i64, hd: i64, rd: i64, n: i32, s: cudaStream_t));
    stub!(silu_and_mul_bf16(x: *const c_void, o: *mut c_void, t: i64, i: i64, s: cudaStream_t));
    stub!(embedding_bf16(ids: *const i32, tb: *const c_void, o: *mut c_void, t: i64, h: i64, v: i64, s: cudaStream_t));
    stub!(argmax_f32(l: *const f32, o: *mut i32, b: i64, v: i64, s: cudaStream_t));
    stub!(topk_topp_mask_f32(l: *mut f32, k: *const i32, p: *const f32, b: i64, v: i64, s: cudaStream_t));
    stub!(sample_f32(l: *const f32, t: *const f32, sd: *const u64, of: *const u64, o: *mut i32, b: i64, v: i64, s: cudaStream_t));
}
#[cfg(dinfer_no_nvcc)]
pub use stubs::*;
