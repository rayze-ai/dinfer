//! `cuda/dinfer_kernels.h` 的 Rust 声明。手写而非 bindgen，保持签名可审计。

#![allow(non_camel_case_types)]

use std::ffi::c_void;

pub type cudaStream_t = *mut c_void;

#[cfg(not(dinfer_no_nvcc))]
extern "C" {
    pub fn dinfer_add_f32(a: *const f32, b: *const f32, out: *mut f32, n: i64, stream: cudaStream_t) -> i32;
}

#[cfg(dinfer_no_nvcc)]
pub unsafe fn dinfer_add_f32(
    _a: *const f32,
    _b: *const f32,
    _out: *mut f32,
    _n: i64,
    _s: cudaStream_t,
) -> i32 {
    panic!("dinfer-kernels built with `no-nvcc`: CUDA kernels unavailable")
}
