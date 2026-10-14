//! dinfer-layers：两模型共用的层实现（基于 dinfer-kernels）。本 crate 之上不允许出现 `unsafe`。
//!
//! 层清单见 `docs/design.md` §4.3。P3 提供：Tensor、WeightSource、Linear、RmsNorm、Rope、Embedding、
//! Mlp（silu）、MlaAttention（含 W_UK/W_UV 吸收、paged KV、prefill/decode 分流）、Sampler（greedy）。

// 唯一例外：weights.rs 的只读 mmap（见 `weights::map_readonly`）。其余代码禁止 unsafe。
#![deny(unsafe_code)]

pub mod mla;
pub mod moe;
pub mod nn;
pub mod tensor;
pub mod weights;

pub use dinfer_kernels::{bf16, Device, Gemm, RopeStyle, Stream};
pub use mla::{MlaAttention, MlaConfig, PagedKv};
pub use moe::{Moe, MoeConfig};
pub use nn::{Embedding, Linear, Mlp, RmsNorm, Rope};
pub use tensor::Tensor;
pub use weights::WeightSource;
