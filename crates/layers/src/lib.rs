//! dinfer-layers：两模型共用的层实现。P0 仅占位；P2 起填充。
//!
//! 层清单见 `docs/design.md` §4.3。本 crate 之上不允许出现 `unsafe`。

#![forbid(unsafe_code)]

pub use dinfer_kernels::{Device, Stream};
