//! dinfer-kernels：CUDA 算子 FFI 与设备运行时封装。
//!
//! 本 crate 是整个工作区唯一允许出现 `unsafe` 的地方；上层 `dinfer-layers` 之上不再有 `unsafe`。
//! 每个 kernel 一个 `extern "C"` 入口（见 `cuda/dinfer_kernels.h`），这里做类型安全的包装。

pub mod device;
pub mod shared;
pub use shared::{RegionKind, SharedRegion};
pub mod ffi;
pub mod gemm;
pub mod ops;
pub mod opset;

pub use cudarc;
pub use device::{Device, Graph, Stream, Event, TimingEvent, PinnedUploader, PinnedSlot, HostArena, PinnedPool, PinnedBuf};
pub use gemm::Gemm;
pub use opset::{Act, Attention, DefaultOps, Norm, OpSet, QkvPost};
pub use half::bf16;
pub use ops::*;
