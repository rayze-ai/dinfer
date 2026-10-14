//! dinfer-core：模型无关的推理引擎核心。
//!
//! 本 crate 不依赖 CUDA 工具链；GPU 相关抽象只以 trait 形式出现，
//! 由 `dinfer-layers` / 模型 crate 提供实现。详见 `docs/design.md` §4.1。

pub mod config;
pub mod kv;
pub mod request;
pub mod sample;
pub mod sched;

pub use config::EngineConfig;
pub use request::{Request, RequestId, RequestStatus};
