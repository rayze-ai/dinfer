//! Kaon-V3（Gemma4 MoE，NVFP4 专家 + FP8 其余）单卡推理。
//! 设计见 `docs/loops/kaon-v3.md`。
pub mod config;
pub mod executor;
pub mod layer;
pub mod model;
pub mod service;
pub mod weights;

pub const MODEL_NAME: &str = "kaon3";
