//! dinfer-kimik3：Kimi-K3（`moonshotai/Kimi-K3`）模型 crate。里程碑 C-1：配置解析 + 权重名表 + KDA 层；
//! C-2：latent MoE（MXFP4 packed 常驻）+ 单层 `KimiDecoderLayer`（含 AttnRes）；C-3：MLA 层 + 多层 `KimiK3Model`。

#![forbid(unsafe_code)]

pub mod config;
pub mod kda;
pub mod latent_moe;
pub mod layer;
pub mod mla;
pub mod model;
pub mod weights;

pub const MODEL_NAME: &str = "kimi-k3";
