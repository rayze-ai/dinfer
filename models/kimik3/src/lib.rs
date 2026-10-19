//! dinfer-kimik3：Kimi-K3（`moonshotai/Kimi-K3`）模型 crate。里程碑 C-1：配置解析 + 权重名表 + KDA 层。

#![forbid(unsafe_code)]

pub mod config;
pub mod kda;
pub mod weights;

pub const MODEL_NAME: &str = "kimi-k3";
