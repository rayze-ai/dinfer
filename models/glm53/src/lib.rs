//! dinfer-glm53：GLM-5.3（`zai-org/GLM-5.3`，`GlmMoeDsaForCausalLM`）模型 crate。
//! P3/P4：MLA + dense/MoE FFN 主干（`model::Glm53Model`）；P5 DSA 索引器 + 稀疏 MLA；
//! P6：tokenizer（`tokenizer`）、chat 模板（`chat`）、流式输出解析（`parser`）、MTP（`mtp`，后续）。

#![forbid(unsafe_code)]

pub mod chat;
pub mod config;
pub mod executor;
pub mod model;
pub mod mtp;
pub mod parser;
pub mod service;
pub mod tokenizer;

pub const MODEL_NAME: &str = "glm-5.3";
