//! dinfer-glm53：GLM-5.3（`zai-org/GLM-5.3`，`GlmMoeDsaForCausalLM`）模型 crate。
//! P3/P4：MLA + dense/MoE FFN 主干（`model::Glm53Model`）；DSA / MTP 后续阶段加。

#![forbid(unsafe_code)]

pub mod config;
pub mod executor;
pub mod model;

pub const MODEL_NAME: &str = "glm-5.3";
