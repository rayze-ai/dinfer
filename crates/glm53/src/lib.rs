//! dinfer-glm53：GLM-5.3（`zai-org/GLM-5.3`，`GlmMoeDsaForCausalLM`）模型 crate。
//! P3：稠密 MLA 子集（`model::Glm53Dense`）；MoE / DSA / MTP 后续阶段加。

#![forbid(unsafe_code)]

pub mod config;
pub mod model;

pub const MODEL_NAME: &str = "glm-5.3";
