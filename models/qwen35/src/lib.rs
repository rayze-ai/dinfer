//! dinfer-qwen35：Qwen35（Qwen3.5-27B，`/data/models/Qwen/Qwen3.8-27B`，`model_type qwen3_5`）模型 crate。
//! 只做 `text_config`（language_model_only）：64 层 = 48 × Gated DeltaNet（线性注意力）+ 16 × 全注意力（每 4 层一层），
//! 全注意力 24 q / 4 kv 头 × head_dim 256、partial RoPE 0.25、attention 后输出门；GDN 16 k 头 × 128 / 48 v 头 × 128、conv 4；
//! lm_head 不 tie；MTP 不做。设计与结构事实 `docs/design-qwen35.md`，接入方法论 `docs/model-onboarding.md`。
//! kernel 已对拍：GDN（`docs/loops/qwen35-gdn.md`）、attention D=256 / GROUP 6 / qkv_post_ext / mul_sigmoid（`docs/loops/qwen35-attn.md`）。
#![forbid(unsafe_code)]
pub mod config;
pub mod executor;
pub mod full_attn;
pub mod gdn;
pub mod layer;
pub mod model;
pub mod parser;
pub mod scratch;
pub mod service;
pub mod weights;
pub const MODEL_NAME: &str = "qwen35";
