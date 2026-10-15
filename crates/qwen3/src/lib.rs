//! dinfer-qwen3：Qwen3（`Qwen/Qwen3-0.6B`，`Qwen3ForCausalLM`）模型 crate。
//! Q1：config、GQA 主干（`model`）、executor、tokenizer、chat 模板、解析器、Service。主流程验证模型（见 design.md §5b）。

#![forbid(unsafe_code)]

pub mod config;
pub mod executor;
pub mod model;
pub mod parser;
pub mod service;

pub const MODEL_NAME: &str = "qwen3-0.6b";
