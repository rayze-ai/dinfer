//! 投机解码的提案器（draft 来源）。目前只有不需要额外权重的 n-gram（vLLM `ngram_proposer.py`）；
//! 验证 / 接受走 runner ↔ executor 既有的 `spec_tokens` / `drafts` 通道（见 `runner.rs` 头注）。

pub mod ngram;

pub use ngram::NgramProposer;
