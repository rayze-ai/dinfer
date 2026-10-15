//! dinfer-text：模型无关的文本层。
//!
//! - `HfTokenizer`：`tokenizers` 加载 `tokenizer.json`；eos 集合（generation_config / tokenizer_config）；增量 detokenize。
//! - `ChatTemplate`：minijinja 跑 HF `chat_template`（`chat_template.jinja` 或 `tokenizer_config.json` 里的字段），
//!   补齐 HF/Jinja2+Python 方言（见 `chat.rs` 头注释）。渲染文本必须与 HF `apply_chat_template(tokenize=False)` 逐字符一致。
//!
//! 模型 crate 只需提供：目录、模板变量（`reasoning_effort` / `enable_thinking` …）、输出解析器。

#![forbid(unsafe_code)]

pub mod chat;
pub mod tokenizer;

pub use chat::{ChatTemplate, RenderParams};
pub use tokenizer::{HfTokenizer, IncrementalDecoder};
