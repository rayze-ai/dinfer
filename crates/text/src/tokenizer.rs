//! 通用 HF tokenizer：`tokenizers` 加载 `tokenizer.json`；eos 集合来自 `generation_config.json`；
//! 增量 detokenize 用 `tokenizers::DecodeStream`（处理 byte-level BPE 的 UTF-8 半字符）。
//!
//! 对齐口径：`encode(text, add_special_tokens=false)` / `decode` 逐 id / 逐字符等于 HF（各模型 crate 的 `chat_parity`）。
//! chat 模板渲染出的文本里特殊 token 以字面出现，HF 把它们当 added token 整体切分，`tokenizers` 同样 →
//! "渲染文本 → encode" 就是正确的 prompt id 序列。

use anyhow::{anyhow, Context, Result};
use std::path::Path;
use tokenizers::Tokenizer as Inner;

pub struct HfTokenizer {
    inner: Inner,
    eos_ids: Vec<u32>,
    pub pad_id: Option<u32>,
}

impl HfTokenizer {
    /// `dir` 含 `tokenizer.json`；eos 从 `generation_config.json`（`eos_token_id` 可为数或数组）读，缺省回退
    /// `tokenizer_config.json` 的 `eos_token`。
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let inner =
            Inner::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("load tokenizer.json: {e}"))?;
        let mut eos_ids = Vec::new();
        let mut pad_id = None;
        if let Ok(s) = std::fs::read_to_string(dir.join("generation_config.json")) {
            let v: serde_json::Value = serde_json::from_str(&s).context("generation_config.json")?;
            match &v["eos_token_id"] {
                serde_json::Value::Number(n) => eos_ids.push(n.as_u64().unwrap() as u32),
                serde_json::Value::Array(a) => {
                    eos_ids.extend(a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32))
                }
                _ => {}
            }
            pad_id = v["pad_token_id"].as_u64().map(|x| x as u32);
        }
        if eos_ids.is_empty() {
            let s = std::fs::read_to_string(dir.join("tokenizer_config.json"))
                .context("tokenizer_config.json")?;
            let v: serde_json::Value = serde_json::from_str(&s)?;
            if let Some(t) = v["eos_token"].as_str() {
                if let Some(id) = inner.token_to_id(t) {
                    eos_ids.push(id);
                }
            }
        }
        anyhow::ensure!(!eos_ids.is_empty(), "no eos token id found in {}", dir.display());
        Ok(Self { inner, eos_ids, pad_id })
    }

    pub fn eos_ids(&self) -> &[u32] {
        &self.eos_ids
    }
    pub fn is_eos(&self, id: u32) -> bool {
        self.eos_ids.contains(&id)
    }
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
    pub fn token_to_id(&self, tok: &str) -> Option<u32> {
        self.inner.token_to_id(tok)
    }
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// 等价 HF `encode(text, add_special_tokens=False)`。
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self.inner.encode(text, false).map_err(|e| anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// 等价 HF `decode(ids)`（默认 `skip_special_tokens=False`，特殊 token 以字面输出）。
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner.decode(ids, false).map_err(|e| anyhow!("decode: {e}"))
    }

    /// 增量 detokenize：每 `step(id)` 返回本 token 新增的文本（UTF-8 不完整时返回 None，等后续 token 补齐）。
    pub fn decode_stream(&self) -> IncrementalDecoder<'_> {
        IncrementalDecoder { inner: self.inner.decode_stream(false) }
    }
}

pub struct IncrementalDecoder<'a> {
    inner: tokenizers::DecodeStream<
        'a,
        tokenizers::ModelWrapper,
        tokenizers::NormalizerWrapper,
        tokenizers::PreTokenizerWrapper,
        tokenizers::PostProcessorWrapper,
        tokenizers::DecoderWrapper,
    >,
}

impl IncrementalDecoder<'_> {
    pub fn step(&mut self, id: u32) -> Result<Option<String>> {
        self.inner.step(id).map_err(|e| anyhow!("decode_stream: {e}"))
    }
}
