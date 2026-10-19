//! 受限解码后端：`llguidance`（vLLM `guidance` 后端同源，crates.io 1.9），实现 core 的 `GrammarState`。
//!
//! - `GrammarFactory`：启动时从模型 tokenizer 建一次 `TokEnv`（toktrie，~几十 ms）+ `ParserFactory`；
//! - `GrammarSpec`：请求层能表达的文法（JSON schema / 任意 JSON / lark）；
//! - `tool_call_lark`：把 OpenAI tools 的 parameters schema 包成模型 chat 模板的 tool_call 文本格式（lark + `%json`）。
//!
//! 每步：runner 在 `SampleBatch::push` 里 `consume_token` 追赶输出序列 → `fill_mask` 得 bitmask → `logits_mask_f32`。
//! TODO(api)：思考段（`<think>…</think>`）期间 vLLM 会跳过文法（reasoning parser 感知）；这里文法从第一个生成 token 起
//! 就生效，所以带约束的请求应 `enable_thinking=false`（server 侧在 response_format / tool_choice 约束时自动置）。

use anyhow::{anyhow, bail, Context, Result};
use dinfer_core::sample::{GrammarHandle, GrammarState};
use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::TokEnv;
use llguidance::{Matcher, ParserFactory};
use std::sync::Arc;
use toktrie_hf_tokenizers::ByteTokenizer;

/// 请求层的文法规格。
#[derive(Debug, Clone, PartialEq)]
pub enum GrammarSpec {
    /// `response_format: {"type":"json_schema", ...}`
    JsonSchema(serde_json::Value),
    /// `response_format: {"type":"json_object"}` → 任意 JSON 对象（vLLM 同样用 `{"type":"object"}`）。
    JsonObject,
    /// llguidance lark 文法（tool_choice 用）。
    Lark(String),
}

/// 一个模型一份：tokenizer 的 toktrie + llguidance ParserFactory。
pub struct GrammarFactory {
    factory: ParserFactory,
    vocab: usize,
}

impl GrammarFactory {
    /// `tok`：HF tokenizer（与 server 用的同一个）；`eos_ids`：eos 集合；`vocab`：**模型** 词表大小（≥ tokenizer 的），
    /// mask 按它分配，与 executor 的 `mask_words = ceil(vocab/32)` 对齐。
    pub fn new(tok: &tokenizers::Tokenizer, eos_ids: &[u32], vocab: usize) -> Result<Self> {
        let t0 = std::time::Instant::now();
        let mut bt = ByteTokenizer::from_tokenizer(tok.clone()).context("llguidance ByteTokenizer")?;
        if !eos_ids.is_empty() {
            bt.set_eos_tokens(eos_ids);
        }
        let env: TokEnv = bt.into_tok_env(Some(vocab)).context("llguidance TokEnv")?;
        let mut factory = ParserFactory::new_simple(&env).context("llguidance ParserFactory")?;
        factory.quiet();
        tracing::info!(vocab, elapsed_ms = t0.elapsed().as_millis() as u64, "guided: llguidance tokenizer env built");
        Ok(Self { factory, vocab })
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// 编译文法（错误 = 请求 400）。编译在 HTTP 线程里做，不占 engine。
    pub fn compile(&self, spec: &GrammarSpec) -> Result<GrammarHandle> {
        let g = match spec {
            GrammarSpec::JsonSchema(schema) => TopLevelGrammar::from_json_schema(schema.clone()),
            GrammarSpec::JsonObject => TopLevelGrammar::from_json_schema(serde_json::json!({"type": "object"})),
            GrammarSpec::Lark(l) => TopLevelGrammar::from_lark(l.clone()),
        };
        let parser = self.factory.create_parser(g).map_err(|e| anyhow!("grammar: {e}"))?;
        let m = Matcher::new(Ok(parser));
        if let Some(e) = m.get_error() {
            bail!("grammar: {e}");
        }
        Ok(GrammarHandle::new(Box::new(LlgGrammar { m, words: self.vocab.div_ceil(32) })))
    }

    /// 只校验（`/v1/chat/completions` 的 400 要在 submit 前给）。
    pub fn validate(&self, spec: &GrammarSpec) -> Result<()> {
        self.compile(spec).map(|_| ())
    }
}

struct LlgGrammar {
    m: Matcher,
    words: usize,
}

impl GrammarState for LlgGrammar {
    fn mask_words(&self) -> usize {
        self.words
    }
    fn fill_mask(&mut self, out: &mut [u32]) -> Result<()> {
        // 文法终止（接受态且无后续）→ 只允许 eos；否则当前允许集
        let vob = self.m.compute_mask_or_eos().map_err(|e| anyhow!("compute_mask: {e}"))?;
        let src = vob.as_slice();
        let n = src.len().min(out.len());
        out[..n].copy_from_slice(&src[..n]);
        out[n..].fill(0);
        Ok(())
    }
    fn consume_token(&mut self, tok: u32) -> Result<()> {
        if self.m.is_stopped() {
            return Ok(()); // 文法已结束（只剩 eos）：后续 token（eos 本身）不再喂
        }
        self.m.consume_token(tok).map_err(|e| anyhow!("consume_token {tok}: {e}"))
    }
    fn reset(&mut self) -> Result<()> {
        self.m.reset().map_err(|e| anyhow!("reset: {e}"))
    }
}

/// lark 字符串字面量（Python 风格转义）。
fn lark_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// 包裹文本 → lark 片段：`<tool_call>` 这类在 tokenizer 里是 **special token**（Qwen3：151657/151658）的部分用 lark 的
/// 裸 `<name>` 语法（匹配该 special token id，而不是它的字面字节——字面字节在 toktrie 里不存在）；其余（`"\n"`）按字符串字面量。
fn tag_lark(tag: &str) -> String {
    let mut parts = Vec::new();
    let mut rest = tag;
    while !rest.is_empty() {
        if let Some(st) = rest.strip_prefix('<') {
            if let Some(end) = st.find('>') {
                let name = &st[..end];
                if !name.is_empty() && !name.contains(|c: char| c == '<' || c.is_whitespace()) {
                    parts.push(format!("<{name}>"));
                    rest = &st[end + 1..];
                    continue;
                }
            }
        }
        // 到下一个 '<' 为止的字面
        let next = rest[1..].find('<').map(|i| i + 1).unwrap_or(rest.len());
        parts.push(lark_str(&rest[..next]));
        rest = &rest[next..];
    }
    parts.join(" ")
}

/// 一个可调用函数：名字 + parameters JSON schema（None = 任意对象）。
pub struct ToolFn<'a> {
    pub name: &'a str,
    pub parameters: Option<&'a serde_json::Value>,
}

/// 把 tools 包成 "`open` + `{"name": "<fn>", "arguments": <schema>}` + `close`" 的 lark 文法（hermes / Qwen3 风格：
/// `open = "<tool_call>\n"`，`close = "\n</tool_call>"`）。`parallel = true` 允许 ≥ 1 个连续 tool_call（之间 `"\n"`），
/// 否则恰好 1 个。`only`：只允许这个函数名（`tool_choice = {"type":"function",...}`）。
/// 与 `models/qwen3/src/parser.rs` 解析的文本格式一致。
pub fn tool_call_lark(tools: &[ToolFn<'_>], open: &str, close: &str, parallel: bool, only: Option<&str>) -> Result<String> {
    let mut rules = Vec::new();
    let mut alts = Vec::new();
    for (i, t) in tools.iter().enumerate() {
        if only.is_some_and(|o| o != t.name) {
            continue;
        }
        let schema = t.parameters.cloned().unwrap_or_else(|| serde_json::json!({"type": "object"}));
        let head = lark_str(&format!("{{\"name\": {}, \"arguments\": ", serde_json::Value::String(t.name.to_string())));
        rules.push(format!("fn{i}: {head} %json {} \"}}\"", serde_json::to_string(&schema)?));
        alts.push(format!("fn{i}"));
    }
    if alts.is_empty() {
        bail!("tool_choice names a function that is not in `tools`");
    }
    let one = format!("call: {} ({}) {}", tag_lark(open), alts.join(" | "), tag_lark(close));
    let start = if parallel { "start: call (\"\\n\" call)*".to_string() } else { "start: call".to_string() };
    Ok(format!("{start}\n{one}\n{}\n", rules.join("\n")))
}

/// 共享工厂类型（server 的 ChatBackend 返回它）。
pub type SharedGrammarFactory = Arc<GrammarFactory>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lark_for_two_tools_mentions_both_and_format() {
        let p = serde_json::json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]});
        let tools = [ToolFn { name: "get_weather", parameters: Some(&p) }, ToolFn { name: "noop", parameters: None }];
        let l = tool_call_lark(&tools, "<tool_call>\n", "\n</tool_call>", false, None).unwrap();
        assert!(l.contains("fn0") && l.contains("fn1") && l.contains("<tool_call> \"\\n\""), "{l}");
        assert_eq!(tag_lark("\n</tool_call>"), "\"\\n\" </tool_call>");
        assert_eq!(tag_lark("plain"), "\"plain\"");
        assert!(l.contains("{\\\"name\\\": \\\"get_weather\\\", \\\"arguments\\\": "));
        let only = tool_call_lark(&tools, "<tool_call>\n", "\n</tool_call>", true, Some("noop")).unwrap();
        assert!(!only.contains("get_weather") && only.contains("(\"\\n\" call)*"));
        assert!(tool_call_lark(&tools, "", "", true, Some("nope")).is_err());
    }
}
