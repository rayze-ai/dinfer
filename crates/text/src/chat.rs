//! 通用 HF chat 模板：用 `minijinja` 跑模型自带的 Jinja 模板（`chat_template.jinja`，或 `tokenizer_config.json` 的
//! `chat_template` 字段），输入 OpenAI 风格 messages / tools + 模型特定变量（`extra`，如 GLM 的 `reasoning_effort` /
//! `clear_thinking`、Qwen 的 `enable_thinking`），输出 prompt 文本（再交 tokenizer 编码）。
//!
//! 对齐口径：渲染文本与 HF `apply_chat_template(tokenize=False)` **逐字符一致**（`tests/chat_parity.rs`，
//! golden 由 `scripts/verify/gen_chat_golden.py` 产出）。模板用到的 Jinja 方言里 minijinja 缺的部分在这里补：
//! - `tojson(ensure_ascii=False)`：minijinja 的 `tojson` 不认 `ensure_ascii`，且默认不转义非 ASCII（与 Python
//!   `ensure_ascii=False` 一致）；但 Python 的 `json.dumps` 分隔符是 `", "` / `": "`，minijinja 是紧凑的 → 自定义 filter
//!   用 Python 风格分隔符。
//! - `is mapping` / `is iterable` / `is string` / `capitalize` / `.split` / `.strip` / `.items()`：minijinja `builtins` 已有
//!   （见 chat_parity 中的覆盖用例）。

use anyhow::{anyhow, Context, Result};
use minijinja::value::{Kwargs, Value};
use minijinja::{Environment, Error, ErrorKind};
use std::path::Path;

pub struct ChatTemplate {
    env: Environment<'static>,
}

/// `apply_chat_template` 的参数子集（OpenAI chat 请求能映射到的部分）。
#[derive(Debug, Clone)]
pub struct RenderParams<'a> {
    /// OpenAI 格式 messages（role / content / reasoning_content / tool_calls / tool_call_id …），原样传给模板。
    pub messages: &'a [serde_json::Value],
    /// OpenAI 格式 tools（`{"type":"function","function":{...}}`），None = 不带。
    pub tools: Option<&'a [serde_json::Value]>,
    pub add_generation_prompt: bool,
    /// vLLM `continue_final_message`：最后一条是 assistant 时不闭合它（去掉模板在其 content 之后追加的结束符，
    /// 模型从该 content 末尾续写）。要求 add_generation_prompt=false。
    pub continue_final_message: bool,
    /// 模型特定模板变量，原样进上下文（GLM：`reasoning_effort`、`clear_thinking`；Qwen3：`enable_thinking`）。
    pub extra: &'a serde_json::Map<String, serde_json::Value>,
}

impl ChatTemplate {
    /// 优先 `chat_template.jinja`，否则 `tokenizer_config.json` 的 `chat_template` 字段。
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let jinja = dir.join("chat_template.jinja");
        if jinja.exists() {
            return Self::from_source(std::fs::read_to_string(&jinja).context("chat_template.jinja")?);
        }
        let cfg =
            std::fs::read_to_string(dir.join("tokenizer_config.json")).context("tokenizer_config.json")?;
        let v: serde_json::Value = serde_json::from_str(&cfg)?;
        let t =
            v["chat_template"].as_str().ok_or_else(|| anyhow!("no chat_template in {}", dir.display()))?;
        Self::from_source(t.to_string())
    }

    pub fn from_source(src: String) -> Result<Self> {
        let src = jinja2_to_minijinja(&src);
        let mut env = Environment::new();
        // HF `apply_chat_template` 的 jinja 环境：trim_blocks=True, lstrip_blocks=True, keep_trailing_newline（见
        // transformers/utils/chat_template_utils.py `_compile_jinja_template`）。
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_keep_trailing_newline(true);
        env.add_filter("tojson", tojson_py);
        env.set_unknown_method_callback(py_methods);
        env.add_template_owned("chat", src).map_err(|e| anyhow!("parse chat template: {e:#}"))?;
        Ok(Self { env })
    }

    pub fn render(&self, p: &RenderParams<'_>) -> Result<String> {
        let tmpl = self.env.get_template("chat")?;
        let tools: Value = match p.tools {
            Some(t) if !t.is_empty() => Value::from_serialize(t),
            _ => Value::from(()),
        };
        let mut ctx: std::collections::BTreeMap<&str, Value> = std::collections::BTreeMap::new();
        for (k, v) in p.extra {
            ctx.insert(k.as_str(), Value::from_serialize(v));
        }
        ctx.insert("messages", Value::from_serialize(p.messages));
        ctx.insert("tools", tools);
        ctx.insert("add_generation_prompt", Value::from(p.add_generation_prompt));
        let text = tmpl.render(Value::from_serialize(&ctx)).map_err(|e| anyhow!("render chat template: {e:#}"))?;
        if p.continue_final_message {
            // vLLM `apply_hf_chat_template(continue_final_message=True)`：按最后一条 assistant 的 content 文本把其后的
            // 结束符截掉（HF 的实现同样是 "render 后 rstrip 到 final message content"）。
            anyhow::ensure!(!p.add_generation_prompt, "continue_final_message requires add_generation_prompt=false");
            let last = p.messages.last().context("continue_final_message: no messages")?;
            anyhow::ensure!(last["role"] == "assistant", "continue_final_message requires the last message to be assistant");
            let content = last["content"].as_str().unwrap_or("");
            let trimmed = content.trim_end();
            if let Some(i) = text.rfind(trimmed) {
                return Ok(text[..i + trimmed.len()].to_string());
            }
            anyhow::bail!("continue_final_message: final assistant content not found in rendered prompt");
        }
        Ok(text)
    }
}

/// Python 字符串 / dict 方法（HF 模板按 Jinja2+Python 语义写：`s.strip()`、`s.split(x)`、`d.items()` …），
/// minijinja 2.10 没有内建；用 `set_unknown_method_callback` 补上模板实际用到的那几个。
fn py_methods(
    _state: &minijinja::State,
    value: &Value,
    method: &str,
    args: &[Value],
) -> Result<Value, Error> {
    use minijinja::value::ValueKind;
    let bad = || {
        Error::new(
            ErrorKind::UnknownMethod,
            format!("no python-compat method `{method}` for {}", value.kind()),
        )
    };
    if let Some(s) = value.as_str() {
        return match method {
            "strip" => Ok(Value::from(s.trim())),
            "lstrip" => Ok(Value::from(s.trim_start())),
            "rstrip" => Ok(Value::from(s.trim_end())),
            "lower" => Ok(Value::from(s.to_lowercase())),
            "upper" => Ok(Value::from(s.to_uppercase())),
            "split" => {
                let parts: Vec<Value> = match args.first().and_then(|a| a.as_str()) {
                    Some(sep) => s.split(sep).map(Value::from).collect(),
                    None => s.split_whitespace().map(Value::from).collect(),
                };
                Ok(Value::from(parts))
            }
            "startswith" => {
                Ok(Value::from(args.first().and_then(|a| a.as_str()).is_some_and(|p| s.starts_with(p))))
            }
            "endswith" => {
                Ok(Value::from(args.first().and_then(|a| a.as_str()).is_some_and(|p| s.ends_with(p))))
            }
            "replace" => {
                let (a, b) = (args.first().and_then(|a| a.as_str()), args.get(1).and_then(|a| a.as_str()));
                match (a, b) {
                    (Some(a), Some(b)) => Ok(Value::from(s.replace(a, b))),
                    _ => Err(bad()),
                }
            }
            _ => Err(bad()),
        };
    }
    if value.kind() == ValueKind::Map {
        let obj = value.as_object().ok_or_else(bad)?;
        return match method {
            "items" => {
                let pairs: Vec<Value> =
                    obj.try_iter_pairs().ok_or_else(bad)?.map(|(k, v)| Value::from(vec![k, v])).collect();
                Ok(Value::from(pairs))
            }
            "keys" => Ok(Value::from(obj.try_iter().ok_or_else(bad)?.collect::<Vec<_>>())),
            "values" => {
                Ok(Value::from(obj.try_iter_pairs().ok_or_else(bad)?.map(|(_, v)| v).collect::<Vec<_>>()))
            }
            "get" => {
                let k = args.first().cloned().unwrap_or_default();
                Ok(obj.get_value(&k).unwrap_or_else(|| args.get(1).cloned().unwrap_or_default()))
            }
            _ => Err(bad()),
        };
    }
    Err(bad())
}

/// Jinja2 方言 → minijinja 方言的最小改写：
/// - `x.0.type`（整数属性访问，Jinja2 允许）→ `x[0].type`（minijinja 把 `.0.type` 词法成 float `0.`）。
fn jinja2_to_minijinja(src: &str) -> String {
    // 只改 `<ident-or-)>.<digits>.<ident>` 与 `<ident-or-)>.<digits>` 后跟非标识符的情形；模板里没有浮点字面量。
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len() + 16);
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'.' && i > 0 && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
            let prev = bytes[i - 1];
            let prev_ok = prev.is_ascii_alphanumeric() || prev == b'_' || prev == b')' || prev == b']';
            if prev_ok {
                let mut j = i + 1;
                while j < bytes.len() && bytes[j].is_ascii_digit() {
                    j += 1;
                }
                // 下一个字符不能是数字/字母（排除真浮点如 0.5 —— 模板里没有，但保守处理）
                let next_is_float_cont = j < bytes.len()
                    && bytes[j] == b'.'
                    && j + 1 < bytes.len()
                    && bytes[j + 1].is_ascii_digit();
                if !next_is_float_cont {
                    out.push('[');
                    out.push_str(&src[i + 1..j]);
                    out.push(']');
                    i = j;
                    continue;
                }
            }
        }
        out.push(c as char);
        i += 1;
    }
    out
}

/// Python `json.dumps(v, ensure_ascii=False)`：`", "` / `": "` 分隔符，非 ASCII 原样，key 顺序保持插入序。
fn tojson_py(v: Value, kwargs: Kwargs) -> Result<Value, Error> {
    let _ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    kwargs.assert_all_used()?;
    let j: serde_json::Value = serde_json::to_value(&v)
        .map_err(|e| Error::new(ErrorKind::InvalidOperation, format!("tojson: {e}")))?;
    let mut out = String::new();
    py_dumps(&j, &mut out);
    Ok(Value::from_safe_string(out))
}

fn py_dumps(v: &serde_json::Value, out: &mut String) {
    use serde_json::Value as J;
    match v {
        J::Null => out.push_str("null"),
        J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        J::Number(n) => out.push_str(&n.to_string()),
        J::String(s) => py_str(s, out),
        J::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_dumps(x, out);
            }
            out.push(']');
        }
        J::Object(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_str(k, out);
                out.push_str(": ");
                py_dumps(x, out);
            }
            out.push('}');
        }
    }
}

/// Python json 字符串转义（ensure_ascii=False）：只转义 `"` `\\` 和控制字符，其中 \b \f \n \r \t 用短形式。
fn py_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}
