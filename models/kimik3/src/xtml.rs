//! Kimi-K3 XTML 聊天渲染：HF `encoding_k3.py::build_chat_segments` 的移植（K3 没有 jinja 模板）。
//!
//! 输出 `Vec<Segment{text, allow_special}>`：结构标记（`<|open|> <|close|> <|sep|> <|end_of_msg|>`）走
//! `allow_special=true` 编成特殊 id；用户 / 工具文本与属性值走 `false`，用户打的 `<|open|>` 字样永远是普通 BPE。
//! 函数名与 Python 一一对应，字符串（空格、转义、顺序、英文正文）以保真为先，不要"整理"。
//!
//! ## 两种方言（`Dialect`）
//! K3 有两份渲染参考实现，嵌入 prompt 的 JSON 字节不同（见 `docs/design-kimik3.md` §1.6、`docs/loops/k3-chat.md`）：
//! - `Hf`：HF `encoding_k3.py` 原样 = 我们的分析基线 vLLM `de8291e`（`renderers/kimi_k3.py` 直接调 HF
//!   `apply_chat_template`）。浮点 `json.dumps`（`1e+16`）、tool_calls 字串参数一层解析保字面量、消息 `name` 渲染成属性。
//! - `Serving`：Moonshot 线上 vLLM 的 Rust renderer（`llm-cache-prediction` 项目 100+ 轮逐 token 对出的口径）：
//!   声明 / schema 路径 pydantic 把整数浮点塌成整数再 serde（`100.0`→`100`、`1e16`→`10000000000000000`、`1e-6` 保留）；
//!   tool_calls 字串参数 serde 完整解析后重序列化（`1e2`→`100.0`、`[1,  2]`→`[1,2]`）；消息 `name` 被前端丢弃。
//!
//! 默认 `Hf`（对拍基线可用 transformers 直接生成 golden）；`Serving` 由 golden 脚本的 monkeypatch 口径生成。

use anyhow::{anyhow, bail, Result};
use serde_json::{Map, Value};

pub const OPEN: &str = "<|open|>";
pub const CLOSE: &str = "<|close|>";
pub const SEP: &str = "<|sep|>";
pub const END_OF_MSG: &str = "<|end_of_msg|>";
/// 图片占位（content part 为 image 时插入；视觉塔接入前只是一个 allow_special 文本段）。
pub const IMAGE_PLACEHOLDER: &str = "<|kimi_image_placeholder|>";

pub const VALID_THINKING_EFFORTS: [&str; 3] = ["low", "high", "max"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub text: String,
    pub allow_special: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Dialect {
    #[default]
    Hf,
    Serving,
}

#[derive(Debug, Clone)]
pub struct RenderOptions<'a> {
    pub tools: Option<&'a [Value]>,
    pub add_generation_prompt: bool,
    /// 思考通道（K3 默认恒开；`reasoning_effort="none"` 关）。
    pub thinking: bool,
    /// `low` / `high` / `max`；None = 不注入 thinking-effort 系统消息。HF `apply_chat_template` 默认 `max`。
    pub thinking_effort: Option<&'a str>,
    /// `"required"` / `"none"`；其他值（含 auto）不渲染。
    pub tool_choice: Option<&'a str>,
    /// OpenAI `response_format`：`{"type":"json_object"}` 或 `{"type":"json_schema","json_schema":{"schema":…}}`。
    pub response_format: Option<&'a Value>,
    pub dialect: Dialect,
}

impl Default for RenderOptions<'_> {
    fn default() -> Self {
        Self {
            tools: None,
            add_generation_prompt: true,
            thinking: true,
            thinking_effort: Some("max"),
            tool_choice: None,
            response_format: None,
            dialect: Dialect::Hf,
        }
    }
}

/// `build_chat_segments(messages, tools, add_generation_prompt, thinking, **kwargs)`。
pub fn build_chat_segments(messages: &[Value], opts: &RenderOptions<'_>) -> Result<Vec<Segment>> {
    if let Some(e) = opts.thinking_effort {
        if opts.thinking && !VALID_THINKING_EFFORTS.contains(&e) {
            bail!("Unsupported thinking_effort={e:?}; supported values are {:?}", VALID_THINKING_EFFORTS);
        }
    }
    let messages = normalize_xtml_tool_result_messages(messages);
    let tools = opts.tools.map(|t| deep_sort(&Value::Array(t.to_vec())));

    let mut seg: Vec<Segment> = Vec::new();
    if let Some(Value::Array(t)) = &tools {
        if !t.is_empty() {
            render_tool_declare(&mut seg, &Value::Array(t.clone()), false, opts.dialect);
        }
    }
    if opts.thinking {
        if let Some(effort) = opts.thinking_effort {
            internal_system_message(
                &mut seg,
                "thinking-effort",
                &format!(
                    "`thinking_effort` guides on how much to think in your thinking channel (not including the response channel), supported values include `low`, `medium`, `high`, and `max`.\nNow the system is invoked with `thinking_effort={effort}`."
                ),
            );
        }
    }

    let mut tool_calls: Option<Vec<Value>> = None;
    let mut tool_index = 0usize;
    for m in &messages {
        let Some(obj) = m.as_object() else { continue };
        let role = obj.get("role").and_then(|r| r.as_str()).ok_or_else(|| anyhow!("message 缺 role"))?;
        let name_attr = |attrs: &mut Vec<(String, String)>| {
            if opts.dialect == Dialect::Hf {
                if let Some(n) = obj.get("name").filter(|n| truthy(n)) {
                    attrs.push(("name".into(), py_str(n)));
                }
            }
        };
        match role {
            "user" => {
                let mut attrs = vec![("role".to_string(), "user".to_string())];
                name_attr(&mut attrs);
                open_tag(&mut seg, "message", &attrs);
                render_content_segments(&mut seg, obj.get("content"))?;
                close_tag(&mut seg, "message");
                control(&mut seg, END_OF_MSG);
            }
            "system" if obj.get("tools").map(truthy).unwrap_or(false) => {
                render_tool_declare(&mut seg, &deep_sort(&obj["tools"]), true, opts.dialect);
            }
            "system" => {
                let mut attrs = vec![("role".to_string(), "system".to_string())];
                name_attr(&mut attrs);
                open_tag(&mut seg, "message", &attrs);
                render_content_segments(&mut seg, obj.get("content"))?;
                close_tag(&mut seg, "message");
                control(&mut seg, END_OF_MSG);
            }
            "tool" => {
                tool_index += 1;
                let mut tool_name: Option<String> =
                    obj.get("tool").or_else(|| obj.get("name")).filter(|v| !v.is_null()).map(py_str);
                if tool_name.is_none() {
                    if let Some(tcs) = &tool_calls {
                        if tool_index <= tcs.len() {
                            let tc = &tcs[tool_index - 1];
                            let f = tc.get("function").unwrap_or(tc);
                            tool_name = f.get("name").map(py_str);
                        }
                    }
                }
                let tool_name = tool_name.ok_or_else(|| {
                    anyhow!("Kimi K3 tool messages need a resolvable tool name: carry `tool`/`name`, or match a preceding assistant tool_call by order.")
                })?;
                let attrs = vec![
                    ("role".to_string(), "tool".to_string()),
                    ("tool".to_string(), tool_name),
                    ("index".to_string(), tool_index.to_string()),
                ];
                open_tag(&mut seg, "message", &attrs);
                render_content_segments(&mut seg, obj.get("content"))?;
                close_tag(&mut seg, "message");
                control(&mut seg, END_OF_MSG);
            }
            "assistant" => {
                tool_calls = obj.get("tool_calls").and_then(|t| t.as_array()).cloned();
                tool_index = 0;
                let mut attrs = vec![("role".to_string(), "assistant".to_string())];
                name_attr(&mut attrs);
                open_tag(&mut seg, "message", &attrs);
                render_assistant_segments(&mut seg, obj, opts.thinking, opts.dialect)?;
                close_tag(&mut seg, "message");
                control(&mut seg, END_OF_MSG);
            }
            other => bail!("unknown role {other:?}"),
        }
    }

    match opts.tool_choice {
        Some("required") => internal_system_message(
            &mut seg,
            "tool-choice",
            "The system is invoked with `tool_choice=required`.\nYou MUST call tools in the next message.",
        ),
        Some("none") => internal_system_message(
            &mut seg,
            "tool-choice",
            "The system is invoked with `tool_choice=none`.\nYou MUST NOT call any tools in the next message.",
        ),
        _ => {}
    }

    if let Some(rf) = opts.response_format {
        let rf_type = rf.get("type").and_then(|t| t.as_str()).or_else(|| rf.as_str());
        match rf_type {
            Some("json_object") => internal_system_message(
                &mut seg,
                "response-format",
                "The system is invoked with `response_format=json_object`.\nYour response must be raw JSON data without markdown code blocks (```json) or any additional formatting.",
            ),
            Some("json_schema") => {
                let schema = extract_response_schema(rf).map(|s| deep_sort(&s)).unwrap_or(Value::Null);
                let schema = json_compact(&schema, opts.dialect);
                internal_system_message(
                    &mut seg,
                    "response-format",
                    &format!(
                        "The system is invoked with `response_format=json_schema`.\nYour response must be raw JSON data without markdown code blocks (```json) or any additional formatting.\nThe JSON data must match the following schema:\n```json\n{schema}\n```"
                    ),
                );
            }
            _ => {}
        }
    }

    if opts.add_generation_prompt {
        open_tag(&mut seg, "message", &[("role".to_string(), "assistant".to_string())]);
        open_tag(&mut seg, if opts.thinking { "think" } else { "response" }, &[]);
    }
    Ok(seg)
}

/// 段落文本拼接（HF `apply_chat_template(tokenize=False)`）。
pub fn join_text(segments: &[Segment]) -> String {
    segments.iter().map(|s| s.text.as_str()).collect()
}

/// `extract_response_schema`：`response_format.json_schema.schema`（或 `.json_schema`，或整个 json_schema）。
pub fn extract_response_schema(rf: &Value) -> Option<Value> {
    let js = rf.get("json_schema")?;
    if let Some(o) = js.as_object() {
        if let Some(s) = o.get("schema") {
            return Some(s.clone());
        }
        if let Some(s) = o.get("json_schema") {
            return Some(s.clone());
        }
    }
    Some(js.clone())
}

// ---- 基本段 ----

fn push(seg: &mut Vec<Segment>, text: impl Into<String>, allow_special: bool) {
    let text = text.into();
    if !text.is_empty() {
        seg.push(Segment { text, allow_special });
    }
}
fn control(seg: &mut Vec<Segment>, s: &str) {
    push(seg, s, true);
}
fn text(seg: &mut Vec<Segment>, s: impl Into<String>) {
    push(seg, s, false);
}

/// `_escape_attr_value`：只转 `&` 与 `"`。
pub fn escape_attr_value(v: &str) -> String {
    v.replace('&', "&amp;").replace('"', "&quot;")
}

fn open_tag(seg: &mut Vec<Segment>, tag: &str, attrs: &[(String, String)]) {
    control(seg, OPEN);
    text(seg, tag);
    for (k, v) in attrs {
        text(seg, format!(" {k}"));
        text(seg, "=\"");
        text(seg, escape_attr_value(v));
        text(seg, "\"");
    }
    control(seg, SEP);
}

fn close_tag(seg: &mut Vec<Segment>, tag: &str) {
    control(seg, CLOSE);
    text(seg, tag);
    control(seg, SEP);
}

fn internal_system_message(seg: &mut Vec<Segment>, msg_type: &str, body: &str) {
    open_tag(seg, "message", &[("role".into(), "system".into()), ("type".into(), msg_type.into())]);
    text(seg, body.trim());
    close_tag(seg, "message");
    control(seg, END_OF_MSG);
}

fn render_tool_declare(seg: &mut Vec<Segment>, tools: &Value, dynamic: bool, d: Dialect) {
    let js = json_compact(tools, d);
    let body = if dynamic {
        format!("## New Tools Available\nThe system dynamically extends the toolset via lazy-loading.\nYou have access to all existing and extended tools.\nHere are the specs for the extended tools.\n\n```json\n{js}\n```")
    } else {
        format!("# Tools\nHere are the available tools, described in JSONSchema.\n\n```json\n{js}\n```")
    };
    open_tag(seg, "message", &[("role".into(), "system".into()), ("type".into(), "tool-declare".into())]);
    text(seg, body);
    close_tag(seg, "message");
    control(seg, END_OF_MSG);
}

/// `_render_content_segments`：str 直接；list 里 `image`/`image_url` → 占位（allow_special），其余取 `text`。
fn render_content_segments(seg: &mut Vec<Segment>, content: Option<&Value>) -> Result<()> {
    match content {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) => text(seg, s.as_str()),
        Some(Value::Array(parts)) => {
            for p in parts {
                match p.get("type").and_then(|t| t.as_str()) {
                    Some("image") | Some("image_url") => push(seg, IMAGE_PLACEHOLDER, true),
                    _ => {
                        let t = p.get("text").ok_or_else(|| anyhow!("content part 缺 text"))?;
                        text(seg, py_str(t));
                    }
                }
            }
        }
        Some(other) => text(seg, py_str(other)),
    }
    Ok(())
}

fn render_assistant_segments(
    seg: &mut Vec<Segment>,
    m: &Map<String, Value>,
    thinking: bool,
    d: Dialect,
) -> Result<()> {
    if thinking {
        let rc = m.get("reasoning_content").filter(|v| truthy(v)).or_else(|| m.get("reasoning"));
        open_tag(seg, "think", &[]);
        if let Some(rc) = rc.filter(|v| !v.is_null()) {
            let s = py_str(rc);
            if !s.trim().is_empty() {
                text(seg, s);
            }
        }
        close_tag(seg, "think");
    }
    open_tag(seg, "response", &[]);
    render_content_segments(seg, m.get("content"))?;
    close_tag(seg, "response");

    if let Some(tcs) = m.get("tool_calls").and_then(|t| t.as_array()).filter(|t| !t.is_empty()) {
        open_tag(seg, "tools", &[]);
        for (i, tc) in tcs.iter().enumerate() {
            let f = tc.get("function").unwrap_or(tc);
            let name = f.get("name").ok_or_else(|| anyhow!("tool_call 缺 function.name"))?;
            open_tag(seg, "call", &[("tool".into(), py_str(name)), ("index".into(), (i + 1).to_string())]);
            match normalize_tool_arguments(f.get("arguments"), d)? {
                Args::Block(block) => {
                    open_tag(seg, "json", &[("type".into(), "object".into())]);
                    text(seg, block);
                    close_tag(seg, "json");
                }
                Args::Map(args) => {
                    for (k, v) in &args {
                        open_tag(
                            seg,
                            "argument",
                            &[("key".into(), k.clone()), ("type".into(), xtml_type(v).into())],
                        );
                        text(seg, xtml_value(v, d));
                        close_tag(seg, "argument");
                    }
                }
            }
            close_tag(seg, "call");
        }
        close_tag(seg, "tools");
    }
    Ok(())
}

enum Args {
    Map(Map<String, Value>),
    /// 字串参数不是合法 JSON：整段作为 `<|open|>json type="object"<|sep|>…` 块。
    Block(String),
}

/// `normalize_tool_arguments`：None → {}；dict 原样；str → json 解析（非 object 报错）；解析失败 → Block。
fn normalize_tool_arguments(a: Option<&Value>, d: Dialect) -> Result<Args> {
    match a {
        None | Some(Value::Null) => Ok(Args::Map(Map::new())),
        Some(Value::Object(m)) => Ok(Args::Map(m.clone())),
        Some(Value::String(s)) => {
            if s.trim().is_empty() {
                return Ok(Args::Map(Map::new()));
            }
            match serde_json::from_str::<Value>(s) {
                Ok(Value::Object(m)) => Ok(Args::Map(m)),
                Ok(_) => match d {
                    Dialect::Hf => bail!("Kimi K3 tool call arguments must be a JSON object."),
                    Dialect::Serving => Ok(Args::Block(s.clone())),
                },
                Err(_) => Ok(Args::Block(s.clone())),
            }
        }
        Some(other) => {
            bail!("Kimi K3 tool call arguments must be a dict or a JSON object string, got {other}")
        }
    }
}

/// `_xtml_type`。
pub fn xtml_type(v: &Value) -> &'static str {
    match v {
        Value::Bool(_) => "boolean",
        Value::Null => "null",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Object(_) => "object",
        Value::Array(_) => "array",
    }
}

/// `_xtml_value`：str 原样；其他 `json.dumps(value, ensure_ascii=False)`（Hf 默认分隔符 `, ` / `: `；Serving 紧凑 serde）。
pub fn xtml_value(v: &Value, d: Dialect) -> String {
    match v {
        Value::String(s) => s.clone(),
        _ => match d {
            Dialect::Hf => py_json_dumps(v, false, false),
            Dialect::Serving => serde_json::to_string(v).unwrap_or_default(),
        },
    }
}

/// `_json_compact`：声明 / schema 路径。Hf = `json.dumps(separators=(",",":"), ensure_ascii=False)`；
/// Serving = pydantic 塌整数浮点 + serde。
pub fn json_compact(v: &Value, d: Dialect) -> String {
    match d {
        Dialect::Hf => py_json_dumps(v, true, false),
        Dialect::Serving => serde_json::to_string(&collapse_integral_floats(v)).unwrap_or_default(),
    }
}

/// Python `str(value)`：字串原样、bool `True/False`、None `None`、浮点 `repr`。
pub fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        Value::Null => "None".to_string(),
        Value::Number(n) => py_number(n),
        other => py_json_dumps(other, false, false),
    }
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// `deep_sort_dict`：递归按 key 排序（Python 对 str 按码点排 = Rust String 字节序）。
pub fn deep_sort(v: &Value) -> Value {
    match v {
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            let mut out = Map::new();
            for k in keys {
                out.insert(k.clone(), deep_sort(&m[k]));
            }
            Value::Object(out)
        }
        Value::Array(a) => Value::Array(a.iter().map(deep_sort).collect()),
        other => other.clone(),
    }
}

/// Serving 方言声明路径：pydantic round-trip 把整数值浮点塌成整数（`100.0`→`100`）。
pub fn collapse_integral_floats(v: &Value) -> Value {
    match v {
        Value::Number(n) if n.is_f64() => {
            let f = n.as_f64().unwrap();
            if f.fract() == 0.0 && f.abs() < 9.007_199_254_740_992e15 {
                Value::Number(serde_json::Number::from(f as i64))
            } else {
                v.clone()
            }
        }
        Value::Object(m) => {
            Value::Object(m.iter().map(|(k, x)| (k.clone(), collapse_integral_floats(x))).collect())
        }
        Value::Array(a) => Value::Array(a.iter().map(collapse_integral_floats).collect()),
        other => other.clone(),
    }
}

/// CPython `json.dumps(v, ensure_ascii=False)`；`compact` = `separators=(",", ":")`，否则默认 `", "` / `": "`。
pub fn py_json_dumps(v: &Value, compact: bool, _ascii: bool) -> String {
    let mut out = String::new();
    write_py_json(v, compact, &mut out);
    out
}

fn write_py_json(v: &Value, compact: bool, out: &mut String) {
    let (item_sep, kv_sep) = if compact { (",", ":") } else { (", ", ": ") };
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&py_number(n)),
        Value::String(s) => write_py_json_str(s, out),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                write_py_json(x, compact, out);
            }
            out.push(']');
        }
        Value::Object(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                write_py_json_str(k, out);
                out.push_str(kv_sep);
                write_py_json(x, compact, out);
            }
            out.push('}');
        }
    }
}

/// Python json 字串转义（ensure_ascii=False）：`"` `\` 与 < 0x20 控制字符（\n \r \t \b \f 简写，其余 \u00XX）。
fn write_py_json_str(s: &str, out: &mut String) {
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

/// JSON 数字 → Python 文本：整数原样；浮点按 `repr(float)`（最短往返；指数 < -4 或 ≥ 16 用 `e±XX`）。
pub fn py_number(n: &serde_json::Number) -> String {
    if let Some(i) = n.as_i64() {
        return i.to_string();
    }
    if let Some(u) = n.as_u64() {
        return u.to_string();
    }
    py_float_repr(n.as_f64().unwrap_or(0.0))
}

/// CPython `repr(float)`。
pub fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    // Rust `{:e}` 给最短往返的 "d.ddde±x"
    let e = format!("{f:e}");
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    if !(-4..16).contains(&exp) {
        return format!("{mant}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
    // 定点：Rust `{}` 对 f64 用最短往返定点，但整数值不带 ".0"
    let s = format!("{f}");
    if s.contains('.') || s.contains('e') {
        s
    } else {
        format!("{s}.0")
    }
}

/// `normalize_xtml_tool_result_messages`：每段连续 tool 消息按最近 assistant `tool_calls[].id` 匹配
/// `tool_call_id`，按 1-based 位置排序，并把 `tool`（和已有的 `name`）改成匹配到的函数名；有一条匹配不上则整段原样。
pub fn normalize_xtml_tool_result_messages(messages: &[Value]) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::with_capacity(messages.len());
    let mut index: Map<String, Value> = Map::new(); // id → [position, name]
    let mut i = 0;
    let n = messages.len();
    while i < n {
        let m = &messages[i];
        let role = m.get("role").and_then(|r| r.as_str());
        if role == Some("assistant") {
            index = Map::new();
            if let Some(tcs) = m.get("tool_calls").and_then(|t| t.as_array()) {
                for (pos, tc) in tcs.iter().enumerate() {
                    let Some(tco) = tc.as_object() else { continue };
                    let Some(id) = tco.get("id").filter(|v| !v.is_null()) else { continue };
                    let key = py_str(id);
                    if index.contains_key(&key) {
                        continue;
                    }
                    let name = match tco.get("function") {
                        Some(Value::Object(f)) => f.get("name").cloned().unwrap_or(Value::Null),
                        _ => tco.get("name").cloned().unwrap_or(Value::Null),
                    };
                    index.insert(key, serde_json::json!([pos + 1, name]));
                }
            }
            out.push(m.clone());
            i += 1;
            continue;
        }
        if role != Some("tool") {
            out.push(m.clone());
            i += 1;
            continue;
        }
        // 一段连续 tool 消息
        let mut run: Vec<(Option<usize>, usize, &Value, Option<String>)> = Vec::new();
        let mut unresolved = false;
        let mut offset = 0;
        while i < n && messages[i].get("role").and_then(|r| r.as_str()) == Some("tool") {
            let tm = &messages[i];
            let call_id = tm.get("tool_call_id").or_else(|| tm.get("id")).filter(|v| !v.is_null());
            let matched = call_id.and_then(|c| index.get(&py_str(c)));
            match matched {
                None => {
                    unresolved = true;
                    run.push((None, offset, tm, None));
                }
                Some(v) => {
                    let pos = v[0].as_u64().unwrap() as usize;
                    let name = v[1].as_str().map(|s| s.to_string());
                    run.push((Some(pos), offset, tm, name));
                }
            }
            offset += 1;
            i += 1;
        }
        if unresolved {
            out.extend(run.iter().map(|r| r.2.clone()));
        } else {
            run.sort_by_key(|r| (r.0, r.1));
            for (_, _, tm, name) in run {
                match name {
                    None => out.push(tm.clone()),
                    Some(name) => {
                        let mut resolved = tm.clone();
                        if let Some(o) = resolved.as_object_mut() {
                            o.insert("tool".into(), Value::String(name.clone()));
                            if o.contains_key("name") {
                                o.insert("name".into(), Value::String(name));
                            }
                        }
                        out.push(resolved);
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn float_repr_matches_cpython() {
        for (f, want) in [
            (1e16, "1e+16"),
            (100.0, "100.0"),
            (1e-5, "1e-05"),
            (1e-6, "1e-06"),
            (1.5, "1.5"),
            (-2.5, "-2.5"),
            (0.1, "0.1"),
            (1e22, "1e+22"),
            (123456789012345678.0, "1.2345678901234568e+17"),
            (0.0001, "0.0001"),
            (1234567890123456.0, "1234567890123456.0"),
        ] {
            assert_eq!(py_float_repr(f), want, "{f}");
        }
    }

    #[test]
    fn json_dumps_default_and_compact() {
        let v = json!({"b": [1, 2.0, "x\"y\n"], "a": null, "c": true, "中": "文"});
        assert_eq!(py_json_dumps(&v, true, false), r#"{"b":[1,2.0,"x\"y\n"],"a":null,"c":true,"中":"文"}"#);
        assert_eq!(
            py_json_dumps(&v, false, false),
            r#"{"b": [1, 2.0, "x\"y\n"], "a": null, "c": true, "中": "文"}"#
        );
        assert_eq!(json_compact(&json!({"a": 100.0, "t": 1e-6}), Dialect::Serving), r#"{"a":100,"t":1e-6}"#);
        assert_eq!(json_compact(&json!({"a": 100.0, "t": 1e-6}), Dialect::Hf), r#"{"a":100.0,"t":1e-06}"#);
    }

    #[test]
    fn single_turn_shape() {
        let msgs = vec![json!({"role": "user", "content": "hi"})];
        let seg = build_chat_segments(&msgs, &RenderOptions::default()).unwrap();
        let t = join_text(&seg);
        assert!(t.starts_with("<|open|>message role=\"system\" type=\"thinking-effort\"<|sep|>"));
        assert!(t.ends_with("<|open|>message role=\"user\"<|sep|>hi<|close|>message<|sep|><|end_of_msg|><|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>"));
        // 用户文本是普通段
        assert!(seg.iter().any(|s| s.text == "hi" && !s.allow_special));
        assert!(seg.iter().filter(|s| s.allow_special).all(|s| s.text.starts_with("<|")));
    }

    #[test]
    fn name_dropped_in_serving_dialect_only() {
        let msgs = vec![json!({"role": "user", "name": "a&b", "content": "hi"})];
        let hf = join_text(&build_chat_segments(&msgs, &RenderOptions::default()).unwrap());
        assert!(hf.contains(r#"role="user" name="a&amp;b""#));
        let sv = join_text(
            &build_chat_segments(&msgs, &RenderOptions { dialect: Dialect::Serving, ..Default::default() })
                .unwrap(),
        );
        assert!(!sv.contains("name="));
    }

    #[test]
    fn tool_results_reordered_by_id() {
        let msgs = vec![
            json!({"role": "assistant", "content": null, "tool_calls": [
                {"id": "c1", "function": {"name": "f1", "arguments": "{}"}},
                {"id": "c2", "function": {"name": "f2", "arguments": "{}"}}]}),
            json!({"role": "tool", "tool_call_id": "c2", "name": "stale", "content": "r2"}),
            json!({"role": "tool", "tool_call_id": "c1", "content": "r1"}),
        ];
        let t = join_text(&build_chat_segments(&msgs, &RenderOptions::default()).unwrap());
        let i1 = t.find(r#"role="tool" tool="f1" index="1"<|sep|>r1"#).unwrap();
        let i2 = t.find(r#"role="tool" tool="f2" index="2"<|sep|>r2"#).unwrap();
        assert!(i1 < i2);
    }

    #[test]
    fn invalid_effort_rejected() {
        let msgs = vec![json!({"role": "user", "content": "hi"})];
        assert!(build_chat_segments(
            &msgs,
            &RenderOptions { thinking_effort: Some("medium"), ..Default::default() }
        )
        .is_err());
    }
}
