//! Kimi-K3 输出流式解析（XTML）：
//! `<|open|>think<|sep|>…<|close|>think<|sep|>` → reasoning；
//! `<|open|>response<|sep|>…<|close|>response<|sep|>` → content；
//! `<|open|>tools<|sep|> <|open|>call tool="…" index="…"<|sep|> <|open|>argument key=… type=…<|sep|>…<|close|>argument<|sep|> …
//! <|close|>call<|sep|> … <|close|>tools<|sep|>` → tool_calls；`<|close|>message<|sep|>` / `<|end_of_msg|>` 结束。
//!
//! 口径 = vLLM `reasoning/kimi_k3_reasoning_parser.py` + `tool_parsers/kimi_k3_tool_parser.py`：
//! - 生成提示以 `<|open|>think<|sep|>` 结尾（thinking 恒开），所以初始态 = Think；若模型输出里又出现 open 标记则吞掉；
//! - 标记跨 token 边界：尾部是任一标记的真前缀就 hold back（`_partial_tag_overlap`）；
//! - 工具调用整块闭合（`<|close|>call<|sep|>`）才吐，`type="string"` 原文，其他 JSON 解码失败退回原文，
//!   属性值做 `&quot;` / `&amp;` 反转义；arguments 以 JSON 文本输出；
//! - 解析器只认字面标记（detokenize 必须 `skip_special_tokens=false`），不输出任何 XTML 标记。
//!
//! 与 vLLM 的差别（有意）：vLLM 对 tool_choice=none 的请求只走 reasoning parser 并剥 response 包裹；我们单一状态机同时
//! 处理两类，行为与"带 tools 的 vLLM"一致（tool 段恒解析）。

use serde::Serialize;
use serde_json::Value;

const OPEN: &str = "<|open|>";
const CLOSE: &str = "<|close|>";
const SEP: &str = "<|sep|>";
const END_OF_MSG: &str = "<|end_of_msg|>";

const THINK_OPEN: &str = "<|open|>think<|sep|>";
const THINK_CLOSE: &str = "<|close|>think<|sep|>";
const RESPONSE_OPEN: &str = "<|open|>response<|sep|>";
const RESPONSE_CLOSE: &str = "<|close|>response<|sep|>";
const TOOLS_OPEN: &str = "<|open|>tools<|sep|>";
const TOOLS_CLOSE: &str = "<|close|>tools<|sep|>";
const MESSAGE_CLOSE: &str = "<|close|>message<|sep|>";
const CALL_CLOSE: &str = "<|close|>call<|sep|>";
const ARG_CLOSE: &str = "<|close|>argument<|sep|>";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ToolCallDelta {
    pub index: usize,
    pub name: Option<String>,
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Delta {
    pub reasoning: String,
    pub content: String,
    pub tool_calls: Vec<ToolCallDelta>,
}

impl Delta {
    pub fn is_empty(&self) -> bool {
        self.reasoning.is_empty() && self.content.is_empty() && self.tool_calls.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Think,
    /// think 已闭合，等 response / tools 的 open 标记（或散文本 → content）。
    AfterThink,
    Response,
    /// response 已闭合，等 tools open / message close。
    AfterResponse,
    Tools,
    Done,
}

#[derive(Debug)]
pub struct OutputParser {
    state: State,
    buf: String,
    /// 本状态刚进入、尚未吐过字：允许吞掉模型重复输出的 open 标记。
    fresh: bool,
    tc_index: usize,
}

impl Default for OutputParser {
    fn default() -> Self {
        Self::new(true)
    }
}

impl OutputParser {
    /// `thinking=true`：生成提示以 `<|open|>think<|sep|>` 结尾，初始 Think；否则以 `<|open|>response<|sep|>` 结尾，初始 Response。
    pub fn new(thinking: bool) -> Self {
        Self {
            state: if thinking { State::Think } else { State::Response },
            buf: String::new(),
            fresh: true,
            tc_index: 0,
        }
    }

    pub fn feed(&mut self, text: &str) -> Delta {
        self.buf.push_str(text);
        let mut d = Delta::default();
        self.drain(&mut d, false);
        d
    }

    pub fn finish(&mut self) -> Delta {
        let mut d = Delta::default();
        self.drain(&mut d, true);
        // 冲尾巴
        let rest = std::mem::take(&mut self.buf);
        match self.state {
            State::Think => d.reasoning.push_str(&rest),
            State::Response => d.content.push_str(&rest),
            State::AfterThink | State::AfterResponse => d.content.push_str(&strip_markers(&rest)),
            State::Tools | State::Done => {}
        }
        d
    }

    fn drain(&mut self, d: &mut Delta, flush: bool) {
        loop {
            let before = self.buf.len();
            let st = self.state;
            match st {
                State::Think => {
                    self.drain_channel(d, THINK_OPEN, THINK_CLOSE, State::AfterThink, true, flush)
                }
                State::Response => {
                    self.drain_channel(d, RESPONSE_OPEN, RESPONSE_CLOSE, State::AfterResponse, false, flush)
                }
                State::AfterThink | State::AfterResponse => self.drain_between(d, flush),
                State::Tools => self.drain_tools(d),
                State::Done => {
                    self.buf.clear();
                }
            }
            if self.state == st && self.buf.len() == before {
                break;
            }
        }
    }

    /// Think / Response 通道：吞重复 open，正文吐到 close 之前（尾部 hold back 可能的标记前缀）。
    fn drain_channel(
        &mut self,
        d: &mut Delta,
        open: &str,
        close: &str,
        next: State,
        reasoning: bool,
        flush: bool,
    ) {
        if self.fresh {
            if let Some(stripped) = self.buf.strip_prefix(open) {
                self.buf = stripped.to_string();
                self.fresh = false;
            } else if open.starts_with(self.buf.as_str()) && !self.buf.is_empty() && !flush {
                return; // 可能是 open 的前缀，等
            } else if !self.buf.is_empty() {
                self.fresh = false;
            }
        }
        let sink =
            |d: &mut Delta, s: &str| if reasoning { d.reasoning.push_str(s) } else { d.content.push_str(s) };
        if let Some(pos) = self.buf.find(close) {
            sink(d, &self.buf[..pos]);
            self.buf.drain(..pos + close.len());
            self.state = next;
            self.fresh = true;
            return;
        }
        // 也可能直接 <|close|>message / <|end_of_msg|>（模型漏写 close）
        for m in [MESSAGE_CLOSE, END_OF_MSG] {
            if let Some(pos) = self.buf.find(m) {
                sink(d, &self.buf[..pos]);
                self.buf.clear();
                self.state = State::Done;
                return;
            }
        }
        let hold = partial_overlap(&self.buf, &[close, MESSAGE_CLOSE, END_OF_MSG, OPEN, CLOSE]);
        let emit = self.buf.len() - hold;
        if emit > 0 {
            let s: String = self.buf.drain(..emit).collect();
            sink(d, &s);
        }
    }

    /// 通道之间：等 `<|open|>response|tools<|sep|>` / `<|close|>message<|sep|>`；散文本当 content。
    fn drain_between(&mut self, d: &mut Delta, flush: bool) {
        let allow_response = self.state == State::AfterThink;
        loop {
            if self.buf.is_empty() {
                return;
            }
            if allow_response && self.buf.starts_with(RESPONSE_OPEN) {
                self.state = State::Response;
                self.fresh = false;
                self.buf.drain(..RESPONSE_OPEN.len());
                return;
            }
            if self.buf.starts_with(TOOLS_OPEN) {
                self.state = State::Tools;
                self.fresh = true;
                self.buf.drain(..TOOLS_OPEN.len());
                return;
            }
            if self.buf.starts_with(MESSAGE_CLOSE) || self.buf.starts_with(END_OF_MSG) {
                self.state = State::Done;
                self.buf.clear();
                return;
            }
            let candidates: &[&str] = if allow_response {
                &[RESPONSE_OPEN, TOOLS_OPEN, MESSAGE_CLOSE, END_OF_MSG]
            } else {
                &[TOOLS_OPEN, MESSAGE_CLOSE, END_OF_MSG]
            };
            if !flush && candidates.iter().any(|m| m.starts_with(self.buf.as_str())) {
                return; // 还是某标记的前缀，等
            }
            // 散文本：吐到下一个 `<|` 之前（或整段）
            let next = self.buf[1..].find("<|").map(|p| p + 1);
            match next {
                Some(p) if p > 0 => {
                    let s: String = self.buf.drain(..p).collect();
                    d.content.push_str(&s);
                }
                _ if flush => {
                    let s = std::mem::take(&mut self.buf);
                    d.content.push_str(&s);
                    return;
                }
                _ => {
                    // 以 `<|` 开头但不是任何已知标记前缀：吐一个字符，继续
                    if self.buf.starts_with("<|")
                        && !candidates.iter().any(|m| m.starts_with(self.buf.as_str()))
                    {
                        let c = self.buf.chars().next().unwrap();
                        self.buf.drain(..c.len_utf8());
                        d.content.push(c);
                    } else {
                        return;
                    }
                }
            }
        }
    }

    /// tools 通道：每个完整 `<|open|>call …<|sep|> … <|close|>call<|sep|>` 吐一个 tool_call；`<|close|>tools<|sep|>` 结束。
    fn drain_tools(&mut self, d: &mut Delta) {
        loop {
            if let Some(stripped) = self.buf.strip_prefix(TOOLS_CLOSE) {
                self.buf = stripped.to_string();
                self.state = State::AfterResponse;
                self.fresh = true;
                return;
            }
            if self.buf.starts_with(MESSAGE_CLOSE) || self.buf.starts_with(END_OF_MSG) {
                self.state = State::Done;
                self.buf.clear();
                return;
            }
            let Some(call_start) = self.buf.find(&format!("{OPEN}call")) else { return };
            let Some(close_pos) = self.buf[call_start..].find(CALL_CLOSE).map(|p| p + call_start) else {
                return;
            };
            let block = self.buf[call_start + OPEN.len()..close_pos].to_string();
            let block = block.as_str();
            // block = "call tool=… index=…<|sep|>body"
            let Some(sep) = block.find(SEP) else {
                self.buf.drain(..close_pos + CALL_CLOSE.len());
                continue;
            };
            let attrs = parse_attrs(&block[..sep]);
            let body = &block[sep + SEP.len()..];
            let name = attr(&attrs, "tool").cloned().unwrap_or_default();
            let args = parse_arguments(body);
            self.buf.drain(..close_pos + CALL_CLOSE.len());
            if name.is_empty() {
                continue;
            }
            d.tool_calls.push(ToolCallDelta {
                index: self.tc_index,
                name: Some(name),
                arguments: Some(serde_json::to_string(&Value::Object(args)).unwrap_or_else(|_| "{}".into())),
            });
            self.tc_index += 1;
        }
    }
}

/// `key="value"` 属性列表（值已在编码侧转义，这里反转义 `&quot;` `&amp;`）。
fn parse_attrs(s: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        while i < b.len() && !(b[i].is_ascii_alphanumeric() || b[i] == b'_') {
            i += 1;
        }
        let ks = i;
        while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
            i += 1;
        }
        if ks == i {
            break;
        }
        let key = &s[ks..i];
        if !s[i..].starts_with("=\"") {
            continue;
        }
        i += 2;
        let vs = i;
        while i < b.len() && b[i] != b'"' {
            i += 1;
        }
        let val = &s[vs..i];
        i += 1;
        out.push((key.to_string(), unescape_attr(val)));
    }
    out
}

fn attr<'a>(attrs: &'a [(String, String)], k: &str) -> Option<&'a String> {
    attrs.iter().find(|(key, _)| key == k).map(|(_, v)| v)
}

pub fn unescape_attr(v: &str) -> String {
    v.replace("&quot;", "\"").replace("&amp;", "&")
}

/// body 里的 `<|open|>argument key="…" type="…"<|sep|>VAL<|close|>argument<|sep|>` 序列 → JSON object。
fn parse_arguments(body: &str) -> serde_json::Map<String, Value> {
    let mut args = serde_json::Map::new();
    let open_arg = format!("{OPEN}argument");
    let mut rest = body;
    while let Some(p) = rest.find(&open_arg) {
        let after = &rest[p + open_arg.len()..];
        let Some(sep) = after.find(SEP) else { break };
        let attrs = parse_attrs(&after[..sep]);
        let val_start = &after[sep + SEP.len()..];
        let Some(end) = val_start.find(ARG_CLOSE) else { break };
        let raw = &val_start[..end];
        let key = attr(&attrs, "key").cloned().unwrap_or_default();
        let ty = attr(&attrs, "type").cloned().unwrap_or_else(|| "string".into());
        let v = if ty == "string" {
            Value::String(raw.to_string())
        } else {
            serde_json::from_str::<Value>(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
        };
        args.insert(key, v);
        rest = &val_start[end + ARG_CLOSE.len()..];
    }
    args
}

/// 尾部与任一标记真前缀的最长重叠（标记全 ASCII，按字节切安全）。
fn partial_overlap(text: &str, markers: &[&str]) -> usize {
    let mut best = 0;
    for m in markers {
        let max = (m.len() - 1).min(text.len());
        for n in (1..=max).rev() {
            if text.is_char_boundary(text.len() - n) && text.ends_with(&m[..n]) {
                best = best.max(n);
                break;
            }
        }
    }
    best
}

/// 剥掉散文本里的完整 XTML 标记（flush 时用）。
fn strip_markers(s: &str) -> String {
    let mut out = s.to_string();
    for m in [RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE, END_OF_MSG, TOOLS_OPEN, TOOLS_CLOSE] {
        out = out.replace(m, "");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(chunks: &[&str], thinking: bool) -> Delta {
        let mut p = OutputParser::new(thinking);
        let mut acc = Delta::default();
        for c in chunks {
            let d = p.feed(c);
            acc.reasoning.push_str(&d.reasoning);
            acc.content.push_str(&d.content);
            acc.tool_calls.extend(d.tool_calls);
        }
        let d = p.finish();
        acc.reasoning.push_str(&d.reasoning);
        acc.content.push_str(&d.content);
        acc.tool_calls.extend(d.tool_calls);
        acc
    }

    const FULL: &str = "I think.<|close|>think<|sep|><|open|>response<|sep|>Hello <|x|> world<|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool=\"get_weather\" index=\"1\"<|sep|><|open|>argument key=\"city\" type=\"string\"<|sep|>北京<|close|>argument<|sep|><|open|>argument key=\"days\" type=\"number\"<|sep|>3<|close|>argument<|sep|><|open|>argument key=\"q\" type=\"string\"<|sep|>a \"b\" & c<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>";

    #[test]
    fn whole_and_per_char_agree() {
        let whole = run(&[FULL], true);
        assert_eq!(whole.reasoning, "I think.");
        assert_eq!(whole.content, "Hello <|x|> world");
        assert_eq!(whole.tool_calls.len(), 1);
        assert_eq!(whole.tool_calls[0].name.as_deref(), Some("get_weather"));
        let args: Value = serde_json::from_str(whole.tool_calls[0].arguments.as_deref().unwrap()).unwrap();
        assert_eq!(args, serde_json::json!({"city": "北京", "days": 3, "q": "a \"b\" & c"}));
        // 逐字符喂
        let chars: Vec<String> = FULL.chars().map(|c| c.to_string()).collect();
        let refs: Vec<&str> = chars.iter().map(|s| s.as_str()).collect();
        let per = run(&refs, true);
        assert_eq!(per, whole);
        // 随机切块
        let mut seed = 12345u64;
        let mut chunks = Vec::new();
        let mut i = 0;
        let bytes = FULL.as_bytes();
        while i < bytes.len() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let mut n = 1 + (seed >> 33) as usize % 7;
            while i + n < bytes.len() && !FULL.is_char_boundary(i + n) {
                n += 1;
            }
            let end = (i + n).min(bytes.len());
            chunks.push(&FULL[i..end]);
            i = end;
        }
        assert_eq!(run(&chunks, true), whole);
    }

    #[test]
    fn redundant_open_and_missing_close() {
        let d = run(&["<|open|>think<|sep|>r<|close|>think<|sep|><|open|>response<|sep|>c"], true);
        assert_eq!((d.reasoning.as_str(), d.content.as_str()), ("r", "c"));
        let d = run(&["only reasoning"], true);
        assert_eq!((d.reasoning.as_str(), d.content.as_str()), ("only reasoning", ""));
        let d = run(&["<|open|>response<|sep|>plain<|end_of_msg|>"], false);
        assert_eq!((d.reasoning.as_str(), d.content.as_str()), ("", "plain"));
    }

    #[test]
    fn tool_call_only_when_closed() {
        let mut p = OutputParser::new(true);
        let d = p.feed("<|close|>think<|sep|><|open|>response<|sep|><|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool=\"f\" index=\"1\"<|sep|>");
        assert!(d.tool_calls.is_empty() && d.content.is_empty());
        let d = p.feed("<|close|>call<|sep|>");
        assert_eq!(d.tool_calls.len(), 1);
        assert_eq!(d.tool_calls[0].arguments.as_deref(), Some("{}"));
        let d = p.feed("<|close|>tools<|sep|><|close|>message<|sep|>");
        assert!(d.is_empty());
    }
}
