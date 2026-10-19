//! Qwen3.5 输出流式解析（沿用 Qwen3 模板族：`<think>` / `<tool_call>`）：`<think>…</think>` → reasoning；`<tool_call>\n{"name": …, "arguments": {…}}\n</tool_call>` → tool_calls。
//!
//! 与 GLM 的差别：① 生成提示 **不** 以 `<think>` 结尾（`enable_thinking=true` 时模型自己先输出 `<think>`；
//! `enable_thinking=false` 时模板已塞了空 `<think>\n\n</think>\n\n`），所以初始态是 Content，看到 `<think>` 才进 Reasoning；
//! ② tool_call 体是 JSON（vLLM `hermes` 风格 parser）。标签跨 token 边界同样用 "尾部可能是标签前缀" 的保留策略。

use serde::Serialize;

const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";
const TC_START: &str = "<tool_call>";
const TC_END: &str = "</tool_call>";

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
    Content,
    Reasoning,
    ToolCall,
}

#[derive(Debug)]
pub struct OutputParser {
    state: State,
    buf: String,
    tc_index: usize,
    /// 刚离开 think 段：吞掉紧跟的换行（HF 模板把 `</think>\n\n` 当分隔，不算 content）。
    strip_leading_newlines: bool,
    /// Reasoning 态：是否已吐出过非换行内容（开头的 `\n` 是模板分隔符，不算 reasoning）。
    reasoning_started: bool,
}

impl Default for OutputParser {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputParser {
    pub fn new() -> Self {
        Self {
            state: State::Content,
            buf: String::new(),
            tc_index: 0,
            strip_leading_newlines: true,
            reasoning_started: false,
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
        d
    }

    fn emit_content(&mut self, d: &mut Delta, s: &str) {
        let s = if self.strip_leading_newlines { s.trim_start_matches('\n') } else { s };
        if !s.is_empty() {
            self.strip_leading_newlines = false;
            d.content.push_str(s);
        }
    }

    fn drain(&mut self, d: &mut Delta, eof: bool) {
        loop {
            match self.state {
                State::Content => {
                    let a = self.buf.find(THINK_START);
                    let b = self.buf.find(TC_START);
                    match (a, b) {
                        (Some(i), Some(j)) if j < i => {
                            let head = self.buf[..j].to_string();
                            self.emit_content(d, &head);
                            self.buf.drain(..j + TC_START.len());
                            self.state = State::ToolCall;
                        }
                        (Some(i), _) => {
                            let head = self.buf[..i].to_string();
                            self.emit_content(d, &head);
                            self.buf.drain(..i + THINK_START.len());
                            self.state = State::Reasoning;
                        }
                        (None, Some(j)) => {
                            let head = self.buf[..j].to_string();
                            self.emit_content(d, &head);
                            self.buf.drain(..j + TC_START.len());
                            self.state = State::ToolCall;
                        }
                        (None, None) => {
                            let keep =
                                if eof { 0 } else { tail_prefix_len(&self.buf, &[THINK_START, TC_START]) };
                            let emit = self.buf.len() - keep;
                            let head = self.buf[..emit].to_string();
                            self.emit_content(d, &head);
                            self.buf.drain(..emit);
                            return;
                        }
                    }
                }
                State::Reasoning => {
                    if let Some(i) = self.buf.find(THINK_END) {
                        d.reasoning.push_str(self.buf[..i].trim_start_matches('\n').trim_end_matches('\n'));
                        self.buf.drain(..i + THINK_END.len());
                        self.state = State::Content;
                        self.strip_leading_newlines = true;
                        self.reasoning_started = false;
                        continue;
                    }
                    // 尾部的 `\n` 可能是 `</think>` 前的分隔符（HF 模板 `\n</think>`），和未完成标签前缀一样先保留；eof 时丢弃
                    let keep = if eof {
                        0
                    } else {
                        let tag = tail_prefix_len(&self.buf, &[THINK_END]);
                        let nl = self.buf[..self.buf.len() - tag].len()
                            - self.buf[..self.buf.len() - tag].trim_end_matches('\n').len();
                        tag + nl
                    };
                    let emit = self.buf.len() - keep;
                    // 开头的换行是模板分隔符，不算 reasoning
                    let s = &self.buf[..emit];
                    let s = if eof { s.trim_end_matches('\n') } else { s };
                    let s = if d.reasoning.is_empty() && !self.reasoning_started {
                        s.trim_start_matches('\n')
                    } else {
                        s
                    };
                    if !s.is_empty() {
                        self.reasoning_started = true;
                    }
                    d.reasoning.push_str(s);
                    self.buf.drain(..emit);
                    return;
                }
                State::ToolCall => {
                    if let Some(i) = self.buf.find(TC_END) {
                        let body = self.buf[..i].trim().to_string();
                        self.buf.drain(..i + TC_END.len());
                        self.emit_tool_call(d, &body);
                        self.state = State::Content;
                        self.strip_leading_newlines = true;
                        continue;
                    }
                    if eof {
                        self.buf.clear(); // 未闭合，丢弃
                    }
                    return;
                }
            }
        }
    }

    fn emit_tool_call(&mut self, d: &mut Delta, body: &str) {
        let (name, args) = match serde_json::from_str::<serde_json::Value>(body) {
            Ok(j) => {
                let name = j["name"].as_str().unwrap_or("").to_string();
                let args = match &j["arguments"] {
                    serde_json::Value::Null => "{}".to_string(),
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                (name, args)
            }
            Err(_) => (String::new(), body.to_string()),
        };
        d.tool_calls.push(ToolCallDelta { index: self.tc_index, name: Some(name), arguments: Some(args) });
        self.tc_index += 1;
    }
}

fn tail_prefix_len(s: &str, tags: &[&str]) -> usize {
    let mut best = 0;
    for tag in tags {
        let max = tag.len().saturating_sub(1).min(s.len());
        for n in (1..=max).rev() {
            if s.is_char_boundary(s.len() - n) && tag.starts_with(&s[s.len() - n..]) {
                best = best.max(n);
                break;
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_whole(s: &str) -> Delta {
        let mut p = OutputParser::new();
        let mut d = p.feed(s);
        let e = p.finish();
        d.reasoning.push_str(&e.reasoning);
        d.content.push_str(&e.content);
        d.tool_calls.extend(e.tool_calls);
        d
    }

    fn run_all_splits(s: &str) -> Delta {
        let whole = run_whole(s);
        let bounds: Vec<usize> = s.char_indices().map(|(i, _)| i).chain(std::iter::once(s.len())).collect();
        for &cut in &bounds {
            let mut p = OutputParser::new();
            let mut d = p.feed(&s[..cut]);
            let d2 = p.feed(&s[cut..]);
            let e = p.finish();
            for x in [d2, e] {
                d.reasoning.push_str(&x.reasoning);
                d.content.push_str(&x.content);
                d.tool_calls.extend(x.tool_calls);
            }
            assert_eq!(d, whole, "split at {cut}");
        }
        let mut p = OutputParser::new();
        let mut d = Delta::default();
        for ch in s.chars() {
            let x = p.feed(&ch.to_string());
            d.reasoning.push_str(&x.reasoning);
            d.content.push_str(&x.content);
            d.tool_calls.extend(x.tool_calls);
        }
        let e = p.finish();
        d.reasoning.push_str(&e.reasoning);
        d.content.push_str(&e.content);
        d.tool_calls.extend(e.tool_calls);
        assert_eq!(d, whole, "char-by-char");
        whole
    }

    #[test]
    fn parser_think_then_answer() {
        let d = run_all_splits("<think>\n好的，用户问 1+1。\n</think>\n\n2");
        assert_eq!(d.reasoning, "好的，用户问 1+1。");
        assert_eq!(d.content, "2");
        assert!(d.tool_calls.is_empty());
    }

    #[test]
    fn parser_no_think_plain_answer() {
        // enable_thinking=false：模板已放空 think，模型直接出正文
        let d = run_all_splits("The answer is 2.");
        assert_eq!(d.reasoning, "");
        assert_eq!(d.content, "The answer is 2.");
    }

    #[test]
    fn parser_empty_think_block_from_template_echo() {
        let d = run_all_splits("<think>\n\n</think>\n\nHello");
        assert_eq!(d.reasoning, "");
        assert_eq!(d.content, "Hello");
    }

    #[test]
    fn parser_tool_call_json() {
        let d = run_all_splits(
            "<think>\nneed weather\n</think>\n\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Beijing\"}}\n</tool_call>",
        );
        assert_eq!(d.reasoning, "need weather");
        assert_eq!(d.content, "");
        assert_eq!(d.tool_calls.len(), 1);
        assert_eq!(d.tool_calls[0].name.as_deref(), Some("get_weather"));
        let a: serde_json::Value =
            serde_json::from_str(d.tool_calls[0].arguments.as_deref().unwrap()).unwrap();
        assert_eq!(a["city"], "Beijing");
    }

    #[test]
    fn parser_two_tool_calls_and_text() {
        let d = run_all_splits(
            "Let me check.\n<tool_call>\n{\"name\": \"a\", \"arguments\": {\"x\": 1}}\n</tool_call>\n<tool_call>\n{\"name\": \"b\", \"arguments\": {}}\n</tool_call>\ndone",
        );
        assert_eq!(d.content, "Let me check.\ndone");
        assert_eq!(d.tool_calls.len(), 2);
        assert_eq!(d.tool_calls[1].index, 1);
        assert_eq!(d.tool_calls[1].name.as_deref(), Some("b"));
        assert_eq!(d.tool_calls[1].arguments.as_deref(), Some("{}"));
    }

    #[test]
    fn parser_unclosed_tool_call_dropped() {
        let d = run_whole("x<tool_call>\n{\"name\": \"a\"");
        assert_eq!(d.content, "x");
        assert!(d.tool_calls.is_empty());
    }

    #[test]
    fn parser_holds_back_tag_prefix() {
        let mut p = OutputParser::new();
        let d = p.feed("abc<thi");
        assert_eq!(d.content, "abc");
        let d = p.feed("nk>\nr</think>\n\nz");
        assert_eq!(d.reasoning, "r");
        assert_eq!(d.content, "z");
    }
}
