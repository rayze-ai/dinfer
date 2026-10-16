//! GLM-5.3 输出流式解析：把模型 token → 文本流拆成 `reasoning_content` / `content` / `tool_calls` 三类增量。
//!
//! 格式（对照 vLLM `parser/glm47_moe.py`，GLM-5.3 复用）：
//! - 生成提示以 `<think>` 结尾，所以输出一开始就在思考段；`</think>` 之后是回复。若模型没输出 `</think>`，
//!   全部算 reasoning（vLLM 同）。
//! - 工具调用：`<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>…</tool_call>`，可多个；NAME 后可直接
//!   接 `<arg_key>`（无换行），也可能有空白。`arg_value` 原文：按 tools schema，字符串类型原样，其他类型是 JSON。
//!   这里不做类型推断（server 层拿到 tools 后自己决定是否 `json.loads`）——输出 `arguments` 为 JSON 对象字符串，
//!   值一律先尝试 `serde_json` 解析，失败则当字符串（与 vLLM `_glm47_arg_converter` 的 partial=False 行为一致）。
//! - 流式：标签可能被切在任意 token 边界 → 内部用 "尾部可能是未完成标签前缀" 的保留策略：文本末尾若是某个
//!   标签的真前缀就先不吐出。
//!
//! 单测覆盖：纯文本 / 只 think / think+content / 1 个 & 2 个 tool_call / 值为 JSON 对象或字符串 / 标签被切在每个字符位置。

use serde::Serialize;

const THINK_END: &str = "</think>";
const TC_START: &str = "<tool_call>";
const TC_END: &str = "</tool_call>";
const AK_START: &str = "<arg_key>";
const AK_END: &str = "</arg_key>";
const AV_START: &str = "<arg_value>";
const AV_END: &str = "</arg_value>";
const TAGS: [&str; 7] = [THINK_END, TC_START, TC_END, AK_START, AK_END, AV_START, AV_END];

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ToolCallDelta {
    /// 本次输出里第几个 tool call（0 起）。
    pub index: usize,
    /// 首次出现时带 name。
    pub name: Option<String>,
    /// `arguments` 的增量（JSON 文本片段）。完整 JSON 在 `</tool_call>` 时一次给出（见 `Delta::tool_calls`）。
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
    Reasoning,
    Content,
    /// `<tool_call>` 之后、第一个 `<arg_key>` 之前：读 name。
    ToolName,
    /// 在 tool_call 内、各 arg 之间（期待 `<arg_key>` 或 `</tool_call>`）。
    ToolBetween,
    ArgKey,
    /// `</arg_key>` 之后、`<arg_value>` 之前（允许空白）。
    ArgAfterKey,
    ArgValue,
}

#[derive(Debug)]
pub struct OutputParser {
    state: State,
    /// 尚未消费的文本（可能以未完成标签前缀结尾）。
    buf: String,
    /// 当前 tool call 的累积。
    tc_index: usize,
    tc_name: String,
    tc_args: Vec<(String, String)>,
    cur_key: String,
    cur_val: String,
    /// 一旦进入 content，是否已经吐过（用于 finish 时判断）。
    pub finished: bool,
}

impl Default for OutputParser {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputParser {
    /// 生成提示以 `<think>` 结尾 → 初始态是 Reasoning。
    pub fn new() -> Self {
        Self {
            state: State::Reasoning,
            buf: String::new(),
            tc_index: 0,
            tc_name: String::new(),
            tc_args: Vec::new(),
            cur_key: String::new(),
            cur_val: String::new(),
            finished: false,
        }
    }

    /// 喂一段新文本（一个或多个 token 的 detokenize 结果），返回可以立刻下发的增量。
    pub fn feed(&mut self, text: &str) -> Delta {
        self.buf.push_str(text);
        let mut d = Delta::default();
        self.drain(&mut d, false);
        d
    }

    /// 流结束：把保留的尾巴全部按当前状态吐出（未闭合的 tool_call 丢弃其残片，与 vLLM 一致：不产出半个 call）。
    pub fn finish(&mut self) -> Delta {
        let mut d = Delta::default();
        self.drain(&mut d, true);
        self.finished = true;
        d
    }

    fn drain(&mut self, d: &mut Delta, eof: bool) {
        loop {
            match self.state {
                State::Reasoning => {
                    if let Some(i) = self.buf.find(THINK_END) {
                        d.reasoning.push_str(&self.buf[..i]);
                        self.buf.drain(..i + THINK_END.len());
                        self.state = State::Content;
                        continue;
                    }
                    let keep = if eof { 0 } else { tail_prefix_len(&self.buf, &[THINK_END]) };
                    let emit = self.buf.len() - keep;
                    d.reasoning.push_str(&self.buf[..emit]);
                    self.buf.drain(..emit);
                    return;
                }
                State::Content => {
                    if let Some(i) = self.buf.find(TC_START) {
                        d.content.push_str(&self.buf[..i]);
                        self.buf.drain(..i + TC_START.len());
                        self.tc_name.clear();
                        self.tc_args.clear();
                        self.state = State::ToolName;
                        continue;
                    }
                    let keep = if eof { 0 } else { tail_prefix_len(&self.buf, &[TC_START]) };
                    let emit = self.buf.len() - keep;
                    d.content.push_str(&self.buf[..emit]);
                    self.buf.drain(..emit);
                    return;
                }
                State::ToolName => {
                    // name 到第一个 <arg_key> 或 </tool_call>（无参数调用）
                    let a = self.buf.find(AK_START);
                    let b = self.buf.find(TC_END);
                    match (a, b) {
                        (Some(i), Some(j)) if j < i => {
                            self.take_name_then(d, j, TC_END, State::ToolBetween, true)
                        }
                        (Some(i), _) => self.take_name_then(d, i, AK_START, State::ArgKey, false),
                        (None, Some(j)) => self.take_name_then(d, j, TC_END, State::ToolBetween, true),
                        (None, None) => {
                            if eof {
                                self.buf.clear(); // 未闭合，丢弃
                            }
                            return;
                        }
                    }
                    if self.state == State::ToolBetween && self.tc_name.is_empty() {
                        // 已在 take_name_then 里闭合并输出
                    }
                    continue;
                }
                State::ToolBetween => {
                    // 期待 <arg_key> 或 </tool_call>；中间允许空白
                    let trimmed = self.buf.trim_start();
                    let skipped = self.buf.len() - trimmed.len();
                    if trimmed.starts_with(AK_START) {
                        self.buf.drain(..skipped + AK_START.len());
                        self.state = State::ArgKey;
                        continue;
                    }
                    if trimmed.starts_with(TC_END) {
                        self.buf.drain(..skipped + TC_END.len());
                        self.emit_tool_call(d);
                        self.state = State::Content;
                        continue;
                    }
                    if eof {
                        self.buf.clear();
                        return;
                    }
                    // 还不足以判断（可能是标签前缀或空白）
                    if trimmed.is_empty() || AK_START.starts_with(trimmed) || TC_END.starts_with(trimmed) {
                        return;
                    }
                    // 非法内容：当作 content 原样吐出，退回 Content 态（容错）
                    self.state = State::Content;
                    continue;
                }
                State::ArgKey => {
                    if let Some(i) = self.buf.find(AK_END) {
                        self.cur_key = self.buf[..i].to_string();
                        self.buf.drain(..i + AK_END.len());
                        self.state = State::ArgAfterKey;
                        continue;
                    }
                    if eof {
                        self.buf.clear();
                    }
                    return;
                }
                State::ArgAfterKey => {
                    let trimmed = self.buf.trim_start();
                    let skipped = self.buf.len() - trimmed.len();
                    if trimmed.starts_with(AV_START) {
                        self.buf.drain(..skipped + AV_START.len());
                        self.cur_val.clear();
                        self.state = State::ArgValue;
                        continue;
                    }
                    if eof {
                        self.buf.clear();
                    }
                    return;
                }
                State::ArgValue => {
                    if let Some(i) = self.buf.find(AV_END) {
                        self.cur_val.push_str(&self.buf[..i]);
                        self.buf.drain(..i + AV_END.len());
                        self.tc_args
                            .push((std::mem::take(&mut self.cur_key), std::mem::take(&mut self.cur_val)));
                        self.state = State::ToolBetween;
                        continue;
                    }
                    if eof {
                        self.buf.clear();
                    }
                    return;
                }
            }
        }
    }

    fn take_name_then(&mut self, d: &mut Delta, at: usize, tag: &str, next: State, close: bool) {
        self.tc_name = self.buf[..at].trim().to_string();
        self.buf.drain(..at + tag.len());
        self.state = next;
        if close {
            self.emit_tool_call(d);
            self.state = State::Content;
        }
    }

    /// `</tool_call>`：一次性产出 name + arguments（JSON 对象文本）。
    fn emit_tool_call(&mut self, d: &mut Delta) {
        let mut obj = serde_json::Map::new();
        for (k, v) in self.tc_args.drain(..) {
            let val = match serde_json::from_str::<serde_json::Value>(v.trim()) {
                Ok(j) => j,
                Err(_) => serde_json::Value::String(v),
            };
            obj.insert(k, val);
        }
        let args = serde_json::Value::Object(obj).to_string();
        d.tool_calls.push(ToolCallDelta {
            index: self.tc_index,
            name: Some(std::mem::take(&mut self.tc_name)),
            arguments: Some(args),
        });
        self.tc_index += 1;
    }
}

/// `s` 末尾与任一 tag 的真前缀重合的最长长度（用于流式保留）。
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
    let _ = TAGS; // 文档用
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

    /// 把 s 在每个字符边界切成两段分别 feed，结果必须与整体一致（标签跨 token 边界）。
    fn run_all_splits(s: &str) -> Delta {
        let whole = run_whole(s);
        let bounds: Vec<usize> = s.char_indices().map(|(i, _)| i).chain(std::iter::once(s.len())).collect();
        for &cut in &bounds {
            let mut p = OutputParser::new();
            let mut d = p.feed(&s[..cut]);
            let d2 = p.feed(&s[cut..]);
            let e = p.finish();
            d.reasoning.push_str(&d2.reasoning);
            d.reasoning.push_str(&e.reasoning);
            d.content.push_str(&d2.content);
            d.content.push_str(&e.content);
            d.tool_calls.extend(d2.tool_calls);
            d.tool_calls.extend(e.tool_calls);
            assert_eq!(d, whole, "split at {cut}");
        }
        // 也按单字符逐个 feed
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
    fn parser_plain_reasoning_only() {
        let d = run_all_splits("I am thinking…");
        assert_eq!(d.reasoning, "I am thinking…");
        assert_eq!(d.content, "");
        assert!(d.tool_calls.is_empty());
    }

    #[test]
    fn parser_think_then_content() {
        let d = run_all_splits("step 1</think>Hello <b>world</b>");
        assert_eq!(d.reasoning, "step 1");
        assert_eq!(d.content, "Hello <b>world</b>");
    }

    #[test]
    fn parser_single_tool_call_json_and_string_values() {
        let d = run_all_splits(
            "plan</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value><arg_key>days</arg_key><arg_value>3</arg_value><arg_key>opts</arg_key><arg_value>{\"unit\": \"c\"}</arg_value></tool_call>",
        );
        assert_eq!(d.reasoning, "plan");
        assert_eq!(d.content, "");
        assert_eq!(d.tool_calls.len(), 1);
        let tc = &d.tool_calls[0];
        assert_eq!(tc.index, 0);
        assert_eq!(tc.name.as_deref(), Some("get_weather"));
        let args: serde_json::Value = serde_json::from_str(tc.arguments.as_deref().unwrap()).unwrap();
        assert_eq!(args["city"], "Beijing");
        assert_eq!(args["days"], 3);
        assert_eq!(args["opts"]["unit"], "c");
    }

    #[test]
    fn parser_two_tool_calls_with_text_around() {
        let d = run_all_splits(
            "</think>Let me check.\n<tool_call>a<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>\n<tool_call>b\n<arg_key>y</arg_key>\n<arg_value>two</arg_value>\n</tool_call>done",
        );
        assert_eq!(d.reasoning, "");
        assert_eq!(d.content, "Let me check.\n\ndone");
        assert_eq!(d.tool_calls.len(), 2);
        assert_eq!(d.tool_calls[0].name.as_deref(), Some("a"));
        assert_eq!(d.tool_calls[0].arguments.as_deref(), Some("{\"x\":1}"));
        assert_eq!(d.tool_calls[1].index, 1);
        assert_eq!(d.tool_calls[1].name.as_deref(), Some("b"));
        assert_eq!(d.tool_calls[1].arguments.as_deref(), Some("{\"y\":\"two\"}"));
    }

    #[test]
    fn parser_tool_call_without_args() {
        let d = run_all_splits("</think><tool_call>noop</tool_call>");
        assert_eq!(d.tool_calls.len(), 1);
        assert_eq!(d.tool_calls[0].name.as_deref(), Some("noop"));
        assert_eq!(d.tool_calls[0].arguments.as_deref(), Some("{}"));
    }

    #[test]
    fn parser_unclosed_tool_call_is_dropped_at_eof() {
        let d = run_whole("</think>text<tool_call>a<arg_key>x</arg_key><arg_value>1");
        assert_eq!(d.content, "text");
        assert!(d.tool_calls.is_empty());
    }

    #[test]
    fn parser_angle_bracket_in_value_and_lt_in_content() {
        let d = run_all_splits(
            "</think>1 < 2 and <tool_call>f<arg_key>k</arg_key><arg_value>a<b</arg_value></tool_call>",
        );
        assert_eq!(d.content, "1 < 2 and ");
        assert_eq!(d.tool_calls[0].arguments.as_deref(), Some("{\"k\":\"a<b\"}"));
    }

    #[test]
    fn parser_streaming_holds_back_tag_prefix() {
        let mut p = OutputParser::new();
        let d = p.feed("abc</thi");
        assert_eq!(d.reasoning, "abc"); // "</thi" 可能是 </think> 前缀，先不吐
        let d = p.feed("nk>xyz");
        assert_eq!(d.reasoning, "");
        assert_eq!(d.content, "xyz");
        let mut p = OutputParser::new();
        let d = p.feed("abc</thi");
        assert_eq!(d.reasoning, "abc");
        let d = p.finish(); // 不是标签，eof 时原样吐出
        assert_eq!(d.reasoning, "</thi");
    }
}
