//! 增量 stop 字符串匹配（语义对齐 vLLM `check_stop_strings` + `stop_buffer_length`，实现只扫增量）。
//!
//! - 每次 `push(new_text)` 只在 `保留尾巴(≤ max_len-1 字节) + new_text` 上找，O(新增 + max_len)，不重扫全文。
//! - 多个 stop 同时命中：**结束位置最早**的优先，平局按 stop 列表顺序（等价逐 token 追加时的判定）。
//! - 可能是某个 stop 前缀的尾巴先扣住不下发（最多 max_len-1 字节，流式下发延迟仅此而已）；命中后
//!   `include_stop_str_in_output=false` 截到 stop 之前，true 截到 stop 之后；`finish()` 把扣住的尾巴放出。
//! - 没有 stop 时 `push` 直接原样返回，零开销。

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StopHit {
    /// 命中的 stop 在列表里的下标。
    pub index: usize,
}

#[derive(Debug, Clone)]
pub struct StopChecker {
    stops: Vec<String>,
    include_in_output: bool,
    /// 所有 stop 的最大字节长；0 = 无 stop。
    max_len: usize,
    /// 扣住未下发的尾巴（可能是某个 stop 的前缀）。
    pending: String,
    stopped: bool,
}

impl StopChecker {
    pub fn new(stops: &[String], include_in_output: bool) -> Self {
        let stops: Vec<String> = stops.iter().filter(|s| !s.is_empty()).cloned().collect();
        let max_len = stops.iter().map(|s| s.len()).max().unwrap_or(0);
        Self { stops, include_in_output, max_len, pending: String::new(), stopped: false }
    }

    pub fn is_empty(&self) -> bool {
        self.max_len == 0
    }
    pub fn stopped(&self) -> bool {
        self.stopped
    }

    /// 喂新文本；返回 (可下发文本, 是否命中)。命中后再 push 返回空。
    pub fn push(&mut self, new_text: &str) -> (String, Option<StopHit>) {
        if self.stopped || new_text.is_empty() {
            return (String::new(), None);
        }
        if self.max_len == 0 {
            return (new_text.to_string(), None);
        }
        let prev_len = self.pending.len();
        self.pending.push_str(new_text);
        let text = &self.pending;
        // 只找"能跨进新增部分"的匹配：起点 ≥ prev_len - (stop_len - 1)
        let mut best: Option<(usize, usize, usize)> = None; // (end, start, idx)
        for (i, s) in self.stops.iter().enumerate() {
            let from = prev_len.saturating_sub(s.len() - 1);
            let from = floor_char_boundary(text, from);
            if let Some(off) = text[from..].find(s.as_str()) {
                let start = from + off;
                let end = start + s.len();
                if best.is_none_or(|(e, _, _)| end < e) {
                    best = Some((end, start, i));
                }
            }
        }
        if let Some((end, start, idx)) = best {
            self.stopped = true;
            let cut = if self.include_in_output { end } else { start };
            let out = text[..cut].to_string();
            self.pending.clear();
            return (out, Some(StopHit { index: idx }));
        }
        // 无命中：放出除可能的前缀尾巴之外的部分（include 模式同样要扣，否则 stop 跨 token 时前半截已下发无法收回）
        let keep = self.max_len - 1;
        if text.len() <= keep {
            return (String::new(), None);
        }
        let cut = floor_char_boundary(text, text.len() - keep);
        let out = self.pending[..cut].to_string();
        self.pending.drain(..cut);
        (out, None)
    }

    /// 流结束：把扣住的尾巴放出。
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(stops: &[&str], include: bool, pieces: &[&str]) -> (String, Option<usize>) {
        let stops: Vec<String> = stops.iter().map(|s| s.to_string()).collect();
        let mut c = StopChecker::new(&stops, include);
        let mut out = String::new();
        let mut hit = None;
        for p in pieces {
            let (t, h) = c.push(p);
            out.push_str(&t);
            if let Some(h) = h {
                hit = Some(h.index);
                break;
            }
        }
        if hit.is_none() {
            out.push_str(&c.finish());
        }
        (out, hit)
    }

    #[test]
    fn no_stops_is_passthrough() {
        assert_eq!(run(&[], false, &["a", "b", "c"]), ("abc".into(), None));
    }

    #[test]
    fn stop_across_tokens() {
        // "<|e" + "nd|>" 跨 token 组成 stop
        assert_eq!(run(&["<|end|>"], false, &["hello ", "<|e", "nd|>", " tail"]), ("hello ".into(), Some(0)));
    }

    #[test]
    fn earliest_completing_stop_wins_then_list_order() {
        // 一步里同时出现 "xyz"（先结束）和 "yz!"（后结束）→ 选 xyz
        assert_eq!(run(&["yz!", "xyz"], false, &["abxyz!"]), ("ab".into(), Some(1)));
        // 同位置同长度：列表顺序
        assert_eq!(run(&["ab", "ab"], false, &["xab"]), ("x".into(), Some(0)));
    }

    #[test]
    fn include_stop_str_in_output_both_ways() {
        assert_eq!(run(&["END"], true, &["foo E", "ND bar"]), ("foo END".into(), Some(0)));
        assert_eq!(run(&["END"], false, &["foo E", "ND bar"]), ("foo ".into(), Some(0)));
    }

    #[test]
    fn stop_exactly_at_end_and_never_appears() {
        assert_eq!(run(&["."], false, &["done", "."]), ("done".into(), Some(0)));
        // 永不出现：尾巴（可能的前缀）在 finish 时放出，内容完整
        assert_eq!(run(&["</s>"], false, &["a<", "/s", "x"]), ("a</sx".into(), None));
    }

    #[test]
    fn holds_back_only_possible_prefix_and_is_char_safe() {
        let stops = vec!["STOP".to_string()];
        let mut c = StopChecker::new(&stops, false);
        let (t, h) = c.push("hello wor");
        assert!(h.is_none());
        assert_eq!(t, "hello ");
        assert_eq!(c.pending, "wor"); // max_len-1 = 3
                                      // 多字节：不在字符中间切
        let mut c = StopChecker::new(&stops, false);
        let (t, _) = c.push("你好世界");
        assert!(t.is_char_boundary(t.len()) && ("你好世界".starts_with(&t)));
        assert_eq!(format!("{t}{}", c.finish()), "你好世界");
    }
}
