//! Kimi-K3 tokenizer：tiktoken BPE（`tiktoken.model`）+ 自定义 `pat_str` + 256 特殊 token。
//! 口径 = HF `tokenization_kimi.py::TikTokenTokenizer`（vLLM `tokenizer_mode="kimi_k3"` 也用它），逐条对应：
//!
//! - ranks 文件每行 `<base64 bytes> <rank>`；特殊 token 空间 `[len(ranks), len(ranks)+256)`，名字来自
//!   `tokenizer_config.json::added_tokens_decoder`（16 个有名），其余为 `<|reserved_token_{i}|>`。
//! - `encode(text, allow_special)`：`allow_special=true` = tiktoken `allowed_special="all"`（文本里出现的特殊 token 字样
//!   编成特殊 id）；`false` = `disallowed_special=()`（用户文本里的 `<|open|>` 只能是普通 BPE，绝不能变控制 token）。
//! - 长文本切分 **必须复刻**（影响 BPE 边界）：400k 字符窗口，再按"连续（非）空白 run ≤ 25k 字符"切，每段独立 BPE。
//! - BOS 不自动前置；`decode` 以 bytes 拼接后 UTF-8 lossy；增量 detokenize 只在 UTF-8 边界吐字。
//!
//! 预分词正则由 `fancy-regex` 执行（tiktoken 自身也是 `fancy_regex`，类交集 `&&` 与 `(?!\S)` 语义一致；
//! Python 的 `regex` 模块默认不认 `&&`，不能拿它当参照）。

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;

/// `tokenization_kimi.py::pat_str`，逐字照抄。
pub const PAT_STR: &str = concat!(
    r"[\p{Han}]+",
    "|",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    "|",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    "|",
    r"\p{N}{1,3}",
    "|",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*",
    "|",
    r"\s*[\r\n]+",
    "|",
    r"\s+(?!\S)",
    "|",
    r"\s+",
);

/// `TIKTOKEN_MAX_ENCODE_CHARS`：按字符数切窗。
const MAX_ENCODE_CHARS: usize = 400_000;
/// `MAX_NO_WHITESPACES_CHARS`：连续（非）空白 run 上限。
const MAX_RUN_CHARS: usize = 25_000;
/// `num_reserved_special_tokens`。
pub const NUM_SPECIAL: u32 = 256;

pub struct Tokenizer {
    /// bytes → rank（= id）。
    encoder: HashMap<Vec<u8>, u32>,
    /// id → bytes（仅基础 token）。
    decoder: Vec<Vec<u8>>,
    /// 特殊 token：名字 → id。
    special_to_id: HashMap<String, u32>,
    /// 特殊 token 名，下标 = id - n_base。
    special_names: Vec<String>,
    n_base: u32,
    pat: fancy_regex::Regex,
    /// 所有特殊 token 名的字面交替，用于 `allow_special` 分段。
    special_pat: fancy_regex::Regex,
    eos_ids: Vec<u32>,
}

impl Tokenizer {
    /// 从模型目录加载：`tiktoken.model` + `tokenizer_config.json`（有名特殊 token）+ `generation_config.json`（eos）。
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let ranks = std::fs::read_to_string(dir.join("tiktoken.model"))
            .with_context(|| format!("read {}", dir.join("tiktoken.model").display()))?;
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer_config.json"))?)
                .context("tokenizer_config.json")?;
        let mut named: HashMap<u32, String> = HashMap::new();
        if let Some(m) = cfg["added_tokens_decoder"].as_object() {
            for (k, v) in m {
                let id: u32 = k.parse().with_context(|| format!("added_tokens_decoder key {k}"))?;
                let content = v["content"].as_str().ok_or_else(|| anyhow!("added token {k} 无 content"))?;
                named.insert(id, content.to_string());
            }
        }
        let mut eos_ids = Vec::new();
        if let Ok(s) = std::fs::read_to_string(dir.join("generation_config.json")) {
            let g: serde_json::Value = serde_json::from_str(&s).context("generation_config.json")?;
            match &g["eos_token_id"] {
                serde_json::Value::Number(n) => eos_ids.push(n.as_u64().unwrap_or(0) as u32),
                serde_json::Value::Array(a) => {
                    eos_ids.extend(a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32))
                }
                _ => {}
            }
        }
        Self::from_parts(&ranks, &named, eos_ids)
    }

    pub fn from_parts(
        ranks_text: &str,
        named_specials: &HashMap<u32, String>,
        eos_ids: Vec<u32>,
    ) -> Result<Self> {
        let b64 = base64::engine::general_purpose::STANDARD;
        let mut encoder: HashMap<Vec<u8>, u32> = HashMap::new();
        let mut pairs: Vec<(u32, Vec<u8>)> = Vec::new();
        for (ln, line) in ranks_text.lines().enumerate() {
            let line = line.trim_end();
            if line.is_empty() {
                continue;
            }
            let (tok, rank) =
                line.split_once(' ').ok_or_else(|| anyhow!("tiktoken.model 第 {} 行格式错", ln + 1))?;
            let bytes = b64.decode(tok).with_context(|| format!("tiktoken.model 第 {} 行 base64", ln + 1))?;
            let rank: u32 =
                rank.trim().parse().with_context(|| format!("tiktoken.model 第 {} 行 rank", ln + 1))?;
            pairs.push((rank, bytes));
        }
        pairs.sort_by_key(|p| p.0);
        let n_base = pairs.len() as u32;
        let mut decoder = Vec::with_capacity(pairs.len());
        for (i, (rank, bytes)) in pairs.into_iter().enumerate() {
            if rank != i as u32 {
                bail!("tiktoken.model ranks 不连续：第 {i} 个是 {rank}");
            }
            encoder.insert(bytes.clone(), rank);
            decoder.push(bytes);
        }
        let mut special_names = Vec::with_capacity(NUM_SPECIAL as usize);
        let mut special_to_id = HashMap::new();
        for i in n_base..n_base + NUM_SPECIAL {
            let name = named_specials.get(&i).cloned().unwrap_or_else(|| format!("<|reserved_token_{i}|>"));
            special_to_id.insert(name.clone(), i);
            special_names.push(name);
        }
        for (&id, name) in named_specials {
            if !(n_base..n_base + NUM_SPECIAL).contains(&id) {
                bail!("有名特殊 token {name} id {id} 不在 [{n_base}, {})", n_base + NUM_SPECIAL);
            }
        }
        // 长名优先（避免 `<|a|>` 吞掉 `<|a|>b`，当前名字表互不为前缀，排序只是保险）
        let mut alts: Vec<&String> = special_names.iter().collect();
        alts.sort_by(|a, b| b.len().cmp(&a.len()).then(a.cmp(b)));
        let special_src =
            alts.iter().map(|s| fancy_regex::escape(s).into_owned()).collect::<Vec<_>>().join("|");
        Ok(Self {
            encoder,
            decoder,
            special_to_id,
            special_names,
            n_base,
            pat: fancy_regex::Regex::new(PAT_STR).context("pat_str")?,
            special_pat: fancy_regex::Regex::new(&special_src).context("special alternation")?,
            eos_ids,
        })
    }

    pub fn n_base(&self) -> u32 {
        self.n_base
    }
    pub fn vocab_size(&self) -> usize {
        (self.n_base + NUM_SPECIAL) as usize
    }
    pub fn eos_ids(&self) -> &[u32] {
        &self.eos_ids
    }
    pub fn is_eos(&self, id: u32) -> bool {
        self.eos_ids.contains(&id)
    }
    pub fn is_special(&self, id: u32) -> bool {
        id >= self.n_base && id < self.n_base + NUM_SPECIAL
    }
    pub fn special_id(&self, name: &str) -> Option<u32> {
        self.special_to_id.get(name).copied()
    }
    /// 特殊 token 名（非特殊 id 返回 None）。
    pub fn special_name(&self, id: u32) -> Option<&str> {
        self.is_special(id).then(|| self.special_names[(id - self.n_base) as usize].as_str())
    }
    /// 单 token 的字节（特殊 token = 名字的 UTF-8）。
    pub fn token_bytes(&self, id: u32) -> Option<&[u8]> {
        if let Some(name) = self.special_name(id) {
            return Some(name.as_bytes());
        }
        self.decoder.get(id as usize).map(|v| v.as_slice())
    }
    /// 单 token 文本（logprobs 展示用；半个 UTF-8 用 lossy）。
    pub fn token_text(&self, id: u32) -> Option<String> {
        self.token_bytes(id).map(|b| String::from_utf8_lossy(b).into_owned())
    }

    /// `TikTokenTokenizer.encode(text, allow_special_tokens)`。
    pub fn encode(&self, text: &str, allow_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        self.encode_into(text, allow_special, &mut out);
        out
    }

    /// 只做普通 BPE（= `allow_special=false`）。
    pub fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        self.encode(text, false)
    }

    pub fn encode_into(&self, text: &str, allow_special: bool, out: &mut Vec<u32>) {
        // `_encode_text_piece`：400k 字符窗口 → 25k run 切分 → 每段独立编码
        for window in char_windows(text, MAX_ENCODE_CHARS) {
            for piece in split_ws_runs(window, MAX_RUN_CHARS) {
                if allow_special {
                    self.encode_with_specials(piece, out);
                } else {
                    self.bpe_text(piece, out);
                }
            }
        }
    }

    fn encode_with_specials(&self, text: &str, out: &mut Vec<u32>) {
        let mut start = 0;
        for m in self.special_pat.find_iter(text) {
            let m = match m {
                Ok(m) => m,
                Err(_) => break,
            };
            self.bpe_text(&text[start..m.start()], out);
            out.push(self.special_to_id[m.as_str()]);
            start = m.end();
        }
        self.bpe_text(&text[start..], out);
    }

    fn bpe_text(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        for m in self.pat.find_iter(text) {
            let m = match m {
                Ok(m) => m,
                Err(_) => break,
            };
            let piece = m.as_str().as_bytes();
            if let Some(&id) = self.encoder.get(piece) {
                out.push(id);
            } else {
                self.byte_pair_merge(piece, out);
            }
        }
    }

    /// tiktoken `_byte_pair_merge`：按 rank 合并直到无可合并。
    fn byte_pair_merge(&self, piece: &[u8], out: &mut Vec<u32>) {
        let rank_of = |a: usize, b: usize| -> u32 {
            if b <= piece.len() {
                self.encoder.get(&piece[a..b]).copied().unwrap_or(u32::MAX)
            } else {
                u32::MAX
            }
        };
        // parts[i] = (start, rank(piece[start..parts[i+2].start]))
        let mut parts: Vec<(usize, u32)> = Vec::with_capacity(piece.len() + 1);
        for i in 0..piece.len().saturating_sub(1) {
            parts.push((i, rank_of(i, i + 2)));
        }
        parts.push((piece.len() - 1, u32::MAX));
        parts.push((piece.len(), u32::MAX));
        loop {
            if parts.len() <= 2 {
                break;
            }
            let mut best = (u32::MAX, usize::MAX);
            for (i, p) in parts.iter().enumerate().take(parts.len() - 1) {
                if p.1 < best.0 {
                    best = (p.1, i);
                }
            }
            if best.0 == u32::MAX {
                break;
            }
            let i = best.1;
            // 合并 parts[i] 与 parts[i+1]
            parts[i].1 = if i + 3 < parts.len() { rank_of(parts[i].0, parts[i + 3].0) } else { u32::MAX };
            if i > 0 {
                parts[i - 1].1 =
                    if i + 2 < parts.len() { rank_of(parts[i - 1].0, parts[i + 2].0) } else { u32::MAX };
            }
            parts.remove(i + 1);
        }
        for w in parts.windows(2) {
            out.push(self.encoder[&piece[w[0].0..w[1].0]]);
        }
    }

    /// `decode(ids)`：bytes 拼接后 UTF-8（非法字节 → U+FFFD，同 tiktoken `errors="replace"`）。
    pub fn decode(&self, ids: &[u32]) -> String {
        self.decode_with(ids, false)
    }

    /// `skip_special=true` 时丢掉特殊 token（HF `skip_special_tokens`）。K3 解析器需要 XTML 标记字面，默认不跳。
    pub fn decode_with(&self, ids: &[u32], skip_special: bool) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if skip_special && self.is_special(id) {
                continue;
            }
            if let Some(b) = self.token_bytes(id) {
                bytes.extend_from_slice(b);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn decode_stream(&self) -> IncrementalDecoder<'_> {
        IncrementalDecoder { tok: self, pending: Vec::new(), skip_special: false }
    }
}

/// 增量 detokenize：逐 token 吐出新增文本，半个 UTF-8 字符留到下一次。
pub struct IncrementalDecoder<'a> {
    tok: &'a Tokenizer,
    pending: Vec<u8>,
    skip_special: bool,
}

impl IncrementalDecoder<'_> {
    pub fn set_skip_special(&mut self, skip: bool) {
        self.skip_special = skip;
    }

    /// 喂一个 token，返回可下发的新文本（可能为空）。
    pub fn step(&mut self, id: u32) -> String {
        if self.skip_special && self.tok.is_special(id) {
            return String::new();
        }
        if let Some(b) = self.tok.token_bytes(id) {
            self.pending.extend_from_slice(b);
        }
        self.drain(false)
    }

    /// 流结束：把残留字节按 lossy 吐出。
    pub fn finish(&mut self) -> String {
        self.drain(true)
    }

    fn drain(&mut self, flush: bool) -> String {
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let s = s.to_string();
                self.pending.clear();
                s
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if e.error_len().is_some() || flush {
                    // 非法序列（不是"未完"）：整体 lossy，不再等
                    let s = String::from_utf8_lossy(&self.pending).into_owned();
                    self.pending.clear();
                    s
                } else {
                    let s = std::str::from_utf8(&self.pending[..valid]).unwrap().to_string();
                    self.pending.drain(..valid);
                    s
                }
            }
        }
    }
}

/// Python `str.isspace()`：White_Space 之外还含 \x1c–\x1f（双向类 B/S）。
fn py_isspace(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

/// 按字符数切窗（Python `text[i:i+N]`）。
fn char_windows(s: &str, n: usize) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut count = 0;
    for (i, _) in s.char_indices() {
        if count == n {
            out.push(&s[start..i]);
            start = i;
            count = 0;
        }
        count += 1;
    }
    out.push(&s[start..]);
    out
}

/// `_split_whitespaces_or_nonwhitespaces`：每段里连续空白 / 连续非空白不超过 `max_run` 个字符。
fn split_ws_runs(s: &str, max_run: usize) -> Vec<&str> {
    let mut out = Vec::new();
    let mut run_len = 0usize;
    let mut run_is_space = s.chars().next().map(py_isspace).unwrap_or(false);
    let mut slice_start = 0usize;
    for (i, c) in s.char_indices() {
        let now_space = py_isspace(c);
        if run_is_space ^ now_space {
            run_len = 1;
            run_is_space = now_space;
        } else {
            run_len += 1;
            if run_len > max_run {
                out.push(&s[slice_start..i]);
                slice_start = i;
                run_len = 1;
            }
        }
    }
    out.push(&s[slice_start..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Tokenizer {
        // 字节级 ranks：0..=255 单字节 + 几条合并
        let b64 = base64::engine::general_purpose::STANDARD;
        let mut lines: Vec<String> =
            (0u8..=255).map(|b| format!("{} {}", b64.encode([b]), b as u32)).collect();
        for (i, tok) in ["ab", "abc", " w", "or", "ld", " wor"].iter().enumerate() {
            lines.push(format!("{} {}", b64.encode(tok.as_bytes()), 256 + i as u32));
        }
        let mut named = HashMap::new();
        named.insert(262, "<|open|>".to_string());
        named.insert(263, "<|sep|>".to_string());
        Tokenizer::from_parts(&lines.join("\n"), &named, vec![263]).unwrap()
    }

    #[test]
    fn specials_and_reserved_names() {
        let t = tiny();
        assert_eq!(t.n_base(), 262);
        assert_eq!(t.special_id("<|open|>"), Some(262));
        assert_eq!(t.special_name(264), Some("<|reserved_token_264|>"));
        assert_eq!(t.vocab_size(), 262 + 256);
        assert!(t.is_eos(263));
    }

    #[test]
    fn bpe_merges_by_rank_and_special_modes() {
        let t = tiny();
        // "abc" 直接命中；" world" → " wor" + "ld"
        assert_eq!(t.encode("abc", false), vec![257]);
        assert_eq!(t.encode(" world", false), vec![261, 260]);
        let sp = t.encode("<|open|>abc<|sep|>", true);
        assert_eq!(sp, vec![262, 257, 263]);
        let plain = t.encode("<|open|>", false);
        assert!(plain.iter().all(|&id| id < 262), "普通文本里的 <|open|> 不得变特殊 token: {plain:?}");
        assert_eq!(t.decode(&sp), "<|open|>abc<|sep|>");
        assert_eq!(t.decode_with(&sp, true), "abc");
    }

    #[test]
    fn incremental_holds_partial_utf8() {
        let t = tiny();
        let ids = t.encode("你a", false); // 你 = 3 字节，字节级 token
        assert_eq!(ids.len(), 4);
        let mut d = t.decode_stream();
        let mut got = Vec::new();
        for &id in &ids {
            got.push(d.step(id));
        }
        assert_eq!(got, vec!["", "", "你", "a"]);
    }

    #[test]
    fn ws_run_split_matches_python_semantics() {
        assert_eq!(split_ws_runs("aaaa  bb", 3), vec!["aaa", "a  bb"]);
        assert_eq!(split_ws_runs("", 3), vec![""]);
        assert_eq!(char_windows("héllo", 2), vec!["hé", "ll", "o"]);
    }
}
