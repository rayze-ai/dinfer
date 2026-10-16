//! P6-a：真 GLM-5.3 tokenizer / chat 模板 vs HF golden（`scripts/verify/gen_chat_golden.py`）。
//! 口径：模板渲染文本逐字符一致；encode 的 id 逐个一致；增量 decode 的前缀文本一致。
//! 跑法：TINY_GLM_DIR=$(realpath test/tiny_glm_dsa) GLM_TOKENIZER_DIR=/data/ref/models/glm53 \
//!       cargo test -p dinfer-glm53 --features ref-tests --test chat_parity -- --nocapture
#![cfg(feature = "ref-tests")]

use dinfer_glm53::chat::{render, ChatTemplate, RenderParams};
use dinfer_glm53::tokenizer::GlmTokenizer;
use std::path::PathBuf;

fn fixture_dir() -> PathBuf {
    let p = PathBuf::from(std::env::var("TINY_GLM_DIR").unwrap_or_else(|_| "test/tiny_glm_dsa".into()));
    if p.is_absolute() {
        p
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join(p)
    }
}
fn tok_dir() -> PathBuf {
    PathBuf::from(std::env::var("GLM_TOKENIZER_DIR").unwrap_or_else(|_| "/data/ref/models/glm53".into()))
}
fn golden() -> serde_json::Value {
    let p = fixture_dir().join("chat_golden.json");
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display())))
        .unwrap()
}

/// 两个字符串第一处不同的位置，带上下文。
fn first_diff(a: &str, b: &str) -> String {
    let i = a
        .chars()
        .zip(b.chars())
        .position(|(x, y)| x != y)
        .unwrap_or(a.chars().count().min(b.chars().count()));
    let ctx = |s: &str| -> String {
        let chars: Vec<char> = s.chars().collect();
        let lo = i.saturating_sub(30);
        let hi = (i + 30).min(chars.len());
        chars[lo..hi].iter().collect::<String>().replace('\n', "\\n")
    };
    format!("first diff at char {i}:\n  ours   …{}…\n  golden …{}…", ctx(a), ctx(b))
}

#[test]
fn tokenizer_specials_and_eos_match_hf() {
    let g = golden();
    let tok = GlmTokenizer::from_dir(tok_dir()).unwrap();
    for (name, id) in g["specials"].as_object().unwrap() {
        assert_eq!(tok.token_to_id(name), Some(id.as_u64().unwrap() as u32), "special token {name}");
    }
    let eos: Vec<u32> =
        g["eos_token_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    assert_eq!(tok.eos_ids(), &eos[..]);
}

#[test]
fn encode_matches_hf() {
    let g = golden();
    let tok = GlmTokenizer::from_dir(tok_dir()).unwrap();
    for c in g["encode"].as_array().unwrap() {
        let text = c["text"].as_str().unwrap();
        let want: Vec<u32> =
            c["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let got = tok.encode(text).unwrap();
        assert_eq!(got, want, "encode({text:?})");
        assert_eq!(tok.decode(&got).unwrap(), text, "decode roundtrip({text:?})");
    }
}

#[test]
fn incremental_decode_matches_prefix_decode() {
    let g = golden();
    let tok = GlmTokenizer::from_dir(tok_dir()).unwrap();
    let ids: Vec<u32> =
        g["incremental"]["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    let prefixes: Vec<&str> =
        g["incremental"]["prefix_texts"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
    let mut dec = tok.decode_stream();
    let mut acc = String::new();
    let mut pending_since: Option<usize> = None;
    for (i, &id) in ids.iter().enumerate() {
        match dec.step(id).unwrap() {
            Some(s) => {
                acc.push_str(&s);
                pending_since = None;
                // 一旦吐出文本，累计文本必须等于 HF 对该前缀的整体 decode
                assert_eq!(acc, prefixes[i], "incremental decode prefix {i}");
            }
            None => {
                // UTF-8 半字符：HF 整体 decode 此时会给出替换符 �，增量流不吐；允许最多连续 3 个 token 悬空
                pending_since.get_or_insert(i);
                assert!(i - pending_since.unwrap() < 4, "decode stream stuck for 4 tokens at {i}");
            }
        }
    }
    assert_eq!(acc, *prefixes.last().unwrap());
}

#[test]
fn chat_template_matches_hf() {
    let g = golden();
    let tok = GlmTokenizer::from_dir(tok_dir()).unwrap();
    let tmpl = ChatTemplate::from_dir(tok_dir()).unwrap();
    let mut n = 0;
    let mut fails = Vec::new();
    for (name, c) in g["cases"].as_object().unwrap() {
        let inp = &c["input"];
        let messages: Vec<serde_json::Value> = inp["messages"].as_array().unwrap().clone();
        let tools: Option<Vec<serde_json::Value>> = inp["tools"].as_array().cloned();
        let p = RenderParams {
            messages: &messages,
            tools: tools.as_deref(),
            reasoning_effort: inp["reasoning_effort"].as_str(),
            clear_thinking: inp["clear_thinking"].as_bool().unwrap_or(false),
            add_generation_prompt: inp["add_generation_prompt"].as_bool().unwrap_or(true),
        };
        let want_text = c["text"].as_str().unwrap();
        let want_ids: Vec<u32> =
            c["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        match render(&tmpl, &p) {
            Err(e) => fails.push(format!("{name}: render error: {e:#}")),
            Ok(text) => {
                if text != want_text {
                    fails.push(format!("{name}: text differs. {}", first_diff(&text, want_text)));
                    continue;
                }
                let ids = tok.encode(&text).unwrap();
                if ids != want_ids {
                    fails.push(format!("{name}: ids differ ({} vs {} tokens)", ids.len(), want_ids.len()));
                }
            }
        }
        n += 1;
    }
    eprintln!("chat template cases checked: {n}, failures: {}", fails.len());
    for f in &fails {
        eprintln!("  {f}");
    }
    assert!(fails.is_empty(), "{} chat template cases differ from HF", fails.len());
}
