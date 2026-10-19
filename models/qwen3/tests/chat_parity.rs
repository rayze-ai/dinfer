//! Q1：Qwen3 tokenizer / chat 模板 vs HF golden（`scripts/verify/gen_qwen3_golden.py` 的 `chat_golden.json`）。
#![cfg(feature = "ref-tests")]

use dinfer_text::{ChatTemplate, HfTokenizer, RenderParams};
use std::path::PathBuf;

fn model_dir() -> PathBuf {
    PathBuf::from(std::env::var("QWEN3_DIR").unwrap_or_else(|_| "/data/ref/models/qwen3-0.6b".into()))
}
fn golden() -> serde_json::Value {
    let p = PathBuf::from(std::env::var("QWEN3_GOLDEN").unwrap_or_else(|_| "test/qwen3_golden".into()));
    let p = if p.is_absolute() { p } else { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join(p) };
    let p = p.join("chat_golden.json");
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display())))
        .unwrap()
}
fn first_diff(a: &str, b: &str) -> String {
    let i = a
        .chars()
        .zip(b.chars())
        .position(|(x, y)| x != y)
        .unwrap_or(a.chars().count().min(b.chars().count()));
    let ctx = |s: &str| -> String {
        let c: Vec<char> = s.chars().collect();
        c[i.saturating_sub(30)..(i + 30).min(c.len())].iter().collect::<String>().replace('\n', "\\n")
    };
    format!("first diff at char {i}:\n  ours   …{}…\n  golden …{}…", ctx(a), ctx(b))
}

#[test]
#[ignore]
fn qwen3_tokenizer_matches_hf() {
    let g = golden();
    let tok = HfTokenizer::from_dir(model_dir()).unwrap();
    for (name, id) in g["specials"].as_object().unwrap() {
        assert_eq!(tok.token_to_id(name), Some(id.as_u64().unwrap() as u32), "special {name}");
    }
    let eos: Vec<u32> =
        g["eos_token_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    assert_eq!(tok.eos_ids(), &eos[..]);
    for c in g["encode"].as_array().unwrap() {
        let text = c["text"].as_str().unwrap();
        let want: Vec<u32> =
            c["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let got = tok.encode(text).unwrap();
        assert_eq!(got, want, "encode({text:?})");
        assert_eq!(tok.decode(&got).unwrap(), text, "decode roundtrip({text:?})");
    }
    let ids: Vec<u32> =
        g["incremental"]["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    let prefixes: Vec<&str> =
        g["incremental"]["prefix_texts"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
    let mut dec = tok.decode_stream();
    let mut acc = String::new();
    for (i, &id) in ids.iter().enumerate() {
        if let Some(s) = dec.step(id).unwrap() {
            acc.push_str(&s);
            assert_eq!(acc, prefixes[i], "incremental prefix {i}");
        }
    }
    assert_eq!(acc, *prefixes.last().unwrap());
    eprintln!("q1: tokenizer specials/eos/encode/incremental all match HF");
}

#[test]
#[ignore]
fn qwen3_chat_template_matches_hf() {
    let g = golden();
    let tok = HfTokenizer::from_dir(model_dir()).unwrap();
    let tmpl = ChatTemplate::from_dir(model_dir()).unwrap();
    let mut fails = Vec::new();
    let mut n = 0;
    for (name, c) in g["cases"].as_object().unwrap() {
        let inp = &c["input"];
        let messages: Vec<serde_json::Value> = inp["messages"].as_array().unwrap().clone();
        let tools: Option<Vec<serde_json::Value>> = inp["tools"].as_array().cloned();
        let mut extra = serde_json::Map::new();
        if let Some(b) = inp.get("enable_thinking") {
            extra.insert("enable_thinking".into(), b.clone());
        }
        let p = RenderParams {
            messages: &messages,
            tools: tools.as_deref(),
            add_generation_prompt: inp["add_generation_prompt"].as_bool().unwrap_or(true),
            continue_final_message: false,
            extra: &extra,
        };
        let want = c["text"].as_str().unwrap();
        let want_ids: Vec<u32> =
            c["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        match tmpl.render(&p) {
            Err(e) => fails.push(format!("{name}: render error: {e:#}")),
            Ok(text) => {
                if text != want {
                    fails.push(format!("{name}: {}", first_diff(&text, want)));
                } else if tok.encode(&text).unwrap() != want_ids {
                    fails.push(format!("{name}: ids differ"));
                }
            }
        }
        n += 1;
    }
    eprintln!("q1: chat template cases {n}, failures {}", fails.len());
    for f in &fails {
        eprintln!("  {f}");
    }
    assert!(fails.is_empty());
}
