//! C-4a：Kimi-K3 tokenizer / XTML 渲染 / 输出解析 vs Python golden（`scripts/verify/gen_k3_chat_golden.py`）。
//! 跑法：KIMIK3_TOK_DIR=/data/ref/models/kimik3 cargo test -p dinfer-kimik3 --test chat_parity
//! golden 默认 `test/kimik3/golden/chat.json`（进仓库；可用 KIMIK3_CHAT_GOLDEN 覆盖）。

use std::path::PathBuf;

use dinfer_kimik3::parser::{Delta, OutputParser};
use dinfer_kimik3::tokenizer::Tokenizer;
use dinfer_kimik3::xtml::{build_chat_segments, join_text, Dialect, RenderOptions};
use serde_json::Value;

fn tok_dir() -> PathBuf {
    PathBuf::from(std::env::var("KIMIK3_TOK_DIR").unwrap_or_else(|_| "/data/ref/models/kimik3".into()))
}

fn golden() -> Value {
    let p =
        PathBuf::from(std::env::var("KIMIK3_CHAT_GOLDEN").unwrap_or_else(|_| {
            format!("{}/../../test/kimik3/golden/chat.json", env!("CARGO_MANIFEST_DIR"))
        }));
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display())))
        .unwrap()
}

fn tokenizer() -> Tokenizer {
    let d = tok_dir();
    if !d.join("tiktoken.model").exists() {
        eprintln!("skip: {} 无 tiktoken.model", d.display());
        std::process::exit(0);
    }
    Tokenizer::from_dir(&d).unwrap()
}

fn ids_of(v: &Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

fn first_diff(a: &str, b: &str) -> String {
    let i = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
    let ctx = |s: &str| -> String {
        let start = s.char_indices().nth(i.saturating_sub(40)).map(|x| x.0).unwrap_or(0);
        let end = s.char_indices().nth(i + 40).map(|x| x.0).unwrap_or(s.len());
        s[start..end].replace('\n', "\\n")
    };
    format!("first diff at char {i}:\n  ours   …{}…\n  golden …{}…", ctx(a), ctx(b))
}

#[test]
fn specials_and_eos_match() {
    let g = golden();
    let t = tokenizer();
    assert_eq!(t.n_base() as u64, g["n_base"].as_u64().unwrap());
    for (name, id) in g["specials"].as_object().unwrap() {
        assert_eq!(t.special_id(name), Some(id.as_u64().unwrap() as u32), "{name}");
        assert_eq!(t.special_name(id.as_u64().unwrap() as u32), Some(name.as_str()));
    }
    assert_eq!(t.eos_ids().to_vec(), ids_of(&g["eos_token_ids"]));
}

#[test]
fn encode_matches_python_both_modes() {
    let g = golden();
    let t = tokenizer();
    let mut fails = Vec::new();
    for c in g["encode"].as_array().unwrap() {
        let text = c["text"].as_str().unwrap();
        let allow = c["allow_special"].as_bool().unwrap();
        let want = ids_of(&c["ids"]);
        let got = t.encode(text, allow);
        if got != want {
            fails.push(format!("{text:?} allow_special={allow}: ours {got:?} vs {want:?}"));
        }
        if !allow || want.iter().all(|&id| !t.is_special(id)) {
            assert_eq!(t.decode(&got), text, "decode roundtrip {text:?}");
        }
    }
    assert!(fails.is_empty(), "{} encode cases differ:\n{}", fails.len(), fails.join("\n"));
}

#[test]
fn incremental_decode_matches_prefix_decode() {
    let g = golden();
    let t = tokenizer();
    let ids = ids_of(&g["incremental"]["ids"]);
    let prefixes: Vec<&str> =
        g["incremental"]["prefix_texts"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
    let mut d = t.decode_stream();
    let mut acc = String::new();
    for (i, &id) in ids.iter().enumerate() {
        acc.push_str(&d.step(id));
        let want = prefixes[i];
        // 增量不能吐出半字符，所以 acc 是 want 去掉尾部 U+FFFD 后的前缀
        let want_clean = want.trim_end_matches('\u{fffd}');
        assert!(want_clean.starts_with(&acc), "step {i}: acc {acc:?} 不是 {want_clean:?} 的前缀");
        if !want.ends_with('\u{fffd}') {
            assert_eq!(acc, want, "step {i}");
        }
    }
    acc.push_str(&d.finish());
    assert_eq!(acc, *prefixes.last().unwrap());
}

fn dialect_of(s: &str) -> Dialect {
    match s {
        "hf" => Dialect::Hf,
        "serving" => Dialect::Serving,
        other => panic!("dialect {other}"),
    }
}

#[test]
fn chat_render_matches_python_both_dialects() {
    let g = golden();
    let t = tokenizer();
    let mut n = 0;
    let mut fails = Vec::new();
    for (name, c) in g["cases"].as_object().unwrap() {
        let inp = &c["input"];
        let messages: Vec<Value> = inp["messages"].as_array().unwrap().clone();
        let tools: Option<Vec<Value>> = inp["tools"].as_array().cloned();
        let o = &inp["options"];
        let rf = o.get("response_format").filter(|v| !v.is_null());
        let opts = RenderOptions {
            tools: tools.as_deref(),
            add_generation_prompt: o["add_generation_prompt"].as_bool().unwrap_or(true),
            thinking: o["thinking"].as_bool().unwrap_or(true),
            thinking_effort: o["thinking_effort"].as_str(),
            tool_choice: o["tool_choice"].as_str(),
            response_format: rf,
            dialect: dialect_of(c["dialect"].as_str().unwrap()),
        };
        n += 1;
        let r = build_chat_segments(&messages, &opts);
        if c["error"].as_bool() == Some(true) {
            if r.is_ok() {
                fails.push(format!("{name}: expected error, got Ok"));
            }
            continue;
        }
        let seg = match r {
            Err(e) => {
                fails.push(format!("{name}: render error: {e:#}"));
                continue;
            }
            Ok(s) => s,
        };
        let text = join_text(&seg);
        let want_text = c["text"].as_str().unwrap();
        if text != want_text {
            fails.push(format!("{name}: text differs. {}", first_diff(&text, want_text)));
            continue;
        }
        let mut ids = Vec::new();
        for s in &seg {
            t.encode_into(&s.text, s.allow_special, &mut ids);
        }
        let want_ids = ids_of(&c["ids"]);
        if ids != want_ids {
            let i = ids.iter().zip(&want_ids).take_while(|(a, b)| a == b).count();
            fails.push(format!(
                "{name}: ids differ at {i} ({} vs {} tokens): ours {:?} golden {:?}",
                ids.len(),
                want_ids.len(),
                &ids[i..(i + 6).min(ids.len())],
                &want_ids[i..(i + 6).min(want_ids.len())]
            ));
        }
    }
    eprintln!("chat cases checked: {n}, failures: {}", fails.len());
    for f in &fails {
        eprintln!("  {f}");
    }
    assert!(fails.is_empty(), "{} chat cases differ from python", fails.len());
}

fn collect(p: &mut OutputParser, chunks: impl Iterator<Item = String>) -> Delta {
    let mut acc = Delta::default();
    let push = |d: Delta, acc: &mut Delta| {
        acc.reasoning.push_str(&d.reasoning);
        acc.content.push_str(&d.content);
        acc.tool_calls.extend(d.tool_calls);
    };
    for c in chunks {
        let d = p.feed(&c);
        push(d, &mut acc);
    }
    let d = p.finish();
    push(d, &mut acc);
    acc
}

fn check_parsed(name: &str, how: &str, got: &Delta, expect: &Value, fails: &mut Vec<String>) {
    let want_r = expect["reasoning"].as_str().unwrap();
    let want_c = expect["content"].as_str().unwrap();
    if got.reasoning != want_r {
        fails.push(format!("{name}[{how}] reasoning {:?} vs {:?}", got.reasoning, want_r));
    }
    if got.content != want_c {
        fails.push(format!("{name}[{how}] content {:?} vs {:?}", got.content, want_c));
    }
    let want_tc = expect["tool_calls"].as_array().unwrap();
    if got.tool_calls.len() != want_tc.len() {
        fails.push(format!("{name}[{how}] {} tool_calls vs {}", got.tool_calls.len(), want_tc.len()));
        return;
    }
    for (i, (g, w)) in got.tool_calls.iter().zip(want_tc).enumerate() {
        if g.index != i || g.name.as_deref() != w["name"].as_str() {
            fails.push(format!("{name}[{how}] tool_call {i}: {:?}/{:?}", g.index, g.name));
        }
        let args: Value = serde_json::from_str(g.arguments.as_deref().unwrap_or("{}")).unwrap();
        if args != w["arguments"] {
            fails.push(format!("{name}[{how}] tool_call {i} args {args} vs {}", w["arguments"]));
        }
    }
}

#[test]
fn parser_matches_rendered_messages_whole_token_and_char() {
    let g = golden();
    let t = tokenizer();
    let mut fails = Vec::new();
    for c in g["parser_cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let text = c["text"].as_str().unwrap();
        let ids = ids_of(&c["ids"]);
        assert_eq!(t.decode(&ids), text, "{name}: golden 自身 decode 不一致");
        // 整段
        let whole = collect(&mut OutputParser::new(true), std::iter::once(text.to_string()));
        check_parsed(name, "whole", &whole, &c["expect"], &mut fails);
        // 逐字符
        let per_char = collect(&mut OutputParser::new(true), text.chars().map(|c| c.to_string()));
        check_parsed(name, "char", &per_char, &c["expect"], &mut fails);
        // 逐 token 经增量 detokenize（真实链路；eos 不进文本）
        let mut d = t.decode_stream();
        let mut p = OutputParser::new(true);
        let chunks: Vec<String> = ids.iter().filter(|&&id| !t.is_eos(id)).map(|&id| d.step(id)).collect();
        let per_tok = collect(&mut p, chunks.into_iter());
        check_parsed(name, "token", &per_tok, &c["expect"], &mut fails);
    }
    for f in &fails {
        eprintln!("  {f}");
    }
    assert!(fails.is_empty(), "{} parser checks failed", fails.len());
}
