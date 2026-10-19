//! Qwen3 文法走查（受限解码，模型相关的部分放模型 crate；`crates/guided` 只留模型无关单测）：
//! 用真 tokenizer 建 `GrammarFactory`（eos 151645 / 151643，vocab 151936），json_object / json_schema / tool_call lark
//! 的允许集合逐 token 走一遍。跑法：QWEN3_DIR=/data/ref/models/qwen3-0.6b cargo test -p dinfer-qwen3 --test guided_parity
use dinfer_guided::*;
use dinfer_core::sample::GrammarHandle;

fn factory() -> Option<GrammarFactory> {
    let dir = std::env::var("QWEN3_DIR").unwrap_or_else(|_| "/data/ref/models/qwen3-0.6b".into());
    let p = std::path::Path::new(&dir).join("tokenizer.json");
    if !p.exists() {
        return None;
    }
    let tok = tokenizers::Tokenizer::from_file(&p).unwrap();
    Some(GrammarFactory::new(&tok, &[151645, 151643], 151936).unwrap())
}

fn allowed(h: &GrammarHandle, out: &[u32]) -> Vec<u32> {
    let mut slot = h.0.lock();
    let m = match GrammarHandle::advance_and_mask(&mut slot, out) {
        Some(m) => m.to_vec(),
        None => panic!("grammar failed at output {out:?} (consumed {})", slot.consumed),
    };
    (0..m.len() * 32).filter(|&i| (m[i / 32] >> (i % 32)) & 1 == 1).map(|i| i as u32).collect()
}

/// 真 tokenizer：json_object 第一个 token 只能是 `{`（或空白变体）；喂一段合法 JSON 后 eos 被允许；
/// tool lark：开头必须是 `<tool_call>`，`{"name": "get_weather"` 走通。
#[test]
fn qwen3_json_and_tool_grammars_walk() {
    let Some(f) = factory() else { eprintln!("skip: no tokenizer"); return };
    let tok = tokenizers::Tokenizer::from_file(
        std::path::Path::new(&std::env::var("QWEN3_DIR").unwrap_or_else(|_| "/data/ref/models/qwen3-0.6b".into()))
            .join("tokenizer.json"),
    )
    .unwrap();
    let enc = |s: &str| tok.encode(s, false).unwrap().get_ids().to_vec();
    // json_object
    let h = f.compile(&GrammarSpec::JsonObject).unwrap();
    let first = allowed(&h, &[]);
    assert!(first.contains(&enc("{")[0]), "`{{` must be allowed first");
    assert!(!first.contains(&enc("hello")[0]), "`hello` must not be allowed first");
    assert_eq!(f.vocab().div_ceil(32), h.0.lock().state.mask_words());
    let body = enc("{\"a\": 1}");
    let after = allowed(&h, &body);
    assert!(after.contains(&151645), "eos must be allowed after complete JSON: {after:?}");
    // json_schema：required 字段
    let schema = serde_json::json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false});
    let h2 = f.compile(&GrammarSpec::JsonSchema(schema)).unwrap();
    let a = allowed(&h2, &enc("{\"ci"));
    assert!(!a.is_empty());
    // 非法 token → 退化（failed）而不是 panic
    {
        let mut slot = h2.0.lock();
        assert!(GrammarHandle::advance_and_mask(&mut slot, &enc("xyz")).is_none());
        assert!(slot.failed);
    }
    // tool lark
    let p = serde_json::json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]});
    let lark = tool_call_lark(&[ToolFn { name: "get_weather", parameters: Some(&p) }], "<tool_call>\n", "\n</tool_call>", false, None).unwrap();
    let h3 = f.compile(&GrammarSpec::Lark(lark)).unwrap();
    let a0 = allowed(&h3, &[]);
    let tc = enc("<tool_call>");
    assert!(a0.contains(&tc[0]), "<tool_call> first; allowed={a0:?} tc={tc:?}");
    let walk = enc("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Beijing\"}}\n</tool_call>");
    let fin = allowed(&h3, &walk);
    assert!(fin.contains(&151645), "eos after full tool_call: {fin:?}");
    // 坏 schema → 编译错误（400）
    assert!(f.compile(&GrammarSpec::JsonSchema(serde_json::json!({"type":"nope"}))).is_err());
}

