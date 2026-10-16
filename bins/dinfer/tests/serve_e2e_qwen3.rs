//! Q1 e2e：起 `dinfer serve --features qwen3`（真权重），问 "1+1=?"：
//! Q2-b e2e（`e2e_qwen3_request_features`）：stop 字符串（非流式 / 流式 / include_stop_str_in_output）、logprobs 形状、n=3、非法参数 400。
//!非流式 `reasoning_content` 非空、`content` 含 "2"；
//! 流式拼接 == 非流式；`usage.prompt_tokens` == HF 模板 token 数（chat_golden）；`enable_thinking=false` 走 chat_template_kwargs。
//! 跑法：QWEN3_DIR=/data/ref/models/qwen3-0.6b QWEN3_GOLDEN=$(realpath test/qwen3_golden) \
//!   cargo test -p dinfer --features qwen3,e2e --test serve_e2e_qwen3 -- --ignored --nocapture --test-threads=1
#![cfg(all(feature = "e2e", feature = "qwen3"))]

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Server {
    child: Child,
    base: String,
}
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
fn env(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}
fn spawn_server() -> Server {
    let model = env("QWEN3_DIR", "/data/ref/models/qwen3-0.6b");
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let mut child = Command::new(env!("CARGO_BIN_EXE_dinfer"))
        .args([
            "serve",
            "--model",
            &model,
            "--port",
            &port.to_string(),
            "--max-model-len",
            "2048",
            "--num-blocks",
            "256",
            "--max-num-batched-tokens",
            "512",
        ])
        .env("RUST_LOG", "info")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn dinfer serve");
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    let readers: Vec<(&str, Box<dyn std::io::Read + Send>)> = [
        child.stdout.take().map(|r| ("out", Box::new(r) as Box<dyn std::io::Read + Send>)),
        child.stderr.take().map(|r| ("err", Box::new(r) as Box<dyn std::io::Read + Send>)),
    ]
    .into_iter()
    .flatten()
    .collect();
    for (tag, rd) in readers {
        let tx = tx.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(rd).lines().map_while(Result::ok) {
                eprintln!("[serve:{tag}] {line}");
                let _ = tx.send(line);
            }
        });
    }
    drop(tx);
    let t0 = Instant::now();
    loop {
        match rx.recv_timeout(Duration::from_secs(1)) {
            Ok(l) if l.contains("listening") => break,
            Ok(_) => {}
            Err(_) => {
                if let Ok(Some(st)) = child.try_wait() {
                    panic!("server exited early: {st}");
                }
                assert!(t0.elapsed() < Duration::from_secs(180), "server did not start in 180s");
            }
        }
    }
    Server { child, base: format!("http://127.0.0.1:{port}") }
}
fn golden() -> serde_json::Value {
    let p = std::path::Path::new(&env("QWEN3_GOLDEN", "test/qwen3_golden")).join("chat_golden.json");
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display())))
        .unwrap()
}

#[tokio::test]
#[ignore]
async fn e2e_qwen3_real_model_chat() {
    let srv = spawn_server();
    let c = reqwest::Client::new();
    let m: serde_json::Value =
        c.get(format!("{}/v1/models", srv.base)).send().await.unwrap().json().await.unwrap();
    assert_eq!(m["data"][0]["id"], "qwen3-0.6b");

    // usage.prompt_tokens == HF golden（case "user" = [{"role":"user","content":"hi"}] with thinking）
    let g = golden();
    let case = &g["cases"]["user"];
    let want_len = case["ids"].as_array().unwrap().len();
    let r = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(
            &serde_json::json!({"messages": case["input"]["messages"], "max_tokens": 4, "temperature": 0.0}),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let v: serde_json::Value = r.json().await.unwrap();
    assert_eq!(v["usage"]["prompt_tokens"].as_u64().unwrap() as usize, want_len);

    // 真模型回答 1+1：thinking 模式 → reasoning_content 非空；content 含 "2"
    let body = serde_json::json!({"model": "qwen3-0.6b", "messages": [{"role":"user","content":"1+1=? Answer with just the number."}], "max_tokens": 256, "temperature": 0.0});
    let r = c.post(format!("{}/v1/chat/completions", srv.base)).json(&body).send().await.unwrap();
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
    let v: serde_json::Value = r.json().await.unwrap();
    let msg = &v["choices"][0]["message"];
    let reasoning = msg["reasoning_content"].as_str().unwrap_or("").to_string();
    let content = msg["content"].as_str().unwrap_or("").to_string();
    eprintln!(
        "e2e: finish={} reasoning[{}]={:?} content={:?}",
        v["choices"][0]["finish_reason"],
        reasoning.chars().count(),
        reasoning.chars().take(80).collect::<String>(),
        content
    );
    assert!(!reasoning.is_empty(), "thinking mode must produce reasoning_content");
    assert!(content.contains('2'), "answer should contain 2: {content:?}");

    // 流式 == 非流式
    let mut sb = body.clone();
    sb["stream"] = serde_json::Value::Bool(true);
    let text = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&sb)
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let (mut sr, mut sc, mut done) = (String::new(), String::new(), false);
    for line in text.lines() {
        let Some(d) = line.strip_prefix("data: ") else { continue };
        if d.trim() == "[DONE]" {
            done = true;
            continue;
        }
        let j: serde_json::Value = serde_json::from_str(d).unwrap();
        if let Some(ch) = j["choices"].as_array().and_then(|a| a.first()) {
            if let Some(x) = ch["delta"]["reasoning_content"].as_str() {
                sr.push_str(x);
            }
            if let Some(x) = ch["delta"]["content"].as_str() {
                sc.push_str(x);
            }
        }
    }
    assert!(done);
    assert_eq!(sr, reasoning, "stream reasoning != non-stream");
    assert_eq!(sc, content, "stream content != non-stream");

    // enable_thinking=false → 无 reasoning_content，直接回答
    let body = serde_json::json!({"messages": [{"role":"user","content":"1+1=? Answer with just the number."}], "max_tokens": 32, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": false}});
    let v: serde_json::Value = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let msg = &v["choices"][0]["message"];
    eprintln!("e2e: no_think content={:?} reasoning={:?}", msg["content"], msg["reasoning_content"]);
    assert!(msg["reasoning_content"].is_null() || msg["reasoning_content"].as_str() == Some(""));
    assert!(msg["content"].as_str().unwrap_or("").contains('2'));
    eprintln!("e2e: PASS");
    drop(srv);
}

/// 把 SSE 文本按 choice index 聚合为 (content, finish_reason, logprob 条数)。
fn parse_sse(text: &str, n: usize) -> (Vec<String>, Vec<Option<String>>, Vec<usize>, bool) {
    let mut content = vec![String::new(); n];
    let mut finish = vec![None; n];
    let mut lps = vec![0usize; n];
    let mut done = false;
    for line in text.lines() {
        let Some(d) = line.strip_prefix("data: ") else { continue };
        if d.trim() == "[DONE]" {
            done = true;
            continue;
        }
        let j: serde_json::Value = serde_json::from_str(d).unwrap();
        for ch in j["choices"].as_array().into_iter().flatten() {
            let i = ch["index"].as_u64().unwrap() as usize;
            if let Some(x) = ch["delta"]["content"].as_str() {
                content[i].push_str(x);
            }
            if let Some(f) = ch["finish_reason"].as_str() {
                finish[i] = Some(f.to_string());
            }
            if let Some(a) = ch["logprobs"]["content"].as_array() {
                lps[i] += a.len();
            }
        }
    }
    (content, finish, lps, done)
}

#[tokio::test]
#[ignore]
async fn e2e_qwen3_request_features() {
    let srv = spawn_server();
    let c = reqwest::Client::new();
    let url = format!("{}/v1/chat/completions", srv.base);
    // no_think + greedy：让输出稳定可预测
    let base = serde_json::json!({
        "messages": [{"role":"user","content":"Count from 1 to 20 separated by commas, nothing else."}],
        "max_tokens": 64, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": false}
    });

    // 1) 非法参数 → 400
    for (k, v) in [
        ("temperature", serde_json::json!(-1.0)),
        ("top_p", serde_json::json!(1.5)),
        ("n", serde_json::json!(9)),
        ("top_logprobs", serde_json::json!(3)),
    ] {
        let mut b = base.clone();
        b[k] = v;
        let r = c.post(&url).json(&b).send().await.unwrap();
        assert_eq!(r.status(), 400, "{k} should be rejected: {}", r.text().await.unwrap());
    }

    // 2) 无 stop 的参考输出
    let v: serde_json::Value = c.post(&url).json(&base).send().await.unwrap().json().await.unwrap();
    let full = v["choices"][0]["message"]["content"].as_str().unwrap().to_string();
    eprintln!("e2e: full = {full:?}");
    assert!(full.contains("5") && full.contains("10"), "unexpected counting output");

    // 3) stop 字符串：非流式，截到 stop 之前；finish_reason = stop；completion_tokens 少于无 stop
    let mut b = base.clone();
    b["stop"] = serde_json::json!(["10"]);
    let v: serde_json::Value = c.post(&url).json(&b).send().await.unwrap().json().await.unwrap();
    let cut = v["choices"][0]["message"]["content"].as_str().unwrap().to_string();
    eprintln!(
        "e2e: stop=10 → {cut:?} finish={} tokens={}",
        v["choices"][0]["finish_reason"], v["usage"]["completion_tokens"]
    );
    assert!(!cut.contains("10"), "stop string must be excluded: {cut:?}");
    assert!(full.starts_with(&cut), "stop output must be a prefix of the full output");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert!(v["usage"]["completion_tokens"].as_u64().unwrap() < 40, "stop should abort early (KV released)");

    // include_stop_str_in_output → 以 stop 结尾
    b["include_stop_str_in_output"] = serde_json::json!(true);
    let v: serde_json::Value = c.post(&url).json(&b).send().await.unwrap().json().await.unwrap();
    let inc = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(inc.ends_with("10"), "include_stop_str_in_output must keep the stop: {inc:?}");
    assert_eq!(inc.strip_suffix("10").unwrap(), cut);

    // 4) 流式 stop：拼出来与非流式一致，finish_reason=stop，[DONE]
    let mut sb = base.clone();
    sb["stop"] = serde_json::json!(["10"]);
    sb["stream"] = serde_json::json!(true);
    let text = c.post(&url).json(&sb).send().await.unwrap().text().await.unwrap();
    let (content, finish, _, done) = parse_sse(&text, 1);
    assert!(done);
    assert_eq!(content[0], cut, "stream stop output != non-stream");
    assert_eq!(finish[0].as_deref(), Some("stop"));

    // 5) logprobs：每个 token 一条，含 token / logprob / bytes / top_logprobs(3)；logprob ≤ 0；greedy 时 chosen 是 top-1
    let mut b = base.clone();
    b["max_tokens"] = serde_json::json!(6);
    b["logprobs"] = serde_json::json!(true);
    b["top_logprobs"] = serde_json::json!(3);
    let v: serde_json::Value = c.post(&url).json(&b).send().await.unwrap().json().await.unwrap();
    let lp = v["choices"][0]["logprobs"]["content"].as_array().unwrap();
    let ntok = v["usage"]["completion_tokens"].as_u64().unwrap() as usize;
    assert_eq!(lp.len(), ntok, "one logprob entry per completion token");
    for e in lp {
        assert!(e["token"].is_string() && e["bytes"].is_array());
        let l = e["logprob"].as_f64().unwrap();
        assert!(l <= 0.0);
        let top = e["top_logprobs"].as_array().unwrap();
        assert_eq!(top.len(), 3);
        assert_eq!(top[0]["token"], e["token"], "greedy: chosen must be top-1");
        assert!((top[0]["logprob"].as_f64().unwrap() - l).abs() < 1e-6);
    }
    eprintln!("e2e: logprobs ok, first = {}", lp[0]);
    // 流式 logprobs：条数相同
    b["stream"] = serde_json::json!(true);
    let text = c.post(&url).json(&b).send().await.unwrap().text().await.unwrap();
    let (_, _, lps, done) = parse_sse(&text, 1);
    assert!(done);
    assert_eq!(lps[0], ntok, "stream logprobs count");

    // 6) n=3：greedy 三个相同；temperature=1.0 + seed 至少两个不同；index 0..2；流式按 index 交错且各自 finish
    let mut b = base.clone();
    b["n"] = serde_json::json!(3);
    b["max_tokens"] = serde_json::json!(16);
    let v: serde_json::Value = c.post(&url).json(&b).send().await.unwrap().json().await.unwrap();
    let ch = v["choices"].as_array().unwrap();
    assert_eq!(ch.len(), 3);
    for (i, x) in ch.iter().enumerate() {
        assert_eq!(x["index"].as_u64().unwrap() as usize, i);
    }
    let texts: Vec<&str> = ch.iter().map(|x| x["message"]["content"].as_str().unwrap()).collect();
    assert!(texts[0] == texts[1] && texts[1] == texts[2], "greedy n=3 must be identical: {texts:?}");
    let ct = v["usage"]["completion_tokens"].as_u64().unwrap();
    assert!(ct > 0 && ct % 3 == 0, "greedy n=3: usage sums 3 identical choices, got {ct}");
    let mut b = serde_json::json!({
        "messages": [{"role":"user","content":"Write one short sentence about the sea."}],
        "max_tokens": 24, "temperature": 1.0, "top_p": 0.95, "seed": 42, "n": 3,
        "chat_template_kwargs": {"enable_thinking": false}
    });
    let v: serde_json::Value = c.post(&url).json(&b).send().await.unwrap().json().await.unwrap();
    let texts: Vec<String> = v["choices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x["message"]["content"].as_str().unwrap().to_string())
        .collect();
    eprintln!("e2e: n=3 sampled = {texts:?}");
    assert!(texts.iter().collect::<std::collections::HashSet<_>>().len() >= 2, "sampled n=3 should differ");
    b["stream"] = serde_json::json!(true);
    let text = c.post(&url).json(&b).send().await.unwrap().text().await.unwrap();
    let (content, finish, _, done) = parse_sse(&text, 3);
    assert!(done);
    assert_eq!(content, texts, "stream n=3 content per index != non-stream (same seed)");
    assert!(finish.iter().all(|f| f.is_some()), "every choice gets a finish_reason");
    eprintln!("e2e: Q2-b request features PASS");
    drop(srv);
}
