//! Q1 e2e：起 `dinfer serve --features qwen3`（真权重），问 "1+1=?"：非流式 `reasoning_content` 非空、`content` 含 "2"；
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
