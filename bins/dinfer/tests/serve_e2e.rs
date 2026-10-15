//! P6-c e2e：起 `dinfer serve`（tiny DSA GLM 全词表 fixture + 真 GLM-5.3 tokenizer / 模板），用 HTTP 打
//! `/v1/models`、`/v1/chat/completions`（非流式 / 流式），验证：
//! - `usage.prompt_tokens` == HF 模板渲染后的 token 数（chat_golden 里同 messages 的 case）
//! - 流式 SSE 拼接出的 reasoning/content == 非流式结果（greedy）
//! - `finish_reason` 合法；流式最后 `[DONE]`；`stream_options.include_usage` 给 usage chunk
//! - greedy 输出 token 序列（`return_token_ids`）与 fixture 的 HF `expect_greedy_long`（同 prompt 时）一致——这里用
//!   fixture 自带 prompt 不走模板，改为用 `prompt_token_ids` 回传核对模板编码一致即可；token 级对拍已由 p4/p5 覆盖。
//!
//! 跑法：TINY_GLM_DIR=$(realpath test/tiny_glm_dsa_fullvocab) GLM_TOKENIZER_DIR=/data/ref/models/glm53 \
//!   cargo test -p dinfer --features glm53,e2e --test serve_e2e -- --ignored --nocapture --test-threads=1
#![cfg(feature = "e2e")]

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

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

fn spawn_server() -> Server {
    let model = env("TINY_GLM_DIR", "test/tiny_glm_dsa_fullvocab");
    let tok = env("GLM_TOKENIZER_DIR", "/data/ref/models/glm53");
    let port = free_port();
    let bin = env!("CARGO_BIN_EXE_dinfer");
    let mut child = Command::new(bin)
        .args([
            "serve",
            "--model",
            &model,
            "--tokenizer",
            &tok,
            "--port",
            &port.to_string(),
            "--max-model-len",
            "512",
            "--num-blocks",
            "64",
            "--max-num-batched-tokens",
            "256",
        ])
        .env("RUST_LOG", "info")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn dinfer serve");
    // 等 "listening"（tracing 默认写 stdout），最多 120s（含模型加载 + CUDA 初始化）；stdout/stderr 都转发
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
                assert!(t0.elapsed() < Duration::from_secs(120), "server did not start in 120s");
            }
        }
    }
    Server { child, base: format!("http://127.0.0.1:{port}") }
}

fn golden() -> serde_json::Value {
    let p =
        std::path::Path::new(&env("TINY_GLM_DIR", "test/tiny_glm_dsa_fullvocab")).join("chat_golden.json");
    let p = if p.exists() { p } else { std::path::Path::new("test/tiny_glm_dsa").join("chat_golden.json") };
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display())))
        .unwrap()
}

#[tokio::test]
#[ignore]
async fn e2e_models_and_chat() {
    let srv = spawn_server();
    let c = reqwest::Client::new();

    // /health, /v1/models
    let h: serde_json::Value =
        c.get(format!("{}/health", srv.base)).send().await.unwrap().json().await.unwrap();
    assert_eq!(h["status"], "ok");
    let m: serde_json::Value =
        c.get(format!("{}/v1/models", srv.base)).send().await.unwrap().json().await.unwrap();
    assert_eq!(m["data"][0]["id"], "glm-5.3");
    eprintln!("e2e: models ok: {}", m["data"][0]);

    // 非流式：prompt_tokens 必须等于 HF 模板的 token 数（chat_golden single_user）
    let g = golden();
    let case = &g["cases"]["single_user"];
    let want_ids: Vec<u64> = case["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
    let body = serde_json::json!({
        "model": "glm-5.3",
        "messages": case["input"]["messages"],
        "max_tokens": 24,
        "temperature": 0.0,
        "return_token_ids": true,
    });
    let r = c.post(format!("{}/v1/chat/completions", srv.base)).json(&body).send().await.unwrap();
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
    let v: serde_json::Value = r.json().await.unwrap();
    eprintln!(
        "e2e: non-stream: {}",
        serde_json::to_string(&v).unwrap().chars().take(400).collect::<String>()
    );
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["usage"]["prompt_tokens"].as_u64().unwrap() as usize, want_ids.len());
    let got_prompt: Vec<u64> =
        v["prompt_token_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
    assert_eq!(got_prompt, want_ids, "prompt ids must equal HF apply_chat_template ids");
    let fr = v["choices"][0]["finish_reason"].as_str().unwrap();
    assert!(fr == "stop" || fr == "length", "finish_reason {fr}");
    let ct = v["usage"]["completion_tokens"].as_u64().unwrap();
    assert!((1..=24).contains(&ct), "completion_tokens {ct}");
    let msg = &v["choices"][0]["message"];
    assert_eq!(msg["role"], "assistant");
    let ns_reason = msg["reasoning_content"].as_str().unwrap_or("").to_string();
    let ns_content = msg["content"].as_str().unwrap_or("").to_string();
    let ns_ids: Vec<u64> = v["token_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
    assert_eq!(ns_ids.len() as u64, ct);

    // 流式：拼接 == 非流式（greedy 决定性）；首 chunk role；末 [DONE]；include_usage
    let body = serde_json::json!({
        "model": "glm-5.3",
        "messages": case["input"]["messages"],
        "max_tokens": 24,
        "temperature": 0.0,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let r = c.post(format!("{}/v1/chat/completions", srv.base)).json(&body).send().await.unwrap();
    assert_eq!(r.status(), 200);
    assert!(r.headers().get("content-type").unwrap().to_str().unwrap().starts_with("text/event-stream"));
    let text = r.text().await.unwrap();
    let mut s_reason = String::new();
    let mut s_content = String::new();
    let mut saw_role = false;
    let mut saw_done = false;
    let mut finish = None;
    let mut usage = None;
    let mut n_chunks = 0;
    for line in text.lines() {
        let Some(data) = line.strip_prefix("data: ") else { continue };
        if data.trim() == "[DONE]" {
            saw_done = true;
            continue;
        }
        let j: serde_json::Value =
            serde_json::from_str(data).unwrap_or_else(|e| panic!("bad chunk {data}: {e}"));
        assert_eq!(j["object"], "chat.completion.chunk");
        n_chunks += 1;
        if let Some(u) = j.get("usage").filter(|u| !u.is_null()) {
            usage = Some(u.clone());
        }
        if let Some(ch) = j["choices"].as_array().and_then(|a| a.first()) {
            let d = &ch["delta"];
            if d["role"] == "assistant" {
                saw_role = true;
            }
            if let Some(x) = d["reasoning_content"].as_str() {
                s_reason.push_str(x);
            }
            if let Some(x) = d["content"].as_str() {
                s_content.push_str(x);
            }
            if let Some(f) = ch["finish_reason"].as_str() {
                finish = Some(f.to_string());
            }
        }
    }
    eprintln!("e2e: stream chunks {n_chunks}, finish {finish:?}, usage {usage:?}");
    assert!(saw_role, "first chunk must carry role");
    assert!(saw_done, "stream must end with [DONE]");
    assert!(finish.is_some());
    assert_eq!(s_reason, ns_reason, "stream reasoning != non-stream");
    assert_eq!(s_content, ns_content, "stream content != non-stream");
    let u = usage.expect("usage chunk");
    assert_eq!(u["prompt_tokens"].as_u64().unwrap() as usize, want_ids.len());
    assert_eq!(u["completion_tokens"].as_u64().unwrap(), ct);

    // 并发：两个请求同时打，都要正常结束（调度器混批）
    let b1 = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&serde_json::json!({
        "messages": [{"role":"user","content":"one"}], "max_tokens": 16, "temperature": 0.0}))
        .send();
    let b2 = c.post(format!("{}/v1/chat/completions", srv.base)).json(&serde_json::json!({
        "messages": [{"role":"user","content":"two two two"}], "max_tokens": 16, "temperature": 0.0, "reasoning_effort": "low"})).send();
    let (r1, r2) = tokio::join!(b1, b2);
    for r in [r1.unwrap(), r2.unwrap()] {
        assert_eq!(r.status(), 200);
        let v: serde_json::Value = r.json().await.unwrap();
        assert!(v["usage"]["completion_tokens"].as_u64().unwrap() >= 1);
    }
    eprintln!("e2e: concurrent ok");

    // 错误：空 messages → 400
    let r = c
        .post(format!("{}/v1/chat/completions", srv.base))
        .json(&serde_json::json!({"messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);
    eprintln!("e2e: PASS");
    drop(srv);
}
