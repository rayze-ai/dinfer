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
    logs: Vec<String>,
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
/// 固定 256 块、**关 prefix cache**：这些用例把同一 prompt 发两次（流式 vs 非流式 / n=3 vs 流式 n=3）比逐字相同；
/// 开 prefix cache 时第二次命中前缀走 paged 路径，bf16 累加顺序不同会在 near-tie 处翻 token（与 HF 对拍同样的噪声底，
/// 见 pitfalls 2026-10-16），不是功能错误。prefix cache 本身的正确性在 `e2e_qwen3_prefix_cache` 里单独验。
fn spawn_server() -> Server {
    spawn_server_with(&["--num-blocks", "256", "--no-prefix-caching"])
}

/// `extra` 追加在固定参数之后；返回的 Server 带启动日志（`logs`），e2e 可断言 kv_cache 行。
fn spawn_server_with(extra: &[&str]) -> Server {
    let model = env("QWEN3_DIR", "/data/ref/models/qwen3-0.6b");
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let mut args: Vec<String> = [
        "serve",
        "--model",
        &model,
        "--port",
        &port.to_string(),
        "--max-model-len",
        "2048",
        "--max-num-batched-tokens",
        "512",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    // extra 里出现的 flag 覆盖默认值（clap 不允许同一 flag 重复）
    let overridden: Vec<&str> = extra.iter().copied().filter(|s| s.starts_with("--")).collect();
    let mut filtered = Vec::with_capacity(args.len());
    let mut skip_next = false;
    for a in args {
        if skip_next {
            skip_next = false;
            continue;
        }
        if a.starts_with("--") && overridden.contains(&a.as_str()) {
            skip_next = true; // 跳过它的值
            continue;
        }
        filtered.push(a);
    }
    let mut args = filtered;
    args.extend(extra.iter().map(|s| s.to_string()));
    let mut child = Command::new(env!("CARGO_BIN_EXE_dinfer"))
        .args(&args)
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
                let _ = tx.send(strip_ansi(&line)); // tracing 给字段名上色，断言前去掉转义序列
            }
        });
    }
    drop(tx);
    let t0 = Instant::now();
    let mut logs = Vec::new();
    loop {
        match rx.recv_timeout(Duration::from_secs(1)) {
            Ok(l) if l.contains("listening") => {
                logs.push(l);
                break;
            }
            Ok(l) => logs.push(l),
            Err(_) => {
                if let Ok(Some(st)) = child.try_wait() {
                    panic!("server exited early: {st}");
                }
                assert!(t0.elapsed() < Duration::from_secs(180), "server did not start in 180s");
            }
        }
    }
    Server { child, base: format!("http://127.0.0.1:{port}"), logs }
}
/// 去掉 `ESC [ ... m` 颊色序列。
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c == '\x1b' && it.peek() == Some(&'[') {
            for d in it.by_ref() {
                if d.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
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
    // 流式 n=3：按 index 交错、每个 choice 都有 finish_reason、三者不同。不与非流式逐字比：3 个请求的 batch 组成 /
    // 调度顺序两次调用不保证相同，bf16 GEMM 随 batch 形状漂移会翻 near-tie token（同 seed 的逐 token 可复现由
    // `sampling_parity::seed_deterministic_and_batch_independent` 在固定 batch 形状下验证）。
    b["stream"] = serde_json::json!(true);
    let text = c.post(&url).json(&b).send().await.unwrap().text().await.unwrap();
    let (content, finish, _, done) = parse_sse(&text, 3);
    assert!(done);
    assert!(content.iter().all(|t| !t.trim().is_empty()), "every stream choice has content: {content:?}");
    assert!(content.iter().collect::<std::collections::HashSet<_>>().len() >= 2, "stream n=3 should differ");
    assert!(finish.iter().all(|f| f.is_some()), "every choice gets a finish_reason");
    // 单请求同 seed 两次：batch 形状相同（单独 prefill）→ 逐字相同
    let mut one = b.clone();
    one["n"] = serde_json::json!(1);
    one["stream"] = serde_json::json!(false);
    let a: serde_json::Value = c.post(&url).json(&one).send().await.unwrap().json().await.unwrap();
    let b2: serde_json::Value = c.post(&url).json(&one).send().await.unwrap().json().await.unwrap();
    assert_eq!(
        a["choices"][0]["message"]["content"], b2["choices"][0]["message"]["content"],
        "same seed, same single request → same text"
    );
    eprintln!("e2e: Q2-b request features PASS");
    drop(srv);
}

/// Q2-c：不传 `--num-blocks` → 按显存自动算；日志有 `kv_cache: auto`，N ≥ 24 GB 卡估算的 80%；服务能正常回答。
#[tokio::test]
#[ignore]
async fn e2e_qwen3_auto_num_blocks() {
    let srv = spawn_server_with(&[]);
    let line = srv
        .logs
        .iter()
        .find(|l| l.contains("kv_cache: auto"))
        .unwrap_or_else(|| panic!("no kv_cache auto log; got:\n{}", srv.logs.join("\n")))
        .clone();
    eprintln!("e2e: {line}");
    let field = |k: &str| -> f64 {
        let key = format!(" {k}=");
        let i = line.find(&key).unwrap_or_else(|| panic!("{k} missing in {line}")) + key.len();
        line[i..].split(|c: char| !(c.is_ascii_digit() || c == '.')).next().unwrap().parse().unwrap()
    };
    let n = field("num_blocks");
    let total = field("total_gib");
    // Qwen3-0.6B：28 层 × 16 × 8 × 2 × 128 × 2 B = 1.75 MiB / block；24 GB 卡 0.9 利用 − 权重 1.5 GB − 预留 ≈ 17 GB → ≈ 10k 块
    let per_block_gib = 28.0 * 16.0 * 8.0 * 2.0 * 128.0 * 2.0 / (1u64 << 30) as f64;
    let expect = (total * 0.9 - 1.5 - 1.0 - 1.9) / per_block_gib;
    eprintln!("e2e: num_blocks {n} vs 80% of estimate {:.0}", expect * 0.8);
    assert!(n >= expect * 0.8, "auto num_blocks {n} < 80% of estimate {expect:.0}");
    let c = reqwest::Client::new();
    let body = serde_json::json!({"messages": [{"role":"user","content":"Say hi."}], "max_tokens": 8, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": false}});
    let r = c.post(format!("{}/v1/chat/completions", srv.base)).json(&body).send().await.unwrap();
    assert_eq!(r.status(), 200);
    eprintln!("e2e: auto num_blocks PASS");
    drop(srv);
}

/// Q2-d：prefix cache 默认开。同一 system + 长历史发两次（max_tokens=1 让耗时 ≈ prefill）：第二次 TTFT 应明显低于第一次
/// （前缀块命中，只算最后一块）；回答仍正常。`/metrics` 命中率在 Q2-e 接上后补断言。
#[tokio::test]
#[ignore]
async fn e2e_qwen3_prefix_cache() {
    let srv = spawn_server_with(&["--num-blocks", "512"]);
    assert!(srv.logs.iter().any(|l| l.contains("prefix_caching=true")), "prefix caching should default on");
    let c = reqwest::Client::new();
    let url = format!("{}/v1/chat/completions", srv.base);
    let history: Vec<serde_json::Value> = (0..12)
        .flat_map(|i| {
            [
                serde_json::json!({"role":"user","content": format!("Fact {i}: the {i}th planet of system Zeta is named Z{i}. Please remember all facts; reply only OK.")}),
                serde_json::json!({"role":"assistant","content":"OK."}),
            ]
        })
        .collect();
    let mut msgs = vec![
        serde_json::json!({"role":"system","content":"You are a terse assistant. Answer with one word."}),
    ];
    msgs.extend(history);
    msgs.push(serde_json::json!({"role":"user","content":"What is the name of the 7th planet?"}));
    let body = serde_json::json!({"messages": msgs, "max_tokens": 1, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": false}});
    let mut times = Vec::new();
    let mut prompt_tokens = 0;
    for _ in 0..3 {
        let t0 = Instant::now();
        let v: serde_json::Value = c.post(&url).json(&body).send().await.unwrap().json().await.unwrap();
        times.push(t0.elapsed().as_secs_f64() * 1e3);
        prompt_tokens = v["usage"]["prompt_tokens"].as_u64().unwrap();
    }
    eprintln!("e2e: prefix cache prompt_tokens={prompt_tokens} ttft ms = {times:.1?}");
    assert!(prompt_tokens >= 300, "history should be long enough for several blocks");
    let (first, warm) = (times[0], times[1].min(times[2]));
    assert!(warm < first * 0.7, "second request should hit the prefix cache: {first:.1} ms → {warm:.1} ms");
    // 答案仍正常（greedy, 1 token）：拿到 200 且 content 非空
    let mut b = body.clone();
    b["max_tokens"] = serde_json::json!(4);
    let v: serde_json::Value = c.post(&url).json(&b).send().await.unwrap().json().await.unwrap();
    let content = v["choices"][0]["message"]["content"].as_str().unwrap_or("");
    eprintln!("e2e: prefix-cached answer = {content:?}");
    assert!(!content.trim().is_empty());
    eprintln!("e2e: prefix cache PASS");
    drop(srv);
}

/// Q2-e：`/metrics` Prometheus 文本：必备指标名；请求后 prompt/generation 计数与直方图增长；同一 prompt 第二次
/// `prefix_cache_hits_total` 增长；空闲时 `num_requests_running == 0`、`kv_cache_usage_perc == 0`。
#[tokio::test]
#[ignore]
async fn e2e_qwen3_metrics() {
    let srv = spawn_server_with(&["--num-blocks", "512"]);
    let c = reqwest::Client::new();
    let murl = format!("{}/metrics", srv.base);
    let url = format!("{}/v1/chat/completions", srv.base);
    let get = |c: &reqwest::Client| {
        let c = c.clone();
        let murl = murl.clone();
        async move {
            let r = c.get(&murl).send().await.unwrap();
            assert_eq!(r.status(), 200);
            assert!(r.headers()["content-type"].to_str().unwrap().starts_with("text/plain"));
            r.text().await.unwrap()
        }
    };
    let val = |text: &str, name: &str| -> f64 {
        text.lines()
            .find(|l| l.starts_with(name) && l[name.len()..].starts_with(' '))
            .unwrap_or_else(|| panic!("metric {name} missing:\n{text}"))
            .split(' ')
            .nth(1)
            .unwrap()
            .parse()
            .unwrap()
    };
    let m0 = get(&c).await;
    for n in [
        "dinfer:num_requests_running",
        "dinfer:num_requests_waiting",
        "dinfer:prompt_tokens_total",
        "dinfer:generation_tokens_total",
        "dinfer:kv_cache_usage_perc",
        "dinfer:prefix_cache_hits_total",
        "dinfer:prefix_cache_queries_total",
    ] {
        val(&m0, n);
    }
    for n in [
        "dinfer:time_to_first_token_seconds_bucket",
        "dinfer:time_per_output_token_seconds_bucket",
        "dinfer:e2e_request_latency_seconds_bucket",
        "dinfer:request_success_total{finished_reason=\"stop\"}",
    ] {
        assert!(m0.contains(n), "missing {n}");
    }
    assert_eq!(val(&m0, "dinfer:prompt_tokens_total"), 0.0);

    let body = serde_json::json!({"messages": [{"role":"system","content":"You are terse."},{"role":"user","content":"Name three colors, comma separated."}],
        "max_tokens": 12, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": false}});
    let v: serde_json::Value = c.post(&url).json(&body).send().await.unwrap().json().await.unwrap();
    let ptoks = v["usage"]["prompt_tokens"].as_f64().unwrap();
    let gtoks = v["usage"]["completion_tokens"].as_f64().unwrap();
    let m1 = get(&c).await;
    assert_eq!(val(&m1, "dinfer:prompt_tokens_total"), ptoks);
    assert_eq!(val(&m1, "dinfer:generation_tokens_total"), gtoks);
    assert_eq!(val(&m1, "dinfer:time_to_first_token_seconds_count"), 1.0);
    assert_eq!(val(&m1, "dinfer:e2e_request_latency_seconds_count"), 1.0);
    assert!(val(&m1, "dinfer:time_per_output_token_seconds_count") >= gtoks - 1.0);
    let hits1 = val(&m1, "dinfer:prefix_cache_hits_total");
    let q1 = val(&m1, "dinfer:prefix_cache_queries_total");
    assert_eq!(q1, ptoks - 1.0, "first request queries num_tokens-1");

    // 第二次同 prompt：命中增长（整块对齐：≥ (ptoks-1)/16 向下取整 × 16）
    let _: serde_json::Value = c.post(&url).json(&body).send().await.unwrap().json().await.unwrap();
    let m2 = get(&c).await;
    let hits2 = val(&m2, "dinfer:prefix_cache_hits_total");
    let expect_hit = ((ptoks - 1.0) / 16.0).floor() * 16.0;
    eprintln!("e2e: metrics prompt={ptoks} hits {hits1} → {hits2} (expect ≥ {expect_hit})");
    assert!(
        hits2 >= hits1 + expect_hit && expect_hit > 0.0,
        "prefix cache hits should grow on the repeated prompt"
    );
    // 空闲
    assert_eq!(val(&m2, "dinfer:num_requests_running"), 0.0);
    assert_eq!(val(&m2, "dinfer:num_requests_waiting"), 0.0);
    assert_eq!(val(&m2, "dinfer:kv_cache_usage_perc"), 0.0, "all KV blocks must be released when idle");
    assert_eq!(
        val(&m2, "dinfer:request_success_total{finished_reason=\"stop\"}")
            + val(&m2, "dinfer:request_success_total{finished_reason=\"length\"}"),
        2.0
    );
    eprintln!("e2e: metrics PASS");
    drop(srv);
}

/// splitmix64：测试内随机数，不引 rand。
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

async fn metrics_val(c: &reqwest::Client, base: &str, name: &str) -> f64 {
    let text = c.get(format!("{base}/metrics")).send().await.unwrap().text().await.unwrap();
    text.lines()
        .find(|l| l.starts_with(name) && l[name.len()..].starts_with(' '))
        .unwrap_or_else(|| panic!("metric {name} missing"))
        .split(' ')
        .nth(1)
        .unwrap()
        .parse()
        .unwrap()
}

/// 等引擎空闲（running==0 且 kv==0），最多 `secs` 秒。
async fn wait_idle(c: &reqwest::Client, base: &str, secs: u64) -> (f64, f64) {
    let t0 = Instant::now();
    loop {
        let running = metrics_val(c, base, "dinfer:num_requests_running").await;
        let kv = metrics_val(c, base, "dinfer:kv_cache_usage_perc").await;
        if (running == 0.0 && kv == 0.0) || t0.elapsed() > Duration::from_secs(secs) {
            return (running, kv);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Q2-f：健壮性。① 超 max_model_len 的 prompt → 400；② 非法 JSON → 4xx；③ 流式客户端中途断开 → 引擎 abort，KV 归零；
/// ④ 32 并发随机（prompt 10–300 token、max_tokens 1–200、随机采样参数、1/8 流式、1/8 n=3）跑 `STRESS_SECS`（默认 60）秒：
/// 无 5xx、无 panic、结束后 running==0 且 kv==0。
#[tokio::test]
#[ignore]
async fn e2e_qwen3_robustness() {
    let srv = spawn_server_with(&["--num-blocks", "1024", "--max-num-seqs", "32"]);
    let c = reqwest::Client::new();
    let url = format!("{}/v1/chat/completions", srv.base);
    let nothink = serde_json::json!({"enable_thinking": false});

    // ① 超长 prompt（max_model_len 2048）→ 400
    let huge = "word ".repeat(2600);
    let r = c
        .post(&url)
        .json(&serde_json::json!({"messages": [{"role":"user","content": huge}], "max_tokens": 1}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400, "oversized prompt must be rejected: {}", r.text().await.unwrap());
    // ② 非法 JSON / 缺字段
    let r = c.post(&url).header("content-type", "application/json").body("{not json").send().await.unwrap();
    assert!(r.status().is_client_error(), "malformed JSON → 4xx, got {}", r.status());
    let r = c.post(&url).json(&serde_json::json!({"messages": []})).send().await.unwrap();
    assert_eq!(r.status(), 400);

    // ③ 流式中途断开：收到前 3 个 chunk 就 drop 连接
    {
        let body = serde_json::json!({"messages": [{"role":"user","content":"Write a long story about a lighthouse."}],
            "max_tokens": 300, "temperature": 0.0, "stream": true, "chat_template_kwargs": nothink});
        let mut resp = c.post(&url).json(&body).send().await.unwrap();
        assert_eq!(resp.status(), 200);
        let mut got = 0;
        while let Some(chunk) = resp.chunk().await.unwrap() {
            got += chunk.len();
            if got > 200 {
                break;
            }
        }
        drop(resp);
        let (running, kv) = wait_idle(&c, &srv.base, 15).await;
        eprintln!("e2e: after client disconnect → running={running} kv={kv}");
        assert_eq!((running, kv), (0.0, 0.0), "disconnect must abort the request and free its KV");
        assert!(
            metrics_val(&c, &srv.base, "dinfer:request_success_total{finished_reason=\"abort\"}").await
                >= 1.0
        );
    }

    // ④ 并发随机压测
    let secs: u64 = env("STRESS_SECS", "60").parse().unwrap();
    let workers = 32usize;
    let deadline = Instant::now() + Duration::from_secs(secs);
    let mut handles = Vec::new();
    for w in 0..workers {
        let c = c.clone();
        let url = url.clone();
        let nothink = nothink.clone();
        handles.push(tokio::spawn(async move {
            let mut rng = 0xC0FFEE_u64 + w as u64 * 7919;
            let (mut ok, mut client_err, mut server_err, mut toks) = (0u64, 0u64, 0u64, 0u64);
            while Instant::now() < deadline {
                let r = splitmix(&mut rng);
                let words = 2 + (r % 220) as usize; // ≈ 10–300 token
                let prompt = format!("Topic {}: {}", r % 1000, "the quick brown fox jumps over the lazy dog ".repeat(words / 9 + 1));
                let max_tokens = 1 + (splitmix(&mut rng) % 200) as u32;
                let temp = [0.0, 0.7, 1.0, 1.3][(splitmix(&mut rng) % 4) as usize];
                let top_p = [1.0, 0.95, 0.8][(splitmix(&mut rng) % 3) as usize];
                let top_k = [-1, 20, 50][(splitmix(&mut rng) % 3) as usize];
                let stream = splitmix(&mut rng) % 8 == 0;
                let n = if splitmix(&mut rng) % 8 == 0 { 3 } else { 1 };
                let body = serde_json::json!({"messages": [{"role":"user","content": prompt}], "max_tokens": max_tokens,
                    "temperature": temp, "top_p": top_p, "top_k": top_k, "repetition_penalty": 1.1, "n": n,
                    "stream": stream, "chat_template_kwargs": nothink});
                let resp = match c.post(&url).json(&body).send().await {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("e2e: transport error {e}");
                        server_err += 1;
                        continue;
                    }
                };
                let st = resp.status();
                if st.is_server_error() {
                    eprintln!("e2e: 5xx {}", resp.text().await.unwrap_or_default());
                    server_err += 1;
                    break;
                }
                if st.is_client_error() {
                    client_err += 1;
                    continue;
                }
                if stream {
                    let t = resp.text().await.unwrap_or_default();
                    if !t.contains("[DONE]") {
                        eprintln!("e2e: stream without [DONE]");
                        server_err += 1;
                        break;
                    }
                } else {
                    let v: serde_json::Value = resp.json().await.unwrap();
                    toks += v["usage"]["completion_tokens"].as_u64().unwrap_or(0);
                }
                ok += 1;
            }
            (ok, client_err, server_err, toks)
        }));
    }
    let (mut ok, mut cerr, mut serr, mut toks) = (0, 0, 0, 0);
    for h in handles {
        let (a, b, d, t) = h.await.unwrap();
        ok += a;
        cerr += b;
        serr += d;
        toks += t;
    }
    eprintln!("e2e: stress {secs}s × {workers} workers: ok={ok} client_err={cerr} server_err={serr} completion_tokens={toks}");
    assert_eq!(serr, 0, "no 5xx / transport errors under load");
    assert_eq!(cerr, 0, "all generated requests were valid");
    assert!(ok >= workers as u64, "too few completed requests: {ok}");
    let (running, kv) = wait_idle(&c, &srv.base, 30).await;
    let waiting = metrics_val(&c, &srv.base, "dinfer:num_requests_waiting").await;
    eprintln!("e2e: after stress → running={running} waiting={waiting} kv={kv}");
    assert_eq!((running, waiting, kv), (0.0, 0.0, 0.0), "engine must drain and release all KV");
    // 服务仍健康
    let r = c.post(&url).json(&serde_json::json!({"messages": [{"role":"user","content":"Say ok."}], "max_tokens": 4, "temperature": 0.0})).send().await.unwrap();
    assert_eq!(r.status(), 200);
    eprintln!("e2e: robustness PASS");
    drop(srv);
}

/// Q3-r：KV host tier。GPU 只有 64 块（1024 token），host tier 1 GiB。
/// A（长前缀 ~700 token）→ B（另一个长前缀，把 A 挤出 GPU）→ A 再来：应从 host tier 拉回（`host_tier_hits_total` > 0），
/// 且 greedy 输出与第一次逐字相同。
#[tokio::test]
#[ignore]
async fn e2e_qwen3_host_tier() {
    // 64 块 = 1024 token；启动校验要求 max_model_len ≤ 90% 池 → 配 --max-model-len 896（A 的 ~700 前缀 + 6 输出装得下）
    let srv = spawn_server_with(&["--num-blocks", "64", "--max-model-len", "896", "--kv-host-gib", "1", "--max-num-seqs", "4"]);
    assert!(srv.logs.iter().any(|l| l.contains("kv host tier enabled")), "host tier should be enabled: {:?}", srv.logs.last());
    let c = reqwest::Client::new();
    let url = format!("{}/v1/chat/completions", srv.base);
    let murl = format!("{}/metrics", srv.base);
    let long_history = |tag: &str| -> Vec<serde_json::Value> {
        let mut v = vec![serde_json::json!({"role":"system","content":"You are a terse assistant. Answer with one word."})];
        for i in 0..16 {
            v.push(serde_json::json!({"role":"user","content": format!("{tag} fact {i}: the {i}th planet of system {tag} is named {tag}{i}. Please remember it.")}));
            v.push(serde_json::json!({"role":"assistant","content":"OK."}));
        }
        v
    };
    let ask = |tag: &str, q: &str| {
        let mut m = long_history(tag);
        m.push(serde_json::json!({"role":"user","content": q}));
        serde_json::json!({"messages": m, "max_tokens": 6, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": false}})
    };
    let get = |c: &reqwest::Client, body: serde_json::Value| {
        let c = c.clone();
        let url = url.clone();
        async move {
            let v: serde_json::Value = c.post(&url).json(&body).send().await.unwrap().json().await.unwrap();
            let pt = v["usage"]["prompt_tokens"].as_u64().unwrap_or_else(|| panic!("no usage in response: {v}"));
            (v["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string(), pt)
        }
    };
    let metric = |c: &reqwest::Client, name: &str| {
        let c = c.clone();
        let murl = murl.clone();
        let name = name.to_string();
        async move {
            let t = c.get(&murl).send().await.unwrap().text().await.unwrap();
            t.lines().find(|l| l.starts_with(&format!("dinfer:{name} "))).and_then(|l| l.split(' ').nth(1)).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0)
        }
    };
    let (a1, pa) = get(&c, ask("Zeta", "What is the name of the 7th planet?")).await;
    assert!(pa >= 450, "A prompt should be ≥ 450 tokens (≈ 28+ blocks), got {pa}");
    let off1 = metric(&c, "host_tier_offload_blocks_total").await;
    assert!(off1 > 0.0, "A should have been offloaded to host tier (offload={off1})");
    // B：不同前缀、同样长 → 把 A 的块挤出 GPU（GPU 共 64 块，A 占 ~40）
    let (_b1, pb) = get(&c, ask("Omega", "What is the name of the 3rd planet?")).await;
    assert!(pb >= 450);
    let hits0 = metric(&c, "host_tier_hits_total").await;
    // A 再来：GPU 前缀缓存应已丢失大部分，host tier 命中
    let (a2, _) = get(&c, ask("Zeta", "What is the name of the 7th planet?")).await;
    let hits1 = metric(&c, "host_tier_hits_total").await;
    let fetched = metric(&c, "host_tier_fetch_blocks_total").await;
    eprintln!("e2e: host tier offload={off1} hits {hits0}→{hits1} fetched={fetched} a1={a1:?} a2={a2:?}");
    assert!(hits1 > hits0, "second A should hit host tier ({hits0} → {hits1})");
    assert!(fetched > 0.0, "blocks should have been fetched from host");
    assert_eq!(a1, a2, "greedy output must be identical with host-tier-restored KV");
    eprintln!("e2e: host tier PASS");
    drop(srv);
}
