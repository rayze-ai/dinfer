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

/// 起 PD proxy 进程（`dinfer proxy`），等 /health。
fn spawn_proxy(prefill: &str, decode: &str) -> Server {
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let mut child = Command::new(env!("CARGO_BIN_EXE_dinfer"))
        .args(["proxy", "--port", &port.to_string(), "--prefill", prefill, "--decode", decode])
        .env("RUST_LOG", "info")
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn dinfer proxy");
    let base = format!("http://127.0.0.1:{port}");
    let t0 = Instant::now();
    loop {
        if std::net::TcpStream::connect(("127.0.0.1", port)).is_ok() {
            break;
        }
        if let Ok(Some(st)) = child.try_wait() {
            panic!("proxy exited early: {st}");
        }
        assert!(t0.elapsed() < Duration::from_secs(30), "proxy did not start");
        std::thread::sleep(Duration::from_millis(50));
    }
    Server { child, base, logs: Vec::new() }
}

/// PD 分离（design §10.3）：同卡三进程 P（--role prefill）+ D（--role decode）+ proxy。
/// 断言：① 经 proxy 的 greedy 输出与单机实例逐字相同（流式 + 非流式）；② D 侧 `pd_import_total ≥ 1`、
/// `pd_fallback_total == 0`，P 侧 `pd_export_total ≥ 1`；③ 杀掉 P 后再请求，proxy 回退 D 单机，输出仍与单机相同，
/// D 侧 fallback 不增（proxy 侧回退不经 D 的 Import 路径）。
#[tokio::test]
#[ignore]
async fn e2e_qwen3_pd() {
    let common = ["--num-blocks", "128", "--max-model-len", "1024", "--max-num-seqs", "4"];
    let single = spawn_server_with(&common);
    let kv_port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port().to_string();
    // PD_TRANSPORT=tcp|shm|cuda-ipc（design §10.3 b）：D 建区、P 跟随；默认 tcp。
    let transport = std::env::var("PD_TRANSPORT").unwrap_or_else(|_| "tcp".into());
    let mut d_args = common.to_vec();
    d_args.extend(["--role", "decode", "--kv-port", &kv_port, "--pd-wait-ms", "3000", "--kv-transport", &transport]);
    let d = spawn_server_with(&d_args);
    assert!(d.logs.iter().any(|l| l.contains("pd: kv receiver listening")), "D must listen: {:?}", d.logs.last());
    let peer = format!("127.0.0.1:{kv_port}");
    let mut p_args = common.to_vec();
    p_args.extend(["--role", "prefill", "--peer", &peer, "--kv-transport", &transport]);
    let p = spawn_server_with(&p_args);
    assert!(p.logs.iter().any(|l| l.contains("pd: prefill side will push KV")), "P must have peer: {:?}", p.logs.last());
    let proxy = spawn_proxy(&p.base, &d.base);
    let c = reqwest::Client::new();
    let body = |stream: bool| {
        serde_json::json!({
            "messages": [{"role":"user","content":"List three primary colors, comma separated, then stop."}],
            "max_tokens": 24, "temperature": 0.0, "stream": stream,
            "chat_template_kwargs": {"enable_thinking": false}
        })
    };
    let text_of = |c: &reqwest::Client, base: &str, stream: bool| {
        let c = c.clone();
        let url = format!("{base}/v1/chat/completions");
        let b = body(stream);
        async move {
            let r = c.post(&url).json(&b).send().await.unwrap();
            assert!(r.status().is_success(), "{url}: {}", r.status());
            if !stream {
                let v: serde_json::Value = r.json().await.unwrap();
                return v["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();
            }
            let t = r.text().await.unwrap();
            let mut out = String::new();
            for line in t.lines() {
                let Some(d) = line.strip_prefix("data: ") else { continue };
                if d == "[DONE]" {
                    break;
                }
                let v: serde_json::Value = serde_json::from_str(d).unwrap_or_else(|e| panic!("bad sse chunk {d}: {e}"));
                if let Some(s) = v["choices"][0]["delta"]["content"].as_str() {
                    out.push_str(s);
                }
            }
            out
        }
    };
    let want = text_of(&c, &single.base, false).await;
    assert!(!want.trim().is_empty(), "single produced nothing");
    let got_ns = text_of(&c, &proxy.base, false).await;
    assert_eq!(got_ns, want, "PD non-stream must equal single-instance greedy");
    let got_s = text_of(&c, &proxy.base, true).await;
    assert_eq!(got_s, want, "PD stream must equal single-instance greedy");
    let m = |base: &str, name: &str| {
        let c = c.clone();
        let base = base.to_string();
        let name = name.to_string();
        async move { metrics_val(&c, &base, &name).await }
    };
    assert!(m(&p.base, "dinfer:pd_export_total").await >= 2.0, "P exported");
    assert!(m(&d.base, "dinfer:pd_import_total").await >= 2.0, "D imported");
    assert_eq!(m(&d.base, "dinfer:pd_fallback_total").await, 0.0, "no D-side fallback on the happy path");
    // 故障：杀 P → proxy 回退到 D 单机
    drop(p);
    let got_fb = text_of(&c, &proxy.base, false).await;
    assert_eq!(got_fb, want, "with P dead, proxy must fall back to D-only and still match");
    assert_eq!(m(&d.base, "dinfer:pd_fallback_total").await, 0.0, "proxy-side fallback does not go through D Import");
    drop(proxy);
    drop(d);
    drop(single);
}

/// completions 流式：按 choice index 拼 `text`、记 finish_reason、统计 logprobs token 数、看到 [DONE]。
fn parse_cmpl_sse(text: &str, n: usize) -> (Vec<String>, Vec<Option<String>>, Vec<usize>, bool, Option<serde_json::Value>) {
    let mut out = vec![String::new(); n];
    let mut finish = vec![None; n];
    let mut lps = vec![0usize; n];
    let mut done = false;
    let mut usage = None;
    for line in text.lines() {
        let Some(d) = line.strip_prefix("data: ") else { continue };
        if d.trim() == "[DONE]" {
            done = true;
            continue;
        }
        let j: serde_json::Value = serde_json::from_str(d).unwrap();
        assert_eq!(j["object"], "text_completion", "chunk object: {d}");
        if j["usage"].is_object() {
            usage = Some(j["usage"].clone());
        }
        for ch in j["choices"].as_array().into_iter().flatten() {
            let i = ch["index"].as_u64().unwrap() as usize;
            out[i].push_str(ch["text"].as_str().unwrap_or(""));
            if let Some(f) = ch["finish_reason"].as_str() {
                finish[i] = Some(f.to_string());
            }
            if let Some(a) = ch["logprobs"]["tokens"].as_array() {
                lps[i] += a.len();
            }
        }
    }
    (out, finish, lps, done, usage)
}

/// API 补全 e2e（docs/loops/api-complete.md）：每个新端点正例 + 错例；completions 流式 == 非流式（greedy）；
/// echo 前缀；tokenize/detokenize 往返；tokenize(messages).count == chat usage.prompt_tokens；logprobs 形状；501 列表。
#[tokio::test]
#[ignore]
async fn e2e_qwen3_openai_surface() {
    let srv = spawn_server();
    let c = reqwest::Client::new();
    let b = &srv.base;
    let model: serde_json::Value = c.get(format!("{b}/v1/models")).send().await.unwrap().json().await.unwrap();
    let card = &model["data"][0];
    let name = card["id"].as_str().unwrap().to_string();
    for k in ["created", "root", "parent", "permission", "max_model_len", "owned_by"] {
        assert!(card.get(k).is_some(), "/v1/models card missing {k}: {card}");
    }
    assert!(card["created"].as_u64().unwrap() > 1_600_000_000);
    assert!(card["parent"].is_null() && card["permission"].as_array().unwrap().is_empty());

    // /v1/models/{id}
    let r = c.get(format!("{b}/v1/models/{name}")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let one: serde_json::Value = r.json().await.unwrap();
    assert_eq!(one["id"], name);
    let r = c.get(format!("{b}/v1/models/nope")).send().await.unwrap();
    assert_eq!(r.status(), 404);
    let e: serde_json::Value = r.json().await.unwrap();
    assert_eq!(e["error"]["type"], "NotFoundError");
    assert!(e["error"].get("param").is_some() && e["error"]["param"].is_null(), "{e}");
    assert_eq!(e["error"]["code"], 404);

    // /version /ping
    let v: serde_json::Value = c.get(format!("{b}/version")).send().await.unwrap().json().await.unwrap();
    assert_eq!(v["version"].as_str().unwrap(), env!("CARGO_PKG_VERSION"));
    assert_eq!(c.get(format!("{b}/ping")).send().await.unwrap().status(), 200);
    assert_eq!(c.post(format!("{b}/ping")).send().await.unwrap().status(), 200);

    // chat 的 model 不匹配 → 404
    let r = c
        .post(format!("{b}/v1/chat/completions"))
        .json(&serde_json::json!({"model":"nope","messages":[{"role":"user","content":"hi"}],"max_tokens":1}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);

    // 501 列表
    for p in [
        "/v1/embeddings",
        "/pooling",
        "/score",
        "/rerank",
        "/v1/rerank",
        "/v2/rerank",
        "/classify",
        "/v1/audio/transcriptions",
        "/v1/audio/translations",
        "/v1/responses",
    ] {
        let r = c.post(format!("{b}{p}")).json(&serde_json::json!({})).send().await.unwrap();
        assert_eq!(r.status(), 501, "{p}");
        let e: serde_json::Value = r.json().await.unwrap();
        assert_eq!(e["error"]["type"], "not_implemented", "{p}: {e}");
        assert_eq!(e["error"]["code"], 501);
    }

    // tokenize / detokenize
    let text = "The quick brown fox jumps over the lazy dog.";
    let t: serde_json::Value = c
        .post(format!("{b}/tokenize"))
        .json(&serde_json::json!({"prompt": text, "return_token_strs": true}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let toks = t["tokens"].as_array().unwrap().clone();
    assert_eq!(t["count"].as_u64().unwrap() as usize, toks.len());
    assert_eq!(t["max_model_len"], 2048);
    assert_eq!(t["token_strs"].as_array().unwrap().len(), toks.len());
    let d: serde_json::Value = c
        .post(format!("{b}/detokenize"))
        .json(&serde_json::json!({"tokens": toks}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(d["prompt"].as_str().unwrap(), text, "tokenize→detokenize roundtrip");
    // 错例：prompt 与 messages 都没给 / 都给 → 400；model 不匹配 → 404
    let r = c.post(format!("{b}/tokenize")).json(&serde_json::json!({})).send().await.unwrap();
    assert_eq!(r.status(), 400);
    let r = c
        .post(format!("{b}/tokenize"))
        .json(&serde_json::json!({"prompt":"a","messages":[{"role":"user","content":"a"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);
    let r = c.post(format!("{b}/detokenize")).json(&serde_json::json!({"model":"nope","tokens":[1]})).send().await.unwrap();
    assert_eq!(r.status(), 404);
    let r = c.post(format!("{b}/detokenize")).json(&serde_json::json!({"tokens":"x"})).send().await.unwrap();
    assert!(r.status().is_client_error(), "detokenize bad body: {}", r.status());

    // tokenize(messages).count == chat usage.prompt_tokens（同 chat_template_kwargs）
    let msgs = serde_json::json!([{"role":"user","content":"Say hello."}]);
    let kw = serde_json::json!({"enable_thinking": false});
    let t: serde_json::Value = c
        .post(format!("{b}/tokenize"))
        .json(&serde_json::json!({"messages": msgs, "chat_template_kwargs": kw}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let chat: serde_json::Value = c
        .post(format!("{b}/v1/chat/completions"))
        .json(&serde_json::json!({"messages": msgs, "chat_template_kwargs": kw, "max_tokens": 1}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(t["count"], chat["usage"]["prompt_tokens"], "tokenize(messages) vs chat prompt_tokens");

    // /v1/completions：非流式 greedy
    let url = format!("{b}/v1/completions");
    let prompt = "The capital of France is";
    let base = serde_json::json!({"prompt": prompt, "max_tokens": 12, "temperature": 0.0});
    let r = c.post(&url).json(&base).send().await.unwrap();
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
    let v: serde_json::Value = r.json().await.unwrap();
    assert_eq!(v["object"], "text_completion");
    assert_eq!(v["model"], name);
    let full = v["choices"][0]["text"].as_str().unwrap().to_string();
    eprintln!("e2e: completion = {full:?}");
    assert!(!full.is_empty());
    assert_eq!(v["choices"][0]["index"], 0);
    assert!(v["choices"][0]["finish_reason"].is_string());
    assert!(v["choices"][0].get("logprobs").is_none() || v["choices"][0]["logprobs"].is_null());
    let pt = v["usage"]["prompt_tokens"].as_u64().unwrap();
    let ct = v["usage"]["completion_tokens"].as_u64().unwrap();
    assert!(pt > 0 && ct > 0 && ct <= 12);
    assert_eq!(v["usage"]["total_tokens"].as_u64().unwrap(), pt + ct);
    // prompt_tokens == tokenize(prompt).count
    let t: serde_json::Value =
        c.post(format!("{b}/tokenize")).json(&serde_json::json!({"prompt": prompt})).send().await.unwrap().json().await.unwrap();
    assert_eq!(t["count"].as_u64().unwrap(), pt);

    // 默认 max_tokens = 16
    let mut b16 = base.clone();
    b16.as_object_mut().unwrap().remove("max_tokens");
    b16["ignore_eos"] = serde_json::json!(true);
    let v16: serde_json::Value = c.post(&url).json(&b16).send().await.unwrap().json().await.unwrap();
    assert_eq!(v16["usage"]["completion_tokens"], 16, "default max_tokens should be 16");
    assert_eq!(v16["choices"][0]["finish_reason"], "length");

    // 流式 == 非流式（greedy），含 usage chunk
    let mut bs = base.clone();
    bs["stream"] = serde_json::json!(true);
    bs["stream_options"] = serde_json::json!({"include_usage": true});
    let r = c.post(&url).json(&bs).send().await.unwrap();
    assert_eq!(r.status(), 200);
    assert!(r.headers()["content-type"].to_str().unwrap().starts_with("text/event-stream"));
    let (texts, fin, _, done, usage) = parse_cmpl_sse(&r.text().await.unwrap(), 1);
    assert!(done, "no [DONE]");
    assert_eq!(texts[0], full, "stream vs non-stream text differ");
    assert_eq!(fin[0].as_deref(), v["choices"][0]["finish_reason"].as_str());
    assert_eq!(usage.unwrap()["completion_tokens"].as_u64().unwrap(), ct);

    // echo=true：text 以 prompt 开头，其后 == 不 echo 的输出（非流式 + 流式）
    let mut be = base.clone();
    be["echo"] = serde_json::json!(true);
    let ve: serde_json::Value = c.post(&url).json(&be).send().await.unwrap().json().await.unwrap();
    let et = ve["choices"][0]["text"].as_str().unwrap();
    assert!(et.starts_with(prompt), "echo text {et:?} should start with prompt");
    assert_eq!(&et[prompt.len()..], full);
    be["stream"] = serde_json::json!(true);
    let (texts, _, _, done, _) = parse_cmpl_sse(&c.post(&url).json(&be).send().await.unwrap().text().await.unwrap(), 1);
    assert!(done);
    assert_eq!(texts[0], et, "echo stream vs non-stream");

    // token id prompt == 字符串 prompt（同一组 ids）
    let mut bi = base.clone();
    bi["prompt"] = t["tokens"].clone();
    let vi: serde_json::Value = c.post(&url).json(&bi).send().await.unwrap().json().await.unwrap();
    assert_eq!(vi["choices"][0]["text"].as_str().unwrap(), full, "token-id prompt vs string prompt");

    // 多 prompt × n=2：choice index = prompt_idx*n + i，共 4 个
    let bm = serde_json::json!({"prompt": [prompt, "Hello, my name is"], "max_tokens": 4, "n": 2, "temperature": 0.0, "seed": 1});
    let vm: serde_json::Value = c.post(&url).json(&bm).send().await.unwrap().json().await.unwrap();
    let ch = vm["choices"].as_array().unwrap();
    assert_eq!(ch.len(), 4);
    let idx: Vec<u64> = ch.iter().map(|x| x["index"].as_u64().unwrap()).collect();
    assert_eq!(idx, vec![0, 1, 2, 3]);
    assert_eq!(vm["usage"]["prompt_tokens"].as_u64().unwrap(), pt + {
        let t2: serde_json::Value = c
            .post(format!("{b}/tokenize"))
            .json(&serde_json::json!({"prompt": "Hello, my name is"}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        t2["count"].as_u64().unwrap()
    });
    // 流式多 prompt：4 个 index 都收到 finish
    let mut bms = bm.clone();
    bms["stream"] = serde_json::json!(true);
    let (_, fin, _, done, _) = parse_cmpl_sse(&c.post(&url).json(&bms).send().await.unwrap().text().await.unwrap(), 4);
    assert!(done && fin.iter().all(|f| f.is_some()), "{fin:?}");

    // stop 字符串：非流式截断 + finish=stop
    let mut bst = serde_json::json!({"prompt": "1, 2, 3, 4, 5, 6,", "max_tokens": 32, "temperature": 0.0});
    let vst0: serde_json::Value = c.post(&url).json(&bst).send().await.unwrap().json().await.unwrap();
    let s0 = vst0["choices"][0]["text"].as_str().unwrap().to_string();
    eprintln!("e2e: counting = {s0:?}");
    if s0.contains("9") {
        bst["stop"] = serde_json::json!(["9"]);
        let vst: serde_json::Value = c.post(&url).json(&bst).send().await.unwrap().json().await.unwrap();
        let s1 = vst["choices"][0]["text"].as_str().unwrap();
        assert!(!s1.contains("9") && s0.starts_with(s1), "stop: {s1:?} vs {s0:?}");
        assert_eq!(vst["choices"][0]["finish_reason"], "stop");
    }

    // logprobs=2：四列等长 == completion_tokens；每个 top_logprobs 2..=3 项且含 chosen token；text_offset 单调
    let mut bl = base.clone();
    bl["logprobs"] = serde_json::json!(2);
    bl["max_tokens"] = serde_json::json!(6);
    let vl: serde_json::Value = c.post(&url).json(&bl).send().await.unwrap().json().await.unwrap();
    let lp = &vl["choices"][0]["logprobs"];
    let nt = vl["usage"]["completion_tokens"].as_u64().unwrap() as usize;
    for k in ["tokens", "token_logprobs", "top_logprobs", "text_offset"] {
        assert_eq!(lp[k].as_array().unwrap().len(), nt, "logprobs.{k} len");
    }
    let toks = lp["tokens"].as_array().unwrap();
    for (i, top) in lp["top_logprobs"].as_array().unwrap().iter().enumerate() {
        let m = top.as_object().unwrap();
        assert!((1..=3).contains(&m.len()), "top_logprobs[{i}] len {}", m.len());
        assert!(m.len() <= 2 + 1);
        assert!(m.contains_key(toks[i].as_str().unwrap()), "top_logprobs[{i}] missing chosen {}", toks[i]);
    }
    let offs: Vec<u64> = lp["text_offset"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
    assert_eq!(offs[0], 0);
    assert!(offs.windows(2).all(|w| w[0] <= w[1]));
    assert_eq!(lp["tokens"].as_array().unwrap().iter().map(|t| t.as_str().unwrap()).collect::<String>(), vl["choices"][0]["text"]);
    // logprobs=0：仍有 chosen token 的 logprob，top_logprobs 每项恰 1
    bl["logprobs"] = serde_json::json!(0);
    let vl0: serde_json::Value = c.post(&url).json(&bl).send().await.unwrap().json().await.unwrap();
    let lp0 = &vl0["choices"][0]["logprobs"];
    assert!(!lp0["token_logprobs"].as_array().unwrap().is_empty());
    assert!(lp0["top_logprobs"].as_array().unwrap().iter().all(|t| t.as_object().unwrap().len() == 1));
    // 流式 logprobs：每个 chunk 带 tokens，总数 == completion_tokens
    bl["logprobs"] = serde_json::json!(2);
    bl["stream"] = serde_json::json!(true);
    let (_, _, lpn, done, _) = parse_cmpl_sse(&c.post(&url).json(&bl).send().await.unwrap().text().await.unwrap(), 1);
    assert!(done);
    assert_eq!(lpn[0], nt, "stream logprobs token count");

    // 错例：suffix 400 / 空 prompt 400 / 混合数组 400 / logprobs 21 400 / model 不匹配 404 / n=9 400
    for (body, st) in [
        (serde_json::json!({"prompt":"a","suffix":"b"}), 400),
        (serde_json::json!({"prompt":""}), 400),
        (serde_json::json!({"prompt":[]}), 400),
        (serde_json::json!({"prompt":["a", 1]}), 400),
        (serde_json::json!({"prompt":[[1,2],"a"]}), 400),
        (serde_json::json!({"prompt":"a","logprobs":21}), 400),
        (serde_json::json!({"prompt":"a","model":"nope"}), 404),
        (serde_json::json!({"prompt":"a","n":9}), 400),
        (serde_json::json!({"prompt":"a","temperature":-1.0}), 400),
    ] {
        let r = c.post(&url).json(&body).send().await.unwrap();
        assert_eq!(r.status(), st, "{body}: {}", r.text().await.unwrap());
    }
    let r = c.post(&url).json(&serde_json::json!({"prompt":"a","model":"nope"})).send().await.unwrap();
    let e: serde_json::Value = r.json().await.unwrap();
    assert_eq!(e["error"]["type"], "NotFoundError");
    assert_eq!(e["error"]["code"], 404);
    assert!(e["error"]["param"].is_null());
    let _ = srv.logs.len();
}

/// A4 n-gram 投机解码（docs/loops/ngram-spec.md）：同一组 greedy prompt 先关投机、再开投机（`--speculative ngram`）。
/// 正确性口径对齐 vLLM `tests/v1/e2e/test_cpu_spec_decode.py` 的 `check_logprobs_close`：逐 token 比较到**首个不同
/// token**，该处投机侧选的 token 必须在基线的 top-2 里且基线 top-1/top-2 logprob 差 < 0.5（并列）——draft 验证行走 paged
/// 多行 attention / 多行 GEMM，KV 与 logits 与单行 decode 有 bf16 级差异，只会在并列处翻转；在明显非并列处翻转就是 bug。
/// 另要求至少 1 个普通 prompt 逐 token 完全一致（防止"全在并列处翻转"掩盖系统性偏移）。
/// 重复性 prompt 的接受率 `spec_accepted_total / spec_draft_total > 0.5`；普通 prompt 总耗时不比关闭慢 > 10%（+100 ms 噪声余量）。
/// GPU 只有一张：两套 server 顺序起。
#[tokio::test]
#[ignore]
async fn e2e_qwen3_ngram() {
    let c = reqwest::Client::new();
    let passage = "The quick brown fox jumps over the lazy dog while the small grey cat sleeps quietly under the old \
wooden table in the corner of the warm kitchen, and the clock on the wall keeps ticking slowly through the long afternoon.";
    let qs = [
        "Explain what a hash map is in two sentences.",
        "Write a short poem about the sea.",
        "List three uses of Rust and why.",
        "What is the capital of France? Answer briefly.",
        "Describe how a bicycle works in one paragraph.",
    ];
    let normal: Vec<serde_json::Value> = qs
        .iter()
        .map(|q| {
            serde_json::json!({"messages": [{"role":"user","content":q}], "max_tokens": 64, "temperature": 0.0,
                "logprobs": true, "top_logprobs": 2})
        })
        .collect();
    let repeat = serde_json::json!({
        "messages": [{"role":"user","content": format!("Repeat the following three times verbatim: {passage}")}],
        "max_tokens": 200, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": false}
    });
    let common = ["--num-blocks", "512", "--max-num-seqs", "8"];

    /// (文本, completion_tokens, 耗时, 每 token (token, logprob, top-1/top-2 gap, top-2 token 列表))
    struct Out {
        text: String,
        n: u64,
        el: Duration,
        toks: Vec<(String, f64, f64, Vec<String>)>,
    }
    async fn run(c: &reqwest::Client, base: &str, body: &serde_json::Value) -> Out {
        let t = Instant::now();
        let r = c.post(format!("{base}/v1/chat/completions")).json(body).send().await.unwrap();
        assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
        let v: serde_json::Value = r.json().await.unwrap();
        let el = t.elapsed();
        let m = &v["choices"][0]["message"];
        let text = format!("{}|{}", m["reasoning_content"].as_str().unwrap_or(""), m["content"].as_str().unwrap_or(""));
        let toks = v["choices"][0]["logprobs"]["content"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|t| {
                        let lp = t["logprob"].as_f64().unwrap();
                        let tl = t["top_logprobs"].as_array().unwrap();
                        let tops: Vec<f64> = tl.iter().map(|x| x["logprob"].as_f64().unwrap()).collect();
                        let top_toks: Vec<String> = tl.iter().map(|x| x["token"].as_str().unwrap_or("").to_string()).collect();
                        let gap = if tops.len() >= 2 { tops[0] - tops[1] } else { f64::INFINITY };
                        (t["token"].as_str().unwrap_or("").to_string(), lp, gap, top_toks)
                    })
                    .collect()
            })
            .unwrap_or_default();
        Out { text, n: v["usage"]["completion_tokens"].as_u64().unwrap(), el, toks }
    }

    // 1) 关投机：基线
    let (mut base_out, mut base_t) = (Vec::new(), Duration::ZERO);
    let base_rep;
    {
        let srv = spawn_server_with(&common);
        let _ = run(&c, &srv.base, &normal[0]).await; // 预热
        for b in &normal {
            let o = run(&c, &srv.base, b).await;
            base_t += o.el;
            base_out.push(o);
        }
        base_rep = run(&c, &srv.base, &repeat).await;
        assert_eq!(metrics_val(&c, &srv.base, "dinfer:spec_draft_total").await, 0.0, "spec off must not draft");
        drop(srv);
    }
    eprintln!("e2e: ngram off  normal {:?}  repeat {} tok {:?}", base_t, base_rep.n, base_rep.el);

    // 2) 开投机
    let mut args: Vec<&str> = common.to_vec();
    args.extend(["--speculative", "ngram", "--num-speculative-tokens", "3", "--ngram-min", "2", "--ngram-max", "4"]);
    let srv = spawn_server_with(&args);
    let _ = run(&c, &srv.base, &normal[0]).await;
    let mut spec_t = Duration::ZERO;
    let mut identical = 0usize;
    for (i, b) in normal.iter().enumerate() {
        let o = run(&c, &srv.base, b).await;
        spec_t += o.el;
        let bo = &base_out[i];
        if o.text == bo.text && o.n == bo.n {
            identical += 1;
            continue;
        }
        // 分叉：首个不同 token 处，投机侧 token ∈ 基线 top-2 且基线处近似并列（vLLM check_logprobs_close）
        let k = bo.toks.iter().zip(o.toks.iter()).position(|(a, b)| a.0 != b.0).unwrap_or(bo.toks.len().min(o.toks.len()));
        let (tok, lp, gap, tops) = bo.toks.get(k).cloned().unwrap_or_default();
        let spec_tok = o.toks.get(k).map(|t| t.0.clone()).unwrap_or_default();
        eprintln!("e2e: prompt {i} diverges at token {k}: base {tok:?} (lp {lp:.3}, top2 gap {gap:.3}, top2 {tops:?}) vs spec {spec_tok:?}");
        assert!(
            tops.contains(&spec_tok) && gap < 0.5,
            "prompt {i}: spec on/off diverge at token {k} where base is not a near-tie (gap {gap:.3}, top2 {tops:?}): {tok:?} vs {spec_tok:?}"
        );
    }
    eprintln!("e2e: {identical}/{} prompts identical with spec on", normal.len());
    assert!(identical >= 1, "no prompt identical with spec on ({identical}/{})", normal.len());

    let d0 = metrics_val(&c, &srv.base, "dinfer:spec_draft_total").await;
    let a0 = metrics_val(&c, &srv.base, "dinfer:spec_accepted_total").await;
    let rep = run(&c, &srv.base, &repeat).await;
    let d1 = metrics_val(&c, &srv.base, "dinfer:spec_draft_total").await;
    let a1 = metrics_val(&c, &srv.base, "dinfer:spec_accepted_total").await;
    assert_eq!(rep.n, base_rep.n, "repeat: completion_tokens differ");
    assert_eq!(rep.text, base_rep.text, "repeat: text differs (spec on vs off)");
    let rate_all = if d1 > 0.0 { a1 / d1 } else { 0.0 };
    let rate_rep = if d1 > d0 { (a1 - a0) / (d1 - d0) } else { 0.0 };
    eprintln!(
        "e2e: ngram on   normal {:?} ({:+.1}% vs off)  repeat {} tok {:?} ({:+.1}%)  accept: normal {}/{} repeat {:.3} ({}/{}) overall {:.3}",
        spec_t,
        (spec_t.as_secs_f64() / base_t.as_secs_f64() - 1.0) * 100.0,
        rep.n,
        rep.el,
        (rep.el.as_secs_f64() / base_rep.el.as_secs_f64() - 1.0) * 100.0,
        a0,
        d0,
        rate_rep,
        a1 - a0,
        d1 - d0,
        rate_all
    );
    assert!(d1 > d0, "repeat prompt must produce drafts");
    assert!(rate_rep > 0.5, "repeat prompt acceptance {rate_rep:.3} <= 0.5");
    assert!(
        spec_t.as_secs_f64() <= base_t.as_secs_f64() * 1.10 + 0.1,
        "normal prompts slower with spec on: {spec_t:?} vs {base_t:?}"
    );
    eprintln!("e2e: ngram PASS");
    drop(srv);
}

/// 受限解码 + chat 字段补全（docs/loops/guided-decoding.md）。
#[tokio::test]
#[ignore]
async fn e2e_qwen3_guided() {
    let srv = spawn_server();
    let c = reqwest::Client::new();
    let url = format!("{}/v1/chat/completions", srv.base);
    let post = |b: serde_json::Value| {
        let c = c.clone();
        let url = url.clone();
        async move {
            let r = c.post(&url).json(&b).send().await.unwrap();
            let st = r.status();
            let v: serde_json::Value = r.json().await.unwrap();
            (st, v)
        }
    };
    let no_think = serde_json::json!({"enable_thinking": false});
    // ---- 1) response_format=json_object：5 个 prompt 全部能 parse ----
    for q in [
        "Describe a cat as JSON.",
        "Give me a JSON object with your name and age.",
        "List three colors.",
        "Who are you?",
        "Return the weather in Beijing.",
    ] {
        let (st, v) = post(serde_json::json!({
            "messages": [{"role":"user","content": q}], "max_tokens": 96, "temperature": 0.0,
            "response_format": {"type":"json_object"}, "chat_template_kwargs": no_think
        }))
        .await;
        assert_eq!(st, 200, "{v}");
        let txt = v["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();
        eprintln!("e2e: json_object({q:?}) → {txt:?} finish={}", v["choices"][0]["finish_reason"]);
        if v["choices"][0]["finish_reason"] == "length" {
            continue; // 被 max_tokens 截断的 JSON 不完整是预期（文法只保证前缀合法）
        }
        let parsed: serde_json::Value = serde_json::from_str(&txt).unwrap_or_else(|e| panic!("not JSON: {e}: {txt:?}"));
        assert!(parsed.is_object(), "json_object must yield an object: {txt:?}");
    }
    // ---- 2) json_schema：jsonschema 校验 ----
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3}
        },
        "required": ["name", "age"],
        "additionalProperties": false
    });
    let (st, v) = post(serde_json::json!({
        "messages": [{"role":"user","content":"Invent a person with a name, age and some tags."}],
        "max_tokens": 96, "temperature": 0.0,
        "response_format": {"type":"json_schema","json_schema":{"name":"person","schema": schema}},
        "chat_template_kwargs": no_think
    }))
    .await;
    assert_eq!(st, 200, "{v}");
    let txt = v["choices"][0]["message"]["content"].as_str().unwrap().to_string();
    eprintln!("e2e: json_schema → {txt:?}");
    let parsed: serde_json::Value = serde_json::from_str(&txt).expect("schema output parses");
    let validator = jsonschema::validator_for(&schema).unwrap();
    assert!(validator.is_valid(&parsed), "schema violated: {txt}");
    // 坏 schema / 坏 response_format.type → 400
    let (st, _) = post(serde_json::json!({"messages":[{"role":"user","content":"x"}],"max_tokens":1,
        "response_format":{"type":"json_schema","json_schema":{"schema":{"type":"nope"}}}})).await;
    assert_eq!(st, 400);
    let (st, _) = post(serde_json::json!({"messages":[{"role":"user","content":"x"}],"max_tokens":1,
        "response_format":{"type":"yaml"}})).await;
    assert_eq!(st, 400);
    // ---- 3) tool_choice ----
    let tools = serde_json::json!([
        {"type":"function","function":{"name":"get_weather","description":"Get weather of a city",
            "parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["c","f"]}},"required":["city"]}}},
        {"type":"function","function":{"name":"get_time","description":"Current time in a timezone",
            "parameters":{"type":"object","properties":{"tz":{"type":"string"}},"required":["tz"]}}}
    ]);
    let msgs = serde_json::json!([{"role":"user","content":"What's the weather like in Beijing right now?"}]);
    // required → ≥ 1 tool_call，arguments 合法 JSON，finish_reason = tool_calls
    let (st, v) = post(serde_json::json!({"messages": msgs, "tools": tools, "tool_choice": "required",
        "max_tokens": 96, "temperature": 0.0, "chat_template_kwargs": no_think})).await;
    assert_eq!(st, 200, "{v}");
    let tcs = v["choices"][0]["message"]["tool_calls"].as_array().cloned().unwrap_or_default();
    eprintln!("e2e: tool_choice=required → {tcs:?} finish={}", v["choices"][0]["finish_reason"]);
    assert!(!tcs.is_empty(), "required must yield a tool_call: {v}");
    for tc in &tcs {
        let name = tc["function"]["name"].as_str().unwrap();
        assert!(name == "get_weather" || name == "get_time", "unknown fn {name}");
        let args: serde_json::Value = serde_json::from_str(tc["function"]["arguments"].as_str().unwrap()).expect("arguments JSON");
        assert!(args.is_object());
    }
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    assert!(v["choices"][0]["message"]["content"].is_null());
    // 流式：最后一个 chunk 的 finish_reason 也是 tool_calls
    let r = c
        .post(&url)
        .json(&serde_json::json!({"messages": msgs, "tools": tools, "tool_choice": "required", "stream": true,
            "max_tokens": 96, "temperature": 0.0, "chat_template_kwargs": no_think}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let (_, finish, _, done) = parse_sse(&r.text().await.unwrap(), 1);
    assert!(done && finish[0].as_deref() == Some("tool_calls"), "stream finish={finish:?}");
    // 指定函数：name 一致；parallel_tool_calls=false → 恰 1 个
    let (st, v) = post(serde_json::json!({"messages": msgs, "tools": tools, "parallel_tool_calls": false,
        "tool_choice": {"type":"function","function":{"name":"get_time"}},
        "max_tokens": 96, "temperature": 0.0, "chat_template_kwargs": no_think})).await;
    assert_eq!(st, 200, "{v}");
    let tcs = v["choices"][0]["message"]["tool_calls"].as_array().cloned().unwrap_or_default();
    eprintln!("e2e: tool_choice=get_time → {tcs:?}");
    assert_eq!(tcs.len(), 1, "{v}");
    assert_eq!(tcs[0]["function"]["name"], "get_time");
    let args: serde_json::Value = serde_json::from_str(tcs[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert!(args["tz"].is_string(), "{args}");
    // none → 无 tool_call，finish_reason != tool_calls
    let (st, v) = post(serde_json::json!({"messages": msgs, "tools": tools, "tool_choice": "none",
        "max_tokens": 32, "temperature": 0.0, "chat_template_kwargs": no_think})).await;
    assert_eq!(st, 200, "{v}");
    assert!(v["choices"][0]["message"]["tool_calls"].is_null(), "{v}");
    assert_ne!(v["choices"][0]["finish_reason"], "tool_calls");
    // 错例：未知函数名 / 没 tools 的 required
    let (st, _) = post(serde_json::json!({"messages": msgs, "tools": tools, "max_tokens": 1,
        "tool_choice": {"type":"function","function":{"name":"nope"}}})).await;
    assert_eq!(st, 400);
    let (st, _) = post(serde_json::json!({"messages": msgs, "tool_choice": "required", "max_tokens": 1})).await;
    assert_eq!(st, 400);
    // ---- 4) logit_bias：把 greedy 首 token 置 -100 → 输出首 token 变了 ----
    let base = serde_json::json!({
        "messages": [{"role":"user","content":"Count from 1 to 20 separated by commas, nothing else."}],
        "max_tokens": 8, "temperature": 0.0, "chat_template_kwargs": no_think, "return_token_ids": true
    });
    let (st, v0) = post(base.clone()).await;
    assert_eq!(st, 200);
    let ids0 = v0["token_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect::<Vec<_>>();
    let first = ids0[0];
    let mut b = base.clone();
    b["logit_bias"] = serde_json::json!({first.to_string(): -100});
    let (st, v1) = post(b).await;
    assert_eq!(st, 200, "{v1}");
    let ids1 = v1["token_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect::<Vec<_>>();
    eprintln!("e2e: logit_bias first={first} → {:?} vs {:?}", &ids0[..3.min(ids0.len())], &ids1[..3.min(ids1.len())]);
    assert_ne!(ids1[0], first, "logit_bias -100 must suppress the greedy first token");
    assert!(!ids1.contains(&first), "token {first} must not appear with -100 bias: {ids1:?}");
    let mut b = base.clone();
    b["logit_bias"] = serde_json::json!({"abc": 1});
    assert_eq!(post(b).await.0, 400);
    // ---- 5) prompt_logprobs：长度 = prompt_tokens，第 0 项 null，其余含 rank/logprob/decoded_token ----
    let mut b = base.clone();
    b["prompt_logprobs"] = serde_json::json!(2);
    b["max_tokens"] = serde_json::json!(2);
    let (st, v) = post(b).await;
    assert_eq!(st, 200, "{v}");
    let plp = v["prompt_logprobs"].as_array().expect("prompt_logprobs present");
    let pt = v["usage"]["prompt_tokens"].as_u64().unwrap() as usize;
    assert_eq!(plp.len(), pt, "prompt_logprobs len vs prompt_tokens");
    assert!(plp[0].is_null());
    let filled = plp.iter().skip(1).filter(|x| !x.is_null()).count();
    assert_eq!(filled, pt - 1, "all prompt positions ≥ 1 must be filled (prompt fits one chunk)");
    let ids = v["prompt_token_ids"].as_array().unwrap();
    for (i, e) in plp.iter().enumerate().skip(1) {
        let m = e.as_object().unwrap();
        assert!((1..=3).contains(&m.len()), "pos {i}: {m:?}");
        let tok = ids[i].as_u64().unwrap().to_string();
        let mine = &m[&tok];
        assert!(mine["logprob"].as_f64().unwrap() <= 0.0 && mine["rank"].as_u64().unwrap() >= 1 && mine["decoded_token"].is_string(), "pos {i}: {mine}");
    }
    assert!(v["choices"][0]["logprobs"].is_null(), "prompt_logprobs must not turn on completion logprobs");
    // ---- 6) truncate_prompt_tokens=8 → usage.prompt_tokens == 8 ----
    let mut b = base.clone();
    b["truncate_prompt_tokens"] = serde_json::json!(8);
    b["max_tokens"] = serde_json::json!(1);
    let (st, v) = post(b).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["usage"]["prompt_tokens"], 8);
    // ---- 7) continue_final_message / add_generation_prompt；/tokenize 放开 add_generation_prompt=false ----
    let msgs2 = serde_json::json!([{"role":"user","content":"Say hello."},{"role":"assistant","content":"Hello, my name"}]);
    let (st, v) = post(serde_json::json!({"messages": msgs2, "continue_final_message": true, "add_generation_prompt": false,
        "max_tokens": 4, "temperature": 0.0, "return_token_ids": true})).await;
    assert_eq!(st, 200, "{v}");
    let pids: Vec<u32> = v["prompt_token_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    let tail: serde_json::Value = c
        .post(format!("{}/detokenize", srv.base))
        .json(&serde_json::json!({"tokens": pids}))
        .send().await.unwrap().json().await.unwrap();
    let ptxt = tail["prompt"].as_str().unwrap();
    assert!(ptxt.ends_with("Hello, my name"), "continue_final_message prompt must end with assistant prefix: {ptxt:?}");
    assert!(!ptxt.ends_with("<|im_end|>\n"), "{ptxt:?}");
    let (st, _) = post(serde_json::json!({"messages": msgs2, "continue_final_message": true, "add_generation_prompt": true, "max_tokens": 1})).await;
    assert_eq!(st, 400);
    let t1: serde_json::Value = c.post(format!("{}/tokenize", srv.base))
        .json(&serde_json::json!({"messages": [{"role":"user","content":"hi"}], "add_generation_prompt": false}))
        .send().await.unwrap().json().await.unwrap();
    let t2: serde_json::Value = c.post(format!("{}/tokenize", srv.base))
        .json(&serde_json::json!({"messages": [{"role":"user","content":"hi"}], "add_generation_prompt": true}))
        .send().await.unwrap().json().await.unwrap();
    assert!(t1["count"].as_u64().unwrap() < t2["count"].as_u64().unwrap(), "{t1} vs {t2}");
    // user 字段接受忽略；skip_special_tokens=false 时特殊 token 以字面出现（eos 本身不进文本，用 stop_token 之外的 <|im_end|> 不好造 → 只验不报错）
    let (st, _) = post(serde_json::json!({"messages": msgs, "max_tokens": 2, "user": "u1", "skip_special_tokens": false, "spaces_between_special_tokens": false})).await;
    assert_eq!(st, 200);
    // ---- 8) completions：echo + logprobs 的 prompt 段（第 0 项 null）；多 prompt 逐 prompt max_tokens 余量 ----
    let curl = format!("{}/v1/completions", srv.base);
    let v: serde_json::Value = c.post(&curl)
        .json(&serde_json::json!({"prompt": "1, 2, 3, 4,", "max_tokens": 3, "temperature": 0.0, "echo": true, "logprobs": 1}))
        .send().await.unwrap().json().await.unwrap();
    let lp = &v["choices"][0]["logprobs"];
    let pt = v["usage"]["prompt_tokens"].as_u64().unwrap() as usize;
    let ct = v["usage"]["completion_tokens"].as_u64().unwrap() as usize;
    assert_eq!(lp["tokens"].as_array().unwrap().len(), pt + ct, "{lp}");
    assert!(lp["token_logprobs"][0].is_null() && lp["top_logprobs"][0].is_null(), "{lp}");
    assert!(lp["token_logprobs"][1].is_number() && lp["token_logprobs"][pt].is_number(), "{lp}");
    assert_eq!(lp["text_offset"][0], 0);
    assert_eq!(lp["tokens"].as_array().unwrap().iter().map(|t| t.as_str().unwrap()).collect::<String>(), v["choices"][0]["text"]);
    // 多 prompt：短 prompt 的余量比长的多 → max_tokens 很大时 completion_tokens 不同（max_model_len 2048）
    let long: Vec<u32> = (0..2040).map(|i| 1000 + (i % 100) as u32).collect();
    let v: serde_json::Value = c.post(&curl)
        .json(&serde_json::json!({"prompt": [[1, 2, 3], long], "max_tokens": 20, "temperature": 0.0, "ignore_eos": true}))
        .send().await.unwrap().json().await.unwrap();
    let texts = v["choices"].as_array().unwrap();
    assert_eq!(texts.len(), 2, "{v}");
    assert_eq!(texts[0]["finish_reason"], "length");
    assert_eq!(texts[1]["finish_reason"], "length");
    drop(srv);
    // ---- 9) `--speculative ngram` 开着：带文法的请求不提案（runner 门控），输出仍合法；无文法请求照常 ----
    let srv = spawn_server_with(&["--num-blocks", "256", "--no-prefix-caching", "--speculative", "ngram",
        "--num-speculative-tokens", "3", "--ngram-min", "2", "--ngram-max", "4"]);
    let url2 = format!("{}/v1/chat/completions", srv.base);
    for q in ["Describe a cat as JSON.", "Return the weather in Beijing as JSON with city, temp, humidity."] {
        let v: serde_json::Value = c.post(&url2).json(&serde_json::json!({
            "messages": [{"role":"user","content": q}], "max_tokens": 96, "temperature": 0.0,
            "response_format": {"type":"json_object"}, "chat_template_kwargs": no_think
        })).send().await.unwrap().json().await.unwrap();
        let txt = v["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();
        eprintln!("e2e: ngram+json_object({q:?}) → {txt:?} finish={}", v["choices"][0]["finish_reason"]);
        if v["choices"][0]["finish_reason"] != "length" {
            let p: serde_json::Value = serde_json::from_str(&txt).unwrap_or_else(|e| panic!("not JSON with ngram on: {e}: {txt:?}"));
            assert!(p.is_object());
        }
    }
    let v: serde_json::Value = c.post(&url2).json(&serde_json::json!({"messages": msgs, "tools": tools, "tool_choice": "required",
        "max_tokens": 96, "temperature": 0.0, "chat_template_kwargs": no_think})).send().await.unwrap().json().await.unwrap();
    let tcs = v["choices"][0]["message"]["tool_calls"].as_array().cloned().unwrap_or_default();
    assert!(!tcs.is_empty() && v["choices"][0]["finish_reason"] == "tool_calls", "ngram + required: {v}");
    let _: serde_json::Value = serde_json::from_str(tcs[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    eprintln!("e2e: guided PASS");
    drop(srv);
}
