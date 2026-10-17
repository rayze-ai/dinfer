//! 服务压测（OpenAI `/v1/chat/completions`，流式）。对 dinfer 与 vLLM 用同一套口径，口径对齐 vLLM `benchmarks/serve.py`：
//! - TTFT  = 收到第一个**带内容**的 chunk 的时刻 − 发请求时刻；
//! - ITL   = 相邻两个带内容 chunk 的间隔（每个 chunk 一条）；
//! - TPOT  = (请求总时延 − TTFT) / (输出 token 数 − 1)（输出 ≤ 1 token 的请求不计）；
//! - 输出 token 数以服务端 `usage.completion_tokens`（`stream_options.include_usage`）为准，拿不到才退化为 chunk 数；
//! - 吞吐 = 完成请求数 / 总墙钟、输出 token 总数 / 总墙钟、(prompt+输出) / 总墙钟。
//!
//! 负载两种：
//! - 闭环并发（默认）：`--concurrency` 个 worker，各自串行发请求，直到总数 `--num-requests` 发完；
//! - 开环泊松到达：`--request-rate R`（req/s）> 0 时，按指数分布间隔发请求、不等上一个完成（并发不设上限，
//!   `--concurrency` 忽略）。这是线上流量的形态：prefill 与 decode 会在同一 step 里重叠，闭环固定并发做不到
//!   （见 docs/loops/q3-xstream.md 第 7–9 轮）。
//! prompt 长度：`--prompt-len` 固定，或 `--prompt-len-mix 128:0.5,1024:0.3,4096:0.2`（长度:权重，按 seed 抽样）。。
//!
//! prompt 由 `--seed` 决定性生成（同参数两次运行发同样的 prompt），长度以**服务端 `usage.prompt_tokens`** 回报为准。
//!
//! 公平性：两边同 `--max-tokens`、`ignore_eos`（vLLM 扩展字段，dinfer 也接）让每个请求都生成满 max_tokens；
//!
//! 思考模式统一关（`chat_template_kwargs.enable_thinking=false`）。
//!
//! 工具自检：`bench --self-check` 用进程内 mock 服务（固定每 token 10 ms、TTFT 50 ms）验证测得的 TTFT / TPOT / token 数
//! 落在已知值 ±2 ms 内；跑真服务前先跑它。

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::Serialize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Parser, Debug, Clone)]
#[command(about = "OpenAI-compatible serving benchmark (TTFT / TPOT / ITL / throughput)")]
struct Args {
    /// 服务地址，如 http://127.0.0.1:8012
    #[arg(long, default_value = "http://127.0.0.1:8012")]
    url: String,
    /// 不传则取 /v1/models 第一个
    #[arg(long)]
    model: Option<String>,
    #[arg(long, default_value_t = 1)]
    concurrency: usize,
    #[arg(long, default_value_t = 32)]
    num_requests: usize,
    /// 目标 prompt 长度（词数，≈ token 数；实际以服务端 usage.prompt_tokens 为准并回报）
    #[arg(long, default_value_t = 128)]
    prompt_len: usize,
    #[arg(long, default_value_t = 128)]
    max_tokens: u32,
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    /// 不传 ignore_eos（默认传 true：每个请求都生成满 max_tokens，两边输出量一致）
    #[arg(long, default_value_t = false)]
    allow_eos: bool,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// 开环泊松到达速率（req/s）；0 = 闭环并发模式
    #[arg(long, default_value_t = 0.0)]
    request_rate: f64,
    /// 长短混合 prompt："len:weight,len:weight,..."（覆盖 --prompt-len）
    #[arg(long, default_value = "")]
    prompt_len_mix: String,
    /// 预热请求数（不计入统计）
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    /// 结果 JSON 写到这里（不传只打印）
    #[arg(long)]
    out: Option<String>,
    /// 给结果打标签（如 dinfer / vllm）
    #[arg(long, default_value = "")]
    label: String,
    /// 进程内 mock 服务自检
    #[arg(long, default_value_t = false)]
    self_check: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ReqResult {
    ok: bool,
    error: Option<String>,
    prompt_tokens: u32,
    output_tokens: u32,
    ttft_ms: f64,
    latency_ms: f64,
    itl_ms: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct Summary {
    label: String,
    url: String,
    model: String,
    concurrency: usize,
    num_requests: usize,
    completed: usize,
    failed: usize,
    duration_s: f64,
    prompt_len_target: usize,
    prompt_tokens_mean: f64,
    output_tokens_mean: f64,
    request_throughput_rps: f64,
    output_token_throughput_tps: f64,
    total_token_throughput_tps: f64,
    ttft_ms: Pct,
    tpot_ms: Pct,
    itl_ms: Pct,
    latency_ms: Pct,
    args: serde_json::Value,
}

#[derive(Debug, Serialize, Default)]
struct Pct {
    mean: f64,
    p50: f64,
    p90: f64,
    p99: f64,
    max: f64,
}

fn pct(mut v: Vec<f64>) -> Pct {
    if v.is_empty() {
        return Pct::default();
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    Pct {
        mean: v.iter().sum::<f64>() / v.len() as f64,
        p50: q(0.5),
        p90: q(0.9),
        p99: q(0.99),
        max: *v.last().unwrap(),
    }
}

/// 决定性 prompt：seed + 下标 → 固定词序列（词表 64 个常见词，每个请求不同，避免 prefix cache 全命中）。
fn make_prompt(seed: u64, idx: usize, words: usize) -> String {
    const VOCAB: &[&str] = &[
        "river", "mountain", "quiet", "engine", "yellow", "window", "forest", "signal", "copper", "harbor",
        "candle", "meadow", "silver", "thunder", "garden", "velvet", "marble", "lantern", "orchard",
        "compass", "feather", "glacier", "harvest", "island", "journey", "kettle", "ladder", "mirror",
        "needle", "ocean", "pepper", "quartz", "ribbon", "saddle", "timber", "umbrella", "valley", "walnut",
        "yarn", "zephyr", "anchor", "basket", "canyon", "desert", "ember", "falcon", "granite", "hollow",
        "ivory", "jasmine", "kernel", "lattice", "mosaic", "nectar", "oyster", "pebble", "quiver", "rustic",
        "summit", "tunnel", "upland", "vessel", "willow", "zenith",
    ];
    let mut x = seed ^ (idx as u64).wrapping_mul(0x9E3779B97F4A7C15);
    let mut out = String::with_capacity(words * 8 + 64);
    out.push_str("Write a story using these words in order: ");
    for i in 0..words {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        if i > 0 {
            out.push(' ');
        }
        out.push_str(VOCAB[(x % VOCAB.len() as u64) as usize]);
    }
    out.push('.');
    out
}

/// 解析 "128:0.5,1024:0.3" → [(128, 0.5), (1024, 0.3)]
fn parse_len_mix(spec: &str) -> Result<Vec<(usize, f64)>> {
    let mut v = Vec::new();
    for part in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (l, w) = part.split_once(':').with_context(|| format!("bad --prompt-len-mix item {part:?}"))?;
        v.push((l.trim().parse()?, w.trim().parse()?));
    }
    anyhow::ensure!(!v.is_empty() && v.iter().all(|&(_, w)| w > 0.0), "--prompt-len-mix empty or non-positive weight");
    Ok(v)
}
/// 第 idx 个请求的 prompt 长度：固定值，或按 mix 权重用 (seed, idx) 决定性抽样。
fn prompt_len_for(a: &Args, mix: &[(usize, f64)], idx: usize) -> usize {
    if mix.is_empty() {
        return a.prompt_len;
    }
    let total: f64 = mix.iter().map(|m| m.1).sum();
    let mut x = (a.seed ^ 0xA5A5_0000).wrapping_add((idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    let u = (x >> 11) as f64 / (1u64 << 53) as f64 * total;
    let mut acc = 0.0;
    for &(l, w) in mix {
        acc += w;
        if u < acc {
            return l;
        }
    }
    mix.last().unwrap().0
}
/// 指数分布间隔（均值 1/rate 秒），决定性：由 (seed, idx) 生成。
fn exp_interval_s(seed: u64, idx: usize, rate: f64) -> f64 {
    let mut x = (seed ^ 0x5EED_1234).wrapping_add((idx as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    let u = ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64; // (0,1)
    -u.ln() / rate
}

async fn one_request(client: &reqwest::Client, url: &str, model: &str, prompt: &str, a: &Args) -> ReqResult {
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": a.max_tokens,
        "temperature": a.temperature,
        "ignore_eos": !a.allow_eos,
        "stream": true,
        "stream_options": {"include_usage": true},
        "chat_template_kwargs": {"enable_thinking": false},
    });
    let t0 = Instant::now();
    let mut r = ReqResult {
        ok: false,
        error: None,
        prompt_tokens: 0,
        output_tokens: 0,
        ttft_ms: 0.0,
        latency_ms: 0.0,
        itl_ms: vec![],
    };
    let resp = match client.post(format!("{url}/v1/chat/completions")).json(&body).send().await {
        Ok(x) => x,
        Err(e) => {
            r.error = Some(format!("send: {e}"));
            return r;
        }
    };
    if !resp.status().is_success() {
        r.error = Some(format!("http {}: {}", resp.status(), resp.text().await.unwrap_or_default()));
        return r;
    }
    let mut resp = resp;
    let mut buf = String::new();
    let mut last_content_at: Option<Instant> = None;
    let mut chunks_with_content = 0u32;
    let mut usage_out: Option<u32> = None;
    let mut done = false;
    loop {
        let chunk = match resp.chunk().await {
            Ok(Some(c)) => c,
            Ok(None) => break,
            Err(e) => {
                r.error = Some(format!("stream: {e}"));
                return r;
            }
        };
        let now = Instant::now();
        buf.push_str(&String::from_utf8_lossy(&chunk));
        // SSE：事件以空行分隔；每个事件一行 "data: ..."
        while let Some(pos) = buf.find("\n\n") {
            let event = buf[..pos].to_string();
            buf.drain(..pos + 2);
            for line in event.lines() {
                let Some(d) = line.strip_prefix("data:") else { continue };
                let d = d.trim();
                if d == "[DONE]" {
                    done = true;
                    continue;
                }
                let Ok(j) = serde_json::from_str::<serde_json::Value>(d) else { continue };
                if let Some(u) = j.get("usage").filter(|u| !u.is_null()) {
                    usage_out = u["completion_tokens"].as_u64().map(|x| x as u32);
                    r.prompt_tokens = u["prompt_tokens"].as_u64().unwrap_or(0) as u32;
                }
                let has_content = j["choices"].as_array().is_some_and(|cs| {
                    cs.iter().any(|c| {
                        let d = &c["delta"];
                        d["content"].as_str().is_some_and(|s| !s.is_empty())
                            || d["reasoning_content"].as_str().is_some_and(|s| !s.is_empty())
                    })
                });
                if has_content {
                    chunks_with_content += 1;
                    match last_content_at {
                        None => r.ttft_ms = (now - t0).as_secs_f64() * 1e3,
                        Some(prev) => r.itl_ms.push((now - prev).as_secs_f64() * 1e3),
                    }
                    last_content_at = Some(now);
                }
            }
        }
    }
    r.latency_ms = t0.elapsed().as_secs_f64() * 1e3;
    r.output_tokens = usage_out.unwrap_or(chunks_with_content);
    if !done {
        r.error = Some("stream ended without [DONE]".into());
        return r;
    }
    if chunks_with_content == 0 {
        r.error = Some("no content chunks".into());
        return r;
    }
    r.ok = true;
    r
}

async fn run(a: &Args) -> Result<Summary> {
    let client = reqwest::Client::builder().pool_max_idle_per_host(a.concurrency.max(1)).build()?;
    let model = match &a.model {
        Some(m) => m.clone(),
        None => {
            let v: serde_json::Value = client
                .get(format!("{}/v1/models", a.url))
                .send()
                .await
                .context("GET /v1/models")?
                .json()
                .await?;
            v["data"][0]["id"].as_str().context("no model id")?.to_string()
        }
    };
    // 预热
    for i in 0..a.warmup {
        let p = make_prompt(a.seed ^ 0xFFFF, i, a.prompt_len);
        let r = one_request(&client, &a.url, &model, &p, a).await;
        if !r.ok {
            bail!("warmup request failed: {:?}", r.error);
        }
    }
    let mix = if a.prompt_len_mix.is_empty() { Vec::new() } else { parse_len_mix(&a.prompt_len_mix)? };
    let t0 = Instant::now();
    let mut handles = Vec::with_capacity(a.concurrency.max(a.num_requests.min(4096)));
    if a.request_rate > 0.0 {
        // 开环：按泊松过程到达，每个请求独立任务，不等前一个完成
        let mut t_next = 0.0f64;
        for i in 0..a.num_requests {
            t_next += exp_interval_s(a.seed, i, a.request_rate);
            let wait = Duration::from_secs_f64(t_next).saturating_sub(t0.elapsed());
            if !wait.is_zero() {
                tokio::time::sleep(wait).await;
            }
            let p = make_prompt(a.seed, i, prompt_len_for(&a, &mix, i));
            let (client, a, model) = (client.clone(), a.clone(), model.clone());
            handles.push(tokio::spawn(async move { vec![one_request(&client, &a.url, &model, &p, &a).await] }));
        }
    } else {
        // 闭环固定并发
        let next = Arc::new(AtomicUsize::new(0));
        for _ in 0..a.concurrency {
            let (client, a, model, next, mix) = (client.clone(), a.clone(), model.clone(), next.clone(), mix.clone());
            handles.push(tokio::spawn(async move {
                let mut out = Vec::new();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= a.num_requests {
                        break;
                    }
                    let p = make_prompt(a.seed, i, prompt_len_for(&a, &mix, i));
                    out.push(one_request(&client, &a.url, &model, &p, &a).await);
                }
                out
            }));
        }
    }
    let mut results = Vec::with_capacity(a.num_requests);
    for h in handles {
        results.extend(h.await?);
    }
    let dur = t0.elapsed().as_secs_f64();
    let ok: Vec<&ReqResult> = results.iter().filter(|r| r.ok).collect();
    let failed = results.len() - ok.len();
    let out_tokens: u64 = ok.iter().map(|r| r.output_tokens as u64).sum();
    let in_tokens: u64 = ok.iter().map(|r| r.prompt_tokens as u64).sum();
    let tpots: Vec<f64> = ok
        .iter()
        .filter(|r| r.output_tokens > 1)
        .map(|r| (r.latency_ms - r.ttft_ms) / (r.output_tokens as f64 - 1.0))
        .collect();
    let itls: Vec<f64> = ok.iter().flat_map(|r| r.itl_ms.iter().copied()).collect();
    let n_ok = ok.len().max(1) as f64;
    Ok(Summary {
        label: a.label.clone(),
        url: a.url.clone(),
        model,
        concurrency: a.concurrency,
        num_requests: a.num_requests,
        completed: ok.len(),
        failed,
        duration_s: dur,
        prompt_len_target: a.prompt_len,
        prompt_tokens_mean: in_tokens as f64 / n_ok,
        output_tokens_mean: out_tokens as f64 / n_ok,
        request_throughput_rps: ok.len() as f64 / dur,
        output_token_throughput_tps: out_tokens as f64 / dur,
        total_token_throughput_tps: (in_tokens + out_tokens) as f64 / dur,
        ttft_ms: pct(ok.iter().map(|r| r.ttft_ms).collect()),
        tpot_ms: pct(tpots),
        itl_ms: pct(itls),
        latency_ms: pct(ok.iter().map(|r| r.latency_ms).collect()),
        args: serde_json::json!({"max_tokens": a.max_tokens, "temperature": a.temperature, "ignore_eos": !a.allow_eos, "seed": a.seed, "warmup": a.warmup, "request_rate": a.request_rate, "prompt_len_mix": a.prompt_len_mix}),
    })
}

fn print_summary(s: &Summary) {
    println!(
        "== {} {} model={} concurrency={} requests={} ({} ok, {} failed) in {:.2}s",
        s.label, s.url, s.model, s.concurrency, s.num_requests, s.completed, s.failed, s.duration_s
    );
    println!(
        "   prompt_tokens mean {:.1}  output_tokens mean {:.1}",
        s.prompt_tokens_mean, s.output_tokens_mean
    );
    println!(
        "   throughput: {:.2} req/s   {:.1} output tok/s   {:.1} total tok/s",
        s.request_throughput_rps, s.output_token_throughput_tps, s.total_token_throughput_tps
    );
    let row = |name: &str, p: &Pct| {
        println!(
            "   {name:<8} mean {:>8.2}  p50 {:>8.2}  p90 {:>8.2}  p99 {:>8.2}  max {:>8.2}  (ms)",
            p.mean, p.p50, p.p90, p.p99, p.max
        )
    };
    row("TTFT", &s.ttft_ms);
    row("TPOT", &s.tpot_ms);
    row("ITL", &s.itl_ms);
    row("latency", &s.latency_ms);
}

/// 一帧 SSE（HTTP chunked 编码）。
async fn send(sock: &mut tokio::net::TcpStream, data: String) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;
    let payload = format!("data: {data}\n\n");
    let frame = format!("{:x}\r\n{payload}\r\n", payload.len());
    sock.write_all(frame.as_bytes()).await
}

/// 自检用 mock：TTFT 50 ms，然后每 10 ms 一个 token（含内容的 chunk），max_tokens 个；最后 usage + [DONE]。
async fn mock_server() -> Result<String> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else { break };
            tokio::spawn(async move {
                let mut buf = vec![0u8; 65536];
                let mut req = Vec::new();
                // 读到 body 结束（Content-Length）
                loop {
                    let n = match sock.read(&mut buf).await {
                        Ok(0) | Err(_) => return,
                        Ok(n) => n,
                    };
                    req.extend_from_slice(&buf[..n]);
                    let s = String::from_utf8_lossy(&req);
                    if let Some(h) = s.find("\r\n\r\n") {
                        let cl = s
                            .lines()
                            .find_map(|l| {
                                l.strip_prefix("Content-Length: ").or(l.strip_prefix("content-length: "))
                            })
                            .and_then(|v| v.trim().parse::<usize>().ok())
                            .unwrap_or(0);
                        if req.len() >= h + 4 + cl {
                            break;
                        }
                    }
                }
                let s = String::from_utf8_lossy(&req).to_string();
                if s.starts_with("GET /v1/models") {
                    let body = r#"{"data":[{"id":"mock"}]}"#;
                    let _ = sock.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await;
                    return;
                }
                let body_start = s.find("\r\n\r\n").unwrap() + 4;
                let j: serde_json::Value = serde_json::from_str(&s[body_start..]).unwrap_or_default();
                let n = j["max_tokens"].as_u64().unwrap_or(8) as usize;
                let _ = sock.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n").await;
                let _ = send(
                    &mut sock,
                    r#"{"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}"#.into(),
                )
                .await
                .ok();
                // 绝对时间表：第 i 个 token 在 t0 + 50 ms + i × 10 ms 发出（相对 sleep 的 ~1 ms 粒度误差会逐 token 累积）
                let t0 = Instant::now();
                for i in 0..n {
                    tokio::time::sleep_until(tokio::time::Instant::from_std(
                        t0 + Duration::from_millis(50 + 10 * i as u64),
                    ))
                    .await;
                    if send(
                        &mut sock,
                        format!(r#"{{"choices":[{{"index":0,"delta":{{"content":"t{i} "}}}}]}}"#),
                    )
                    .await
                    .is_err()
                    {
                        return;
                    }
                }
                let _ = send(
                    &mut sock,
                    r#"{"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}"#.to_string(),
                )
                .await;
                let _ = send(&mut sock, format!(r#"{{"choices":[],"usage":{{"prompt_tokens":17,"completion_tokens":{n},"total_tokens":{}}}}}"#, 17 + n)).await;
                let _ = send(&mut sock, "[DONE]".into()).await;
                let _ = sock.write_all(b"0\r\n\r\n").await;
            });
        }
    });
    Ok(format!("http://{addr}"))
}

async fn self_check() -> Result<()> {
    let url = mock_server().await?;
    let a = Args {
        url,
        model: None,
        concurrency: 4,
        num_requests: 16,
        prompt_len: 8,
        max_tokens: 20,
        temperature: 0.0,
        allow_eos: false,
        seed: 1,
        warmup: 1,
        out: None,
        label: "self-check".into(),
        self_check: true,
    };
    let s = run(&a).await?;
    print_summary(&s);
    let close = |got: f64, want: f64, tol: f64, what: &str| -> Result<()> {
        if (got - want).abs() > tol {
            bail!("self-check FAILED: {what} = {got:.2}, want {want:.2} ± {tol}");
        }
        Ok(())
    };
    if s.completed != 16 || s.failed != 0 {
        bail!("self-check FAILED: completed {} failed {}", s.completed, s.failed);
    }
    close(s.output_tokens_mean, 20.0, 0.0, "output_tokens (from usage)")?;
    close(s.prompt_tokens_mean, 17.0, 0.0, "prompt_tokens (from usage)")?;
    close(s.ttft_ms.p50, 50.0, 3.0, "TTFT p50")?;
    close(s.tpot_ms.p50, 10.0, 1.0, "TPOT p50")?;
    close(s.itl_ms.p50, 10.0, 1.5, "ITL p50")?;
    close(s.latency_ms.p50, 50.0 + 19.0 * 10.0, 6.0, "latency p50")?;
    // 并发 4、每请求 ≈240 ms、16 个请求 → ≈ 4 轮 ≈ 1 s
    if !(0.8..1.6).contains(&s.duration_s) {
        bail!("self-check FAILED: duration {:.2}s (expected ≈1 s for 4 rounds)", s.duration_s);
    }
    println!("self-check PASS (TTFT/TPOT/ITL/latency/usage/concurrency all within tolerance)");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let a = Args::parse();
    if a.self_check {
        return self_check().await;
    }
    let s = run(&a).await?;
    print_summary(&s);
    if let Some(p) = &a.out {
        std::fs::write(p, serde_json::to_string_pretty(&s)?)?;
        println!("   wrote {p}");
    }
    Ok(())
}
