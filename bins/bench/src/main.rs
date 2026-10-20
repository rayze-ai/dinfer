//! 服务压测（OpenAI `/v1/chat/completions`，流式）。对 dinfer 与 vLLM 用同一套口径，口径对齐 vLLM `benchmarks/serve.py`：
//! - TTFT  = 收到第一个**带内容**的 chunk 的时刻 − 发请求时刻；另记 `ttft_first_chunk_ms` = 任意首个 SSE 事件（含空 role delta）
//!   的时刻，这是 vLLM `bench serve` 的口径（它把空 role delta 当首 token → dinfer 侧会出 ~5 ms 伪像，design §15.4），只用于和第三方工具对账；
//! - ITL   = 相邻两个带内容 chunk 的间隔（每个 chunk 一条）；
//! - TPOT  = (请求总时延 − TTFT) / (输出 token 数 − 1)（输出 ≤ 1 token 的请求不计；goodput 判定时按 0 计，同 vLLM）；
//! - 输出 token 数以服务端 `usage.completion_tokens`（`stream_options.include_usage`）为准，拿不到才退化为 chunk 数；
//! - 吞吐 = 完成请求数 / 总墙钟、输出 token 总数 / 总墙钟、(prompt+输出) / 总墙钟；
//! - goodput（vLLM `--goodput ttft:500 tpot:20` 同义，DistServe 定义）= TTFT ≤ `--slo-ttft-ms` **且** TPOT ≤ `--slo-tpot-ms`
//!   的完成请求数 / 总墙钟；只给其中一个就只判一个；都不给不算。
//! - prefix cache 命中：每请求读 `usage.prompt_tokens_details.cached_tokens`（dinfer / vLLM 同字段），
//!   `prefix_cache_hit_ratio` = Σcached / Σprompt_tokens（只算回报了该字段的请求；一个都没报 → null）。
//!
//! 负载两种：
//! - 闭环并发（默认）：`--concurrency` 个 worker，各自串行发请求，直到总数 `--num-requests` 发完；
//! - 开环泊松到达：`--request-rate R`（req/s）> 0 时，按指数分布间隔发请求、不等上一个完成（并发不设上限，
//!   `--concurrency` 忽略）。这是线上流量的形态：prefill 与 decode 会在同一 step 里重叠，闭环固定并发做不到
//!   （见 docs/loops/q3-xstream.md 第 7–9 轮）。
//! 长度分布（引擎层面口径，docs/design.md §15）：
//! - prompt：`--prompt-len` 固定，或 `--prompt-len-mix 128:0.5,1024:0.3,4096:0.2`（长度:权重，按 seed 抽样）；
//! - 输出：`--max-tokens` 固定，或 `--output-len-mix 128:0.5,512:0.3,1024:0.2` 按 seed 抽 max_tokens；配合 `--allow-eos`
//!   时实际输出 = min(eos 位置, 抽样值)，即"不 ignore_eos"模式。
//! 多轮：`--multi-turn K`：到达单位是"会话"，每会话顺序发 K 轮，第 k 轮 messages = 前 k−1 轮的 user/assistant 全文 + 新问题
//!   （真实多轮前缀复用：同一前缀按时间错开到达，而不是同时到）。泊松速率仍按请求计（会话到达率 = R / K）。
//!   `--prefix-repeat` 决定会话首轮是否走共享长前缀。
//!
//! prompt 由 `--seed` 决定性生成（同参数两次运行发同样的 prompt），长度以**服务端 `usage.prompt_tokens`** 回报为准。
//!
//! 公平性：两边同 `--max-tokens`、`ignore_eos`（vLLM 扩展字段，dinfer 也接）让每个请求都生成满 max_tokens；
//! 思考模式统一关（`chat_template_kwargs.enable_thinking=false`）。
//!
//! 测试程序自身的准确性与开销（用户 2026-10-20）：
//! - 时间戳在**收到字节时**打（`resp.chunk()` 返回即取 `Instant::now()`，单调时钟），SSE 切分 / JSON 解析在其后；解析耗时另计入
//!   `client.parse_ms_total`，不进 TTFT / ITL；
//! - 所有 prompt / 到达间隔 / max_tokens 在发压前**一次预生成到内存**（`Plan`），发送期只做 HTTP + 解析；
//! - JSON `client` 段：进程 CPU 时间（getrusage user+sys）、峰值 RSS、线程数、CPU 亲和集合、解析总耗时、运行墙钟 → 压测端占用 =
//!   cpu_s / wall_s 个核；`bench_engine.sh` 用 taskset 把它钉到与服务进程不重叠的核。
//!
//! 工具自检：`bench --self-check` 用进程内 mock 服务（固定每 token 10 ms、TTFT 50 ms、cached_tokens 5/17）验证测得的
//! TTFT / TPOT / token 数 / cache 命中率落在已知值内；跑真服务前先跑它。

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::Serialize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Parser, Debug, Clone)]
#[command(about = "OpenAI-compatible serving benchmark (TTFT / TPOT / ITL / throughput / goodput)")]
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
    /// 输出长度分布："len:weight,..."，按 seed 抽每请求的 max_tokens（覆盖 --max-tokens）
    #[arg(long, default_value = "")]
    output_len_mix: String,
    /// 前缀复用比例 [0,1]：这部分请求共享同一个 `--prompt-len` 词的长前缀，各自只追加 16 个随机词（测 prefix cache / host tier）
    #[arg(long, default_value_t = 0.0)]
    prefix_repeat: f64,
    /// 多轮：每个会话顺序发 K 轮（K=1 为单轮）；--num-requests 仍是请求总数（会话数 = ceil(N/K)）
    #[arg(long, default_value_t = 1)]
    multi_turn: usize,
    /// SLO：TTFT 上限（ms），0 = 不判；与 --slo-tpot-ms 都满足才算 good → goodput
    #[arg(long, default_value_t = 0.0)]
    slo_ttft_ms: f64,
    /// SLO：TPOT 上限（ms），0 = 不判
    #[arg(long, default_value_t = 0.0)]
    slo_tpot_ms: f64,
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

impl Args {
    fn slo(&self) -> Slo {
        Slo { ttft_ms: self.slo_ttft_ms, tpot_ms: self.slo_tpot_ms }
    }
}

#[derive(Debug, Clone, Serialize)]
struct ReqResult {
    ok: bool,
    error: Option<String>,
    prompt_tokens: u32,
    output_tokens: u32,
    /// 本请求发的 max_tokens（固定值或 --output-len-mix 抽样）
    max_tokens: u32,
    /// 服务端 usage.prompt_tokens_details.cached_tokens；没报 → None
    cached_tokens: Option<u32>,
    /// 服务端 finish_reason（"stop" / "length" / ...），流里没带 → None
    finish_reason: Option<String>,
    /// 多轮：第几轮（从 1 起）
    turn: u32,
    ttft_ms: f64,
    /// 任意首个 SSE 数据事件（含空 role delta）的时刻 − 发请求时刻（vLLM bench serve 口径，对账用）
    ttft_first_chunk_ms: f64,
    latency_ms: f64,
    itl_ms: Vec<f64>,
    /// 本请求客户端 SSE 切分 + JSON 解析总耗时（不进 TTFT / ITL；汇总进 client.parse_ms_total）
    #[serde(skip)]
    parse_ms: f64,
    /// 流式拼起来的回答全文（多轮下一轮要用；不进 JSON）
    #[serde(skip)]
    text: String,
}

impl ReqResult {
    fn blank(max_tokens: u32, turn: u32) -> Self {
        Self {
            ok: false,
            error: None,
            prompt_tokens: 0,
            output_tokens: 0,
            max_tokens,
            cached_tokens: None,
            finish_reason: None,
            turn,
            ttft_ms: 0.0,
            ttft_first_chunk_ms: 0.0,
            latency_ms: 0.0,
            itl_ms: vec![],
            parse_ms: 0.0,
            text: String::new(),
        }
    }
    /// vLLM 口径：输出 ≤ 1 token 的 TPOT 记 0（goodput 判定用）；统计分布时跳过。
    fn tpot_ms(&self) -> Option<f64> {
        (self.output_tokens > 1).then(|| (self.latency_ms - self.ttft_ms) / (self.output_tokens as f64 - 1.0))
    }
}

/// 每请求简表（进 JSON，供交叉核对 / 画散点；不含逐 token ITL）。
#[derive(Debug, Serialize)]
struct ReqBrief {
    turn: u32,
    prompt_tokens: u32,
    output_tokens: u32,
    max_tokens: u32,
    cached_tokens: Option<u32>,
    finish_reason: Option<String>,
    ttft_ms: f64,
    ttft_first_chunk_ms: f64,
    tpot_ms: Option<f64>,
    latency_ms: f64,
    good: Option<bool>,
}

#[derive(Debug, Clone, Copy, Serialize, Default)]
struct Slo {
    ttft_ms: f64,
    tpot_ms: f64,
}

impl Slo {
    fn enabled(&self) -> bool {
        self.ttft_ms > 0.0 || self.tpot_ms > 0.0
    }
    /// (ttft 违约, tpot 违约)；未启用的维度不违约。TPOT 按 vLLM：输出 ≤ 1 token 记 0。
    fn violations(&self, r: &ReqResult) -> (bool, bool) {
        let tpot = r.tpot_ms().unwrap_or(0.0);
        (self.ttft_ms > 0.0 && r.ttft_ms > self.ttft_ms, self.tpot_ms > 0.0 && tpot > self.tpot_ms)
    }
    fn is_good(&self, r: &ReqResult) -> bool {
        let (a, b) = self.violations(r);
        !a && !b
    }
}

#[derive(Debug, Serialize, Default)]
struct SloViolations {
    ttft: usize,
    tpot: usize,
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
    /// 发出去的 max_tokens 均值（--output-len-mix 抽样后）
    max_tokens_mean: f64,
    /// finish_reason == "length" 的完成请求数（allow_eos 下看多少条跑满）
    finished_by_length: usize,
    request_throughput_rps: f64,
    output_token_throughput_tps: f64,
    total_token_throughput_tps: f64,
    /// SLO 下的 good 请求数 / 总墙钟（未设 SLO → null）
    goodput_rps: Option<f64>,
    good_requests: Option<usize>,
    slo_violations: Option<SloViolations>,
    /// Σcached / Σprompt（只算回报了 cached_tokens 的请求；一个没报 → null）
    prefix_cache_hit_ratio: Option<f64>,
    cached_tokens_mean: Option<f64>,
    /// 回报了 cached_tokens 的完成请求数
    cached_reported: usize,
    ttft_ms: Pct,
    /// 任意首 chunk 口径的 TTFT（vLLM bench serve 同义；对账用，不进 SLO）
    ttft_first_chunk_ms: Pct,
    tpot_ms: Pct,
    itl_ms: Pct,
    latency_ms: Pct,
    args: serde_json::Value,
    /// 压测端自身开销（测试程序占用资源的影响）
    client: ClientCost,
    requests: Vec<ReqBrief>,
}

/// 压测进程自身的开销快照：bench 结束时 getrusage(RUSAGE_SELF) − 开始时。
#[derive(Debug, Serialize, Default, Clone)]
struct ClientCost {
    /// 发压期间进程 CPU 时间（user + sys，秒）；cpu_s / wall_s = 平均占用核数
    cpu_user_s: f64,
    cpu_sys_s: f64,
    cpu_cores_avg: f64,
    wall_s: f64,
    /// 峰值 RSS（MiB，ru_maxrss）
    max_rss_mib: f64,
    threads: usize,
    /// sched_getaffinity 的 CPU 集合（如 "0-7,16-23"）
    cpus: String,
    ncpus_allowed: usize,
    /// SSE 切分 + JSON 解析的总耗时（ms，所有请求求和；不进 TTFT / ITL）
    parse_ms_total: f64,
    parse_ms_per_request: f64,
}

fn rusage() -> (f64, f64, f64) {
    // (user s, sys s, maxrss MiB)
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    let tv = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
    (tv(ru.ru_utime), tv(ru.ru_stime), ru.ru_maxrss as f64 / 1024.0)
}

/// sched_getaffinity → "0-7,16" 形式 + 个数。
fn cpu_affinity() -> (String, usize) {
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    if unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) } != 0 {
        return ("?".into(), 0);
    }
    let cpus: Vec<usize> = (0..libc::CPU_SETSIZE as usize).filter(|&i| unsafe { libc::CPU_ISSET(i, &set) }).collect();
    (range_str(&cpus), cpus.len())
}

/// [0,1,2,5,6] → "0-2,5-6"
fn range_str(v: &[usize]) -> String {
    let mut out = Vec::new();
    let mut i = 0;
    while i < v.len() {
        let mut j = i;
        while j + 1 < v.len() && v[j + 1] == v[j] + 1 {
            j += 1;
        }
        out.push(if j > i { format!("{}-{}", v[i], v[j]) } else { v[i].to_string() });
        i = j + 1;
    }
    out.join(",")
}

fn thread_count() -> usize {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find_map(|l| l.strip_prefix("Threads:").and_then(|v| v.trim().parse().ok())))
        .unwrap_or(0)
}

struct CostStart {
    user: f64,
    sys: f64,
    t0: Instant,
}
fn cost_start() -> CostStart {
    let (user, sys, _) = rusage();
    CostStart { user, sys, t0: Instant::now() }
}
fn cost_end(c: &CostStart, parse_ms_total: f64, n_requests: usize) -> ClientCost {
    let (user, sys, rss) = rusage();
    let wall = c.t0.elapsed().as_secs_f64();
    let (cpus, n) = cpu_affinity();
    ClientCost {
        cpu_user_s: user - c.user,
        cpu_sys_s: sys - c.sys,
        cpu_cores_avg: if wall > 0.0 { (user - c.user + sys - c.sys) / wall } else { 0.0 },
        wall_s: wall,
        max_rss_mib: rss,
        threads: thread_count(),
        cpus,
        ncpus_allowed: n,
        parse_ms_total,
        parse_ms_per_request: parse_ms_total / n_requests.max(1) as f64,
    }
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

const PROMPT_HEAD: &str = "Write a story using these words in order: ";

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
    out.push_str(PROMPT_HEAD);
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
        let (l, w) = part.split_once(':').with_context(|| format!("bad len-mix item {part:?} (want len:weight)"))?;
        v.push((l.trim().parse()?, w.trim().parse()?));
    }
    anyhow::ensure!(!v.is_empty() && v.iter().all(|&(_, w)| w > 0.0), "len-mix empty or non-positive weight");
    Ok(v)
}

/// (seed, salt, idx) → [0,1) 的决定性均匀数（splitmix 终混）。
fn uniform01(seed: u64, salt: u64, idx: usize) -> f64 {
    let mut x = (seed ^ salt).wrapping_add((idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// 按权重从 mix 里抽一个长度（u ∈ [0,1)）。
fn sample_mix(mix: &[(usize, f64)], u: f64) -> usize {
    let total: f64 = mix.iter().map(|m| m.1).sum();
    let u = u * total;
    let mut acc = 0.0;
    for &(l, w) in mix {
        acc += w;
        if u < acc {
            return l;
        }
    }
    mix.last().unwrap().0
}

/// 第 idx 个请求是否走 "共享长前缀" 分支（按 seed 决定性抽样，比例 = --prefix-repeat）。
fn is_prefix_repeat(a: &Args, idx: usize) -> bool {
    if a.prefix_repeat <= 0.0 {
        return false;
    }
    let mut x = (a.seed ^ 0x7E7E_0000).wrapping_add((idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    x ^= x >> 31;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 29;
    ((x >> 11) as f64 / (1u64 << 53) as f64) < a.prefix_repeat
}
/// 共享前缀 + 唯一尾部：前缀由 seed 固定（所有请求同一个），尾部由 idx 变化。
fn make_prompt_shared_prefix(seed: u64, idx: usize, prefix_words: usize) -> String {
    let mut p = make_prompt(seed ^ 0x5A5A_5A5A, 0, prefix_words);
    p.truncate(p.len() - 1); // 去掉句号
    let tail = make_prompt(seed, idx, 16);
    let tail = tail.trim_start_matches(PROMPT_HEAD);
    format!("{p} {tail}")
}
/// 第 idx 个请求的 prompt 长度：固定值，或按 mix 权重用 (seed, idx) 决定性抽样。
fn prompt_len_for(a: &Args, mix: &[(usize, f64)], idx: usize) -> usize {
    if mix.is_empty() {
        a.prompt_len
    } else {
        sample_mix(mix, uniform01(a.seed, 0xA5A5_0000, idx))
    }
}
/// 第 idx 个请求的 max_tokens：固定值，或按 --output-len-mix 抽样（与 prompt 抽样独立：不同 salt）。
fn max_tokens_for(a: &Args, omix: &[(usize, f64)], idx: usize) -> u32 {
    if omix.is_empty() {
        a.max_tokens
    } else {
        sample_mix(omix, uniform01(a.seed, 0x0C0C_7777, idx)) as u32
    }
}
/// 多轮第 k（≥2）轮的新问题：16 个随机词。
fn follow_up(seed: u64, idx: usize) -> String {
    let t = make_prompt(seed ^ 0x3333_0000, idx, 16);
    format!("Continue the story using these words in order: {}", t.trim_start_matches(PROMPT_HEAD))
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

async fn one_request(
    client: &reqwest::Client,
    url: &str,
    model: &str,
    messages: &serde_json::Value,
    max_tokens: u32,
    turn: u32,
    a: &Args,
) -> ReqResult {
    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": a.temperature,
        "ignore_eos": !a.allow_eos,
        "stream": true,
        "stream_options": {"include_usage": true},
        "chat_template_kwargs": {"enable_thinking": false},
    });
    let t0 = Instant::now();
    let mut r = ReqResult::blank(max_tokens, turn);
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
        // 时间戳在收到字节的瞬间打；下面的切分 / 解析耗时计入 parse_ms，不进 TTFT / ITL
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
                if r.ttft_first_chunk_ms == 0.0 {
                    r.ttft_first_chunk_ms = (now - t0).as_secs_f64() * 1e3;
                }
                if let Some(u) = j.get("usage").filter(|u| !u.is_null()) {
                    usage_out = u["completion_tokens"].as_u64().map(|x| x as u32);
                    r.prompt_tokens = u["prompt_tokens"].as_u64().unwrap_or(0) as u32;
                    r.cached_tokens = u["prompt_tokens_details"]["cached_tokens"].as_u64().map(|x| x as u32);
                }
                let mut has_content = false;
                if let Some(cs) = j["choices"].as_array() {
                    for c in cs {
                        let d = &c["delta"];
                        for k in ["content", "reasoning_content"] {
                            if let Some(s) = d[k].as_str().filter(|s| !s.is_empty()) {
                                has_content = true;
                                if k == "content" {
                                    r.text.push_str(s);
                                }
                            }
                        }
                        if let Some(f) = c["finish_reason"].as_str() {
                            r.finish_reason = Some(f.to_string());
                        }
                    }
                }
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
        r.parse_ms += now.elapsed().as_secs_f64() * 1e3;
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

/// 请求生成参数（解析好的两个 mix）。
#[derive(Clone, Default)]
struct Gen {
    mix: Vec<(usize, f64)>,
    omix: Vec<(usize, f64)>,
}

/// 一个会话的预生成计划：K 轮的 user 文本与 max_tokens（发压期不再生成 prompt / 抽样）。
#[derive(Clone, Debug)]
struct SessionPlan {
    users: Vec<String>,
    max_tokens: Vec<u32>,
}

/// 预生成全部会话 + 泊松到达时刻（秒，相对 t0）。idx = s*K + (k−1)，与单轮模式 K=1 时 idx = s 一致。
fn make_plan(a: &Args, g: &Gen, n_sessions: usize) -> (Vec<SessionPlan>, Vec<f64>) {
    let k_turns = a.multi_turn.max(1);
    let mut sessions = Vec::with_capacity(n_sessions);
    for s in 0..n_sessions {
        let mut users = Vec::with_capacity(k_turns);
        let mut mt = Vec::with_capacity(k_turns);
        for k in 0..k_turns {
            let idx = s * k_turns + k;
            users.push(if k == 0 {
                if is_prefix_repeat(a, idx) {
                    make_prompt_shared_prefix(a.seed, idx, a.prompt_len)
                } else {
                    make_prompt(a.seed, idx, prompt_len_for(a, &g.mix, idx))
                }
            } else {
                follow_up(a.seed, idx)
            });
            mt.push(max_tokens_for(a, &g.omix, idx));
        }
        sessions.push(SessionPlan { users, max_tokens: mt });
    }
    let mut arrivals = Vec::with_capacity(n_sessions);
    if a.request_rate > 0.0 {
        let session_rate = a.request_rate / k_turns as f64;
        let mut t = 0.0;
        for s in 0..n_sessions {
            t += exp_interval_s(a.seed, s, session_rate);
            arrivals.push(t);
        }
    }
    (sessions, arrivals)
}

/// 第 s 个会话：K 轮顺序请求，第 k 轮 messages = 全部历史 + 预生成的新问题。
async fn run_session(client: &reqwest::Client, url: &str, model: &str, a: &Args, plan: &SessionPlan) -> Vec<ReqResult> {
    let k_turns = plan.users.len();
    let mut out = Vec::with_capacity(k_turns);
    let mut messages: Vec<serde_json::Value> = Vec::with_capacity(2 * k_turns);
    for k in 0..k_turns {
        messages.push(serde_json::json!({"role": "user", "content": plan.users[k]}));
        let r = one_request(client, url, model, &serde_json::Value::Array(messages.clone()), plan.max_tokens[k], k as u32 + 1, a).await;
        let ok = r.ok;
        let answer = r.text.clone();
        out.push(r);
        if !ok {
            break; // 会话里一轮失败，后续轮没有上下文，不再发
        }
        messages.push(serde_json::json!({"role": "assistant", "content": answer}));
    }
    out
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
    anyhow::ensure!(a.multi_turn >= 1, "--multi-turn must be >= 1");
    // 预热
    for i in 0..a.warmup {
        let p = make_prompt(a.seed ^ 0xFFFF, i, a.prompt_len);
        let msgs = serde_json::json!([{"role": "user", "content": p}]);
        let r = one_request(&client, &a.url, &model, &msgs, a.max_tokens, 1, a).await;
        if !r.ok {
            bail!("warmup request failed: {:?}", r.error);
        }
    }
    let g = Gen {
        mix: if a.prompt_len_mix.is_empty() { Vec::new() } else { parse_len_mix(&a.prompt_len_mix)? },
        omix: if a.output_len_mix.is_empty() { Vec::new() } else { parse_len_mix(&a.output_len_mix)? },
    };
    let k_turns = a.multi_turn;
    let n_sessions = a.num_requests.div_ceil(k_turns);
    // 预生成：prompt / max_tokens / 到达时刻全部在发压前算好并驻留内存
    let (plan, arrivals) = make_plan(a, &g, n_sessions);
    let plan = Arc::new(plan);
    let cost = cost_start();
    let t0 = Instant::now();
    let mut handles = Vec::with_capacity(a.concurrency.max(n_sessions.min(4096)));
    if a.request_rate > 0.0 {
        // 开环：按泊松过程到达，每个会话独立任务，不等前一个完成。到达率按请求计 → 会话到达率 = R / K。
        for s in 0..n_sessions {
            let wait = Duration::from_secs_f64(arrivals[s]).saturating_sub(t0.elapsed());
            if !wait.is_zero() {
                tokio::time::sleep(wait).await;
            }
            let (client, a, model, plan) = (client.clone(), a.clone(), model.clone(), plan.clone());
            handles.push(tokio::spawn(async move { run_session(&client, &a.url, &model, &a, &plan[s]).await }));
        }
    } else {
        // 闭环固定并发：worker 领会话
        let next = Arc::new(AtomicUsize::new(0));
        for _ in 0..a.concurrency {
            let (client, a, model, next, plan) = (client.clone(), a.clone(), model.clone(), next.clone(), plan.clone());
            handles.push(tokio::spawn(async move {
                let mut out = Vec::new();
                loop {
                    let s = next.fetch_add(1, Ordering::Relaxed);
                    if s >= n_sessions {
                        break;
                    }
                    out.extend(run_session(&client, &a.url, &model, &a, &plan[s]).await);
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
    let parse_total: f64 = results.iter().map(|r| r.parse_ms).sum();
    let client_cost = cost_end(&cost, parse_total, results.len());
    let mut s = summarize(a, model, &results, dur);
    s.client = client_cost;
    Ok(s)
}

/// 从每请求结果汇总（纯函数，单测用）。`num_requests` 以实际发出的条数为准（多轮会话中途失败会少发）。
fn summarize(a: &Args, model: String, results: &[ReqResult], dur: f64) -> Summary {
    let ok: Vec<&ReqResult> = results.iter().filter(|r| r.ok).collect();
    let failed = results.len() - ok.len();
    let out_tokens: u64 = ok.iter().map(|r| r.output_tokens as u64).sum();
    let in_tokens: u64 = ok.iter().map(|r| r.prompt_tokens as u64).sum();
    let tpots: Vec<f64> = ok.iter().filter_map(|r| r.tpot_ms()).collect();
    let itls: Vec<f64> = ok.iter().flat_map(|r| r.itl_ms.iter().copied()).collect();
    let n_ok = ok.len().max(1) as f64;
    let slo = a.slo();
    let (good, viol) = if slo.enabled() {
        let mut v = SloViolations::default();
        let mut good = 0usize;
        for r in &ok {
            let (vt, vp) = slo.violations(r);
            v.ttft += vt as usize;
            v.tpot += vp as usize;
            good += (!vt && !vp) as usize;
        }
        (Some(good), Some(v))
    } else {
        (None, None)
    };
    let cached: Vec<(u32, u32)> = ok.iter().filter_map(|r| r.cached_tokens.map(|c| (c, r.prompt_tokens))).collect();
    let cached_sum: u64 = cached.iter().map(|&(c, _)| c as u64).sum();
    let cached_prompt: u64 = cached.iter().map(|&(_, p)| p as u64).sum();
    let requests = results
        .iter()
        .map(|r| ReqBrief {
            turn: r.turn,
            prompt_tokens: r.prompt_tokens,
            output_tokens: r.output_tokens,
            max_tokens: r.max_tokens,
            cached_tokens: r.cached_tokens,
            finish_reason: r.finish_reason.clone(),
            ttft_ms: r.ttft_ms,
            ttft_first_chunk_ms: r.ttft_first_chunk_ms,
            tpot_ms: r.tpot_ms(),
            latency_ms: r.latency_ms,
            good: (r.ok && slo.enabled()).then(|| slo.is_good(r)),
        })
        .collect();
    Summary {
        label: a.label.clone(),
        url: a.url.clone(),
        model,
        concurrency: a.concurrency,
        num_requests: results.len(),
        completed: ok.len(),
        failed,
        duration_s: dur,
        prompt_len_target: a.prompt_len,
        prompt_tokens_mean: in_tokens as f64 / n_ok,
        output_tokens_mean: out_tokens as f64 / n_ok,
        max_tokens_mean: ok.iter().map(|r| r.max_tokens as f64).sum::<f64>() / n_ok,
        finished_by_length: ok.iter().filter(|r| r.finish_reason.as_deref() == Some("length")).count(),
        request_throughput_rps: ok.len() as f64 / dur,
        output_token_throughput_tps: out_tokens as f64 / dur,
        total_token_throughput_tps: (in_tokens + out_tokens) as f64 / dur,
        goodput_rps: good.map(|g| g as f64 / dur),
        good_requests: good,
        slo_violations: viol,
        prefix_cache_hit_ratio: (cached_prompt > 0).then(|| cached_sum as f64 / cached_prompt as f64),
        cached_tokens_mean: (!cached.is_empty()).then(|| cached_sum as f64 / cached.len() as f64),
        cached_reported: cached.len(),
        ttft_ms: pct(ok.iter().map(|r| r.ttft_ms).collect()),
        ttft_first_chunk_ms: pct(ok.iter().map(|r| r.ttft_first_chunk_ms).collect()),
        tpot_ms: pct(tpots),
        itl_ms: pct(itls),
        latency_ms: pct(ok.iter().map(|r| r.latency_ms).collect()),
        args: serde_json::json!({
            "max_tokens": a.max_tokens, "temperature": a.temperature, "ignore_eos": !a.allow_eos, "seed": a.seed,
            "warmup": a.warmup, "request_rate": a.request_rate, "prompt_len_mix": a.prompt_len_mix,
            "output_len_mix": a.output_len_mix, "prefix_repeat": a.prefix_repeat, "multi_turn": a.multi_turn,
            "slo": slo, "num_requests_requested": a.num_requests,
        }),
        client: ClientCost::default(),
        requests,
    }
}

fn print_summary(s: &Summary) {
    println!(
        "== {} {} model={} concurrency={} requests={} ({} ok, {} failed) in {:.2}s",
        s.label, s.url, s.model, s.concurrency, s.num_requests, s.completed, s.failed, s.duration_s
    );
    println!(
        "   prompt_tokens mean {:.1}  output_tokens mean {:.1}  max_tokens mean {:.1}  finished_by_length {}",
        s.prompt_tokens_mean, s.output_tokens_mean, s.max_tokens_mean, s.finished_by_length
    );
    println!(
        "   throughput: {:.2} req/s   {:.1} output tok/s   {:.1} total tok/s",
        s.request_throughput_rps, s.output_token_throughput_tps, s.total_token_throughput_tps
    );
    if let (Some(g), Some(n), Some(v)) = (s.goodput_rps, s.good_requests, &s.slo_violations) {
        println!("   goodput: {g:.2} req/s   good {n}/{}   violations ttft {} tpot {}", s.completed, v.ttft, v.tpot);
    }
    match s.prefix_cache_hit_ratio {
        Some(h) => println!(
            "   prefix cache: hit ratio {:.3}  cached_tokens mean {:.1}  (reported by {}/{} requests)",
            h,
            s.cached_tokens_mean.unwrap_or(0.0),
            s.cached_reported,
            s.completed
        ),
        None => println!("   prefix cache: server did not report usage.prompt_tokens_details.cached_tokens"),
    }
    let row = |name: &str, p: &Pct| {
        println!(
            "   {name:<8} mean {:>8.2}  p50 {:>8.2}  p90 {:>8.2}  p99 {:>8.2}  max {:>8.2}  (ms)",
            p.mean, p.p50, p.p90, p.p99, p.max
        )
    };
    let c = &s.client;
    println!(
        "   client: cpu {:.2}s (user {:.2} sys {:.2}) = {:.2} cores avg over {:.1}s wall; rss {:.0} MiB; {} threads; cpus {} ({}); parse {:.1} ms total / {:.3} ms per req",
        c.cpu_user_s + c.cpu_sys_s, c.cpu_user_s, c.cpu_sys_s, c.cpu_cores_avg, c.wall_s, c.max_rss_mib, c.threads, c.cpus, c.ncpus_allowed, c.parse_ms_total, c.parse_ms_per_request
    );
    row("TTFT", &s.ttft_ms);
    row("TTFT1st", &s.ttft_first_chunk_ms);
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

/// 自检用 mock：TTFT 50 ms，然后每 10 ms 一个 token（含内容的 chunk），max_tokens 个；最后 usage（prompt 17，cached 5）+ [DONE]。
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
                let _ = send(&mut sock, format!(r#"{{"choices":[],"usage":{{"prompt_tokens":17,"completion_tokens":{n},"total_tokens":{},"prompt_tokens_details":{{"cached_tokens":5}}}}}}"#, 17 + n)).await;
                let _ = send(&mut sock, "[DONE]".into()).await;
                let _ = sock.write_all(b"0\r\n\r\n").await;
            });
        }
    });
    Ok(format!("http://{addr}"))
}

fn test_args(url: String) -> Args {
    Args {
        url,
        model: None,
        concurrency: 4,
        num_requests: 16,
        prompt_len: 8,
        max_tokens: 20,
        temperature: 0.0,
        allow_eos: false,
        seed: 1,
        request_rate: 0.0,
        prompt_len_mix: String::new(),
        output_len_mix: String::new(),
        prefix_repeat: 0.0,
        multi_turn: 1,
        slo_ttft_ms: 0.0,
        slo_tpot_ms: 0.0,
        warmup: 1,
        out: None,
        label: "self-check".into(),
        self_check: true,
    }
}

async fn self_check() -> Result<()> {
    let url = mock_server().await?;
    // SLO 取 TTFT 60 / TPOT 10.5：mock 的 50 / 10 都应满足 → goodput == 吞吐
    let a = Args { slo_ttft_ms: 60.0, slo_tpot_ms: 10.5, ..test_args(url) };
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
    // 口径自校验：mock 的 TTFT 50 ms、ITL 10 ms 为已知真值 → 报测量误差（µs）。mock 用绝对时间表发送，
    // 误差来源 = tokio 定时器粒度 + 本机回环 + 客户端收包；真服务上客户端口径的附加误差不会小于这个数。
    let err_us = |got: f64, want: f64| (got - want) * 1e3;
    println!(
        "   measurement error vs mock truth: TTFT p50 {:+.0} µs (max {:+.0}), TPOT p50 {:+.0} µs, ITL p50 {:+.0} µs (p99 {:+.0}); parse {:.1} µs/req",
        err_us(s.ttft_ms.p50, 50.0), err_us(s.ttft_ms.max, 50.0), err_us(s.tpot_ms.p50, 10.0), err_us(s.itl_ms.p50, 10.0), err_us(s.itl_ms.p99, 10.0),
        s.client.parse_ms_per_request * 1e3
    );
    close(s.ttft_ms.p50, 50.0, 3.0, "TTFT p50")?;
    if !(s.ttft_first_chunk_ms.p50 < 10.0) {
        bail!("self-check FAILED: ttft_first_chunk p50 {:.2} (mock sends empty role delta at once, want < 10 ms)", s.ttft_first_chunk_ms.p50);
    }
    close(s.tpot_ms.p50, 10.0, 1.0, "TPOT p50")?;
    close(s.itl_ms.p50, 10.0, 1.5, "ITL p50")?;
    close(s.latency_ms.p50, 50.0 + 19.0 * 10.0, 6.0, "latency p50")?;
    close(s.prefix_cache_hit_ratio.unwrap_or(-1.0), 5.0 / 17.0, 1e-9, "prefix_cache_hit_ratio")?;
    if s.cached_reported != 16 || s.finished_by_length != 16 {
        bail!("self-check FAILED: cached_reported {} finished_by_length {}", s.cached_reported, s.finished_by_length);
    }
    if s.good_requests != Some(16) || s.goodput_rps != Some(s.request_throughput_rps) {
        bail!("self-check FAILED: goodput {:?}/{:?} vs {:.2}", s.good_requests, s.goodput_rps, s.request_throughput_rps);
    }
    // 并发 4、每请求 ≈240 ms、16 个请求 → ≈ 4 轮 ≈ 1 s
    if !(0.8..1.6).contains(&s.duration_s) {
        bail!("self-check FAILED: duration {:.2}s (expected ≈1 s for 4 rounds)", s.duration_s);
    }
    // 多轮 2 × 8 会话：每轮都有回答文本接进下一轮 messages；mock 不看 messages，只核条数与 turn 标记
    let a2 = Args { multi_turn: 2, num_requests: 16, warmup: 0, ..test_args(s.url.clone()) };
    let s2 = run(&a2).await?;
    if s2.completed != 16 || s2.requests.iter().filter(|r| r.turn == 2).count() != 8 {
        bail!("self-check FAILED: multi-turn completed {} turn2 {}", s2.completed, s2.requests.iter().filter(|r| r.turn == 2).count());
    }
    println!("self-check PASS (TTFT/TPOT/ITL/latency/usage/cached/goodput/concurrency/multi-turn all within tolerance)");
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

#[cfg(test)]
mod tests {
    use super::*;

    fn req(ok: bool, ttft: f64, out: u32, latency: f64, cached: Option<u32>, prompt: u32) -> ReqResult {
        ReqResult {
            ok,
            prompt_tokens: prompt,
            output_tokens: out,
            cached_tokens: cached,
            ttft_ms: ttft,
            latency_ms: latency,
            ..ReqResult::blank(128, 1)
        }
    }

    #[test]
    fn len_mix_parses_and_rejects() {
        assert_eq!(parse_len_mix("128:0.5, 1024:0.3").unwrap(), vec![(128, 0.5), (1024, 0.3)]);
        assert!(parse_len_mix("").is_err());
        assert!(parse_len_mix("128:0").is_err());
        assert!(parse_len_mix("128").is_err());
    }

    #[test]
    fn mix_sampling_matches_weights_and_is_deterministic() {
        let mix = parse_len_mix("128:0.5,512:0.35,1024:0.15").unwrap();
        let n = 20_000;
        let mut cnt = std::collections::HashMap::new();
        for i in 0..n {
            *cnt.entry(sample_mix(&mix, uniform01(7, 0x0C0C_7777, i))).or_insert(0usize) += 1;
        }
        for &(l, w) in &mix {
            let got = cnt[&l] as f64 / n as f64;
            assert!((got - w).abs() < 0.015, "len {l}: got {got:.3} want {w}");
        }
        assert_eq!(uniform01(7, 1, 3), uniform01(7, 1, 3));
        assert_ne!(uniform01(7, 1, 3), uniform01(7, 2, 3), "prompt / output 抽样用不同 salt 应独立");
    }

    #[test]
    fn output_len_mix_overrides_max_tokens() {
        let mut a = test_args(String::new());
        a.output_len_mix = "64:1".into();
        let omix = parse_len_mix(&a.output_len_mix).unwrap();
        assert_eq!(max_tokens_for(&a, &omix, 0), 64);
        assert_eq!(max_tokens_for(&a, &[], 0), 20);
    }

    #[test]
    fn goodput_counts_only_requests_meeting_both_slos() {
        // 4 ok：好 / TTFT 违约 / TPOT 违约 / 两者都违约；1 failed 不计
        let rs = vec![
            req(true, 100.0, 11, 300.0, Some(0), 100),  // tpot (300-100)/10 = 20 → 好
            req(true, 600.0, 11, 800.0, Some(0), 100),  // ttft 违约, tpot 20
            req(true, 100.0, 11, 400.0, Some(0), 100),  // tpot 30 违约
            req(true, 900.0, 11, 1400.0, Some(0), 100), // 都违约 (tpot 50)
            req(false, 0.0, 0, 0.0, None, 0),
        ];
        let a = Args { slo_ttft_ms: 500.0, slo_tpot_ms: 20.0, ..test_args(String::new()) };
        let s = summarize(&a, "m".into(), &rs, 2.0);
        assert_eq!(s.completed, 4);
        assert_eq!(s.failed, 1);
        assert_eq!(s.good_requests, Some(1));
        assert_eq!(s.goodput_rps, Some(0.5));
        let v = s.slo_violations.unwrap();
        assert_eq!((v.ttft, v.tpot), (2, 2));
        assert_eq!(s.request_throughput_rps, 2.0);
        // 只判 TTFT：3 条 ttft ≤ 500 中 2 条好
        let a = Args { slo_ttft_ms: 500.0, ..test_args(String::new()) };
        assert_eq!(summarize(&a, "m".into(), &rs, 2.0).good_requests, Some(2));
        // 无 SLO → null
        let a = test_args(String::new());
        let s = summarize(&a, "m".into(), &rs, 2.0);
        assert!(s.goodput_rps.is_none() && s.good_requests.is_none() && s.slo_violations.is_none());
    }

    #[test]
    fn single_token_output_counts_tpot_as_zero_for_goodput() {
        // vLLM：output_len ≤ 1 → tpot 0 → 不违约；但不进 TPOT 分布
        let rs = vec![req(true, 100.0, 1, 5000.0, None, 10)];
        let a = Args { slo_ttft_ms: 500.0, slo_tpot_ms: 20.0, ..test_args(String::new()) };
        let s = summarize(&a, "m".into(), &rs, 1.0);
        assert_eq!(s.good_requests, Some(1));
        assert_eq!(s.tpot_ms.p50, 0.0, "空分布");
    }

    #[test]
    fn prefix_cache_ratio_uses_only_reporting_requests() {
        let rs = vec![
            req(true, 1.0, 2, 2.0, Some(50), 100),
            req(true, 1.0, 2, 2.0, Some(0), 100),
            req(true, 1.0, 2, 2.0, None, 1000), // 没报 → 不进分母
        ];
        let s = summarize(&test_args(String::new()), "m".into(), &rs, 1.0);
        assert_eq!(s.prefix_cache_hit_ratio, Some(0.25));
        assert_eq!(s.cached_tokens_mean, Some(25.0));
        assert_eq!(s.cached_reported, 2);
        let s = summarize(&test_args(String::new()), "m".into(), &[req(true, 1.0, 2, 2.0, None, 10)], 1.0);
        assert!(s.prefix_cache_hit_ratio.is_none());
    }

    #[test]
    fn poisson_intervals_have_right_mean() {
        let n = 50_000;
        let sum: f64 = (0..n).map(|i| exp_interval_s(3, i, 8.0)).sum();
        assert!((sum / n as f64 - 0.125).abs() < 0.003);
    }

    #[test]
    fn multi_turn_session_count() {
        // 向上取整：N=10, K=4 → 3 个会话（12 条）；N=16, K=2 → 8
        assert_eq!(10usize.div_ceil(4), 3);
        assert_eq!(16usize.div_ceil(2), 8);
        assert!(follow_up(1, 5).starts_with("Continue the story"));
        assert_ne!(follow_up(1, 5), follow_up(1, 6));
    }
}
