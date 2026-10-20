//! Kimi-K3 的 `ChatBackend`（里程碑 C-4b）：加载前 N 层 + 顶层、tiktoken tokenizer，起引擎线程；
//! 渲染走 `xtml`（`Dialect::{Hf,Serving}`，默认 Hf = 基线 vLLM 的 HF `apply_chat_template` 路径），
//! 输出流经 `parser::OutputParser`（XTML 思考段 / 回复段 / 工具调用；思考恒开，`reasoning_effort` 进 xtml）。
//!
//! **前 N 层模式（`--layers N` / `KIMIK3_LAYERS=N`）只为冒烟**：输出无语义，但调度 / KV / 采样 / HTTP 全流程要通；
//! 启动日志明示 `TRUNCATED MODEL`。全 93 层单卡放不下（专家 1.45 TB）→ C-5 EP/TP。
//!
//! KV：MLA 层 paged latent（`KvCacheSpec::Mla`，576 维 / token），块数 `--num-blocks` 或按 `max_num_seqs × max_model_len`
//! 算（MLA 每 token 1.1 KB，24 层 × 16 请求 × 4k = 1.7 GB，不必按剩余显存压满）；KDA 层状态槽 = `max_num_seqs`（+1 废纸篓），
//! 不经过 core 的 `KvCacheSpec::Mamba`（与 Qwen3.5 同口径，design-qwen35 §3 待办）。prefix cache 强制关（递归状态不可按块复用）。
use crate::config::KimiK3Config;
use crate::executor::KimiK3Executor;
use crate::tp::TpExecutor;
use crate::weights::Shard;
use crate::layer::LayerBudget;
use crate::model::KimiK3Model;
use crate::parser::OutputParser;
use crate::tokenizer::Tokenizer;
use crate::xtml::{build_chat_segments, Dialect, RenderOptions, VALID_THINKING_EFFORTS};
use anyhow::{bail, Context, Result};
use dinfer_core::config::{CacheConfig, SchedulerConfig};
use dinfer_core::engine::{self, EngineEvent, EngineHandle, Submission};
use dinfer_core::kv::{GroupInfo, KvCacheSpec};
use dinfer_core::request::RequestId;
use dinfer_core::runner::Runner;
use dinfer_core::sample::SamplingParams;
use dinfer_core::sched::Scheduler;
use dinfer_layers::Device;
use dinfer_server::backend::{ChatBackend, OutDelta, OutputStream, Prompt, RenderRequest};
use std::path::Path;
use std::sync::Arc;

#[derive(Clone)]
pub struct ServeOptions {
    pub model_dir: String,
    pub tokenizer_dir: Option<String>,
    pub max_model_len: u32,
    pub max_num_seqs: u32,
    pub max_num_batched_tokens: u32,
    /// MLA 层的 paged KV 块数（None = 按 `max_num_seqs × max_model_len` 算）。
    pub num_blocks: Option<u32>,
    pub block_size: u32,
    pub gpu_memory_utilization: f32,
    pub enable_chunked_prefill: bool,
    pub enable_prefix_caching: bool,
    pub decode_exempt_from_budget: bool,
    /// 单条 prefill 每步最多 token（0 = 不限）；vLLM `long_prefill_token_threshold`
    pub long_prefill_token_threshold: u32,
    pub device: usize,
    pub enable_cuda_graph: bool,
    /// 只装前 N 层（None = `KIMIK3_LAYERS` 环境变量，再无 = 全部）。**冒烟用**。
    pub layers: Option<usize>,
    /// 聊天渲染方言：`hf`（默认，基线 vLLM 路径）| `serving`（Moonshot 线上 Rust renderer 口径）。
    pub chat_dialect: String,
    /// TP/EP 组大小（design-kimik3 §2.4）：用卡 `device .. device+tp`，每 rank 一线程 lockstep；1 = 单卡。
    pub tp: usize,
}

impl Default for ServeOptions {
    fn default() -> Self {
        Self {
            model_dir: String::new(),
            tokenizer_dir: None,
            max_model_len: 4096,
            max_num_seqs: 16,
            max_num_batched_tokens: 2048,
            num_blocks: None,
            block_size: 16,
            gpu_memory_utilization: 0.9,
            enable_chunked_prefill: true,
            enable_prefix_caching: false,
            decode_exempt_from_budget: true,
            long_prefill_token_threshold: 0,
            device: 0,
            enable_cuda_graph: true,
            layers: None,
            chat_dialect: "hf".into(),
            tp: 1,
        }
    }
}

pub struct KimiK3Service {
    tok: Arc<Tokenizer>,
    dialect: Dialect,
    handle: EngineHandle,
    name: String,
    max_model_len: u32,
    defaults: SamplingParams,
    _engine: std::thread::JoinHandle<Result<()>>,
}

fn parse_dialect(s: &str) -> Result<Dialect> {
    match s.to_ascii_lowercase().as_str() {
        "hf" => Ok(Dialect::Hf),
        "serving" => Ok(Dialect::Serving),
        other => bail!("--chat-dialect {other}: 只认 hf | serving"),
    }
}

impl KimiK3Service {
    pub fn start(opt: ServeOptions, served_name: String) -> Result<Self> {
        let mut t_startup = dinfer_layers::StartupTimer::start();
        let model_dir = Path::new(&opt.model_dir);
        let tok_dir = opt.tokenizer_dir.as_deref().map(Path::new).unwrap_or(model_dir).to_path_buf();
        let dialect = parse_dialect(&opt.chat_dialect)?;
        let tok_thread = std::thread::spawn(move || -> Result<Tokenizer> {
            Tokenizer::from_dir(&tok_dir).with_context(|| format!("tokenizer in {}", tok_dir.display()))
        });
        let defaults = read_generation_defaults(model_dir);

        let cfg = KimiK3Config::from_json(&std::fs::read_to_string(model_dir.join("config.json"))?)?;
        let tc = &cfg.text_config;
        let total_layers = tc.num_hidden_layers as usize;
        let n_layers = opt
            .layers
            .or_else(|| std::env::var("KIMIK3_LAYERS").ok().and_then(|s| s.parse().ok()))
            .unwrap_or(total_layers);
        anyhow::ensure!((1..=total_layers).contains(&n_layers), "--layers {n_layers} 不在 1..={total_layers}");
        if n_layers < total_layers {
            tracing::warn!(
                n_layers,
                total_layers,
                "kimik3: TRUNCATED MODEL —— 只装前 {n_layers}/{total_layers} 层，输出无语义，仅用于流程冒烟"
            );
        }

        let num_spec = 0usize;
        let max_tokens = if opt.decode_exempt_from_budget {
            (opt.max_num_batched_tokens + opt.max_num_seqs) as usize
        } else {
            opt.max_num_batched_tokens.max(opt.max_num_seqs) as usize
        } + opt.max_num_seqs as usize * num_spec;
        let max_rows = opt.max_num_seqs as usize * (1 + num_spec);
        let bs = opt.block_size as usize;
        let max_blocks_per_seq = (opt.max_model_len as usize).div_ceil(bs);
        // MLA paged latent：每 token entry = kv_lora + rope（576）bf16。块数不按剩余显存压满（见文件头）。
        let entry = (tc.kv_lora_rank + tc.qk_rope_head_dim) as usize;
        let kv_blocks = match opt.num_blocks {
            Some(n) => n as usize,
            None => opt.max_num_seqs as usize * max_blocks_per_seq + 1,
        };
        let n_mla = (0..n_layers as u32).filter(|&l| !tc.is_kda_layer(l)).count();
        let n_kda = n_layers - n_mla;
        // KDA 状态槽：max_num_seqs 个活请求 + 1 个废纸篓（图捕获 / padding 行用）
        let budget = LayerBudget { max_tokens, num_slots: max_rows + 1, kv_blocks, block_size: bs };
        tracing::info!(
            cuda_graphs = opt.enable_cuda_graph,
            chunked_prefill = opt.enable_chunked_prefill,
            n_layers,
            n_kda,
            n_mla,
            max_tokens,
            kv_blocks,
            kv_tokens = kv_blocks * bs,
            mla_kv_mib = (n_mla * kv_blocks * bs * entry * 2) as f64 / 1048576.0,
            kda_slots = budget.num_slots,
            "kimik3: single-stream executor（decode CUDA graph lazy per bucket；no TP / EP / PD / MTP yet）"
        );

        anyhow::ensure!(
            max_blocks_per_seq < kv_blocks,
            "KV pool too small: max_model_len {} needs {} blocks, pool has {}",
            opt.max_model_len,
            max_blocks_per_seq,
            kv_blocks
        );
        // 每个 rank 一张卡：Device + 模型分片 + executor（+ NCCL comm）。rank 0 在本线程，其余各起一线程并行装载（同 qwen3 service）。
        let tp_world = opt.tp.max(1);
        let comm_id = if tp_world > 1 { Some(dinfer_kernels::comm::CommId::new()?) } else { None };
        let build_rank = {
            let tc = tc.clone();
            let model_dir = model_dir.to_path_buf();
            let (dev0, graphs, act_est) =
                (opt.device, opt.enable_cuda_graph, (max_tokens * tc.hidden_size as usize * 2 * 16) as f64 / 1048576.0);
            move |rank: usize| -> Result<KimiK3Executor> {
                let dev = Device::new(dev0 + rank)?;
                let s = dev.default_stream();
                let gemm = dinfer_layers::Gemm::new(&s)?;
                let (free0, total) = s.mem_info()?;
                let t_load = std::time::Instant::now();
                let shard = Shard::new(rank, tp_world)?;
                let model = KimiK3Model::load_tp(&s, &model_dir, &tc, n_layers, budget, max_rows, shard)
                    .with_context(|| format!("kimik3 model load (rank {rank})"))?;
                s.synchronize()?;
                let (free1, _) = s.mem_info()?;
                let used_gb = free0.saturating_sub(free1) as f64 / 1e9;
                tracing::info!(
                    rank,
                    n_layers,
                    secs = format!("{:.1}", t_load.elapsed().as_secs_f32()),
                    gb = format!("{used_gb:.2}"),
                    act_mib_est = act_est,
                    free_gb = format!("{:.1}", free1 as f64 / 1e9),
                    total_gb = format!("{:.1}", total as f64 / 1e9),
                    "kimik3 model loaded（权重 + 激活 + KV 合计；激活估算 = max_tokens × hidden × bf16 × ~16 份）"
                );
                KimiK3Executor::new_with_tp(
                    s.clone(),
                    gemm,
                    model,
                    max_blocks_per_seq,
                    graphs,
                    comm_id.map(|id| (rank, tp_world, id)),
                )
            }
        };
        let other_ranks: Vec<_> = (1..tp_world)
            .map(|r| {
                let b = build_rank.clone();
                std::thread::Builder::new().name(format!("tp-load{r}")).spawn(move || b(r)).expect("spawn tp load")
            })
            .collect();
        let exec0 = build_rank(0)?;
        let mut others = Vec::with_capacity(other_ranks.len());
        for jh in other_ranks {
            others.push(jh.join().map_err(|_| anyhow::anyhow!("tp load thread panicked"))??);
        }
        t_startup.model_loaded();
        let startup_phases = exec0.startup_phases();

        let tok = Arc::new(tok_thread.join().map_err(|_| anyhow::anyhow!("tokenizer thread panicked"))??);
        anyhow::ensure!(tc.vocab_size as usize >= tok.vocab_size(), "model vocab {} < tokenizer {}", tc.vocab_size, tok.vocab_size());
        anyhow::ensure!(!tok.eos_ids().is_empty(), "tokenizer: generation_config.json 没有 eos_token_id");
        tracing::info!(eos = ?tok.eos_ids(), eos_name = ?tok.eos_ids().iter().map(|&e| tok.special_name(e).unwrap_or("?")).collect::<Vec<_>>(), "kimik3 eos");

        if opt.enable_prefix_caching {
            tracing::warn!("kimik3: prefix caching 对 KDA 混合模型未实现（递归状态不可按块复用），已强制关闭");
        }
        let params = engine::scheduler_params(
            SchedulerConfig {
                max_num_seqs: opt.max_num_seqs,
                max_num_batched_tokens: opt.max_num_batched_tokens,
                enable_chunked_prefill: opt.enable_chunked_prefill,
                long_prefill_token_threshold: opt.long_prefill_token_threshold,
                decode_exempt_from_budget: opt.decode_exempt_from_budget,
            },
            CacheConfig {
                block_size: opt.block_size,
                enable_prefix_caching: false,
                hash_block_size: opt.block_size,
                gpu_memory_utilization: opt.gpu_memory_utilization,
            },
            opt.max_model_len,
            tok.eos_ids().to_vec(),
            num_spec as u32,
        );
        let groups = vec![GroupInfo {
            spec: KvCacheSpec::Mla { block_size: opt.block_size, head_size: entry as u32, dtype_bytes: 2 },
            block_size: opt.block_size,
        }];
        let sched = Scheduler::new(params, groups, kv_blocks as u32);
        let (handle, jh) = if tp_world > 1 {
            tracing::info!(tp = tp_world, "kimik3: TP/EP lockstep executor（rank 0 = engine 线程）");
            engine::spawn(sched, Runner::new(TpExecutor::new(exec0, others), bs))
        } else {
            engine::spawn(sched, Runner::new(exec0, bs))
        };
        tracing::info!(
            model = %served_name,
            max_model_len = opt.max_model_len,
            num_blocks = kv_blocks,
            kv_tokens = kv_blocks * bs,
            num_slots = max_rows,
            dialect = ?dialect,
            truncated = n_layers < total_layers,
            "kimik3 engine started"
        );
        t_startup.done(startup_phases.0, startup_phases.1);
        Ok(Self { tok, dialect, handle, name: served_name, max_model_len: opt.max_model_len, defaults, _engine: jh })
    }

    pub fn shutdown(&self) {
        self.handle.shutdown();
    }
}

fn read_generation_defaults(dir: &Path) -> SamplingParams {
    let mut sp = SamplingParams::default();
    if let Ok(s) = std::fs::read_to_string(dir.join("generation_config.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
            if let Some(t) = v["temperature"].as_f64() {
                sp.temperature = t as f32;
            }
            if let Some(p) = v["top_p"].as_f64() {
                sp.top_p = p as f32;
            }
            if let Some(k) = v["top_k"].as_i64() {
                sp.top_k = k as i32;
            }
        }
    }
    sp
}

/// `reasoning_effort` / `chat_template_kwargs.{thinking,thinking_effort}` → xtml 的 (thinking, effort)。
/// 规则（C-4a 核对）：默认 thinking=true、effort=max；`none` → thinking=false；非法 effort 报错（HF assert / vLLM 400）。
fn thinking_opts<'a>(req: &RenderRequest<'a>) -> Result<(bool, Option<&'a str>)> {
    let mut thinking = true;
    let mut effort: Option<&'a str> = Some("max");
    if let Some(extra) = req.extra {
        if let Some(b) = extra.get("thinking").and_then(|v| v.as_bool()) {
            thinking = b;
        }
        if let Some(e) = extra.get("thinking_effort").and_then(|v| v.as_str()) {
            effort = Some(e);
        }
        if let Some(e) = extra.get("thinking").and_then(|v| v.get("effort")).and_then(|v| v.as_str()) {
            effort = Some(e);
        }
    }
    if let Some(e) = req.reasoning_effort {
        if e.eq_ignore_ascii_case("none") {
            thinking = false;
        } else {
            effort = Some(e);
        }
    }
    if let Some(e) = effort {
        anyhow::ensure!(
            VALID_THINKING_EFFORTS.contains(&e),
            "reasoning_effort `{e}` 非法：只认 {:?} 或 none",
            VALID_THINKING_EFFORTS
        );
    }
    Ok((thinking, effort))
}

/// 增量 detokenize（bytes 缓冲，半个 UTF-8 不吐）+ XTML 解析。**特殊 token 不跳过**：parser 靠 `<|open|>` 等标记切段；
/// 只有 eos（`<|end_of_msg|>`）不进文本。
struct K3OutputStream {
    tok: Arc<Tokenizer>,
    parser: OutputParser,
    pending: Vec<u8>,
}

impl K3OutputStream {
    fn drain(&mut self, flush: bool) -> String {
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let s = s.to_string();
                self.pending.clear();
                s
            }
            Err(e) => {
                let ok = e.valid_up_to();
                if e.error_len().is_none() && !flush {
                    let s = String::from_utf8_lossy(&self.pending[..ok]).into_owned();
                    self.pending.drain(..ok);
                    s
                } else {
                    let s = String::from_utf8_lossy(&self.pending).into_owned();
                    self.pending.clear();
                    s
                }
            }
        }
    }
}

impl OutputStream for K3OutputStream {
    fn push_text(&mut self, ids: &[u32]) -> Result<String> {
        for &id in ids {
            if self.tok.is_eos(id) {
                continue;
            }
            if let Some(b) = self.tok.token_bytes(id) {
                self.pending.extend_from_slice(b);
            }
        }
        Ok(self.drain(false))
    }
    fn parse(&mut self, text: &str) -> OutDelta {
        if text.is_empty() {
            return OutDelta::default();
        }
        let d = self.parser.feed(text);
        OutDelta {
            reasoning: d.reasoning,
            content: d.content,
            tool_calls: d.tool_calls.into_iter().map(|t| (t.index, t.name, t.arguments)).collect(),
        }
    }
    fn finish(&mut self) -> Result<OutDelta> {
        let tail = self.drain(true);
        let mut d = self.parser.feed(&tail);
        let f = self.parser.finish();
        d.reasoning.push_str(&f.reasoning);
        d.content.push_str(&f.content);
        d.tool_calls.extend(f.tool_calls);
        Ok(OutDelta {
            reasoning: d.reasoning,
            content: d.content,
            tool_calls: d.tool_calls.into_iter().map(|t| (t.index, t.name, t.arguments)).collect(),
        })
    }
    fn token_text(&self, id: u32) -> Option<String> {
        self.tok.token_text(id)
    }
    fn set_skip_special_tokens(&mut self, _skip: bool) {
        // XTML 标记本身就是特殊 token，parser 需要看到它们；忽略开关。
    }
}

impl ChatBackend for KimiK3Service {
    fn model_name(&self) -> &str {
        &self.name
    }
    fn max_model_len(&self) -> u32 {
        self.max_model_len
    }
    fn default_sampling(&self) -> SamplingParams {
        self.defaults.clone()
    }
    fn render(&self, req: &RenderRequest<'_>) -> Result<Prompt> {
        let (thinking, thinking_effort) = thinking_opts(req)?;
        let tool_choice = req.extra.and_then(|m| m.get("tool_choice")).and_then(|v| v.as_str());
        let response_format = req.extra.and_then(|m| m.get("response_format"));
        let opts = RenderOptions {
            tools: req.tools,
            add_generation_prompt: req.add_generation_prompt && !req.continue_final_message,
            thinking,
            thinking_effort,
            tool_choice,
            response_format,
            dialect: self.dialect,
        };
        let segs = build_chat_segments(req.messages, &opts)?;
        let mut text = String::new();
        let mut ids = Vec::new();
        for s in &segs {
            text.push_str(&s.text);
            self.tok.encode_into(&s.text, s.allow_special, &mut ids);
        }
        Ok(Prompt { text, ids })
    }
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tok.encode(text, false))
    }
    fn decode(&self, ids: &[u32]) -> Result<String> {
        Ok(self.tok.decode_with(ids, true))
    }
    fn token_strs(&self, ids: &[u32]) -> Option<Vec<String>> {
        ids.iter().map(|&id| self.tok.token_text(id)).collect()
    }
    fn submit(
        &self,
        prompt_ids: Vec<u32>,
        sampling: SamplingParams,
        pd: Option<dinfer_core::engine::PdSubmit>,
    ) -> (RequestId, tokio::sync::mpsc::UnboundedReceiver<EngineEvent>) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let id = self.handle.submit(Submission { prompt_ids, sampling, tx, pd });
        (id, rx)
    }
    fn abort(&self, id: RequestId) {
        self.handle.abort(id);
    }
    fn metrics(&self) -> Option<Arc<dinfer_core::metrics::Metrics>> {
        Some(self.handle.metrics())
    }
    fn new_output_stream(&self) -> Box<dyn OutputStream> {
        // 生成提示恒以 `<|open|>think<|sep|>` 结尾（thinking=false 时是 response）；parser 初态按默认 thinking=true。
        // reasoning_effort=none 的请求初态应为 Response：server 的 OutputStream 接口没带 render 结果，parser 会在看到
        // 文本不含 think 段时自行走 Response（`fresh` 吞 open 标记）；C-5 若有偏差再加钩子。
        Box::new(K3OutputStream { tok: self.tok.clone(), parser: OutputParser::new(true), pending: Vec::new() })
    }
}
