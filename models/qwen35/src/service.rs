//! Qwen3.5 的 `ChatBackend`：加载权重 + tokenizer + chat 模板（同 Qwen3 模板族），起引擎线程，
//! 提供渲染 / 提交 / 输出流（`parser::OutputParser`）。
//!
//! **里程碑 4 第一版**：单卡、无 TP / PD / host tier / CUDA graph（选项保留在 `ServeOptions` 里但会被忽略或报错，
//! 见 `start` 里的 `ensure!`）。KV 组：全注意力层用 paged `FullAttention`；GDN 层状态在模型内按 slot 自管，
//! 不经过 core 的 `KvCacheSpec::Mamba`（design-qwen35 §3 待办）。
use crate::config::Qwen35Config;
use crate::executor::Qwen35Executor;
use crate::model::Qwen35Model;
use crate::parser::OutputParser;
use anyhow::{Context, Result};
use dinfer_core::config::{CacheConfig, SchedulerConfig};
use dinfer_core::engine::{self, EngineEvent, EngineHandle, Submission};
use dinfer_core::kv::{GroupInfo, KvCacheSpec};
use dinfer_core::request::RequestId;
use dinfer_core::runner::Runner;
use dinfer_core::sample::SamplingParams;
use dinfer_core::sched::Scheduler;
use dinfer_layers::{Device, WeightSource};
use dinfer_server::backend::{ChatBackend, OutDelta, OutputStream, Prompt, RenderRequest};
use dinfer_text::{ChatTemplate, HfTokenizer, RenderParams};
use std::path::Path;
use std::sync::Arc;

#[derive(Clone)]
pub struct ServeOptions {
    pub model_dir: String,
    pub tokenizer_dir: Option<String>,
    pub max_model_len: u32,
    pub max_num_seqs: u32,
    pub max_num_batched_tokens: u32,
    /// 全注意力层的 KV 块数（自动算：按剩余显存）。
    pub num_blocks: Option<u32>,
    pub block_size: u32,
    pub gpu_memory_utilization: f32,
    pub enable_chunked_prefill: bool,
    pub enable_prefix_caching: bool,
    pub decode_exempt_from_budget: bool,
    pub device: usize,
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
            enable_prefix_caching: true,
            decode_exempt_from_budget: true,
            device: 0,
        }
    }
}

pub struct Qwen35Service {
    tok: Arc<HfTokenizer>,
    tmpl: ChatTemplate,
    handle: EngineHandle,
    name: String,
    max_model_len: u32,
    defaults: SamplingParams,
    _engine: std::thread::JoinHandle<Result<()>>,
}

impl Qwen35Service {
    pub fn start(opt: ServeOptions, served_name: String) -> Result<Self> {
        let t_start = std::time::Instant::now();
        let model_dir = Path::new(&opt.model_dir);
        let tok_dir = opt.tokenizer_dir.as_deref().map(Path::new).unwrap_or(model_dir);
        let tok_dir_owned = tok_dir.to_path_buf();
        let tok_thread = std::thread::spawn(move || -> Result<(HfTokenizer, ChatTemplate)> {
            let tok = HfTokenizer::from_dir(&tok_dir_owned)
                .with_context(|| format!("tokenizer in {}", tok_dir_owned.display()))?;
            let tmpl =
                ChatTemplate::from_dir(&tok_dir_owned).with_context(|| format!("chat template in {}", tok_dir_owned.display()))?;
            Ok((tok, tmpl))
        });
        let defaults = read_generation_defaults(tok_dir);
        // 里程碑 4 的 executor 是 eager 单流：前缀缓存能命中（scheduler 侧），但没有 decode CUDA graph，
        // 也没有跨流 / host tier / PD / 投机。这些在 `bin/dinfer` 里被忽略——启动时明确打一行，不静默。
        tracing::info!(
            cuda_graphs = 0,
            chunked_prefill = opt.enable_chunked_prefill,
            prefix_caching = opt.enable_prefix_caching,
            "qwen35: eager single-stream executor (no CUDA graph / xstream / host tier / PD / speculative yet)"
        );
        let cfg = Qwen35Config::from_json(&std::fs::read_to_string(model_dir.join("config.json"))?)?;
        let src = Arc::new(WeightSource::open_dir(model_dir)?);

        let num_spec = 0usize; // 里程碑 4 不做投机
        let max_tokens = if opt.decode_exempt_from_budget {
            (opt.max_num_batched_tokens + opt.max_num_seqs) as usize
        } else {
            opt.max_num_batched_tokens.max(opt.max_num_seqs) as usize
        } + opt.max_num_seqs as usize * num_spec;
        let max_logits_rows = opt.max_num_seqs as usize * (1 + num_spec);
        let bs = opt.block_size as usize;
        let max_blocks_per_seq = (opt.max_model_len as usize).div_ceil(bs);

        let dev = Device::new(opt.device)?;
        let s = dev.default_stream();
        let gemm = dinfer_layers::Gemm::new(&s)?;
        // KV 块数：`--num-blocks` 给定就用；否则权重 + scratch 分配完后按剩余显存自动算（`load_with_budget` 内部）。
        let model: Qwen35Model = Qwen35Model::load_with_budget(
            &s,
            &src,
            cfg.clone(),
            max_tokens,
            max_logits_rows,
            opt.num_blocks.map(|n| n as usize),
            bs,
            opt.max_num_seqs as usize * (1 + num_spec),
            (0, 1),
            opt.gpu_memory_utilization,
            512 << 20,
        )?;
        s.synchronize()?;
        let num_blocks = model.num_blocks();
        let exec = Qwen35Executor::new(s.clone(), gemm, model, max_blocks_per_seq)?;
        let need = max_blocks_per_seq;
        anyhow::ensure!(
            need as f64 <= num_blocks as f64 * 0.9,
            "KV pool too small: max_model_len {} needs {} blocks, pool has {}; raise --num-blocks / --gpu-memory-utilization",
            opt.max_model_len,
            need,
            num_blocks
        );
        let t_model = t_start.elapsed();

        let (tok, tmpl) = tok_thread.join().map_err(|_| anyhow::anyhow!("tokenizer thread panicked"))??;
        let tok = Arc::new(tok);
        anyhow::ensure!(cfg.vocab_size as usize >= tok.vocab_size(), "model vocab {} < tokenizer {}", cfg.vocab_size, tok.vocab_size());

        let params = engine::scheduler_params(
            SchedulerConfig {
                max_num_seqs: opt.max_num_seqs,
                max_num_batched_tokens: opt.max_num_batched_tokens,
                enable_chunked_prefill: opt.enable_chunked_prefill,
                long_prefill_token_threshold: 0,
                decode_exempt_from_budget: opt.decode_exempt_from_budget,
            },
            CacheConfig {
                block_size: opt.block_size,
                enable_prefix_caching: opt.enable_prefix_caching,
                hash_block_size: opt.block_size,
                gpu_memory_utilization: opt.gpu_memory_utilization,
            },
            opt.max_model_len,
            tok.eos_ids().to_vec(),
            num_spec as u32,
        );
        let groups = vec![GroupInfo {
            spec: KvCacheSpec::FullAttention {
                block_size: opt.block_size,
                kv_heads: cfg.num_key_value_heads,
                head_size: cfg.head_dim,
                dtype_bytes: 2,
            },
            block_size: opt.block_size,
        }];
        let sched = Scheduler::new(params, groups, num_blocks as u32);
        let (handle, jh) = engine::spawn(sched, Runner::new(exec, bs));
        tracing::info!(
            model = %served_name,
            max_model_len = opt.max_model_len,
            num_blocks,
            kv_tokens = num_blocks * bs,
            num_slots = opt.max_num_seqs,
            "qwen35 engine started"
        );
        tracing::info!(
            model_load_ms = t_model.as_millis() as u64,
            total_ms = t_start.elapsed().as_millis() as u64,
            "startup timing"
        );
        Ok(Self { tok, tmpl, handle, name: served_name, max_model_len: opt.max_model_len, defaults, _engine: jh })
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

struct Qwen35OutputStream {
    tok: Arc<HfTokenizer>,
    parser: OutputParser,
    ids: Vec<u32>,
    prefix_len: usize,
    emitted: String,
    skip_special: bool,
}

impl Qwen35OutputStream {
    /// 增量解码（与 Qwen3 同法：整段解码后按已发前缀切差量，处理多字节字符跨 token）。
    fn decode_new(&mut self) -> Result<String> {
        let text = self.tok.decode_with(&self.ids[self.prefix_len..], self.skip_special)?;
        if text.ends_with('\u{fffd}') {
            return Ok(String::new());
        }
        if !text.starts_with(&self.emitted) {
            self.emitted = text.clone();
            return Ok(text);
        }
        let new = text[self.emitted.len()..].to_string();
        self.emitted = text;
        if self.ids.len() - self.prefix_len > 64 {
            self.prefix_len = self.ids.len();
            self.emitted.clear();
        }
        Ok(new)
    }
}

impl OutputStream for Qwen35OutputStream {
    fn push_text(&mut self, ids: &[u32]) -> Result<String> {
        let mut out = String::new();
        for &id in ids {
            if self.tok.is_eos(id) {
                continue;
            }
            self.ids.push(id);
            out.push_str(&self.decode_new()?);
        }
        Ok(out)
    }
    fn parse(&mut self, text: &str) -> OutDelta {
        if text.is_empty() {
            return OutDelta::default();
        }
        let pd = self.parser.feed(text);
        OutDelta {
            reasoning: pd.reasoning,
            content: pd.content,
            tool_calls: pd.tool_calls.into_iter().map(|t| (t.index, t.name, t.arguments)).collect(),
        }
    }
    fn finish(&mut self) -> Result<OutDelta> {
        let pd = self.parser.finish();
        Ok(OutDelta {
            reasoning: pd.reasoning,
            content: pd.content,
            tool_calls: pd.tool_calls.into_iter().map(|t| (t.index, t.name, t.arguments)).collect(),
        })
    }
    fn token_text(&self, id: u32) -> Option<String> {
        self.tok.token_text(id)
    }
    fn set_skip_special_tokens(&mut self, skip: bool) {
        self.skip_special = skip;
    }
}

impl ChatBackend for Qwen35Service {
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
        let mut extra = req.extra.cloned().unwrap_or_default();
        extra.entry("enable_thinking".to_string()).or_insert(serde_json::Value::Bool(true));
        let text = self.tmpl.render(&RenderParams {
            messages: req.messages,
            tools: req.tools,
            add_generation_prompt: req.add_generation_prompt,
            continue_final_message: req.continue_final_message,
            extra: &extra,
        })?;
        let ids = self.tok.encode(&text)?;
        Ok(Prompt { text, ids })
    }
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tok.encode(text)
    }
    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tok.decode(ids)
    }
    fn token_strs(&self, ids: &[u32]) -> Option<Vec<String>> {
        ids.iter().map(|&id| self.tok.id_to_token(id)).collect()
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
        Box::new(Qwen35OutputStream {
            tok: self.tok.clone(),
            parser: OutputParser::new(),
            ids: Vec::new(),
            prefix_len: 0,
            emitted: String::new(),
            skip_special: true,
        })
    }
    fn tool_call_tags(&self) -> Option<dinfer_server::backend::ToolCallTags> {
        Some(dinfer_server::backend::ToolCallTags { open: "<tool_call>\n", close: "\n</tool_call>" })
    }
    fn grammar_render_extra(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut m = serde_json::Map::new();
        m.insert("enable_thinking".into(), serde_json::Value::Bool(false));
        m
    }
}
