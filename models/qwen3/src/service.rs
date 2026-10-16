//! Qwen3 的 `ChatBackend`：加载真权重 + tokenizer + chat 模板（`enable_thinking` 由请求的 `chat_template_kwargs` 或
//! 默认 true 决定），起引擎线程，提供渲染 / 提交 / 输出流（`parser::OutputParser`）。

use crate::config::Qwen3Config;
use crate::executor::Qwen3Executor;
use crate::model::Qwen3Model;
use crate::parser::OutputParser;
use anyhow::{Context, Result};
use dinfer_core::config::{CacheConfig, SchedulerConfig};
use dinfer_core::engine::{self, EngineEvent, EngineHandle, Submission};
use dinfer_core::kv::{GroupInfo, KvCacheSpec};
use dinfer_core::request::RequestId;
use dinfer_core::runner::Runner;
use dinfer_core::sample::SamplingParams;
use dinfer_core::sched::Scheduler;
use dinfer_layers::{Device, Gemm, WeightSource};
use dinfer_server::backend::{ChatBackend, OutDelta, OutputStream, Prompt, RenderRequest};
use dinfer_text::{ChatTemplate, HfTokenizer, RenderParams};
use std::path::Path;
use std::sync::Arc;

pub struct ServeOptions {
    pub model_dir: String,
    pub tokenizer_dir: Option<String>,
    pub max_model_len: u32,
    pub max_num_seqs: u32,
    pub max_num_batched_tokens: u32,
    /// None = 按显存自动算（见 `Qwen3Model::load_with_budget`）。
    pub num_blocks: Option<u32>,
    pub block_size: u32,
    /// 自动算 num_blocks 时的显存上限比例（vLLM 默认 0.9）。
    pub gpu_memory_utilization: f32,
    pub enable_chunked_prefill: bool,
    pub enable_prefix_caching: bool,
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
            device: 0,
        }
    }
}

pub struct Qwen3Service {
    tok: Arc<HfTokenizer>,
    tmpl: ChatTemplate,
    handle: EngineHandle,
    name: String,
    max_model_len: u32,
    defaults: SamplingParams,
    _engine: std::thread::JoinHandle<Result<()>>,
}

impl Qwen3Service {
    pub fn start(opt: ServeOptions, served_name: String) -> Result<Self> {
        let model_dir = Path::new(&opt.model_dir);
        let tok_dir = opt.tokenizer_dir.as_deref().map(Path::new).unwrap_or(model_dir);
        let tok = Arc::new(
            HfTokenizer::from_dir(tok_dir).with_context(|| format!("tokenizer in {}", tok_dir.display()))?,
        );
        let tmpl = ChatTemplate::from_dir(tok_dir)
            .with_context(|| format!("chat template in {}", tok_dir.display()))?;
        let defaults = read_generation_defaults(tok_dir);
        let cfg = Qwen3Config::from_json(&std::fs::read_to_string(model_dir.join("config.json"))?)?;
        anyhow::ensure!(
            cfg.vocab_size as usize >= tok.vocab_size(),
            "model vocab {} < tokenizer {}",
            cfg.vocab_size,
            tok.vocab_size()
        );
        let dev = Device::new(opt.device)?;
        let s = dev.default_stream();
        let gemm = Gemm::new(&s)?;
        let src = WeightSource::open_dir(model_dir)?;
        let max_tokens = opt.max_num_batched_tokens.max(opt.max_num_seqs) as usize;
        let model = Qwen3Model::load_with_budget(
            &s,
            &src,
            cfg.clone(),
            max_tokens,
            opt.max_num_seqs as usize, // 每请求只取最后一行 logits（无投机）
            opt.num_blocks.map(|n| n as usize),
            opt.block_size as usize,
            opt.gpu_memory_utilization,
            512 << 20, // cuBLAS workspace / 临时张量余量
        )?;
        let num_blocks = model.num_blocks() as u32;
        let exec = Qwen3Executor::new(s.clone(), gemm, model)?;
        let runner = Runner::new(exec, opt.block_size as usize);
        let params = engine::scheduler_params(
            SchedulerConfig {
                max_num_seqs: opt.max_num_seqs,
                max_num_batched_tokens: opt.max_num_batched_tokens,
                enable_chunked_prefill: opt.enable_chunked_prefill,
                long_prefill_token_threshold: 0,
            },
            CacheConfig {
                block_size: opt.block_size,
                enable_prefix_caching: opt.enable_prefix_caching,
                hash_block_size: opt.block_size,
                gpu_memory_utilization: opt.gpu_memory_utilization,
            },
            opt.max_model_len,
            tok.eos_ids().to_vec(),
            0,
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
        let sched = Scheduler::new(params, groups, num_blocks);
        let (handle, jh) = engine::spawn(sched, runner);
        tracing::info!(
            model = %served_name,
            max_model_len = opt.max_model_len,
            num_blocks,
            kv_tokens = num_blocks * opt.block_size,
            prefix_caching = opt.enable_prefix_caching,
            "qwen3 engine started"
        );
        Ok(Self {
            tok,
            tmpl,
            handle,
            name: served_name,
            max_model_len: opt.max_model_len,
            defaults,
            _engine: jh,
        })
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

struct Qwen3OutputStream {
    tok: Arc<HfTokenizer>,
    parser: OutputParser,
    ids: Vec<u32>,
    prefix_len: usize,
    emitted: String,
}

impl Qwen3OutputStream {
    fn decode_new(&mut self) -> Result<String> {
        let text = self.tok.decode(&self.ids[self.prefix_len..])?;
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

impl OutputStream for Qwen3OutputStream {
    fn push_text(&mut self, ids: &[u32]) -> Result<String> {
        let mut out = String::new();
        for &id in ids {
            if self.tok.is_eos(id) {
                continue; // eos 不进文本
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
}

impl ChatBackend for Qwen3Service {
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
        // enable_thinking：chat_template_kwargs 里给了就用；否则默认 true（HF 默认）
        let mut extra = req.extra.cloned().unwrap_or_default();
        extra.entry("enable_thinking".to_string()).or_insert(serde_json::Value::Bool(true));
        let text = self.tmpl.render(&RenderParams {
            messages: req.messages,
            tools: req.tools,
            add_generation_prompt: true,
            extra: &extra,
        })?;
        let ids = self.tok.encode(&text)?;
        Ok(Prompt { text, ids })
    }
    fn submit(
        &self,
        prompt_ids: Vec<u32>,
        sampling: SamplingParams,
    ) -> (RequestId, tokio::sync::mpsc::UnboundedReceiver<EngineEvent>) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let id = self.handle.submit(Submission { prompt_ids, sampling, tx });
        (id, rx)
    }
    fn abort(&self, id: RequestId) {
        self.handle.abort(id);
    }
    fn metrics(&self) -> Option<Arc<dinfer_core::metrics::Metrics>> {
        Some(self.handle.metrics())
    }
    fn new_output_stream(&self) -> Box<dyn OutputStream> {
        Box::new(Qwen3OutputStream {
            tok: self.tok.clone(),
            parser: OutputParser::new(),
            ids: Vec::new(),
            prefix_len: 0,
            emitted: String::new(),
        })
    }
}
