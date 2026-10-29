//! Kaon-V3 的 `ChatBackend`：加载权重 + tokenizer + gemma 系 chat 模板，起引擎线程，提供渲染 / 提交 / 输出流。
//!
//! **第一版**：单卡、greedy、无 TP / PD / CUDA graph / 投机 / prefix caching。KV 由模型内部逐层按 head_dim
//! 分配（`alloc_kv`），scheduler 只管逻辑块号（单 group，FullAttention 规格仅作块数元数据，num_blocks 显式给定）。
//! Gemma4 输出无 `<think>` / `<tool_call>` 标签，OutputStream 直接把增量文本当 content 下发。

use crate::config::KaonConfig;
use crate::executor::KaonExecutor;
use crate::model::{check_weights, KaonModel};
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
    pub num_blocks: u32,
    pub block_size: u32,
    pub enable_chunked_prefill: bool,
    pub enable_prefix_caching: bool,
    pub decode_exempt_from_budget: bool,
    /// 单条 prefill 每步最多 token（0 = 不限）；vLLM `long_prefill_token_threshold`
    pub long_prefill_token_threshold: u32,
    /// 一步内全部 prefill 行合计的新 token 上限（0 = 不限；B47）。
    pub prefill_tokens_per_step: u32,
    pub enable_cuda_graph: bool,
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
            num_blocks: 2048,
            block_size: 32,
            enable_chunked_prefill: true,
            enable_prefix_caching: true,
            decode_exempt_from_budget: true,
            long_prefill_token_threshold: 0,
            prefill_tokens_per_step: 0,
            enable_cuda_graph: true,
            device: 0,
        }
    }
}

pub struct KaonService {
    tok: Arc<HfTokenizer>,
    tmpl: ChatTemplate,
    handle: EngineHandle,
    name: String,
    max_model_len: u32,
    defaults: SamplingParams,
    _engine: std::thread::JoinHandle<Result<()>>,
}

impl KaonService {
    pub fn start(opt: ServeOptions, served_name: String) -> Result<Self> {
        let mut t_startup = dinfer_layers::StartupTimer::start();
        let model_dir = Path::new(&opt.model_dir);
        let tok_dir = opt.tokenizer_dir.as_deref().map(Path::new).unwrap_or(model_dir);
        let tok = Arc::new(
            HfTokenizer::from_dir(tok_dir).with_context(|| format!("tokenizer in {}", tok_dir.display()))?,
        );
        let tmpl = ChatTemplate::from_dir(tok_dir)
            .with_context(|| format!("chat template in {}", tok_dir.display()))?;
        let defaults = read_generation_defaults(tok_dir);

        let cfg = KaonConfig::from_json(&std::fs::read_to_string(model_dir.join("config.json"))?)?;
        let q_path = model_dir.join("hf_quant_config.json");
        let cfg = if q_path.exists() {
            cfg.with_quant_json(&std::fs::read_to_string(&q_path)?)?
        } else {
            cfg
        };
        anyhow::ensure!(
            cfg.vocab_size >= tok.vocab_size(),
            "model vocab {} < tokenizer vocab {}",
            cfg.vocab_size,
            tok.vocab_size()
        );

        let dev = Device::new(opt.device)?;
        let s = dev.default_stream();
        let gemm = Gemm::new(&s)?;
        let src = WeightSource::open_dir(model_dir)?;
        check_weights(&src, &cfg)?;

        let bs = opt.block_size as usize;
        let num_blocks = opt.num_blocks as usize;
        // 本步最多 token 数（decode 每请求 1 行）。decode 豁免预算时，一步可含满额 prefill chunk
        // (max_num_batched_tokens) 再叠所有 running decode 行(≤max_num_seqs)，buffer 要容得下，否则
        // 调度器放行 2048+decode 行会触发 executor 的 max_tokens 上限（engine step failed: batch > max_tokens）。
        let max_tokens = if opt.decode_exempt_from_budget {
            (opt.max_num_batched_tokens + opt.max_num_seqs) as usize
        } else {
            opt.max_num_batched_tokens.max(opt.max_num_seqs) as usize
        };
        let max_rows = opt.max_num_seqs as usize;

        let mut model = KaonModel::load(&s, &src, cfg.clone())?;
        model.alloc_kv(&s, num_blocks, bs)?;
        s.synchronize()?;

        // KV 池要能放下一条 max_model_len 的序列。
        let max_blocks_per_seq = (opt.max_model_len as usize).div_ceil(bs);
        anyhow::ensure!(
            max_blocks_per_seq as f64 <= num_blocks as f64 * 0.95,
            "KV pool too small: max_model_len {} needs {} blocks, pool has {}; raise --num-blocks",
            opt.max_model_len,
            max_blocks_per_seq,
            num_blocks
        );

        let exec = KaonExecutor::new(s.clone(), gemm, model, max_tokens, max_rows, max_blocks_per_seq, opt.enable_cuda_graph)?;
        t_startup.model_loaded();
        let runner = Runner::new(exec, bs);

        let params = engine::scheduler_params(
            SchedulerConfig {
                max_num_seqs: opt.max_num_seqs,
                max_num_batched_tokens: opt.max_num_batched_tokens,
                enable_chunked_prefill: opt.enable_chunked_prefill,
                long_prefill_token_threshold: opt.long_prefill_token_threshold,
                prefill_tokens_per_step: opt.prefill_tokens_per_step,
                decode_exempt_from_budget: opt.decode_exempt_from_budget,
            },
            CacheConfig { block_size: opt.block_size, enable_prefix_caching: opt.enable_prefix_caching, ..Default::default() },
            opt.max_model_len,
            tok.eos_ids().to_vec(),
            0,
        );
        // 单 group：所有层共用 block_size / num_blocks（每层物理 KV 由模型内部按 head_dim 分配）。
        // spec 仅作块数元数据（num_blocks 显式给定，不走显存自动算），用 full 层形状代表。
        let groups = vec![GroupInfo {
            spec: KvCacheSpec::FullAttention {
                block_size: opt.block_size,
                kv_heads: cfg.nkv_full as u32,
                head_size: cfg.head_dim_full as u32,
                dtype_bytes: 2,
            },
            block_size: opt.block_size,
        }];
        let sched = Scheduler::new(params, groups, opt.num_blocks);
        let (handle, jh) = engine::spawn(sched, runner);
        tracing::info!(
            model = %served_name,
            max_model_len = opt.max_model_len,
            num_blocks,
            kv_tokens = num_blocks * bs,
            "kaon3 engine started (single-stream, greedy, no graph/spec/PD)"
        );
        t_startup.done(0.0, 0.0);
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

/// Gemma4 输出无思考 / tool_call 标签：增量 detokenize → 全部进 content。
struct KaonOutputStream {
    tok: Arc<HfTokenizer>,
    ids: Vec<u32>,
    prefix_len: usize,
    emitted: String,
    skip_special: bool,
}

impl KaonOutputStream {
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

impl OutputStream for KaonOutputStream {
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
        OutDelta { content: text.to_string(), ..Default::default() }
    }
    fn finish(&mut self) -> Result<OutDelta> {
        Ok(OutDelta::default())
    }
    fn token_text(&self, id: u32) -> Option<String> {
        self.tok.token_text(id)
    }
    fn set_skip_special_tokens(&mut self, skip: bool) {
        self.skip_special = skip;
    }
}

impl ChatBackend for KaonService {
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
        let extra = req.extra.cloned().unwrap_or_default();
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
        Box::new(KaonOutputStream {
            tok: self.tok.clone(),
            ids: Vec::new(),
            prefix_len: 0,
            emitted: String::new(),
            skip_special: true,
        })
    }
}
