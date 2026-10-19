//! GLM-5.3 的 `ChatBackend` 实现：加载模型 + tokenizer + chat 模板，起引擎线程，提供渲染 / 提交 / 输出流。
//! server crate 只认 `dinfer_server::backend::ChatBackend`，这里是模型侧的唯一接入点。

use crate::chat::{render, ChatTemplate, RenderParams};
use crate::config::GlmMoeDsaConfig;
use crate::executor::Glm53Executor;
use crate::model::Glm53Model;
use crate::mtp::MtpLayer;
use crate::parser::OutputParser;
use crate::tokenizer::GlmTokenizer;
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
use std::path::Path;
use std::sync::Arc;

pub struct ServeOptions {
    /// 模型目录（config.json + safetensors）。
    pub model_dir: String,
    /// tokenizer / chat 模板目录；None = 同 model_dir。
    pub tokenizer_dir: Option<String>,
    pub max_model_len: u32,
    pub max_num_seqs: u32,
    pub max_num_batched_tokens: u32,
    pub num_blocks: u32,
    pub block_size: u32,
    pub enable_chunked_prefill: bool,
    pub device: usize,
    /// MTP 投机解码（nextn=1）；需要 config `num_nextn_predict_layers == 1` 与 `model.layers.{N}.*` 权重。
    pub mtp: bool,
}

impl Default for ServeOptions {
    fn default() -> Self {
        Self {
            model_dir: String::new(),
            tokenizer_dir: None,
            max_model_len: 4096,
            max_num_seqs: 16,
            max_num_batched_tokens: 2048,
            num_blocks: 512,
            block_size: 16,
            enable_chunked_prefill: true,
            device: 0,
            mtp: false,
        }
    }
}

pub struct Glm53Service {
    tok: Arc<GlmTokenizer>,
    tmpl: ChatTemplate,
    handle: EngineHandle,
    name: String,
    max_model_len: u32,
    defaults: SamplingParams,
    _engine: std::thread::JoinHandle<Result<()>>,
}

impl Glm53Service {
    pub fn start(opt: ServeOptions, served_name: String) -> Result<Self> {
        let model_dir = Path::new(&opt.model_dir);
        let tok_dir = opt.tokenizer_dir.as_deref().map(Path::new).unwrap_or(model_dir);
        let tok = Arc::new(
            GlmTokenizer::from_dir(tok_dir).with_context(|| format!("tokenizer in {}", tok_dir.display()))?,
        );
        let tmpl = ChatTemplate::from_dir(tok_dir)
            .with_context(|| format!("chat template in {}", tok_dir.display()))?;
        let defaults = read_generation_defaults(tok_dir);

        let cfg = GlmMoeDsaConfig::from_json(&std::fs::read_to_string(model_dir.join("config.json"))?)?;
        anyhow::ensure!(
            cfg.vocab_size as usize >= tok.vocab_size(),
            "model vocab {} < tokenizer vocab {}",
            cfg.vocab_size,
            tok.vocab_size()
        );
        let dev = Device::new(opt.device)?;
        let s = dev.default_stream();
        let gemm = Gemm::new(&s)?;
        let mut src = WeightSource::open_dir(model_dir)?;
        if let Some((bn, bk)) = cfg.fp8_block_size() {
            src = src.with_quant_block(bn, bk);
        }
        let max_tokens = opt.max_num_batched_tokens.max(opt.max_num_seqs) as usize;
        let model = Glm53Model::load(
            &s,
            &src,
            cfg.clone(),
            max_tokens,
            opt.num_blocks as usize,
            opt.block_size as usize,
        )?;
        let mtp = if opt.mtp {
            anyhow::ensure!(
                cfg.num_nextn_predict_layers == 1,
                "--mtp: model has no MTP layer (num_nextn_predict_layers != 1)"
            );
            Some(MtpLayer::load(
                &s,
                &src,
                &cfg,
                max_tokens,
                opt.num_blocks as usize,
                opt.block_size as usize,
            )?)
        } else {
            None
        };
        let exec = Glm53Executor::new(s.clone(), gemm, model, mtp)?;
        let runner = Runner::new(exec, opt.block_size as usize);

        let params = engine::scheduler_params(
            SchedulerConfig {
                max_num_seqs: opt.max_num_seqs,
                max_num_batched_tokens: opt.max_num_batched_tokens,
                enable_chunked_prefill: opt.enable_chunked_prefill,
                long_prefill_token_threshold: 0,
                decode_exempt_from_budget: false,
            },
            CacheConfig { block_size: opt.block_size, enable_prefix_caching: false, ..Default::default() },
            opt.max_model_len,
            tok.eos_ids().to_vec(),
            if opt.mtp { 1 } else { 0 },
        );
        let entry = cfg.kv_lora_rank + cfg.qk_rope_head_dim;
        let groups = vec![GroupInfo {
            spec: KvCacheSpec::Mla { block_size: opt.block_size, head_size: entry, dtype_bytes: 2 },
            block_size: opt.block_size,
        }];
        let sched = Scheduler::new(params, groups, opt.num_blocks);
        let (handle, jh) = engine::spawn(sched, runner);
        tracing::info!(model = %served_name, max_model_len = opt.max_model_len, num_blocks = opt.num_blocks, mtp = opt.mtp, "glm53 engine started");
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

/// generation_config.json 的 temperature / top_p / top_k 作为默认采样。
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

struct GlmOutputStream {
    tok: Arc<GlmTokenizer>,
    parser: OutputParser,
    /// 增量 decode 的内部状态：`DecodeStream` 借用 tokenizer，这里用 ids 累积 + 前缀 decode 的等价实现以避免自引用。
    ids: Vec<u32>,
    prefix_len: usize,
    emitted: String,
}

impl GlmOutputStream {
    /// 等价 `tokenizers::DecodeStream`：对累计 ids 做整体 decode，吐出相对上次的新增文本；若尾部是 U+FFFD（半字符）则先不吐。
    fn decode_new(&mut self) -> Result<String> {
        let text = self.tok.decode(&self.ids[self.prefix_len..])?;
        if text.ends_with('\u{fffd}') {
            return Ok(String::new());
        }
        let new = text[self.emitted.len().min(text.len())..].to_string();
        if !text.starts_with(&self.emitted) {
            // 极少数情况下前缀改写（规范化）：全量重发差异不可表达，退化为把当前段整体当新增
            self.emitted = text.clone();
            return Ok(text);
        }
        self.emitted = text;
        // 段落足够长就截断，防止 O(n^2)：在吐出完整文本后重置窗口
        if self.ids.len() - self.prefix_len > 64 {
            self.prefix_len = self.ids.len();
            self.emitted.clear();
        }
        Ok(new)
    }
}

impl OutputStream for GlmOutputStream {
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

impl ChatBackend for Glm53Service {
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
        let p = RenderParams {
            messages: req.messages,
            tools: req.tools,
            reasoning_effort: req.reasoning_effort,
            clear_thinking: req.clear_thinking,
            add_generation_prompt: req.add_generation_prompt,
        };
        let text = render(&self.tmpl, &p)?;
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
    fn new_output_stream(&self) -> Box<dyn OutputStream> {
        Box::new(GlmOutputStream {
            tok: self.tok.clone(),
            parser: OutputParser::new(),
            ids: Vec::new(),
            prefix_len: 0,
            emitted: String::new(),
        })
    }
}
