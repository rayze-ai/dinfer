//! Qwen3 的 `ChatBackend`：加载真权重 + tokenizer + chat 模板（`enable_thinking` 由请求的 `chat_template_kwargs` 或
//! 默认 true 决定），起引擎线程，提供渲染 / 提交 / 输出流（`parser::OutputParser`）。

use crate::config::Qwen3Config;
use crate::executor::Qwen3Executor;
use crate::model::Qwen3Model;
use crate::parser::OutputParser;
use anyhow::{Context, Result};
use dinfer_core::config::{CacheConfig, SchedulerConfig};
use dinfer_core::pd::KvTransport as _;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PdRoleOpt {
    #[default]
    Single,
    Prefill,
    Decode,
}
impl std::str::FromStr for PdRoleOpt {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "single" => Ok(Self::Single),
            "prefill" | "p" => Ok(Self::Prefill),
            "decode" | "d" => Ok(Self::Decode),
            o => Err(format!("role must be single|prefill|decode, got {o}")),
        }
    }
}

#[derive(Clone)]
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
    /// decode 行不占 token 预算（见 SchedulerConfig）。默认开。
    pub decode_exempt_from_budget: bool,
    /// KV host tier 容量（GiB pinned 内存）；0 = 关。
    pub kv_host_gib: f64,
    /// PD 分离角色（design §10.3）：single / prefill / decode。
    pub role: PdRoleOpt,
    /// prefill / decode 都监听这个端口收 KV（decode 用；prefill 侧预留给 v2 回传）。0 = 不监听。
    pub kv_port: u16,
    /// prefill 侧：对端 decode 的 `host:kv_port`。
    pub peer: Option<String>,
    /// decode 侧：等 KV 最长多少 ms，超时本地重算。
    pub pd_wait_ms: u64,
    /// PD：pinned 缓冲个数（每块 = max_model_len 的全层 KV；导出 / 导入在飞上限）。
    pub pd_pool_bufs: usize,
    /// PD KV 传输后端：tcp | shm | cuda-ipc（design §10.3 b；D 侧建区，P 侧跟随握手）。
    pub kv_transport: String,
    /// 共享槽区槽数；0 = 自动。
    pub kv_slots: usize,
    /// 纯 decode 步用 CUDA graph（启动时按 batch 桶捕获；vLLM `cudagraph_mode=FULL_DECODE_ONLY` 对齐）。
    pub enable_cuda_graph: bool,
    pub device: usize,
    /// TP 组大小（design §13）：用卡 `device .. device+tp`；1 = 单卡。
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
            enable_prefix_caching: true,
            decode_exempt_from_budget: true,
            kv_host_gib: 0.0,
            role: PdRoleOpt::Single,
            kv_port: 0,
            peer: None,
            pd_wait_ms: 2000,
            pd_pool_bufs: 8,
            kv_transport: "tcp".into(),
            kv_slots: 0,
            enable_cuda_graph: true,
            device: 0,
            tp: 1,
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
    pd_incoming: Option<Arc<dinfer_core::pd::IncomingKv>>,
    pd_wait: std::time::Duration,
    _engine: std::thread::JoinHandle<Result<()>>,
}

impl Qwen3Service {
    pub fn start(opt: ServeOptions, served_name: String) -> Result<Self> {
        let t_start = std::time::Instant::now();
        let model_dir = Path::new(&opt.model_dir);
        let tok_dir = opt.tokenizer_dir.as_deref().map(Path::new).unwrap_or(model_dir);
        // tokenizer（BPE 合并表 ~0.4 s）与 CUDA 初始化 / 权重加载无依赖 → 另起线程并行（启动 loop 第 5 轮）
        let tok_dir_owned = tok_dir.to_path_buf();
        let tok_thread = std::thread::spawn(move || -> Result<(HfTokenizer, ChatTemplate)> {
            let tok = HfTokenizer::from_dir(&tok_dir_owned)
                .with_context(|| format!("tokenizer in {}", tok_dir_owned.display()))?;
            let tmpl = ChatTemplate::from_dir(&tok_dir_owned)
                .with_context(|| format!("chat template in {}", tok_dir_owned.display()))?;
            Ok((tok, tmpl))
        });
        let defaults = read_generation_defaults(tok_dir);
        let cfg = Qwen3Config::from_json(&std::fs::read_to_string(model_dir.join("config.json"))?)?;
        let t_tok = t_start.elapsed(); // 此处只剩 config / 模板默认值读取；tokenizer 耗时见 "tokenizer thread join"
        // 算子实现选择（design §10.1）：目前只有 default；未知名字直接报错
        let ops_name = dinfer_layers::OpSet::name(&dinfer_layers::DefaultOps);
        let want = dinfer_kernels::opset::selected_ops_name();
        anyhow::ensure!(want == ops_name, "unknown OPS={want:?}; available: {ops_name}");
        let tp_world = opt.tp.max(1);
        // TP + PD（design §13.7）：各 rank 分片导出 / 导入，P 与 D 同 tp。cuda-ipc 区是 rank 0 那张卡的设备内存，其它 rank 读不到。
        anyhow::ensure!(
            tp_world == 1 || opt.role == PdRoleOpt::Single || opt.kv_transport != "cuda-ipc",
            "--tp > 1 与 --kv-transport cuda-ipc 不兼容（其它 rank 无 P2P 读不到 rank 0 的设备区）；用 shm 或 tcp"
        );
        anyhow::ensure!(tp_world == 1 || opt.kv_host_gib == 0.0, "--tp > 1 与 host tier 暂不兼容");
        if tp_world > 1 {
            // 副流上没有 comm，TP 下关闭跨流 prefill/decode 并行（design §13）
            std::env::set_var("XSTREAM", "0");
        }
        let src = Arc::new({
            let t_open = std::time::Instant::now();
            let src = WeightSource::open_dir(model_dir)?;
            tracing::info!(open_ms = t_open.elapsed().as_millis() as u64, "load timing: safetensors open (mmap populate + header)");
            src
        });
        let max_tokens = if opt.decode_exempt_from_budget {
            (opt.max_num_batched_tokens + opt.max_num_seqs) as usize
        } else {
            opt.max_num_batched_tokens.max(opt.max_num_seqs) as usize
        };
        let max_blocks_per_seq = (opt.max_model_len as usize).div_ceil(opt.block_size as usize);
        let comm_id = if tp_world > 1 { Some(dinfer_kernels::comm::CommId::new()?) } else { None };
        // 每个 rank 一张卡：Device + 模型分片 + executor（+ NCCL comm）。rank 0 在本线程，其余各起一线程并行装载。
        let build_rank = {
            let cfg = cfg.clone();
            let opt_dev = opt.device;
            let (num_blocks_opt, block_size, gmu, graphs) =
                (opt.num_blocks.map(|n| n as usize), opt.block_size as usize, opt.gpu_memory_utilization, opt.enable_cuda_graph);
            let max_num_seqs = opt.max_num_seqs as usize;
            let pd_role = opt.role != PdRoleOpt::Single;
            move |rank: usize, src: Arc<WeightSource>| -> Result<(Qwen3Executor, usize, std::time::Duration)> {
                let t0 = std::time::Instant::now();
                let dev = Device::new(opt_dev + rank)?;
                let s = dev.default_stream();
                let t_cuda = t0.elapsed();
                let model = Qwen3Model::load_with_budget_tp(
                    &s,
                    &src,
                    cfg.clone(),
                    max_tokens,
                    max_num_seqs, // 每请求只取最后一行 logits（无投机）
                    num_blocks_opt,
                    block_size,
                    gmu,
                    512 << 20, // cuBLAS workspace / 临时张量余量
                    (rank, tp_world),
                )?;
                let nb = model.num_blocks();
                s.synchronize()?;
                let mut exec = Qwen3Executor::with_graphs_and_blocks(s.clone(), model, graphs, max_blocks_per_seq)?;
                if let Some(id) = comm_id {
                    exec.init_tp(rank, tp_world, id)?;
                }
                if pd_role && rank > 0 {
                    // 非 0 rank：只要 pack/unpack staging + copy stream，不要自己的 pinned 池（用 rank 0 的缓冲 / 槽按偏移读写）
                    exec.enable_pd(0, tp_world)?;
                }
                Ok((exec, nb, t_cuda))
            }
        };
        let other_ranks: Vec<_> = (1..tp_world)
            .map(|r| {
                let b = build_rank.clone();
                let src = src.clone();
                std::thread::Builder::new().name(format!("tp-load{r}")).spawn(move || b(r, src)).expect("spawn tp load")
            })
            .collect();
        let (exec, num_blocks, t_cuda_d) = build_rank(0, src.clone())?;
        let t_cuda = t_tok + t_cuda_d;
        let mut others = Vec::with_capacity(other_ranks.len());
        for jh in other_ranks {
            let (ex, nb, _) = jh.join().map_err(|_| anyhow::anyhow!("tp load thread panicked"))??;
            anyhow::ensure!(nb == num_blocks, "tp ranks disagree on num_blocks ({nb} vs {num_blocks}); pass --num-blocks");
            others.push(ex);
        }
        let num_blocks = num_blocks as u32;
        let need = (opt.max_model_len as usize).div_ceil(opt.block_size as usize);
        anyhow::ensure!(
            need as f64 <= num_blocks as f64 * 0.9,
            "KV pool too small: max_model_len {} needs {} blocks, pool has {} (need ≤ 90%); lower --max-model-len or raise --num-blocks / --gpu-memory-utilization",
            opt.max_model_len,
            need,
            num_blocks
        );
        let t_model = t_start.elapsed();
        let num_graphs = exec.num_graphs();
        let mut exec = exec;
        let host_slots = if opt.kv_host_gib > 0.0 { exec.enable_host_tier(opt.kv_host_gib)? } else { 0 };
        // PD（design §10.3）：P/D 都需要 pack/unpack staging + pinned 中转槽；传输后端 v1 TCP（NIXL 后端换 transport 即可）。
        let mut pd_incoming_keep: Option<Arc<dinfer_core::pd::IncomingKv>> = None;
        let pd_state = if opt.role != PdRoleOpt::Single {
            let (base, len) = exec.enable_pd(opt.pd_pool_bufs, tp_world)?;
            let pool = dinfer_core::runner::ModelExecutor::kv_host_pool(&exec).context("pd pool")?;
            // tag 只含 wire 布局相关量（层数 / block_size / 全局 kv 头数 / head_dim / dtype）；payload 是 canonical 按头布局，
            // 与 tp 无关 ⇒ P / D 可异构 tp（design §13.7）。cfg 是全局配置（TP 切分在 load_with_budget_tp 内部做）。
            let model_tag = format!(
                "qwen3:L{}:bs{}:kvh{}:hd{}:bf16",
                cfg.num_hidden_layers, opt.block_size, cfg.num_key_value_heads, cfg.head_dim
            );
            let incoming = Arc::new(dinfer_core::pd::IncomingKv::default());
            pd_incoming_keep = Some(incoming.clone());
            // 共享槽区后端（design §10.3 b）：D 侧建区（shm / cuda-ipc），握手时发给 P；P 侧跟随对端的区类型，不看本地 kv_transport。
            let region_kind = match opt.kv_transport.as_str() {
                "tcp" => None,
                k => Some(dinfer_kernels::RegionKind::parse(k).with_context(|| format!("unknown --kv-transport {k} (tcp | shm | cuda-ipc)"))?),
            };
            let local_region = match (opt.role, region_kind) {
                (PdRoleOpt::Decode, Some(kind)) => {
                    let slots = if opt.kv_slots > 0 { opt.kv_slots } else if kind == dinfer_kernels::RegionKind::Shm { 4 } else { 2 };
                    Some(Arc::new(exec.enable_pd_region(kind, slots, tp_world)?))
                }
                _ => None,
            };
            if opt.kv_port > 0 {
                let port = dinfer_core::pd::spawn_receiver_shared(&format!("0.0.0.0:{}", opt.kv_port), model_tag.clone(), incoming.clone(), pool.clone(), local_region)?;
                tracing::info!(port, role = ?opt.role, transport = %opt.kv_transport, "pd: kv receiver listening");
            }
            let transport: Option<Arc<dyn dinfer_core::pd::KvTransport>> = match (opt.role, &opt.peer) {
                (PdRoleOpt::Prefill, Some(peer)) => {
                    let t: Arc<dyn dinfer_core::pd::KvTransport> = if region_kind.is_some() {
                        dinfer_core::pd::SharedTransport::new(peer.clone(), std::time::Duration::from_secs(2))
                    } else {
                        Arc::new(dinfer_core::pd::TcpTransport { addr: peer.clone(), connect_timeout: std::time::Duration::from_secs(2) })
                    };
                    t.register(dinfer_core::pd::MemDesc { ptr: base, len })?;
                    tracing::info!(peer = %peer, transport = t.name(), "pd: prefill side will push KV to peer");
                    Some(t)
                }
                (PdRoleOpt::Prefill, None) => anyhow::bail!("--role prefill requires --peer host:kv_port"),
                _ => None,
            };
            Some(dinfer_core::engine::PdState::new(incoming, transport, model_tag, std::time::Duration::from_millis(opt.pd_wait_ms)))
        } else {
            None
        };
        enum ExecKind {
            Single(Qwen3Executor),
            Tp(crate::tp::TpExecutor),
        }
        let exec_kind = if others.is_empty() {
            ExecKind::Single(exec)
        } else {
            let tpx = crate::tp::TpExecutor::new(exec, others);
            if pd_state.is_some() {
                tpx.share_region()?; // D 侧：rank 0 建的 shm 区交给其它 rank（tcp 下为 None，no-op）
            }
            ExecKind::Tp(tpx)
        };
        let t_join = std::time::Instant::now();
        let (tok, tmpl) = tok_thread.join().map_err(|_| anyhow::anyhow!("tokenizer thread panicked"))??;
        let tok = Arc::new(tok);
        anyhow::ensure!(
            cfg.vocab_size as usize >= tok.vocab_size(),
            "model vocab {} < tokenizer {}",
            cfg.vocab_size,
            tok.vocab_size()
        );
        tracing::info!(
            wait_ms = t_join.elapsed().as_millis() as u64,
            "load timing: tokenizer thread join (0 = 完全被 CUDA/权重加载掩盖)"
        );
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
        let mut sched = Scheduler::new(params, groups, num_blocks);
        if host_slots > 0 {
            sched.enable_host_tier(host_slots, Box::new(dinfer_core::kv::host_tier::FullAttentionPolicy));
        }
        let bs = opt.block_size as usize;
        let (handle, jh) = match (exec_kind, pd_state) {
            (ExecKind::Tp(tpx), None) => engine::spawn(sched, Runner::new(tpx, bs)),
            (ExecKind::Tp(tpx), Some(st)) => engine::spawn_with_pd(sched, Runner::new(tpx, bs), st),
            (ExecKind::Single(ex), Some(st)) => engine::spawn_with_pd(sched, Runner::new(ex, bs), st),
            (ExecKind::Single(ex), None) => engine::spawn(sched, Runner::new(ex, bs)),
        };
        tracing::info!(
            model = %served_name,
            max_model_len = opt.max_model_len,
            num_blocks,
            kv_tokens = num_blocks * opt.block_size,
            prefix_caching = opt.enable_prefix_caching,
            cuda_graphs = num_graphs,
            "qwen3 engine started"
        );
        // 加载各阶段耗时（用户要求记录，看是否值得优化）：tokenizer+模板 / CUDA 初始化 / 权重+scratch+KV / executor+engine
        tracing::info!(
            tokenizer_ms = t_tok.as_millis() as u64,
            cuda_init_ms = (t_cuda - t_tok).as_millis() as u64,
            model_load_ms = (t_model - t_cuda).as_millis() as u64,
            engine_ms = (t_start.elapsed() - t_model).as_millis() as u64,
            total_ms = t_start.elapsed().as_millis() as u64,
            "startup timing"
        );
        Ok(Self {
            tok,
            tmpl,
            handle,
            name: served_name,
            max_model_len: opt.max_model_len,
            defaults,
            pd_incoming: pd_incoming_keep,
            pd_wait: std::time::Duration::from_millis(opt.pd_wait_ms),
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
        pd: Option<dinfer_core::engine::PdSubmit>,
    ) -> (RequestId, tokio::sync::mpsc::UnboundedReceiver<EngineEvent>) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let id = self.handle.submit(Submission { prompt_ids, sampling, tx, pd });
        (id, rx)
    }
    fn abort(&self, id: RequestId) {
        self.handle.abort(id);
    }
    fn pd_wait_for(&self, remote_id: RequestId, timeout: std::time::Duration) -> bool {
        let Some(inc) = self.pd_incoming.as_ref() else { return false };
        let timeout = timeout.min(self.pd_wait);
        let t0 = std::time::Instant::now();
        while !inc.contains(remote_id) {
            if t0.elapsed() >= timeout {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        true
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
