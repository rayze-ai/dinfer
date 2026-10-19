//! `Qwen3Model` 的 `ModelExecutor`：上传 BatchPlan → forward → 采样 → D2H。
//! 投机解码（A4，n-gram）：draft 由 runner 在 host 侧提案（`drafts` 这里永远为空）；带 draft 的请求本步输入
//! `[base, draft…]`，logits / 采样取 `1 + 草稿数` 行（eager 路径），`verify_rows` 在 host 逐行比对：行 j 的 argmax 等于
//! draft[j] 则接受，首个不匹配行（或全接受时最后一行）的采样值为 bonus；只对 greedy 请求提案所以逐位相等即 vLLM 口径。
//!
//! 采样（Q2-a）与 vLLM `Sampler.forward` 同序：penalties（prompt+output 历史）→ temperature → min_p → top_k/top_p →
//! Philox 多项式采样（greedy 行 argmax）；logprobs 用 forward 的 **原始** logits（惩罚 / 温度之前）做 log_softmax。
//!
//! Q3-c：
//! - 所有每 step 输入（ids / positions / slot_mapping / logits_rows / cu_seqlens / seq_lens / block_table / 采样列）
//!   走 **固定地址** 设备缓冲（按上限一次分配），每 step 各一次异步 H2D，不再 `Tensor::from_vec`；
//! - **纯 decode 步**（`BatchMode::Decode`，无惩罚、无 logprobs）按 batch 桶 {1,2,4,8,16,32,64,…} 捕获为 CUDA graph：
//!   图内 = 整个 forward + temperature/min_p/topk-topp/sample 四个采样 kernel；batch 不满桶用 padding 行
//!   （slot=-1 不写 cache、seq_len=1、温度 0）；启动时对每个桶捕获一次（对齐 vLLM `cudagraph_mode=FULL_DECODE_ONLY`）。
//!   其他情况（prefill / 混批 / 有惩罚 / 要 logprobs）走 eager，同一套固定缓冲。
//! - executor 自己建一条非默认流（capture 要求）并在其上建 cuBLASLt 句柄；cudarc 的自动事件跟踪关闭
//!   （见 `Device::disable_event_tracking`）：全程单流顺序执行。

use crate::model::{AttnMeta, Forward, Qwen3Model, ModelScratch};
use anyhow::{ensure, Context, Result};
use dinfer_core::runner::{BatchMode, BatchPlan, ModelExecutor, PairLogprobs};
use dinfer_core::sampler::{SampleBatch, TokenLogprobs};
use dinfer_kernels as k;
use dinfer_layers::exec::{async_sched_enabled, decode_capture_plan, graph_capture_eager, AsyncResultRing, DecodeGraphs, PlanBuffers, Ticket};
use dinfer_layers::{Gemm, Stream, Tensor, Event, TimingEvent, PinnedUploader, HostArena, PinnedPool, PinnedBuf};

/// 固定地址的每 step 输入缓冲。
struct StepBuffers {
    /// plan 列（ids / pos / slot_mapping / rows / cu / seq / block_table + 异步 gather）：公共件 `PlanBuffers`。
    plan: PlanBuffers,
    // 采样列（[max_rows]）
    temperature: Tensor<f32>,
    temp01: Tensor<f32>, // sample_f32 用：随机行 1.0 / greedy 行 0
    top_k: Tensor<i32>,
    top_p: Tensor<f32>,
    min_p: Tensor<f32>,
    rep: Tensor<f32>,
    freq: Tensor<f32>,
    pres: Tensor<f32>,
    seeds: Tensor<u64>,
    offsets: Tensor<u64>,
    hist_offsets: Tensor<i32>,    // [max_rows + 1]
    hist_prompt_len: Tensor<i32>, // [max_rows]
    hist_ids: Tensor<i32>,        // 可增长（惩罚历史，eager 路径）
    chosen: Tensor<i32>,          // [max_rows]
    /// 异步调度：本步每 token 行从上一步 chosen 取 id 的行号（-1 = 用 ids 里的值）；上一步 chosen 的副本。
    lp_ids: Tensor<i32>,          // [max_rows * 33]
    lp_val: Tensor<f32>,
    lp_rank: Tensor<i32>,
    /// 受限解码 / logit_bias（docs/loops/guided-decoding.md）：图内常驻的 `logits_mask_f32` 读它们。
    /// `mask_row[b] < 0` 无 mask；`mask_bits` 每行 `mask_words` 个 u32；bias 为 CSR（容量 max_rows × 300）。
    mask_row: Tensor<i32>,    // [max_rows]
    mask_bits: Tensor<u32>,   // [max_rows * mask_words]
    mask_words: usize,
    bias_offsets: Tensor<i32>, // [max_rows + 1]
    bias_ids: Tensor<i32>,     // [max_rows * 300]
    bias_vals: Tensor<f32>,
    /// 上一步是否写过非空 mask / bias：true 时下一步若无约束要把 mask_row / bias_offsets 清回 -1 / 0（图内 kernel 常读）；
    /// 连续无约束的步不发任何额外 H2D。
    mask_dirty: bool,
    /// prompt_logprobs：本步要再算 logits 的 hidden 行号与目标 token（[max_tokens]）。
    plp_rows: Tensor<i32>,
    plp_targets: Tensor<i32>,
    /// 每 step 小输入的打包上传器（pinned 一次 H2D + D2D 散开）。
    up: PinnedUploader,
}

impl StepBuffers {
    fn alloc(s: &Stream, mt: usize, rows: usize, max_blocks: usize, vocab: usize) -> Result<Self> {
        let z32 = |n: usize| Tensor::<i32>::zeros(s, &[n.max(1)]);
        let zf = |n: usize| Tensor::<f32>::zeros(s, &[n.max(1)]);
        let mask_words = vocab.div_ceil(32);
        Ok(StepBuffers {
            plan: PlanBuffers::alloc(s, mt, rows, max_blocks)?,
            temperature: zf(rows)?,
            temp01: zf(rows)?,
            top_k: z32(rows)?,
            top_p: zf(rows)?,
            min_p: zf(rows)?,
            rep: zf(rows)?,
            freq: zf(rows)?,
            pres: zf(rows)?,
            seeds: Tensor::<u64>::zeros(s, &[rows])?,
            offsets: Tensor::<u64>::zeros(s, &[rows])?,
            hist_offsets: z32(rows + 1)?,
            hist_prompt_len: z32(rows)?,
            hist_ids: z32(1024)?,
            chosen: z32(rows)?,
            lp_ids: z32(rows * 33)?,
            lp_val: zf(rows * 33)?,
            lp_rank: z32(rows)?,
            mask_row: Tensor::<i32>::from_vec(s, vec![-1; rows.max(1)], &[rows.max(1)])?,
            mask_bits: Tensor::<u32>::zeros(s, &[(rows * mask_words).max(1)])?,
            mask_words,
            bias_offsets: z32(rows + 1)?,
            bias_ids: z32(rows * 300)?,
            bias_vals: zf(rows * 300)?,
            mask_dirty: false,
            plp_rows: z32(mt)?,
            plp_targets: z32(mt)?,
            up: PinnedUploader::new(s, (mt * 16 + rows * (10 * 4 + 2 * 8) + rows * max_blocks * 4 + (1 << 20)).max(1 << 20))?,
        })
    }
}

/// 跨流并行的第二套执行资源（副流 + cuBLASLt + 输入缓冲 + logits 工作区 + 模型激活 scratch）。
struct AltSet {
    stream: Stream,
    gemm: Gemm,
    buf: StepBuffers,
    logits: Tensor<f32>,
    work: Tensor<f32>,
    raw: Tensor<f32>,
    scratch: ModelScratch,
}

pub struct Qwen3Executor {
    pub stream: Stream,
    pub gemm: Gemm,
    pub model: Qwen3Model,
    logits: Tensor<f32>,
    /// 采样工作区 [max_rows, vocab]（原地改）。
    work: Tensor<f32>,
    /// logprobs 用的原始 logits 副本（eager 路径）。
    raw: Tensor<f32>,
    buf: StepBuffers,
    graphs: DecodeGraphs,
    last_logprobs: Vec<Vec<TokenLogprobs>>,
    /// prompt_logprobs（本步）：(采样行号, hidden 行号, 目标 token, prompt 位置)，upload_plan 收集、sample_eager 消费。
    plp_pending: Vec<(usize, i32, i32, u32)>,
    last_rows: usize,
    alt: Option<AltSet>,
    /// 当前 plan 是否是插队 prefill（采样后不写 prev_chosen）。enqueue 时从 plan 读入。
    skip_prev_chosen: bool,
    /// KV host tier（design §10.2）：pinned 竞技场 + copy stream；None = 关。
    host_tier: Option<HostTierStore>,
    /// PD 分离（design §10.3 v2）：异步导出 / 导入资源。`enable_pd` 建。
    pd: Option<PdStore>,
    /// 异步调度：pinned 槽环（公共件）；在飞步的计时 (t0, t_launch, t_wait) 与票据平行。
    async_ring: AsyncResultRing,
    async_inflight: std::collections::VecDeque<(Ticket, std::time::Instant, std::time::Instant, std::time::Instant)>,
    /// 当前主字段是否是副套（swap_alt 后为 true）；副套上不走图。
    alt_active: bool,
    pub pair_steps: u64,
    /// 主流每 step 末尾记录；副流下一步开工前等它（KV 一致）。
    main_done: Option<Event>,
    /// 统计：走图 / 走 eager 的 step 数。
    pub graph_steps: u64,
    pub eager_steps: u64,
    /// 每 step 分段累计（µs）：H2D 上传 / kernel 入队（launch，不含执行）/ 等待 GPU 并 D2H。每 1000 步打一行 info。
    pub t_upload_us: u64,
    pub t_launch_us: u64,
    pub t_wait_us: u64,
    /// GPU 忙时（主流上 launch 前 / 后两个事件的差，step 结束后读）累计 µs。
    pub t_gpu_ms: f32,
    pub t_upload_sampling_us: u64,
    pub t_graph_launch_us: u64,
    t_gpu_err: u64,
    ev_gpu: Option<(TimingEvent, TimingEvent)>,
    /// GPU_TIMING=1 才每步记两个带时间戳的事件（怀疑 cuEventRecord 本身有 host 开销，默认关）。
    gpu_timing: bool,
}

pub use dinfer_layers::exec::graph_buckets;

impl Qwen3Executor {
    /// `enable_graphs`：是否在构造时为每个 batch 桶捕获 decode 图。
    pub fn new(stream: Stream, gemm: Gemm, model: Qwen3Model) -> Result<Self> {
        let _ = gemm;
        Self::with_graphs(stream, model, true)
    }

    pub fn with_graphs(stream: Stream, model: Qwen3Model, enable_graphs: bool) -> Result<Self> {
        let n = model.kv.first().map_or(1, |kv| kv.num_blocks).max(1);
        Self::with_graphs_and_blocks(stream, model, enable_graphs, n)
    }

    /// `max_blocks_per_seq`：单请求 block_table 的最大长度（= max_model_len / block_size）。它决定 StepBuffers
    /// 里 block_table 的行跨度与每步上传量；之前误用 KV 池总块数（4090 上 9972 → 每步 1.3 MB 零填充 + H2D，
    /// 占 decode 步 host 时间的大头，Q3 第 53 轮）。
    pub fn with_graphs_and_blocks(
        stream: Stream,
        model: Qwen3Model,
        enable_graphs: bool,
        max_blocks_per_seq: usize,
    ) -> Result<Self> {
        // capture 要求非默认流；cuBLASLt 句柄绑流，所以在新流上重建。权重 / scratch 在旧流上分配完并已同步。
        stream.synchronize()?;
        let dev = stream.device();
        dev.disable_event_tracking();
        let s = dev.new_stream()?;
        let gemm = Gemm::new(&s).context("cublasLt on executor stream")?;
        let v = model.cfg.vocab_size as usize;
        let rows = model.max_logits_rows;
        let mt = model.max_tokens;
        let max_blocks = max_blocks_per_seq.max(1);
        let buf = StepBuffers::alloc(&s, mt, rows, max_blocks, v)?;
        let async_ring = AsyncResultRing::new(&s, 2, rows)?;
        let graphs = DecodeGraphs::new(&s, enable_graphs, rows);
        let mut ex = Self {
            logits: Tensor::zeros(&s, &[rows, v])?,
            work: Tensor::zeros(&s, &[rows, v])?,
            raw: Tensor::zeros(&s, &[rows, v])?,
            stream: s,
            gemm,
            model,
            buf,
            graphs,
            last_logprobs: Vec::new(),
            plp_pending: Vec::new(),
            last_rows: 0,
            async_ring,
            async_inflight: std::collections::VecDeque::with_capacity(2),
            alt: None,
            skip_prev_chosen: false,
            host_tier: None,
            pd: None,
            alt_active: false,
            pair_steps: 0,
            main_done: None,
            graph_steps: 0,
            eager_steps: 0,
            t_upload_us: 0,
            t_launch_us: 0,
            t_wait_us: 0,
            t_gpu_ms: 0.0,
            t_upload_sampling_us: 0,
            t_graph_launch_us: 0,
            t_gpu_err: 0,
            ev_gpu: None,
            gpu_timing: std::env::var("GPU_TIMING").is_ok_and(|v| v == "1"),
        };
        // decode 图默认 **按桶懒捕获**（首次用到某个 batch 桶时捕获，~30 ms/桶），启动少 ~210 ms；
        // GRAPH_CAPTURE=eager 恢复启动时全部捕获（对拍 / 压测 TTFT 口径不想含首捕获时用）。
        if enable_graphs && graph_capture_eager() {
            ex.capture_decode_graphs()?;
        }
        // 跨流 prefill/decode 并行（Q3-j）：XSTREAM=0 关。副流一套资源在图捕获之后分配。
        if std::env::var("XSTREAM").map(|v| v != "0").unwrap_or(true) {
            let t0 = std::time::Instant::now();
            let s2 = dev.new_stream()?;
            let gemm2 = Gemm::new(&s2).context("cublasLt on alt stream")?;
            let buf2 = StepBuffers::alloc(&s2, mt, rows, max_blocks, v)?;
            let scratch = ex.model.alloc_scratch(&s2)?;
            ex.alt = Some(AltSet {
                logits: Tensor::zeros(&s2, &[rows, v])?,
                work: Tensor::zeros(&s2, &[rows, v])?,
                raw: Tensor::zeros(&s2, &[rows, v])?,
                stream: s2,
                gemm: gemm2,
                buf: buf2,
                scratch,
            });
            tracing::info!(alloc_ms = t0.elapsed().as_millis() as u64, "xstream: alt stream + scratch ready");
        }
        Ok(ex)
    }

    /// 上一 step 的 **原始** logits（fp32，[num_rows, vocab]）；对拍用。
    /// TP（design §13）：在 executor 自己的流上建 NCCL 通信器并挂到模型。必须在任何 CUDA graph 捕获之前调用
    /// （懒捕获默认满足；`GRAPH_CAPTURE=eager` 与 TP 不兼容），且所有 rank 并发调用（ncclCommInitRank 等全组到齐）。
    pub fn init_tp(&mut self, rank: usize, world: usize, id: k::comm::CommId) -> Result<()> {
        anyhow::ensure!(self.alt.is_none(), "tp: XSTREAM must be off (alt stream has no comm)");
        anyhow::ensure!(self.num_graphs() == 0, "tp: init_tp must run before graph capture");
        let comm = k::comm::Comm::init(&self.stream, rank, world, id)?;
        self.model.comm = Some(comm);
        self.model.tp = (rank, world);
        Ok(())
    }

    // ---- 自定义 one-shot all-reduce（design §13.6 T3）三步握手，由 `TpExecutor::setup_custom_ar` 在进程内编排 ----
    /// 第 1 步：建本 rank 的 cuda-ipc 区，返回公布信息（None = 本 rank 不参与 → 全组 disable）。
    pub fn custom_ar_begin(&mut self) -> Result<Option<k::comm::ArPeerInfo>> {
        anyhow::ensure!(self.num_graphs() == 0, "tp: custom_ar_begin must run before graph capture");
        match self.model.comm.as_mut() {
            Some(c) => c.custom_ar_begin(),
            None => Ok(None),
        }
    }
    /// 第 2 步：拿到全部 rank 的信息后打开 peer 区。false = 本 rank 失败 → 全组 disable。
    pub fn custom_ar_finish(&mut self, peers: &[k::comm::ArPeerInfo]) -> Result<bool> {
        match self.model.comm.as_mut() {
            Some(c) => c.custom_ar_finish(peers),
            None => Ok(false),
        }
    }
    pub fn custom_ar_disable(&mut self) {
        if let Some(c) = self.model.comm.as_mut() {
            c.custom_ar_disable();
        }
    }
    pub fn custom_ar_ready(&self) -> bool {
        self.model.comm.as_ref().is_some_and(|c| c.custom_ar_ready())
    }

    pub fn last_logits(&self, num_rows: usize) -> Result<Vec<f32>> {
        let v = self.model.cfg.vocab_size as usize;
        let all = self.logits.to_vec(&self.stream)?;
        Ok(all[..num_rows * v].to_vec())
    }

    pub fn num_graphs(&self) -> usize {
        self.graphs.len()
    }

    // ---------------------------------------------------------------- 上传

    /// 把 plan 的输入写进固定缓冲（batch 维可按 `pad_to` 补到桶大小：padding 行 slot=-1、seq_len=1、block 0）。
    /// 返回 (token 数含 padding, 请求数含 padding, logits 行数含 padding)。
    fn upload_plan(&mut self, plan: &BatchPlan, pad_to: usize) -> Result<(usize, usize, usize)> {
        let s = &self.stream;
        let n = plan.num_reqs();
        let (tt, nb, nrows) = self.buf.plan.upload(s, &mut self.buf.up, plan, pad_to)?;
        self.buf.up.flush(s)?;
        // prompt_logprobs：记下本步哪些 token 行是 prompt 位置（其 "下一个 token" 仍在 prompt 内），sample_eager 再算
        self.plp_pending.clear();
        if plan.sample.max_prompt_logprobs.is_some() {
            for i in 0..n {
                // sample 列按 logits 行（带 draft 的请求 1+K 行）；prefill 行无 draft → 该请求的行 = logits_row_start[i]
                let srow = plan.logits_row_start[i];
                if plan.sample.prompt_logprobs.get(srow).copied().flatten().is_none() {
                    continue;
                }
                let np = plan.sample.num_prompt[srow] as i32;
                let (r0, r1) = (plan.cu_seqlens[i] as usize, plan.cu_seqlens[i + 1] as usize);
                for r in r0..r1 {
                    let p = plan.positions[r];
                    if p + 1 >= np {
                        break; // 该行预测的是第一个生成 token（不属于 prompt_logprobs）
                    }
                    let target = if r + 1 < r1 {
                        if plan.gather_idx.get(r + 1).is_some_and(|&g| g >= 0) {
                            continue; // 在飞 token（设备侧 gather）：host 不知道目标，跳过
                        }
                        plan.input_ids[r + 1]
                    } else {
                        plan.next_known[i] // chunk 末行：下一个 token 在下一 chunk（runner 已给）
                    };
                    if target < 0 {
                        continue;
                    }
                    self.plp_pending.push((srow, r as i32, target, (p + 1) as u32));
                }
            }
        }
        Ok((tt, nb, nrows))
    }

    fn upload_sampling(&mut self, sb: &SampleBatch, nb: usize) -> Result<()> {
        let s = &self.stream;
        self.buf.up.begin();
        let n = sb.num_rows();
        let padf = |v: &Vec<f32>, fill: f32| {
            let mut x = v.clone();
            x.resize(nb, fill);
            x
        };
        self.buf.up.push(s, &padf(&sb.temperature, 0.0), &mut self.buf.temperature.data)?;
        let t01: Vec<f32> =
            (0..nb).map(|i| if i < n && sb.temperature[i] > 1e-5 { 1.0 } else { 0.0 }).collect();
        self.buf.up.push(s, &t01, &mut self.buf.temp01.data)?;
        let mut tk = sb.top_k.clone();
        tk.resize(nb, -1);
        self.buf.up.push(s, &tk, &mut self.buf.top_k.data)?;
        self.buf.up.push(s, &padf(&sb.top_p, 1.0), &mut self.buf.top_p.data)?;
        self.buf.up.push(s, &padf(&sb.min_p, 0.0), &mut self.buf.min_p.data)?;
        let mut seeds = sb.seed.clone();
        seeds.resize(nb, 0);
        self.buf.up.push(s, &seeds, &mut self.buf.seeds.data)?;
        let mut offs = sb.offset.clone();
        offs.resize(nb, 0);
        self.buf.up.push(s, &offs, &mut self.buf.offsets.data)?;
        if sb.any_penalty {
            self.buf.up.push(s, &padf(&sb.repetition_penalty, 1.0), &mut self.buf.rep.data)?;
            self.buf.up.push(s, &padf(&sb.frequency_penalty, 0.0), &mut self.buf.freq.data)?;
            self.buf.up.push(s, &padf(&sb.presence_penalty, 0.0), &mut self.buf.pres.data)?;
            let mut ho = sb.hist_offsets.clone();
            while ho.len() < nb + 1 {
                ho.push(*ho.last().unwrap());
            }
            self.buf.up.push(s, &ho, &mut self.buf.hist_offsets.data)?;
            let mut hp = sb.hist_prompt_len.clone();
            hp.resize(nb, 0);
            self.buf.up.push(s, &hp, &mut self.buf.hist_prompt_len.data)?;
            if self.buf.hist_ids.numel() < sb.hist_ids.len().max(1) {
                self.buf.hist_ids = Tensor::<i32>::zeros(s, &[sb.hist_ids.len().next_power_of_two()])?;
            }
            if !sb.hist_ids.is_empty() {
                s.htod_into(&sb.hist_ids, &mut self.buf.hist_ids.data)?;
            }
        }
        // 受限解码 mask / logit_bias：有约束才上传；上一步有、这一步没有 → 清回无约束态（图内 kernel 每步都读）
        if sb.any_mask {
            let mut mr = sb.mask_row.clone();
            mr.resize(nb, -1);
            self.buf.up.push(s, &mr, &mut self.buf.mask_row.data)?;
            let mut bo = sb.bias_offsets.clone();
            while bo.len() < nb + 1 {
                bo.push(*bo.last().unwrap());
            }
            self.buf.up.push(s, &bo, &mut self.buf.bias_offsets.data)?;
            if !sb.bias_ids.is_empty() {
                if self.buf.bias_ids.numel() < sb.bias_ids.len() {
                    // 超过 max_rows × 300 的容量（每请求 ≤ 300 已在 validate 限制，这里只防御）
                    self.buf.bias_ids = Tensor::<i32>::zeros(s, &[sb.bias_ids.len().next_power_of_two()])?;
                    self.buf.bias_vals = Tensor::<f32>::zeros(s, &[sb.bias_ids.len().next_power_of_two()])?;
                }
                self.buf.up.push(s, &sb.bias_ids, &mut self.buf.bias_ids.data)?;
                self.buf.up.push(s, &sb.bias_vals, &mut self.buf.bias_vals.data)?;
            }
            if !sb.mask_bits.is_empty() {
                ensure!(
                    sb.mask_words == self.buf.mask_words,
                    "grammar mask words {} != executor {} (tokenizer vocab vs model vocab)",
                    sb.mask_words,
                    self.buf.mask_words
                );
                ensure!(sb.mask_bits.len() <= self.buf.mask_bits.numel(), "grammar mask rows > max_rows");
                s.htod_into(&sb.mask_bits, &mut self.buf.mask_bits.data)?;
            }
            self.buf.mask_dirty = true;
        } else if self.buf.mask_dirty {
            let cap = self.buf.mask_row.numel();
            self.buf.up.push(s, &vec![-1i32; cap], &mut self.buf.mask_row.data)?;
            let cap = self.buf.bias_offsets.numel();
            self.buf.up.push(s, &vec![0i32; cap], &mut self.buf.bias_offsets.data)?;
            self.buf.mask_dirty = false;
        }
        self.buf.up.flush(s)?;
        Ok(())
    }

    // ---------------------------------------------------------------- 设备侧流水（可被 capture）

    /// forward（固定缓冲，显式 t / rows / batch）。
    fn forward_device(&mut self, mode: BatchMode, t: usize, nb: usize, rows: usize, max_seqlen: usize) -> Result<()> {
        // 异步调度：ids[i] = prev_chosen[gather_idx[i]]（gather_idx<0 的行不动）。进图；非异步步全 -1 等于空操作。
        self.buf.plan.gather_inflight(&self.stream, t)?;
        let meta = AttnMeta {
            mode,
            cu_seqlens: &self.buf.plan.cu,
            seq_lens: &self.buf.plan.seq,
            block_table: &self.buf.plan.bt,
            num_reqs: nb,
            max_blocks: self.buf.plan.max_blocks,
            max_seqlen,
        };
        let f = Forward {
            input_ids: &self.buf.plan.ids,
            positions: &self.buf.plan.pos,
            slot_mapping: &self.buf.plan.slots,
            meta,
            logits_rows: &self.buf.plan.rows,
            t,
            rows,
        };
        self.model.forward(&self.stream, &self.gemm, &f, &mut self.logits)
    }

    /// 采样流水（图内版本）：copy → temperature → min_p → topk/topp → sample_f32。全部按 nb 行、固定地址。
    /// 对 greedy 行：温度 0 → temperature kernel 跳过、mask 用中性值跳过、sample 内 argmax；与快路径 argmax 等价。
    fn sample_device_fixed(&mut self, nb: usize) -> Result<()> {
        let s = &self.stream;
        let v = self.model.cfg.vocab_size as usize;
        // logits → work（采样原地改 work，保留 logits 给 logprobs / 对拍）；之后一个融合 kernel 做完四步
        s.dtod_copy_n(&self.logits.data, &mut self.work.data, nb * v)?;
        let StepBuffers {
            chosen, temperature, min_p, top_k, top_p, temp01, seeds, offsets, mask_row, mask_bits, mask_words,
            bias_offsets, bias_ids, bias_vals, ..
        } = &mut self.buf;
        let mut wv = self.work.data.slice_mut(..nb * v);
        // 受限解码 mask + logit_bias：图内常驻；无约束行 kernel 立即返回（每步固定 ~几 µs）
        k::logits_mask_f32(
            s, &mut wv, &mask_row.data, &mask_bits.data, *mask_words,
            Some((&bias_offsets.data, &bias_ids.data, &bias_vals.data)), nb, v,
        )?;
        let mut ch = chosen.data.slice_mut(..nb);
        k::sample_fused_f32(
            s, &mut wv, &temperature.data, &min_p.data, &top_k.data, &top_p.data, &temp01.data, &seeds.data,
            &offsets.data, &mut ch, nb, v,
        )?;
        // 异步调度：保存本步采样给下一步 gather（图内；非异步时无害）
        {
            let StepBuffers { chosen, plan, .. } = &mut self.buf;
            plan.save_chosen(&self.stream, &chosen.data, nb)?;
        }
        Ok(())
    }

    // ---------------------------------------------------------------- CUDA graph

    /// 捕获体（上传假 plan + forward + 采样），`DecodeGraphs` 预热一次再在 capture 内跑一次。
    fn capture_body(&mut self, nb: usize) -> Result<()> {
        let p = decode_capture_plan(nb);
        self.upload_plan(&p, nb)?;
        self.upload_sampling(&p.sample, nb)?;
        self.forward_device(BatchMode::Decode, nb, nb, nb, 1)?;
        self.sample_device_fixed(nb)
    }

    /// 启动时一次捕获全部桶（GRAPH_CAPTURE=eager）。
    fn capture_decode_graphs(&mut self) -> Result<()> {
        let mut graphs = std::mem::replace(&mut self.graphs, DecodeGraphs::new(&self.stream, false, 0));
        let r = graphs.capture_all(|nb| self.capture_body(nb));
        self.graphs = graphs;
        r
    }

    /// 懒捕获：n 行 decode 需要的桶若还没有图，现在捕获（只在主套、图启用时）。
    fn ensure_graph(&mut self, n: usize) -> Result<()> {
        if self.alt_active {
            return Ok(());
        }
        let mut graphs = std::mem::replace(&mut self.graphs, DecodeGraphs::new(&self.stream, false, 0));
        let r = graphs.ensure(n, |nb| self.capture_body(nb));
        self.graphs = graphs;
        r
    }

    // ---------------------------------------------------------------- eager 采样（惩罚 / logprobs / 非 decode）

    fn sample_eager(&mut self, sb: &SampleBatch, nrows: usize) -> Result<()> {
        let v = self.model.cfg.vocab_size as usize;
        let fast = !sb.any_random && !sb.any_penalty && sb.max_logprobs.is_none() && !sb.any_mask;
        if fast {
            let s = &self.stream;
            let lg = self.logits.data.slice(..nrows * v);
            let mut ch = self.buf.chosen.data.slice_mut(..nrows);
            k::argmax_f32_n(s, &lg, &mut ch, nrows, v)?;
            self.last_logprobs = vec![Vec::new(); nrows];
            self.save_prev_chosen(nrows)?;
            return Ok(());
        }
        self.upload_sampling(sb, nrows)?;
        let s = &self.stream;
        s.dtod_copy_n(&self.logits.data, &mut self.work.data, nrows * v)?;
        if sb.max_logprobs.is_some() {
            s.dtod_copy_n(&self.logits.data, &mut self.raw.data, nrows * v)?;
        }
        {
            let b = &self.buf;
            let mut wv = self.work.data.slice_mut(..nrows * v);
            if sb.any_mask {
                let bias = (!sb.bias_ids.is_empty()).then(|| (&b.bias_offsets.data, &b.bias_ids.data, &b.bias_vals.data));
                k::logits_mask_f32(s, &mut wv, &b.mask_row.data, &b.mask_bits.data, b.mask_words, bias, nrows, v)?;
            }
            if sb.any_penalty {
                k::penalties_f32(
                    s,
                    &mut wv,
                    &b.hist_ids.data,
                    &b.hist_offsets.data.slice(..nrows + 1),
                    &b.hist_prompt_len.data.slice(..nrows),
                    &b.rep.data.slice(..nrows),
                    &b.freq.data.slice(..nrows),
                    &b.pres.data.slice(..nrows),
                    v,
                )?;
            }
            if sb.any_temperature {
                k::temperature_f32(s, &mut wv, &b.temperature.data.slice(..nrows), v)?;
            }
            if sb.any_min_p {
                k::min_p_mask_f32(s, &mut wv, &b.min_p.data.slice(..nrows), v)?;
            }
            if sb.any_topk_topp {
                k::topk_topp_mask_f32(
                    s,
                    &mut wv,
                    &b.top_k.data.slice(..nrows),
                    &b.top_p.data.slice(..nrows),
                    v,
                )?;
            }
        }
        {
            let StepBuffers { chosen, temp01, seeds, offsets, .. } = &mut self.buf;
            let wv = self.work.data.slice(..nrows * v);
            let mut ch = chosen.data.slice_mut(..nrows);
            if sb.any_random {
                k::sample_f32(
                    s,
                    &wv,
                    &temp01.data.slice(..nrows),
                    &seeds.data.slice(..nrows),
                    &offsets.data.slice(..nrows),
                    &mut ch,
                    v,
                )?;
            } else {
                k::argmax_f32_n(s, &wv, &mut ch, nrows, v)?;
            }
        }
        self.last_logprobs = vec![Vec::new(); nrows];
        if let Some(kmax) = sb.max_logprobs {
            let kmax = kmax as usize;
            let raw = self.raw.data.slice(..nrows * v);
            let ch = self.buf.chosen.data.slice(..nrows);
            let (ids_v, lp_v, rank_v) = {
                let mut ids = self.buf.lp_ids.data.slice_mut(..nrows * (kmax + 1));
                let mut lp = self.buf.lp_val.data.slice_mut(..nrows * (kmax + 1));
                let mut rank = self.buf.lp_rank.data.slice_mut(..nrows);
                k::logprobs_f32(s, &raw, &ch, kmax, &mut ids, &mut lp, &mut rank, v)?;
                (
                    s.dtoh_sync_copy_view(&self.buf.lp_ids.data.slice(..nrows * (kmax + 1)))?,
                    s.dtoh_sync_copy_view(&self.buf.lp_val.data.slice(..nrows * (kmax + 1)))?,
                    s.dtoh_sync_copy_view(&self.buf.lp_rank.data.slice(..nrows))?,
                )
            };
            for (r, want) in sb.logprobs.iter().enumerate() {
                let Some(kk) = want else { continue };
                let base = r * (kmax + 1);
                let top = (1..=*kk as usize).map(|j| (ids_v[base + j] as u32, lp_v[base + j])).collect();
                self.last_logprobs[r] = vec![TokenLogprobs {
                    token_id: ids_v[base] as u32,
                    logprob: lp_v[base],
                    rank: rank_v[r] as u32,
                    top,
                    prompt_pos: None,
                    prompt: Vec::new(),
                }];
            }
        }
        if let Some(kmax) = sb.max_prompt_logprobs {
            self.prompt_logprobs(sb, kmax as usize)?;
        }
        self.save_prev_chosen(nrows)?;
        Ok(())
    }

    /// `prompt_logprobs`：对 `upload_plan` 收集的 prompt 行（hidden 仍在 model 里）按 max_logits_rows 分块再算
    /// 末层 norm + lm_head → logprobs(目标 = 下一个 prompt token)，结果以 `prompt_pos = Some(p)` 挂在该行 logprobs 前面。
    /// 只在请求显式要了 prompt_logprobs 的 prefill 步发生；其余请求零开销。
    /// TODO(api): chunked prefill 的中间 chunk 行 runner 不回传 logprobs（只有 end_of_seq 行回传）→ 只有整段 prompt
    /// 在一个 chunk 内（≤ max_num_batched_tokens）时 prompt_logprobs 才完整；prefix cache 命中的前缀也没有 hidden。
    fn prompt_logprobs(&mut self, sb: &SampleBatch, kmax: usize) -> Result<()> {
        if self.plp_pending.is_empty() {
            return Ok(());
        }
        let v = self.model.cfg.vocab_size as usize;
        let chunk = self.model.max_logits_rows.min(self.buf.plp_rows.numel()).max(1);
        let pending = std::mem::take(&mut self.plp_pending);
        for part in pending.chunks(chunk) {
            let n = part.len();
            let rows: Vec<i32> = part.iter().map(|x| x.1).collect();
            let targets: Vec<i32> = part.iter().map(|x| x.2).collect();
            let s = &self.stream;
            s.htod_into(&rows, &mut self.buf.plp_rows.data)?;
            s.htod_into(&targets, &mut self.buf.plp_targets.data)?;
            // raw 在 chosen 的 logprobs 之后已无用，借作 logits 工作区
            self.model.logits_for_rows(s, &self.gemm, &self.buf.plp_rows.data, n, &mut self.raw.data)?;
            let (ids_v, lp_v, rank_v) = {
                let raw = self.raw.data.slice(..n * v);
                let tg = self.buf.plp_targets.data.slice(..n);
                let mut ids = self.buf.lp_ids.data.slice_mut(..n * (kmax + 1));
                let mut lp = self.buf.lp_val.data.slice_mut(..n * (kmax + 1));
                let mut rank = self.buf.lp_rank.data.slice_mut(..n);
                k::logprobs_f32(s, &raw, &tg, kmax, &mut ids, &mut lp, &mut rank, v)?;
                (
                    s.dtoh_sync_copy_view(&self.buf.lp_ids.data.slice(..n * (kmax + 1)))?,
                    s.dtoh_sync_copy_view(&self.buf.lp_val.data.slice(..n * (kmax + 1)))?,
                    s.dtoh_sync_copy_view(&self.buf.lp_rank.data.slice(..n))?,
                )
            };
            for (j, &(row, _, _, pos)) in part.iter().enumerate() {
                let Some(kk) = sb.prompt_logprobs.get(row).copied().flatten() else { continue };
                let base = j * (kmax + 1);
                let top = (1..=kk as usize).map(|t| (ids_v[base + t] as u32, lp_v[base + t])).collect();
                let e = TokenLogprobs {
                    token_id: ids_v[base] as u32,
                    logprob: lp_v[base],
                    rank: rank_v[j] as u32,
                    top,
                    prompt_pos: Some(pos),
                    prompt: Vec::new(),
                };
                // 挂在该行生成 token 的条目下（没要生成 logprobs 时放一个占位条目）
                let lp = &mut self.last_logprobs[row];
                if lp.is_empty() {
                    lp.push(TokenLogprobs {
                        token_id: TokenLogprobs::PLACEHOLDER,
                        logprob: 0.0,
                        rank: 0,
                        top: Vec::new(),
                        prompt_pos: None,
                        prompt: Vec::new(),
                    });
                }
                lp[0].prompt.push(e);
            }
        }
        Ok(())
    }
}

impl Qwen3Executor {
    /// 主套 ↔ 副套整体互换（流、GEMM、输入缓冲、logits 工作区、模型激活 scratch）。
    fn swap_alt(&mut self) {
        let mut a = self.alt.take().expect("alt set");
        std::mem::swap(&mut self.stream, &mut a.stream);
        std::mem::swap(&mut self.gemm, &mut a.gemm);
        std::mem::swap(&mut self.buf, &mut a.buf);
        std::mem::swap(&mut self.logits, &mut a.logits);
        std::mem::swap(&mut self.work, &mut a.work);
        std::mem::swap(&mut self.raw, &mut a.raw);
        self.model.swap_scratch(&mut a.scratch);
        self.alt = Some(a);
        self.alt_active = !self.alt_active;
    }

    /// 采样后把 chosen 存到 prev_chosen 供下一步 gather；插队 prefill 步（skip_prev_chosen）不写，
    /// 以免覆盖在飞 decode 步的结果。
    fn save_prev_chosen(&mut self, nrows: usize) -> Result<()> {
        if self.skip_prev_chosen {
            return Ok(());
        }
        let StepBuffers { chosen, plan, .. } = &mut self.buf;
        plan.save_chosen(&self.stream, &chosen.data, nrows)
    }

    /// GPU 忙时测量：主流上 step 工作入队前 / 后各记一个带时间戳的事件；collect 同步后读差值。
    fn mark_gpu_start(&mut self) -> Result<()> {
        if !self.gpu_timing {
            return Ok(());
        }
        if self.ev_gpu.is_none() {
            self.ev_gpu = Some((self.stream.new_timing_event()?, self.stream.new_timing_event()?));
        }
        self.stream.record_timing(&self.ev_gpu.as_ref().unwrap().0)
    }
    fn mark_gpu_end(&mut self) -> Result<()> {
        if !self.gpu_timing {
            return Ok(());
        }
        self.stream.record_timing(&self.ev_gpu.as_ref().unwrap().1)
    }

    /// 启用 KV host tier：`host_gib` 的 pinned 内存，按全层块大小切槽。返回槽位数（scheduler 索引容量）。
    /// 一个块（全部层）在 host 侧的字节数（host tier 槽大小 = PD 导出单元大小）。
    pub fn kv_block_bytes_all_layers(&self) -> (usize, usize) {
        let cfg = &self.model.cfg;
        let bs = self.model.kv.first().map(|k| k.block_size).unwrap_or(16);
        let per_layer = bs * cfg.num_key_value_heads as usize * 2 * cfg.head_dim as usize * 2;
        (per_layer, per_layer * self.model.kv.len())
    }
    /// PD 分离：保证有一套 pack/unpack staging + pinned 槽（`--role prefill/decode` 时即使 host tier 关也要）。
    /// 已开 host tier 则直接复用其 arena；否则建 `HOST_TIER_CHUNK` 槽的小 arena（只做中转，不做 LRU 索引）。
    /// 返回 pinned 区 (基址, 字节) 供 `KvTransport::register`。
    /// PD 分离 v2：建异步导出 / 导入资源（copy stream、设备 staging ×2、pinned 池），返回 pinned 池总区（NIXL 注册用：
    /// 池是多块独立 cuMemAllocHost，v2 TCP 不需要整块注册，这里返回第一块的 (ptr, 总字节) 仅作容量提示）。
    /// `pool_bufs`：pinned 缓冲个数（每块 = max_blocks × 单元字节，Qwen3 4k 上下文 ≈ 448 MB）。
    /// `world`：TP 规模。pinned 缓冲按 **全 world** 的单元算（rank 0 的缓冲要装下所有 rank 的分片，布局 `[rank][block][layer]`）；
    /// 非 0 rank 传 `pool_bufs = 0`（不需要自己的缓冲，只用 staging）。
    pub fn enable_pd(&mut self, pool_bufs: usize, world: usize) -> Result<(*mut u8, usize)> {
        let (per_layer, per_block_all) = self.kv_block_bytes_all_layers();
        let layers = self.model.kv.len();
        let cap = self.buf.plan.max_blocks * per_block_all * world.max(1);
        let copy = self.stream.device().new_stream()?;
        let ptrs: Vec<u64> = self.model.kv.iter().map(|kv| self.stream.ptr(&kv.cache.data)).collect();
        let layer_ptrs = Tensor::from_vec(&self.stream, ptrs, &[layers])?;
        let mk = |st: &Stream| -> Result<(Tensor<i32>, Tensor<u8>)> {
            Ok((Tensor::<i32>::zeros(st, &[PD_CHUNK])?, Tensor::<u8>::uninit(st, &[PD_CHUNK * per_block_all])?))
        };
        let (b0, s0) = mk(&self.stream)?;
        let (b1, s1) = mk(&self.stream)?;
        let pool = PinnedPool::new(&self.stream, pool_bufs, cap)?;
        self.stream.synchronize()?;
        tracing::info!(
            pool_bufs,
            world,
            buf_mib = cap >> 20,
            chunk = PD_CHUNK,
            "pd: async export/import resources built (copy stream + 2 staging + pinned pool)"
        );
        let region = (std::ptr::null_mut(), cap * pool_bufs.max(1));
        self.pd = Some(PdStore {
            copy,
            layer_ptrs,
            bytes_per_layer_block: per_layer,
            staging: [s0, s1],
            blocks: [b0, b1],
            staging_free: [None, None],
            pool,
            inflight_imports: Vec::new(),
            region: None,
        });
        Ok(region)
    }
    /// D 侧：建共享槽区（每槽 = max_blocks × 全层块字节 = 一个最长请求），返回给接收线程握手用。需先 `enable_pd`。
    /// `world`：槽要装下所有 rank 的分片（同 `enable_pd`）。
    pub fn enable_pd_region(&mut self, kind: k::RegionKind, slots: usize, world: usize) -> Result<dinfer_core::pd::LocalRegion> {
        let (_, per_block_all) = self.kv_block_bytes_all_layers();
        let slot_bytes = self.buf.plan.max_blocks * per_block_all * world.max(1);
        let pd = self.pd.as_mut().context("enable_pd_region before enable_pd")?;
        let r = std::sync::Arc::new(k::SharedRegion::create(&self.stream, kind, slots, slot_bytes)?);
        tracing::info!(kind = kind.as_str(), slots, slot_mib = slot_bytes >> 20, "pd: shared slot region created");
        let info = dinfer_core::pd::RegionInfo {
            kind: kind.as_str().into(),
            slots: slots as u32,
            slot_bytes: slot_bytes as u64,
            shm_name: r.shm_name().map(str::to_string),
            ipc_handle: r.ipc_handle().map(|h| h.to_vec()),
            device_uuid: r.uuid().to_vec(),
        };
        let host_base = if r.is_host() { r.slot_addr(0)? as *const u8 } else { std::ptr::null() };
        pd.region = Some(r);
        Ok(dinfer_core::pd::LocalRegion { info, host_base })
    }
    // ---------------------------------------------------------------- PD 分片原语（单 rank；`tp::TpExecutor` 按 rank 组合，design §13.7）
    /// 本 rank 一个块（全层）的导出字节（TP 下 = 单卡的 1/tp：每 rank 只持自己的 KV 头分片）。
    pub fn pd_unit_bytes_local(&self) -> usize {
        self.kv_block_bytes_all_layers().1
    }
    /// 一个块（全层、**全部 KV 头**）在 wire 上的 canonical 字节 = 本地 × world，与 tp 无关（design §13.7 异构 tp：
    /// 布局 `[block][layer][全局 kv_head][slot][K/V][D]`，P 与 D 的 tp 可不同，各 rank 只读写自己头段）。
    pub fn pd_unit_bytes_canonical(&self) -> usize {
        self.pd_unit_bytes_local() * self.model.tp.1.max(1)
    }
    /// pack / unpack kernel 的形状参数：(block_size, 本地 kv_heads)。
    fn kv_pack_dims(&self) -> (usize, usize) {
        let bs = self.model.kv.first().map(|k| k.block_size).unwrap_or(16);
        (bs, self.model.cfg.num_key_value_heads as usize)
    }
    pub fn pd_enabled(&self) -> bool {
        self.pd.is_some()
    }
    /// 从本 rank 的 pinned 池取一块（None = 未 enable_pd 或池空）。
    pub fn pd_take_buf(&self) -> Option<PinnedBuf> {
        self.pd.as_ref().and_then(|p| p.pool.take())
    }
    pub fn pd_region(&self) -> Option<std::sync::Arc<k::SharedRegion>> {
        self.pd.as_ref().and_then(|p| p.region.clone())
    }
    /// 直接设区（TP：rank 0 建 / 映射的 shm 区交给其它 rank 共用；shm 已 PORTABLE 注册，各 ctx 都能当 pinned 用）。
    pub fn pd_set_region(&mut self, r: Option<std::sync::Arc<k::SharedRegion>>) -> Result<()> {
        self.pd.as_mut().context("pd_set_region without enable_pd")?.region = r;
        Ok(())
    }
    /// 登记一次在飞导入：`ev`（copy stream 读完源）完成前 `keep` 不释放（缓冲归池 / 槽 credit 回 P）。
    pub fn pd_push_inflight(&mut self, ev: Event, keep: Box<dyn std::any::Any + Send + Sync>) {
        if let Some(pd) = self.pd.as_mut() {
            pd.inflight_imports.push((ev, keep));
        }
    }
    /// 回收已完成的在飞导入（非阻塞）。
    pub fn pd_gc_inflight(&mut self) {
        let Some(pd) = self.pd.as_mut() else { return };
        if pd.inflight_imports.is_empty() {
            return;
        }
        let mut keep = Vec::with_capacity(pd.inflight_imports.len());
        for (ev, b) in pd.inflight_imports.drain(..) {
            if ev.is_done().unwrap_or(true) { drop(b) } else { keep.push((ev, b)) }
        }
        pd.inflight_imports = keep;
    }
    /// 把 `ids` 的全层 KV（本 rank 分片）写进地址 `dst` 起的 **canonical** 布局 `[block][layer][全局 kv_head][slot][K/V][D]`
    /// （host：已注册 pinned / shm；设备：本 ctx 可见）。本 rank 只写每层里自己头段那一列（2D 拷贝：源行距 = 本地层字节，
    /// 目标行距 = 全局层字节，列偏移 = rank × 本地层字节），所以任意 tp 的各 rank 写同一个 `dst` 互不重叠，D 侧任意 tp 可读。
    /// 主流 pack → staging（双缓冲）→ copy stream 搬走；返回 copy stream 上"全部写完"的事件。调用方保证 `dst` 容量 ≥ ids.len() × canonical 单元。
    pub fn pd_export_addr(&mut self, ids: &[dinfer_core::kv::BlockId], dst: u64, dst_is_host: bool) -> Result<Event> {
        let (bs, hk) = self.kv_pack_dims();
        let (rank, world) = self.model.tp;
        let pd = self.pd.as_mut().context("pd: export without resources (call enable_pd)")?;
        let layers = self.model.kv.len();
        let per = pd.bytes_per_layer_block;
        let gper = per * world.max(1); // canonical 每层字节
        let gper_all = gper * layers;
        let col = (rank * per) as u64;
        for (ci, chunk) in ids.chunks(PD_CHUNK).enumerate() {
            let i = ci & 1;
            // 生产方（主流 pack）写 staging[i] 前，等上一次消费方（copy stream 读走）用完
            if let Some(ev) = pd.staging_free[i].take() {
                self.stream.wait_event(&ev)?;
            }
            let bids: Vec<i32> = chunk.iter().map(|&b| b as i32).collect();
            self.stream.htod_into(&bids, &mut pd.blocks[i].data)?;
            k::kv_pack(&self.stream, &pd.layer_ptrs.data, &pd.blocks[i].data, &mut pd.staging[i].data, chunk.len(), layers, per, bs, hk)?;
            let packed = self.stream.record_event()?;
            pd.copy.wait_event(&packed)?;
            let src = pd.copy.ptr(&pd.staging[i].data);
            let off = (ci * PD_CHUNK * gper_all) as u64;
            // 行 = (块, 层)，每行写本地 per 字节到 canonical 行的 rank 列
            pd.copy.copy2d(dst + off + col, dst_is_host, gper, src, false, per, per, chunk.len() * layers)?;
            pd.staging_free[i] = Some(pd.copy.record_event()?);
        }
        pd.copy.record_event()
    }
    /// `pd_export_addr` 的反向：从地址 `src`（同布局）读回 `ids` 的全层 KV 分片。copy stream 搬进 staging → 主流 unpack；
    /// 返回 copy stream 上"源已全部读完"的事件（之后源缓冲 / 槽才可释放）。主流后续 forward 自然排在 unpack 之后。
    pub fn pd_import_addr(&mut self, ids: &[dinfer_core::kv::BlockId], src: u64, src_is_host: bool) -> Result<Event> {
        let (bs, hk) = self.kv_pack_dims();
        let (rank, world) = self.model.tp;
        let pd = self.pd.as_mut().context("pd: import without resources (call enable_pd)")?;
        let layers = self.model.kv.len();
        let per = pd.bytes_per_layer_block;
        let gper = per * world.max(1);
        let gper_all = gper * layers;
        let col = (rank * per) as u64;
        for (ci, chunk) in ids.chunks(PD_CHUNK).enumerate() {
            let i = ci & 1;
            // 生产方（copy 写 staging）前，等上一次消费方（主流 unpack）用完
            if let Some(ev) = pd.staging_free[i].take() {
                pd.copy.wait_event(&ev)?;
            }
            let dst = pd.copy.ptr(&pd.staging[i].data);
            let off = (ci * PD_CHUNK * gper_all) as u64;
            // 只读 canonical 每行里本 rank 的头段到紧凑 staging
            pd.copy.copy2d(dst, false, per, src + off + col, src_is_host, gper, per, chunk.len() * layers)?;
            let h2d = pd.copy.record_event()?;
            self.stream.wait_event(&h2d)?;
            let bids: Vec<i32> = chunk.iter().map(|&b| b as i32).collect();
            self.stream.htod_into(&bids, &mut pd.blocks[i].data)?;
            k::kv_unpack(&self.stream, &pd.layer_ptrs.data, &pd.blocks[i].data, &pd.staging[i].data, chunk.len(), layers, per, bs, hk)?;
            pd.staging_free[i] = Some(self.stream.record_event()?);
        }
        pd.copy.record_event()
    }
    pub fn enable_host_tier(&mut self, host_gib: f64) -> Result<usize> {
        let (_, per_block_all) = self.kv_block_bytes_all_layers();
        let num_slots = ((host_gib * (1u64 << 30) as f64) as usize) / per_block_all;
        if num_slots == 0 {
            return Ok(0);
        }
        self.build_host_tier(num_slots, per_block_all)?;
        tracing::info!(num_slots, per_block_kib = per_block_all / 1024, host_gib, "kv host tier enabled");
        Ok(num_slots)
    }
    fn build_host_tier(&mut self, num_slots: usize, per_block_all: usize) -> Result<()> {
        let (per_layer, _) = self.kv_block_bytes_all_layers();
        let layers = self.model.kv.len();
        let copy = self.stream.device().new_stream()?;
        let arena = HostArena::new(&self.stream, num_slots, per_block_all)?;
        let ptrs: Vec<u64> = self.model.kv.iter().map(|kv| self.stream.ptr(&kv.cache.data)).collect();
        let layer_ptrs = Tensor::from_vec(&self.stream, ptrs, &[layers])?;
        let blocks_buf = Tensor::<i32>::zeros(&self.stream, &[HOST_TIER_CHUNK])?;
        let blocks_off = Tensor::<i32>::zeros(&self.stream, &[HOST_TIER_CHUNK])?;
        let staging = Tensor::<u8>::uninit(&self.stream, &[HOST_TIER_CHUNK * per_block_all])?;
        let staging_off = Tensor::<u8>::uninit(&self.stream, &[HOST_TIER_CHUNK * per_block_all])?;
        self.stream.synchronize()?;
        self.host_tier = Some(HostTierStore {
            arena,
            bytes_per_layer_block: per_layer,
            copy,
            offload_done: None,
            layer_ptrs,
            blocks_buf,
            staging,
            chunk: HOST_TIER_CHUNK,
            staging_off,
            blocks_off,
        });
        Ok(())
    }

    /// 在副套上执行 `f`（进入前换入副套，结束后无论成败都换回主套）。
    fn on_alt<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        self.swap_alt();
        let r = f(self);
        self.swap_alt();
        r
    }

    /// 上传 + 入队（forward + 采样），不等结果。返回 (请求数, 入队开始时刻)。
    fn enqueue(&mut self, plan: &BatchPlan) -> Result<(usize, std::time::Instant)> {
        self.skip_prev_chosen = plan.skip_prev_chosen;
        let n = plan.num_reqs();
        let t = plan.total_tokens();
        ensure!(t <= self.model.max_tokens, "batch of {t} tokens > max_tokens {}", self.model.max_tokens);
        let nrows = plan.logits_rows.len();
        ensure!(
            nrows <= self.model.max_logits_rows,
            "{nrows} logits rows > max_logits_rows {}",
            self.model.max_logits_rows
        );
        ensure!(
            nrows >= n && plan.sample.num_rows() == nrows,
            "qwen3 executor: rows {nrows} / sample {} vs reqs {n}",
            plan.sample.num_rows()
        );
        let sb = &plan.sample;
        // Decode 模式 ⇒ 每请求 1 行（带 draft 的请求是 Paged）→ 图路径下 nrows == n
        let graph_eligible = plan.mode == BatchMode::Decode && !sb.any_penalty && sb.max_logprobs.is_none();
        if graph_eligible {
            self.ensure_graph(n)?;
        }
        let use_graph = graph_eligible
            && !self.alt_active // 图是在主流上捕获的，副套不走图
            && self.graphs.get(n).is_some();
        let t_launch;
        if use_graph {
            let nb = self.graphs.get(n).unwrap().0;
            let (tt, nb2, nr) = self.upload_plan(plan, nb)?;
            debug_assert_eq!((tt, nb2, nr), (nb, nb, nb));
            let t_us = std::time::Instant::now();
            self.upload_sampling(sb, nb)?;
            self.t_upload_sampling_us += t_us.elapsed().as_micros() as u64;
            t_launch = std::time::Instant::now();
            self.graphs.get(n).unwrap().1.launch().context("launch decode graph")?;
            self.t_graph_launch_us += t_launch.elapsed().as_micros() as u64;
            self.graph_steps += 1;
            self.last_logprobs = vec![Vec::new(); n];
        } else {
            let (tt, nb, nr) = self.upload_plan(plan, n)?;
            debug_assert_eq!((tt, nb, nr), (t, n, nrows));
            t_launch = std::time::Instant::now();
            let max_seqlen = plan.num_tokens.iter().copied().max().unwrap_or(0) as usize;
            self.forward_device(plan.mode, t, n, nrows, max_seqlen)?;
            self.sample_eager(sb, nrows)?;
            self.eager_steps += 1;
        }
        self.last_rows = nrows;
        Ok((nrows, t_launch))
    }

    /// 等本流完成并取回采样结果。
    fn collect(&mut self, n: usize) -> Result<Vec<i32>> {
        self.stream.dtoh_sync_copy_view(&self.buf.chosen.data.slice(..n))
    }

    fn account(&mut self, t0: std::time::Instant, t_launch: std::time::Instant, t_wait: std::time::Instant) {
        let t_end = std::time::Instant::now();
        if let Some((a, b)) = &self.ev_gpu {
            // collect() 已同步主流，两事件都完成；失败（如副套路径未记录）忽略
            match TimingEvent::elapsed_ms(a, b) {
                Ok(ms) => self.t_gpu_ms += ms,
                Err(e) => {
                    if self.t_gpu_err == 0 {
                        tracing::warn!(err = %e, "gpu busy timing: elapsed_ms failed (reported once)");
                    }
                    self.t_gpu_err += 1;
                }
            }
        }
        self.t_upload_us += (t_launch - t0).as_micros() as u64;
        self.t_launch_us += (t_wait - t_launch).as_micros() as u64;
        self.t_wait_us += (t_end - t_wait).as_micros() as u64;
        let steps = self.graph_steps + self.eager_steps;
        if steps % 1000 == 0 {
            tracing::info!(
                steps,
                graph = self.graph_steps,
                eager = self.eager_steps,
                pair = self.pair_steps,
                upload_us_per_step = self.t_upload_us / steps,
                launch_us_per_step = self.t_launch_us / steps,
                wait_us_per_step = self.t_wait_us / steps,
                gpu_busy_us_per_step = (self.t_gpu_ms * 1000.0 / steps as f32) as u64,
                upload_sampling_us_per_graph_step = self.t_upload_sampling_us / self.graph_steps.max(1),
                graph_launch_us_per_graph_step = self.t_graph_launch_us / self.graph_steps.max(1),
                gpu_timing_errors = self.t_gpu_err,
                "executor step breakdown (cumulative avg)"
            );
        }
    }
}

/// KV host tier 的设备侧：pinned 竞技场（`num_slots × 全层块字节`，槽内 `[layer][一层一块]`）+ 独立 copy stream。
/// 与设备 `kv[layer].cache[block]` 布局一致，直接按字节搬。unsafe 全在 `dinfer_kernels::HostArena` / `Stream` 里。
struct HostTierStore {
    arena: HostArena,
    bytes_per_layer_block: usize,
    copy: Stream,
    /// copy stream 上最近一次 offload 结束的事件；fetch 前主流等它（host 槽可能刚被写）。
    offload_done: Option<Event>,
    /// 每层 cache 基址表（设备常量）、块号表、设备 staging（最多 CHUNK 块 × 全层）；
    /// 搬运 = 1 个 pack/unpack kernel + 每块 1 次 H2D/D2H（原来每块 28 次）。
    layer_ptrs: Tensor<u64>,
    blocks_buf: Tensor<i32>,
    staging: Tensor<u8>,
    chunk: usize,
    /// 主流上 "staging 已用完" 的事件：fetch 用主流 staging，offload 用 copy stream staging，两者分开避免互踩。
    staging_off: Tensor<u8>,
    blocks_off: Tensor<i32>,
}
const HOST_TIER_CHUNK: usize = 64;

/// PD 分离 v2 资源。导出：主流 pack → staging[i]（事件 packed_i）→ copy stream 等事件后 D2H → pinned buf（事件 done）。
/// 导入：copy stream H2D pinned → staging[i]（事件 h2d_i）→ 主流等事件后 unpack（事件 unpacked_i，copy stream 复用 staging 前等它）。
/// 两套 staging 交替，pack / D2H 可以跨 chunk 重叠；engine 线程从不等 GPU。
struct PdStore {
    copy: Stream,
    layer_ptrs: Tensor<u64>,
    bytes_per_layer_block: usize,
    staging: [Tensor<u8>; 2],
    blocks: [Tensor<i32>; 2],
    /// staging[i] 上一次被 "消费方" 用完的事件（导出 = copy stream D2H 完；导入 = 主流 unpack 完）；生产方写之前等它。
    staging_free: [Option<Event>; 2],
    pool: PinnedPool,
    /// 导入在飞：(copy stream 上 H2D / D2D 全部完成的事件, 缓冲或槽 payload)。事件完成即 drop（槽 drop = credit 回 P）。
    inflight_imports: Vec<(Event, Box<dyn std::any::Any + Send + Sync>)>,
    /// 共享槽区（design §10.3 b）：D 侧 = 自己建的本地区；P 侧 = 映射的对端区。
    region: Option<std::sync::Arc<k::SharedRegion>>,
}
const PD_CHUNK: usize = 64;

/// 本 crate 的 newtype（孤儿规则：core 的 trait × kernels 的类型只能在这里接）。
pub struct PdBuf(pub PinnedBuf);
#[derive(Clone)]
pub struct PdPool(pub PinnedPool);
impl dinfer_core::pd::HostBuf for PdBuf {
    fn as_slice(&self) -> &[u8] {
        self.0.as_slice()
    }
    fn as_mut_cap(&mut self) -> &mut [u8] {
        self.0.as_mut_cap()
    }
    fn set_len(&mut self, len: usize) -> Result<()> {
        self.0.set_len(len)
    }
    fn capacity(&self) -> usize {
        self.0.capacity()
    }
    fn is_pinned(&self) -> bool {
        true
    }
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send + Sync> {
        self
    }
}
impl dinfer_core::pd::HostBufPool for PdPool {
    fn take(&self, min_bytes: usize) -> Option<Box<dyn dinfer_core::pd::HostBuf>> {
        if self.0.capacity() < min_bytes {
            return None;
        }
        self.0.take().map(|b| Box::new(PdBuf(b)) as Box<dyn dinfer_core::pd::HostBuf>)
    }
}

/// 导出句柄：发送线程 `wait()` 等 copy stream 的 done 事件，然后读 pinned 缓冲。
struct PdExport {
    done: Event,
    buf: PdBuf,
}
impl dinfer_core::pd::KvExportHandle for PdExport {
    fn wait(&mut self) -> Result<&dyn dinfer_core::pd::HostBuf> {
        self.done.synchronize()?;
        Ok(&self.buf)
    }
}

/// 槽导出句柄：等 copy stream 写完对端槽；无 host 字节（header 只带槽号）。
struct PdSlotExport {
    done: Event,
    _region: std::sync::Arc<k::SharedRegion>,
}
impl dinfer_core::pd::KvExportHandle for PdSlotExport {
    fn wait(&mut self) -> Result<&dyn dinfer_core::pd::HostBuf> {
        self.done.synchronize()?;
        Ok(&EMPTY)
    }
}
static EMPTY: Vec<u8> = Vec::new();
/// 每个请求本步 1 个采样 token、无 draft。
/// 无 draft 的 plan（异步 finish 路径）：每行一个 token。
fn push_tokens(generated: &mut Vec<Vec<u32>>, drafts: &mut Vec<Vec<u32>>, chosen: &[i32]) {
    for &c in chosen {
        generated.push(vec![c as u32]);
        drafts.push(Vec::new());
    }
}

/// 按 plan 把逐行采样结果折成每请求的 `generated`（无 draft：1 个；有 draft：接受的前缀 + bonus，GLM MTP 同口径），
/// 并把逐行 logprobs 折成每请求一条（接受的行 + bonus 行按序拼接）。drafts 这里不产（n-gram 在 runner 侧）。
fn verify_rows(
    plan: &BatchPlan,
    chosen: &[i32],
    lps: Vec<Vec<TokenLogprobs>>,
    generated: &mut Vec<Vec<u32>>,
    drafts: &mut Vec<Vec<u32>>,
) -> Vec<Vec<TokenLogprobs>> {
    let n = plan.num_reqs();
    let mut out_lps = Vec::with_capacity(n);
    let mut lps = lps.into_iter();
    for i in 0..n {
        let (r0, r1) = (plan.logits_row_start[i], plan.logits_row_start[i + 1]);
        let spec = &plan.spec_tokens[i];
        let mut gen = Vec::with_capacity(spec.len() + 1);
        let mut last_row = r0;
        for (j, &d) in spec.iter().enumerate() {
            if chosen[r0 + j] as u32 == d {
                gen.push(d);
                last_row = r0 + j + 1;
            } else {
                break;
            }
        }
        debug_assert!(last_row < r1);
        gen.push(chosen[last_row] as u32);
        generated.push(gen);
        drafts.push(Vec::new());
        // logprobs：行 r0..=last_row 各一条（无 logprobs 的请求全空 → 折成空）
        let mut lp = Vec::new();
        for r in r0..r1 {
            let row = lps.next().unwrap_or_default();
            if r <= last_row {
                lp.extend(row);
            }
        }
        out_lps.push(lp);
    }
    out_lps
}

impl ModelExecutor for Qwen3Executor {
    fn supports_pair(&self) -> bool {
        self.alt.is_some()
    }

    fn supports_async(&self) -> bool {
        async_sched_enabled()
    }

    /// host → GPU：主流上同步排在本步 forward 之前（同流顺序即保证可见）。
    fn kv_fetch(&mut self, orders: &[(dinfer_core::kv::BlockId, u32)]) -> Result<()> {
        let Some(ht) = self.host_tier.as_ref() else { return Ok(()) };
        if let Some(ev) = &ht.offload_done {
            self.stream.wait_event(ev)?; // 槽位可能刚被 offload 写入（copy stream）
        }
        let layers = self.model.kv.len();
        let per = ht.bytes_per_layer_block;
        let per_all = per * layers;
        let (bs, hk) = self.kv_pack_dims();
        let ht = self.host_tier.as_mut().unwrap();
        for chunk in orders.chunks(ht.chunk) {
            let base = self.stream.ptr_mut(&mut ht.staging.data);
            for (i, &(_, slot)) in chunk.iter().enumerate() {
                self.stream.htod_from_arena(&ht.arena, slot as usize, 0, base + (i * per_all) as u64, per_all)?;
            }
            let ids: Vec<i32> = chunk.iter().map(|&(b, _)| b as i32).collect();
            self.stream.htod_into(&ids, &mut ht.blocks_buf.data)?;
            k::kv_unpack(&self.stream, &ht.layer_ptrs.data, &ht.blocks_buf.data, &ht.staging.data, chunk.len(), layers, per, bs, hk)?;
        }
        Ok(())
    }

    /// GPU → host：copy stream 等主流（本步 KV 写完）后异步 D2H；记录完成事件供下次 fetch 等待。
    fn kv_offload(&mut self, orders: &[(dinfer_core::kv::BlockId, u32)]) -> Result<()> {
        let (bs, hk) = self.kv_pack_dims();
        let Some(ht) = self.host_tier.as_mut() else { return Ok(()) };
        let main_ev = self.stream.record_event()?;
        ht.copy.wait_event(&main_ev)?;
        let layers = self.model.kv.len();
        let per = ht.bytes_per_layer_block;
        let per_all = per * layers;
        for chunk in orders.chunks(ht.chunk) {
            // 上一轮 offload 的 staging_off 可能还在 D2H：copy stream 顺序执行，同流内自然有序
            let ids: Vec<i32> = chunk.iter().map(|&(b, _)| b as i32).collect();
            ht.copy.htod_into(&ids, &mut ht.blocks_off.data)?;
            k::kv_pack(&ht.copy, &ht.layer_ptrs.data, &ht.blocks_off.data, &mut ht.staging_off.data, chunk.len(), layers, per, bs, hk)?;
            let base = ht.copy.ptr(&ht.staging_off.data);
            for (i, &(_, slot)) in chunk.iter().enumerate() {
                ht.copy.dtoh_to_arena(&ht.arena, slot as usize, 0, base + (i * per_all) as u64, per_all)?;
            }
        }
        ht.offload_done = Some(ht.copy.record_event()?);
        Ok(())
    }

    // ---- PD 分离（design §10.3 v2，异步）：按位置导出 / 导入全部层 KV。engine 线程不等 GPU。
    fn kv_unit_bytes(&self) -> usize {
        if self.pd.is_none() {
            return 0;
        }
        self.pd_unit_bytes_canonical()
    }
    fn kv_host_pool(&self) -> Option<std::sync::Arc<dyn dinfer_core::pd::HostBufPool>> {
        self.pd.as_ref().map(|p| std::sync::Arc::new(PdPool(p.pool.clone())) as std::sync::Arc<dyn dinfer_core::pd::HostBufPool>)
    }
    /// P 侧。块已被 scheduler free 但未复用。主流：pack chunk → 记事件；copy stream：等事件 → D2H 进 pinned。
    /// 主流的后续 forward 排在 pack 之后，所以块复用安全；D2H 与 forward 重叠。返回句柄，发送线程等 `done`。
    fn kv_export(&mut self, blocks: &dinfer_core::kv::KvBlocks) -> Result<Box<dyn dinfer_core::pd::KvExportHandle>> {
        let ids = blocks.group(0);
        let total = ids.len() * self.pd_unit_bytes_canonical();
        let mut buf = self.pd_take_buf().context("pd: no pinned buffer (enable_pd not called, or pool exhausted → raise --pd-pool-bufs)")?;
        ensure!(buf.capacity() >= total, "pd: export {total} B > pinned buf {}", buf.capacity());
        let done = self.pd_export_addr(ids, buf.ptr() as u64, true)?;
        buf.set_len(total)?;
        Ok(Box::new(PdExport { done, buf: PdBuf(buf) }))
    }
    /// D 侧。payload pinned 则 copy stream 直接 H2D；否则先拷进池里一块 pinned。主流等 H2D 事件后 unpack；
    /// 缓冲挂 inflight 直到 H2D 完成事件过了再归还。
    fn kv_import(&mut self, blocks: &dinfer_core::kv::KvBlocks, payload: Box<dyn dinfer_core::pd::HostBuf>) -> Result<()> {
        ensure!(self.pd.is_some(), "pd: kv_import without resources (call enable_pd)");
        self.pd_gc_inflight();
        let ids = blocks.group(0);
        let per_all = self.pd_unit_bytes_canonical();
        let total = ids.len() * per_all;
        ensure!(payload.len() == total, "pd: payload {} != {} blocks × {per_all}", payload.len(), ids.len());
        // 共享槽区：数据已在本地区的槽里（shm = pinned host；cuda-ipc = 本卡设备区），copy stream 直接搬进 staging → unpack。
        if let Some(slot) = payload.slot() {
            let r = self.pd_region().context("pd: slot payload but no local region")?;
            ensure!(total <= r.slot_bytes(), "pd: import {total} B > slot {}", r.slot_bytes());
            let src = r.slot_addr(slot as usize)?;
            let done = self.pd_import_addr(ids, src, r.is_host())?;
            self.pd_push_inflight(done, payload.into_any());
            return Ok(());
        }
        // 拿到 pinned 源：payload 本身是我们池里的 PinnedBuf（接收线程用 kv_host_pool 落的）就直接用；否则 memcpy 进池
        let src: PinnedBuf = if payload.is_pinned() {
            match payload.into_any().downcast::<PdBuf>() {
                Ok(b) => b.0,
                Err(_) => anyhow::bail!("pd: pinned payload is not this executor's PinnedBuf"),
            }
        } else {
            let mut b = self.pd_take_buf().context("pd: pinned pool exhausted for import staging")?;
            ensure!(b.capacity() >= total, "pd: import {total} B > pinned buf {}", b.capacity());
            b.as_mut_cap()[..total].copy_from_slice(payload.as_slice());
            b.set_len(total)?;
            b
        };
        let done = self.pd_import_addr(ids, src.ptr() as u64, true)?;
        self.pd_push_inflight(done, Box::new(src));
        Ok(())
    }
    fn pd_poll(&mut self) {
        self.pd_gc_inflight();
    }
    /// P 侧：映射对端共享区。cuda-ipc 要求同卡（uuid 相等）；shm 任意卡。
    fn pd_map_region(&mut self, info: &dinfer_core::pd::RegionInfo) -> Result<()> {
        let per_block_all = self.pd_unit_bytes_canonical();
        let pd = self.pd.as_mut().context("pd_map_region without enable_pd")?;
        let kind = k::RegionKind::parse(&info.kind).with_context(|| format!("pd: unknown region kind {}", info.kind))?;
        ensure!(info.slot_bytes as usize >= per_block_all, "pd: peer slot {} B < one block {per_block_all}", info.slot_bytes);
        pd.region = None; // 先释放旧代数的映射
        let r = k::SharedRegion::open_remote(
            &self.stream,
            kind,
            info.slots as usize,
            info.slot_bytes as usize,
            info.shm_name.as_deref(),
            info.ipc_handle.as_deref(),
            &info.device_uuid,
        )?;
        pd.region = Some(std::sync::Arc::new(r));
        Ok(())
    }
    /// P 侧：pack → staging（主流）→ copy stream 写进对端槽（shm：D2H 到 pinned 映射；cuda-ipc：D2D）。
    fn kv_export_slot(&mut self, blocks: &dinfer_core::kv::KvBlocks, slot: u32) -> Result<Box<dyn dinfer_core::pd::KvExportHandle>> {
        let r = self.pd_region().context("pd: kv_export_slot without mapped region")?;
        let ids = blocks.group(0);
        let total = ids.len() * self.pd_unit_bytes_canonical();
        ensure!(total <= r.slot_bytes(), "pd: export {total} B > slot {}", r.slot_bytes());
        let dst = r.slot_addr(slot as usize)?;
        let done = self.pd_export_addr(ids, dst, r.is_host())?;
        Ok(Box::new(PdSlotExport { done, _region: r }))
    }
    /// 异步第一段：入队 + 异步 D2H（不等 GPU）。runner 只对 `async_eligible` 的 plan 调用（纯 decode）。
    fn begin(&mut self, plan: &BatchPlan) -> Result<()> {
        let t0 = std::time::Instant::now();
        let (n, t_launch) = self.enqueue(plan)?;
        let t_wait = std::time::Instant::now();
        let ticket = self.async_ring.enqueue(&self.stream, &self.buf.chosen.data, n)?;
        self.async_inflight.push_back((ticket, t0, t_launch, t_wait));
        Ok(())
    }

    fn finish(&mut self, generated: &mut Vec<Vec<u32>>, drafts: &mut Vec<Vec<u32>>) -> Result<()> {
        let (ticket, t0, t_launch, t_wait) = self.async_inflight.pop_front().context("executor finish without begin")?;
        let chosen = self.async_ring.wait(ticket)?;
        self.main_done = Some(self.stream.record_event()?);
        self.account(t0, t_launch, t_wait);
        push_tokens(generated, drafts, &chosen);
        Ok(())
    }

    fn forward_sample(
        &mut self,
        plan: &BatchPlan,
        generated: &mut Vec<Vec<u32>>,
        drafts: &mut Vec<Vec<u32>>,
    ) -> Result<()> {
        let t0 = std::time::Instant::now();
        self.mark_gpu_start()?;
        let (n, t_launch) = self.enqueue(plan)?;
        self.mark_gpu_end()?;
        let t_wait = std::time::Instant::now();
        let chosen = self.collect(n)?;
        self.main_done = Some(self.stream.record_event()?);
        self.account(t0, t_launch, t_wait);
        let lps = std::mem::take(&mut self.last_logprobs);
        self.last_logprobs = verify_rows(plan, &chosen, lps, generated, drafts);
        Ok(())
    }

    /// decode 批（主流，走 CUDA graph）与 prefill 批（副流，eager）并行执行，两边各自 D2H。
    /// 同步关系：副流先等上一步主流结束（历史 KV 完整）；主流下一步前等副流（本步 prefill 写的 KV 可见）。
    /// 返回 (decode 的 logprobs, prefill 的 logprobs)。
    fn forward_sample_pair(
        &mut self,
        decode: &BatchPlan,
        prefill: &BatchPlan,
        gen_d: &mut Vec<Vec<u32>>,
        dr_d: &mut Vec<Vec<u32>>,
        gen_p: &mut Vec<Vec<u32>>,
        dr_p: &mut Vec<Vec<u32>>,
    ) -> Result<PairLogprobs> {
        ensure!(self.alt.is_some(), "forward_sample_pair without alt set");
        let t0 = std::time::Instant::now();

        // 1. 主流：decode 入队（不等结果）
        let (num_decode, t_launch) = self.enqueue(decode)?;
        let lp_decode = std::mem::take(&mut self.last_logprobs);

        // 2. 副流：等上一步主流 → prefill 入队 → 记录完成事件 → 取回结果
        if let Some(prev_main) = &self.main_done {
            self.alt.as_ref().unwrap().stream.wait_event(prev_main)?;
        }
        let (chosen_prefill, lp_prefill, prefill_done) = self.on_alt(|ex| {
            let (num_prefill, _) = ex.enqueue(prefill)?;
            let lp = std::mem::take(&mut ex.last_logprobs);
            let done = ex.stream.record_event()?;
            Ok((ex.collect(num_prefill)?, lp, done))
        })?;
        let t_wait = std::time::Instant::now();

        // 3. 主流：等副流（prefill 写的 KV 对后续 decode 可见）→ 取回 decode 结果 → 记录本步结束
        self.stream.wait_event(&prefill_done)?;
        let chosen_decode = self.collect(num_decode)?;
        self.main_done = Some(self.stream.record_event()?);
        self.pair_steps += 1;
        self.account(t0, t_launch, t_wait);

        let lp_decode = verify_rows(decode, &chosen_decode, lp_decode, gen_d, dr_d);
        let lp_prefill = verify_rows(prefill, &chosen_prefill, lp_prefill, gen_p, dr_p);
        Ok((lp_decode, lp_prefill))
    }

    fn take_logprobs(&mut self, num_reqs: usize) -> Vec<Vec<TokenLogprobs>> {
        let mut v = std::mem::take(&mut self.last_logprobs);
        v.resize(num_reqs, Vec::new());
        v
    }
}
