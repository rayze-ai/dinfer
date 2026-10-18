//! `Qwen3Model` 的 `ModelExecutor`：上传 BatchPlan → forward → 采样 → D2H。无投机解码（drafts 为空）。
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
use dinfer_kernels::Graph;
use dinfer_layers::{Gemm, Stream, Tensor, Event, TimingEvent, PinnedUploader, PinnedSlot, HostArena, PinnedPool, PinnedBuf};

/// 固定地址的每 step 输入缓冲。
struct StepBuffers {
    ids: Tensor<i32>,   // [max_tokens]
    pos: Tensor<i32>,   // [max_tokens]
    slots: Tensor<i64>, // [max_tokens]
    rows: Tensor<i32>,  // [max_rows]
    cu: Tensor<i32>,    // [max_rows + 1]
    seq: Tensor<i32>,   // [max_rows]
    bt: Tensor<i32>,    // [max_rows * max_blocks]
    max_blocks: usize,
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
    gather_idx: Tensor<i32>,      // [max_tokens]
    prev_chosen: Tensor<i32>,     // [max_rows]
    lp_ids: Tensor<i32>,          // [max_rows * 33]
    lp_val: Tensor<f32>,
    lp_rank: Tensor<i32>,
    /// 每 step 小输入的打包上传器（pinned 一次 H2D + D2D 散开）。
    up: PinnedUploader,
}

struct DecodeGraph {
    batch: usize,
    graph: Graph,
}

impl StepBuffers {
    fn alloc(s: &Stream, mt: usize, rows: usize, max_blocks: usize) -> Result<Self> {
        let z32 = |n: usize| Tensor::<i32>::zeros(s, &[n.max(1)]);
        let zf = |n: usize| Tensor::<f32>::zeros(s, &[n.max(1)]);
        Ok(StepBuffers {
            ids: z32(mt)?,
            pos: z32(mt)?,
            slots: Tensor::<i64>::zeros(s, &[mt])?,
            rows: z32(rows)?,
            cu: z32(rows + 1)?,
            seq: z32(rows)?,
            bt: z32(rows * max_blocks)?,
            max_blocks,
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
            gather_idx: Tensor::<i32>::from_vec(s, vec![-1; mt.max(1)], &[mt.max(1)])?,
            prev_chosen: z32(rows)?,
            lp_ids: z32(rows * 33)?,
            lp_val: zf(rows * 33)?,
            lp_rank: z32(rows)?,
            up: PinnedUploader::new(s, (mt * 16 + rows * (8 * 4 + 2 * 8) + rows * max_blocks * 4 + (1 << 20)).max(1 << 20))?,
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
    graphs: Vec<DecodeGraph>,
    /// 是否启用 decode 图（懒捕获：graph_for 未命中时按桶捕获）。
    graphs_enabled: bool,
    last_logprobs: Vec<Vec<TokenLogprobs>>,
    last_rows: usize,
    alt: Option<AltSet>,
    /// 当前 plan 是否是插队 prefill（采样后不写 prev_chosen）。enqueue 时从 plan 读入。
    skip_prev_chosen: bool,
    /// KV host tier（design §10.2）：pinned 竞技场 + copy stream；None = 关。
    host_tier: Option<HostTierStore>,
    /// PD 分离（design §10.3 v2）：异步导出 / 导入资源。`enable_pd` 建。
    pd: Option<PdStore>,
    /// 异步调度：两个 pinned 槽轮转（步 n 的 D2H 在槽 n%2；finish n 必在 begin n+2 之前）。
    async_slots: [PinnedSlot; 2],
    async_slot_idx: usize,
    /// 在飞的异步步（最多 2 个：engine 先 begin n+1 再 finish n）：(槽, 行数, t0, t_launch, t_wait)。
    async_inflight: std::collections::VecDeque<(usize, usize, std::time::Instant, std::time::Instant, std::time::Instant)>,
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

/// CUDA graph 的 batch 桶（vLLM 默认 1,2,4,8 再按 8 递增到上限；这里 2 的幂 + 上限，够用且捕获快）。
pub fn graph_buckets(max_rows: usize) -> Vec<usize> {
    let mut v: Vec<usize> =
        [1usize, 2, 4, 8, 16, 32, 64, 128, 256].into_iter().filter(|&b| b < max_rows).collect();
    v.push(max_rows);
    v
}

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
        let buf = StepBuffers::alloc(&s, mt, rows, max_blocks)?;
        let async_slots = [PinnedSlot::new(&s, rows)?, PinnedSlot::new(&s, rows)?];
        let mut ex = Self {
            logits: Tensor::zeros(&s, &[rows, v])?,
            work: Tensor::zeros(&s, &[rows, v])?,
            raw: Tensor::zeros(&s, &[rows, v])?,
            stream: s,
            gemm,
            model,
            buf,
            graphs: Vec::new(),
            graphs_enabled: false,
            last_logprobs: Vec::new(),
            last_rows: 0,
            async_slots,
            async_slot_idx: 0,
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
        ex.graphs_enabled = enable_graphs;
        if enable_graphs && std::env::var("GRAPH_CAPTURE").is_ok_and(|v| v == "eager") {
            ex.capture_decode_graphs()?;
        }
        // 跨流 prefill/decode 并行（Q3-j）：XSTREAM=0 关。副流一套资源在图捕获之后分配。
        if std::env::var("XSTREAM").map(|v| v != "0").unwrap_or(true) {
            let t0 = std::time::Instant::now();
            let s2 = dev.new_stream()?;
            let gemm2 = Gemm::new(&s2).context("cublasLt on alt stream")?;
            let buf2 = StepBuffers::alloc(&s2, mt, rows, max_blocks)?;
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
    fn upload_plan(&mut self, plan: &BatchPlan, pad_to: usize) -> Result<(usize, usize)> {
        let s = &self.stream;
        self.buf.up.reset_step(); // 每 step 的第一次打包：host 区回卷
        let n = plan.num_reqs();
        let t = plan.total_tokens();
        let nb = pad_to.max(n);
        // token 维
        let pad_tok = (nb - n).max(0);
        let tt = t + pad_tok; // decode 时每请求 1 token；非 decode 时 pad_to == n
        let mut ids = plan.input_ids.clone();
        let mut pos = plan.positions.clone();
        let mut slots = plan.slot_mapping.clone();
        ids.resize(tt, 0);
        pos.resize(tt, 0);
        slots.resize(tt, -1);
        self.buf.up.push(s, &ids, &mut self.buf.ids.data)?;
        // 异步调度：在飞行的 token 行由设备 gather；非异步 plan 的 gather_idx 全 -1（图内 kernel 不改 ids）
        let mut gidx = plan.gather_idx.clone();
        gidx.resize(tt, -1);
        self.buf.up.push(s, &gidx, &mut self.buf.gather_idx.data)?;
        self.buf.up.push(s, &pos, &mut self.buf.pos.data)?;
        self.buf.up.push(s, &slots, &mut self.buf.slots.data)?;
        // 行维
        let mut rows = plan.logits_rows.clone();
        for i in n..nb {
            rows.push((t + (i - n)) as i32);
        }
        self.buf.up.push(s, &rows, &mut self.buf.rows.data)?;
        let mut cu = plan.cu_seqlens.clone();
        for i in n..nb {
            cu.push(cu[i] + 1);
        }
        self.buf.up.push(s, &cu, &mut self.buf.cu.data)?;
        let mut seq = plan.seq_lens.clone();
        seq.resize(nb, 1);
        self.buf.up.push(s, &seq, &mut self.buf.seq.data)?;
        // block_table：图里 max_blocks 固定为缓冲宽度（按每行 stride = buf.max_blocks 布局）
        ensure!(
            plan.max_blocks <= self.buf.max_blocks,
            "block_table wider than buffer: {} > {} (max_model_len / block_size)",
            plan.max_blocks,
            self.buf.max_blocks
        );
        let mb = self.buf.max_blocks;
        let mut bt = vec![0i32; nb * mb];
        for r in 0..n {
            for j in 0..plan.max_blocks {
                bt[r * mb + j] = plan.block_table[r * plan.max_blocks + j];
            }
        }
        self.buf.up.push(s, &bt, &mut self.buf.bt.data)?;
        self.buf.up.flush(s)?;
        Ok((tt, nb))
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
        self.buf.up.flush(s)?;
        Ok(())
    }

    // ---------------------------------------------------------------- 设备侧流水（可被 capture）

    /// forward（固定缓冲，显式 t / rows / batch）。
    fn forward_device(&mut self, mode: BatchMode, t: usize, nb: usize, max_seqlen: usize) -> Result<()> {
        // 异步调度：ids[i] = prev_chosen[gather_idx[i]]（gather_idx<0 的行不动）。进图；非异步步全 -1 等于空操作。
        k::row_gather_i32(&self.stream, &self.buf.prev_chosen.data, &self.buf.gather_idx.data, &mut self.buf.ids.data, t)?;
        let meta = AttnMeta {
            mode,
            cu_seqlens: &self.buf.cu,
            seq_lens: &self.buf.seq,
            block_table: &self.buf.bt,
            num_reqs: nb,
            max_blocks: self.buf.max_blocks,
            max_seqlen,
        };
        let f = Forward {
            input_ids: &self.buf.ids,
            positions: &self.buf.pos,
            slot_mapping: &self.buf.slots,
            meta,
            logits_rows: &self.buf.rows,
            t,
            rows: nb,
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
        let StepBuffers { chosen, temperature, min_p, top_k, top_p, temp01, seeds, offsets, .. } = &mut self.buf;
        let mut wv = self.work.data.slice_mut(..nb * v);
        let mut ch = chosen.data.slice_mut(..nb);
        k::sample_fused_f32(
            s, &mut wv, &temperature.data, &min_p.data, &top_k.data, &top_p.data, &temp01.data, &seeds.data,
            &offsets.data, &mut ch, nb, v,
        )?;
        // 异步调度：保存本步采样给下一步 gather（图内；非异步时无害）
        {
            let StepBuffers { chosen, prev_chosen, .. } = &mut self.buf;
            self.stream.dtod_copy_n(&chosen.data, &mut prev_chosen.data, nb)?;
        }
        Ok(())
    }

    // ---------------------------------------------------------------- CUDA graph

    /// 捕获用的假 decode plan（slot=-1 不写 cache、seq_len=1、温度 0）。
    #[allow(clippy::field_reassign_with_default)]
    fn capture_plan(nb: usize) -> BatchPlan {
        let mut p = BatchPlan::default();
        p.req_ids = (0..nb as u64).collect();
        p.num_tokens = vec![1; nb];
        p.input_ids = vec![0; nb];
        p.positions = vec![0; nb];
        p.slot_mapping = vec![-1; nb];
        p.cu_seqlens = (0..=nb as i32).collect();
        p.seq_lens = vec![1; nb];
        p.block_table = vec![0; nb];
        p.max_blocks = 1;
        p.logits_rows = (0..nb as i32).collect();
        p.logits_row_start = (0..=nb).collect();
        p.spec_tokens = vec![Vec::new(); nb];
        p.next_known = vec![-1; nb];
        p.mode = BatchMode::Decode;
        p.sample.clear();
        for _ in 0..nb {
            p.sample.push(&dinfer_core::sample::SamplingParams { temperature: 0.0, ..Default::default() }, 0, &[0], 1);
        }
        p
    }

    /// 捕获一个 batch 桶的 decode 图：先 eager 跑一次预热（cuBLASLt 启发式、模块加载在 capture 外完成），再捕获。
    fn capture_bucket(&mut self, nb: usize) -> Result<()> {
        let t0 = std::time::Instant::now();
        let p = Self::capture_plan(nb);
        self.upload_plan(&p, nb)?;
        self.upload_sampling(&p.sample, nb)?;
        self.forward_device(BatchMode::Decode, nb, nb, 1)?;
        self.sample_device_fixed(nb)?;
        self.stream.synchronize()?;
        self.stream.begin_capture()?;
        let r = self.forward_device(BatchMode::Decode, nb, nb, 1).and_then(|_| self.sample_device_fixed(nb));
        let g = self.stream.end_capture();
        r?;
        let graph = g.with_context(|| format!("capture decode graph for batch {nb}"))?;
        self.graphs.push(DecodeGraph { batch: nb, graph });
        self.graphs.sort_by_key(|g| g.batch);
        self.stream.synchronize()?;
        tracing::info!(bucket = nb, elapsed_ms = t0.elapsed().as_millis() as u64, "cuda graph captured (decode)");
        Ok(())
    }

    /// 启动时一次捕获全部桶（GRAPH_CAPTURE=eager）。
    fn capture_decode_graphs(&mut self) -> Result<()> {
        let t0 = std::time::Instant::now();
        let buckets = graph_buckets(self.model.max_logits_rows);
        for &nb in &buckets {
            self.capture_bucket(nb)?;
        }
        tracing::info!(buckets = ?buckets, elapsed_ms = t0.elapsed().as_millis() as u64, "cuda graphs captured for decode");
        Ok(())
    }

    /// 懒捕获：n 行 decode 需要的桶若还没有图，现在捕获（只在主套、图启用时）。
    fn ensure_graph(&mut self, n: usize) -> Result<()> {
        if !self.graphs_enabled || self.alt_active || self.graph_for(n).is_some() {
            return Ok(());
        }
        let Some(&nb) = graph_buckets(self.model.max_logits_rows).iter().find(|&&b| b >= n) else { return Ok(()) };
        self.capture_bucket(nb)
    }

    fn graph_for(&self, n: usize) -> Option<&DecodeGraph> {
        self.graphs.iter().find(|g| g.batch >= n)
    }

    // ---------------------------------------------------------------- eager 采样（惩罚 / logprobs / 非 decode）

    fn sample_eager(&mut self, sb: &SampleBatch, nrows: usize) -> Result<()> {
        let v = self.model.cfg.vocab_size as usize;
        let fast = !sb.any_random && !sb.any_penalty && sb.max_logprobs.is_none();
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
                }];
            }
        }
        self.save_prev_chosen(nrows)?;
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
        let StepBuffers { chosen, prev_chosen, .. } = &mut self.buf;
        self.stream.dtod_copy_n(&chosen.data, &mut prev_chosen.data, nrows)
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
        let cap = self.buf.max_blocks * per_block_all * world.max(1);
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
        let slot_bytes = self.buf.max_blocks * per_block_all * world.max(1);
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
    /// 把 `ids` 的全层 KV（本 rank 分片）按 `[block][layer]` 连续写到地址 `dst`（host：已注册 pinned / shm；设备：本 ctx 可见）。
    /// 主流 pack → staging（双缓冲）→ copy stream 搬走；返回 copy stream 上"全部写完"的事件。调用方保证 `dst` 容量 ≥ ids.len() × 单元。
    pub fn pd_export_addr(&mut self, ids: &[dinfer_core::kv::BlockId], dst: u64, dst_is_host: bool) -> Result<Event> {
        let pd = self.pd.as_mut().context("pd: export without resources (call enable_pd)")?;
        let layers = self.model.kv.len();
        let per = pd.bytes_per_layer_block;
        let per_all = per * layers;
        for (ci, chunk) in ids.chunks(PD_CHUNK).enumerate() {
            let i = ci & 1;
            // 生产方（主流 pack）写 staging[i] 前，等上一次消费方（copy stream 读走）用完
            if let Some(ev) = pd.staging_free[i].take() {
                self.stream.wait_event(&ev)?;
            }
            let bids: Vec<i32> = chunk.iter().map(|&b| b as i32).collect();
            self.stream.htod_into(&bids, &mut pd.blocks[i].data)?;
            k::kv_pack(&self.stream, &pd.layer_ptrs.data, &pd.blocks[i].data, &mut pd.staging[i].data, chunk.len(), layers, per)?;
            let packed = self.stream.record_event()?;
            pd.copy.wait_event(&packed)?;
            let src = pd.copy.ptr(&pd.staging[i].data);
            let off = (ci * PD_CHUNK * per_all) as u64;
            let n = chunk.len() * per_all;
            if dst_is_host {
                pd.copy.dtoh_to_ptr(dst + off, src, n)?;
            } else {
                pd.copy.dtod(dst + off, src, n)?;
            }
            pd.staging_free[i] = Some(pd.copy.record_event()?);
        }
        pd.copy.record_event()
    }
    /// `pd_export_addr` 的反向：从地址 `src`（同布局）读回 `ids` 的全层 KV 分片。copy stream 搬进 staging → 主流 unpack；
    /// 返回 copy stream 上"源已全部读完"的事件（之后源缓冲 / 槽才可释放）。主流后续 forward 自然排在 unpack 之后。
    pub fn pd_import_addr(&mut self, ids: &[dinfer_core::kv::BlockId], src: u64, src_is_host: bool) -> Result<Event> {
        let pd = self.pd.as_mut().context("pd: import without resources (call enable_pd)")?;
        let layers = self.model.kv.len();
        let per = pd.bytes_per_layer_block;
        let per_all = per * layers;
        for (ci, chunk) in ids.chunks(PD_CHUNK).enumerate() {
            let i = ci & 1;
            // 生产方（copy 写 staging）前，等上一次消费方（主流 unpack）用完
            if let Some(ev) = pd.staging_free[i].take() {
                pd.copy.wait_event(&ev)?;
            }
            let dst = pd.copy.ptr(&pd.staging[i].data);
            let off = (ci * PD_CHUNK * per_all) as u64;
            let n = chunk.len() * per_all;
            if src_is_host {
                pd.copy.htod_from_ptr(dst, src + off, n)?;
            } else {
                pd.copy.dtod(dst, src + off, n)?;
            }
            let h2d = pd.copy.record_event()?;
            self.stream.wait_event(&h2d)?;
            let bids: Vec<i32> = chunk.iter().map(|&b| b as i32).collect();
            self.stream.htod_into(&bids, &mut pd.blocks[i].data)?;
            k::kv_unpack(&self.stream, &pd.layer_ptrs.data, &pd.blocks[i].data, &pd.staging[i].data, chunk.len(), layers, per)?;
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
        ensure!(
            plan.spec_tokens.iter().all(|d| d.is_empty()),
            "qwen3 executor: speculative decoding not supported yet"
        );
        let nrows = plan.logits_rows.len();
        ensure!(
            nrows <= self.model.max_logits_rows,
            "{nrows} logits rows > max_logits_rows {}",
            self.model.max_logits_rows
        );
        ensure!(
            nrows == n && plan.sample.num_rows() == n,
            "qwen3 executor: rows {nrows} / sample {} vs reqs {n}",
            plan.sample.num_rows()
        );
        let sb = &plan.sample;
        let graph_eligible = plan.mode == BatchMode::Decode && !sb.any_penalty && sb.max_logprobs.is_none();
        if graph_eligible {
            self.ensure_graph(n)?;
        }
        let use_graph = graph_eligible
            && !self.alt_active // 图是在主流上捕获的，副套不走图
            && self.graph_for(n).is_some();
        let t_launch;
        if use_graph {
            let nb = self.graph_for(n).unwrap().batch;
            let (tt, nb2) = self.upload_plan(plan, nb)?;
            debug_assert_eq!((tt, nb2), (nb, nb));
            let t_us = std::time::Instant::now();
            self.upload_sampling(sb, nb)?;
            self.t_upload_sampling_us += t_us.elapsed().as_micros() as u64;
            t_launch = std::time::Instant::now();
            let g = self.graph_for(n).unwrap();
            g.graph.launch().context("launch decode graph")?;
            self.t_graph_launch_us += t_launch.elapsed().as_micros() as u64;
            self.graph_steps += 1;
            self.last_logprobs = vec![Vec::new(); n];
        } else {
            let (tt, nb) = self.upload_plan(plan, n)?;
            debug_assert_eq!((tt, nb), (t, n));
            t_launch = std::time::Instant::now();
            let max_seqlen = plan.num_tokens.iter().copied().max().unwrap_or(0) as usize;
            self.forward_device(plan.mode, t, n, max_seqlen)?;
            self.sample_eager(sb, n)?;
            self.eager_steps += 1;
        }
        self.last_rows = n;
        Ok((n, t_launch))
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
fn push_tokens(generated: &mut Vec<Vec<u32>>, drafts: &mut Vec<Vec<u32>>, chosen: &[i32]) {
    for &c in chosen {
        generated.push(vec![c as u32]);
        drafts.push(Vec::new());
    }
}

impl ModelExecutor for Qwen3Executor {
    fn supports_pair(&self) -> bool {
        self.alt.is_some()
    }

    fn supports_async(&self) -> bool {
        std::env::var("ASYNC_SCHED").map(|v| v != "0").unwrap_or(true)
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
        let ht = self.host_tier.as_mut().unwrap();
        for chunk in orders.chunks(ht.chunk) {
            let base = self.stream.ptr_mut(&mut ht.staging.data);
            for (i, &(_, slot)) in chunk.iter().enumerate() {
                self.stream.htod_from_arena(&ht.arena, slot as usize, 0, base + (i * per_all) as u64, per_all)?;
            }
            let ids: Vec<i32> = chunk.iter().map(|&(b, _)| b as i32).collect();
            self.stream.htod_into(&ids, &mut ht.blocks_buf.data)?;
            k::kv_unpack(&self.stream, &ht.layer_ptrs.data, &ht.blocks_buf.data, &ht.staging.data, chunk.len(), layers, per)?;
        }
        Ok(())
    }

    /// GPU → host：copy stream 等主流（本步 KV 写完）后异步 D2H；记录完成事件供下次 fetch 等待。
    fn kv_offload(&mut self, orders: &[(dinfer_core::kv::BlockId, u32)]) -> Result<()> {
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
            k::kv_pack(&ht.copy, &ht.layer_ptrs.data, &ht.blocks_off.data, &mut ht.staging_off.data, chunk.len(), layers, per)?;
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
        self.kv_block_bytes_all_layers().1
    }
    fn kv_host_pool(&self) -> Option<std::sync::Arc<dyn dinfer_core::pd::HostBufPool>> {
        self.pd.as_ref().map(|p| std::sync::Arc::new(PdPool(p.pool.clone())) as std::sync::Arc<dyn dinfer_core::pd::HostBufPool>)
    }
    /// P 侧。块已被 scheduler free 但未复用。主流：pack chunk → 记事件；copy stream：等事件 → D2H 进 pinned。
    /// 主流的后续 forward 排在 pack 之后，所以块复用安全；D2H 与 forward 重叠。返回句柄，发送线程等 `done`。
    fn kv_export(&mut self, blocks: &dinfer_core::kv::KvBlocks) -> Result<Box<dyn dinfer_core::pd::KvExportHandle>> {
        let ids = blocks.group(0);
        let total = ids.len() * self.pd_unit_bytes_local();
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
        let per_all = self.pd_unit_bytes_local();
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
        let per_block_all = self.pd_unit_bytes_local();
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
        let total = ids.len() * self.pd_unit_bytes_local();
        ensure!(total <= r.slot_bytes(), "pd: export {total} B > slot {}", r.slot_bytes());
        let dst = r.slot_addr(slot as usize)?;
        let done = self.pd_export_addr(ids, dst, r.is_host())?;
        Ok(Box::new(PdSlotExport { done, _region: r }))
    }
    /// 异步第一段：入队 + 异步 D2H（不等 GPU）。runner 只对 `async_eligible` 的 plan 调用（纯 decode）。
    fn begin(&mut self, plan: &BatchPlan) -> Result<()> {
        ensure!(self.async_inflight.len() < 2, "executor begin: two async steps already in flight");
        let t0 = std::time::Instant::now();
        let (n, t_launch) = self.enqueue(plan)?;
        let t_wait = std::time::Instant::now();
        let slot = self.async_slot_idx;
        self.async_slots[slot].enqueue_d2h(&self.stream, &self.buf.chosen.data, n)?;
        self.async_slot_idx ^= 1;
        self.async_inflight.push_back((slot, n, t0, t_launch, t_wait));
        Ok(())
    }

    fn finish(&mut self, generated: &mut Vec<Vec<u32>>, drafts: &mut Vec<Vec<u32>>) -> Result<()> {
        let (slot, n, t0, t_launch, t_wait) =
            self.async_inflight.pop_front().context("executor finish without begin")?;
        let chosen = self.async_slots[slot].wait_read()?;
        ensure!(chosen.len() == n, "async finish: {} values for {n} rows", chosen.len());
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
        push_tokens(generated, drafts, &chosen);
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

        push_tokens(gen_d, dr_d, &chosen_decode);
        push_tokens(gen_p, dr_p, &chosen_prefill);
        Ok((lp_decode, lp_prefill))
    }

    fn take_logprobs(&mut self, num_reqs: usize) -> Vec<Vec<TokenLogprobs>> {
        let mut v = std::mem::take(&mut self.last_logprobs);
        v.resize(num_reqs, Vec::new());
        v
    }
}
