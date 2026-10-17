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
use dinfer_layers::{Gemm, Stream, Tensor, Event, TimingEvent, PinnedUploader};

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
        // capture 要求非默认流；cuBLASLt 句柄绑流，所以在新流上重建。权重 / scratch 在旧流上分配完并已同步。
        stream.synchronize()?;
        let dev = stream.device();
        dev.disable_event_tracking();
        let s = dev.new_stream()?;
        let gemm = Gemm::new(&s).context("cublasLt on executor stream")?;
        let v = model.cfg.vocab_size as usize;
        let rows = model.max_logits_rows;
        let mt = model.max_tokens;
        let max_blocks = model.kv.first().map_or(1, |kv| kv.num_blocks).max(1);
        let buf = StepBuffers::alloc(&s, mt, rows, max_blocks)?;
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
            alt: None,
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
        self.buf.up.begin();
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
        ensure!(plan.max_blocks <= self.buf.max_blocks, "block_table wider than buffer");
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
        s.dtod_copy_n(&self.logits.data, &mut self.work.data, nb * v)?;
        let b = &self.buf;
        let mut wv = self.work.data.slice_mut(..nb * v);
        k::temperature_f32(s, &mut wv, &b.temperature.data.slice(..nb), v)?;
        k::min_p_mask_f32(s, &mut wv, &b.min_p.data.slice(..nb), v)?;
        k::topk_topp_mask_f32(s, &mut wv, &b.top_k.data.slice(..nb), &b.top_p.data.slice(..nb), v)?;
        let wv_ro = self.work.data.slice(..nb * v);
        let StepBuffers { chosen, temp01, seeds, offsets, .. } = &mut self.buf;
        let mut ch = chosen.data.slice_mut(..nb);
        k::sample_f32(
            s,
            &wv_ro,
            &temp01.data.slice(..nb),
            &seeds.data.slice(..nb),
            &offsets.data.slice(..nb),
            &mut ch,
            v,
        )?;
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

    /// GPU 忙时测量：主流上 step 工作入队前 / 后各记一个带时间戳的事件；collect 同步后读差值。
    fn mark_gpu_start(&mut self) -> Result<()> {
        if self.ev_gpu.is_none() {
            self.ev_gpu = Some((self.stream.new_timing_event()?, self.stream.new_timing_event()?));
        }
        self.stream.record_timing(&self.ev_gpu.as_ref().unwrap().0)
    }
    fn mark_gpu_end(&mut self) -> Result<()> {
        self.stream.record_timing(&self.ev_gpu.as_ref().unwrap().1)
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
