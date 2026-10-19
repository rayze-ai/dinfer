//! `Qwen35Model` 的 `ModelExecutor`：上传 BatchPlan → forward → 采样 → D2H。
//!
//! 单流；**decode 步走 CUDA graph**（里程碑 6 第 2 轮，照抄 Qwen3 的桶捕获）：`plan.mode == Decode` 且无惩罚 /
//! logprobs / mask 的步，按 batch 桶（`graph_buckets`）懒捕获一次，之后只换固定地址缓冲的内容再 `Graph::launch`。
//! 图内 = `forward_device`（64 层）+ `sample_device_fixed`（融合采样 kernel，温度 / top-k / top-p 都是设备缓冲参数）；
//! 其余步（prefill / 混批 / 惩罚 / logprobs）eager。还没有跨流 pair / KV host tier / PD 分离 / TP 通信。
//! 采样流水与 Qwen3 同序：penalties → temperature → min_p → top_k/top_p → 多项式采样（greedy 行 argmax）→ logprobs。
//!
//! GDN 状态与图：recurrent / conv1d_update kernel 对每一行都读写 `state[state_indices[i]]`，所以捕获用的假 plan
//! 和 padding 行都指向**废纸篓 slot**（`model.num_slots`，GdnState 多分的那一个），不会碰活请求；`reset_slots`
//! （dtod 清零）在图外、重放之前做，同流有序。
//!
//! GDN 状态：每步按 `RequestId` 分 slot（`SlotTable`）；**本步首个 token 位置为 0 的请求**当作从头开始
//! （新请求 / 被抢占后重算），清零它的 conv + 递推状态，`has_initial = 0`；其余 `has_initial = 1`。
use crate::gdn::SlotTable;
use crate::model::{Forward, ModelScratch, Qwen35Model};
use crate::layer::StepMeta;
use anyhow::{ensure, Context, Result};
use dinfer_core::runner::{BatchMode, BatchPlan, ModelExecutor};
use dinfer_core::sampler::{SampleBatch, TokenLogprobs};
use dinfer_kernels as k;
use dinfer_kernels::Graph;
use dinfer_layers::{Gemm, PinnedUploader, Stream, Tensor};

/// CUDA graph 的 batch 桶（与 Qwen3 同：2 的幂 + 上限）。
pub fn graph_buckets(max_rows: usize) -> Vec<usize> {
    let mut v: Vec<usize> = [1usize, 2, 4, 8, 16, 32, 64, 128, 256].into_iter().filter(|&b| b < max_rows).collect();
    v.push(max_rows);
    v
}

struct DecodeGraph {
    batch: usize,
    graph: Graph,
}

/// 固定地址的每 step 输入缓冲。
struct StepBuffers {
    ids: Tensor<i32>,   // [max_tokens]
    pos: Tensor<i32>,   // [max_tokens]
    slots: Tensor<i64>, // [max_tokens]
    rows: Tensor<i32>,  // [max_rows]
    cu: Tensor<i32>,    // [max_rows + 1]
    seq: Tensor<i32>,   // [max_rows]
    bt: Tensor<i32>,    // [max_rows * max_blocks]
    /// GDN：每请求的 slot 与"是否有历史状态"。
    state_idx: Tensor<i32>, // [max_rows]
    has_init: Tensor<i32>,  // [max_rows]
    max_blocks: usize,
    // 采样列
    temperature: Tensor<f32>,
    temp01: Tensor<f32>,
    top_k: Tensor<i32>,
    top_p: Tensor<f32>,
    min_p: Tensor<f32>,
    rep: Tensor<f32>,
    freq: Tensor<f32>,
    pres: Tensor<f32>,
    seeds: Tensor<u64>,
    offsets: Tensor<u64>,
    hist_offsets: Tensor<i32>,
    hist_prompt_len: Tensor<i32>,
    hist_ids: Tensor<i32>,
    chosen: Tensor<i32>, // [max_rows]
    lp_ids: Tensor<i32>,
    lp_val: Tensor<f32>,
    lp_rank: Tensor<i32>,
    up: PinnedUploader,
}

impl StepBuffers {
    fn alloc(s: &Stream, mt: usize, rows: usize, max_blocks: usize) -> Result<Self> {
        let z32 = |n: usize| Tensor::<i32>::zeros(s, &[n.max(1)]);
        let zf = |n: usize| Tensor::<f32>::zeros(s, &[n.max(1)]);
        Ok(Self {
            ids: z32(mt)?,
            pos: z32(mt)?,
            slots: Tensor::<i64>::zeros(s, &[mt.max(1)])?,
            rows: z32(rows)?,
            cu: z32(rows + 1)?,
            seq: z32(rows)?,
            bt: z32(rows * max_blocks)?,
            state_idx: z32(rows)?,
            has_init: z32(rows)?,
            max_blocks,
            temperature: zf(rows)?,
            temp01: zf(rows)?,
            top_k: z32(rows)?,
            top_p: zf(rows)?,
            min_p: zf(rows)?,
            rep: zf(rows)?,
            freq: zf(rows)?,
            pres: zf(rows)?,
            seeds: Tensor::<u64>::zeros(s, &[rows.max(1)])?,
            offsets: Tensor::<u64>::zeros(s, &[rows.max(1)])?,
            hist_offsets: z32(rows + 1)?,
            hist_prompt_len: z32(rows)?,
            hist_ids: z32(1024)?,
            chosen: z32(rows)?,
            lp_ids: z32(rows * 33)?,
            lp_val: zf(rows * 33)?,
            lp_rank: z32(rows)?,
            up: PinnedUploader::new(s, (mt * 16 + rows * 128 + rows * max_blocks * 4 + (1 << 20)).max(1 << 20))?,
        })
    }
}

pub struct Qwen35Executor {
    pub stream: Stream,
    pub gemm: Gemm,
    pub model: Qwen35Model,
    pub slots: SlotTable,
    logits: Tensor<f32>,
    work: Tensor<f32>,
    raw: Tensor<f32>,
    buf: StepBuffers,
    last_logprobs: Vec<Vec<TokenLogprobs>>,
    last_rows: usize,
    pub steps: u64,
    /// 跨流并行用的第二套激活缓冲（第二版接 `forward_sample_pair`）。
    pub alt: Option<ModelScratch>,
    /// GDN chunk prefill 的共享工作区（48 层共用；按需增长）。`GDN_PREFILL=recurrent` 时关闭。
    gdn_chunk: Option<crate::gdn::GdnChunkWorkspace>,
    gdn_chunk_enabled: bool,
    /// decode 图（按 batch 桶，懒捕获）。
    graphs: Vec<DecodeGraph>,
    graphs_enabled: bool,
    pub graph_steps: u64,
    pub eager_steps: u64,
}

impl Qwen35Executor {
    /// 默认开 decode 图（`NO_GRAPH` 环境变量关）。
    pub fn new(stream: Stream, gemm: Gemm, model: Qwen35Model, max_blocks: usize) -> Result<Self> {
        Self::with_graphs(stream, gemm, model, max_blocks, true)
    }

    /// `enable_graphs`：decode 步是否走 CUDA graph。默认按桶懒捕获；`GRAPH_CAPTURE=eager` 启动时全捕获；
    /// `NO_GRAPH` 环境变量强制关（测试两条路径用）。
    pub fn with_graphs(stream: Stream, gemm: Gemm, model: Qwen35Model, max_blocks: usize, enable_graphs: bool) -> Result<Self> {
        // 同 Qwen3：capture 要求非默认流；cudarc 的自动跨流事件跟踪在 capture 期间会被驱动拒绝
        //（CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED，首个 kernel 就报），关掉——全程单流顺序执行。
        // cuBLASLt 句柄绑流，在新流上重建；权重 / scratch 已在旧流上分配并同步完。
        let _ = gemm;
        stream.synchronize()?;
        let dev = stream.device();
        dev.disable_event_tracking();
        let stream = dev.new_stream()?;
        let gemm = Gemm::new(&stream).context("cublasLt on executor stream")?;
        // GEMM 自调优（docs/loops/qwen35-perf.md 第 3 轮）：模型加载后、graph 捕获前，对每个 (N,K) × M 桶实测
        // cuBLASLt 候选 / small_m kernel 取最快。`GEMM_TUNE=off` 关。耗时与每形状选择进启动日志。
        if gemm.tune_enabled() {
            let report = gemm.autotune(&model.gemm_tune_entries()).context("gemm autotune")?;
            for line in report.summary_lines() {
                tracing::info!("gemm tune: {line}");
            }
            tracing::info!("gemm autotune: {:.0} ms, partial workspace {:.1} MB", report.elapsed_ms, report.partial_bytes as f64 / 1048576.0);
        } else {
            tracing::info!("gemm autotune: off (GEMM_TUNE=off)");
        }
        let (mt, rows, v) = (model.max_tokens, model.max_logits_rows, model.cfg.vocab_size as usize);
        let slots = SlotTable::new(model.num_slots);
        let buf = StepBuffers::alloc(&stream, mt, rows, max_blocks)?;
        let logits = Tensor::zeros(&stream, &[rows * v])?;
        let work = Tensor::zeros(&stream, &[rows * v])?;
        let raw = Tensor::zeros(&stream, &[rows * v])?;
        let mut ex = Self {
            stream,
            gemm,
            model,
            slots,
            logits,
            work,
            raw,
            buf,
            last_logprobs: Vec::new(),
            last_rows: 0,
            steps: 0,
            alt: None,
            gdn_chunk: None,
            gdn_chunk_enabled: !matches!(std::env::var("GDN_PREFILL").as_deref(), Ok("recurrent")),
            graphs: Vec::new(),
            graphs_enabled: enable_graphs && std::env::var("NO_GRAPH").is_err(),
            graph_steps: 0,
            eager_steps: 0,
        };
        if ex.graphs_enabled && std::env::var("GRAPH_CAPTURE").is_ok_and(|v| v == "eager") {
            ex.capture_decode_graphs()?;
        }
        Ok(ex)
    }

    /// GDN 状态的废纸篓 slot（GdnState 比 `num_slots` 多分的那一个）。
    fn scratch_slot(&self) -> i32 {
        self.model.num_slots as i32
    }

    /// 上传 plan 并执行一步（不取结果）。
    fn enqueue(&mut self, plan: &BatchPlan) -> Result<usize> {
        let n = plan.num_reqs();
        let t = plan.total_tokens();
        let nrows = plan.logits_rows.len();
        ensure!(t <= self.model.max_tokens, "batch of {t} tokens > max_tokens {}", self.model.max_tokens);
        ensure!(nrows <= self.model.max_logits_rows, "rows {nrows} > max_logits_rows");
        ensure!(nrows >= n && plan.sample.num_rows() == nrows, "rows {nrows} / sample {}", plan.sample.num_rows());
        ensure!(plan.max_blocks <= self.buf.max_blocks, "block_table {} > buffer {}", plan.max_blocks, self.buf.max_blocks);
        ensure!(!plan.sample.any_mask, "qwen35 executor: guided decoding 未接（里程碑 4 不做）");

        if self.model.trace {
            tracing::info!(?plan.mode, ?plan.req_ids, ?plan.num_tokens, ?plan.cu_seqlens, ?plan.positions, ?plan.seq_lens, ?plan.logits_rows, "qwen35 plan");
        }
        // GDN slot：分配 + 判定"从头开始"的请求
        let req_ids: Vec<u64> = plan.req_ids.clone();
        self.slots.release(&plan.finished_req_ids);
        let fresh_slots = self.slots.begin_step(&req_ids)?;
        let state_idx = self.slots.indices(&req_ids);
        // 首个 token 位置 == 0 → 从头开始（新请求 / 被抢占后的重算），状态清零。
        // 分到全新 slot 的请求也必须清（旧占用者的状态不能用）；若它带着历史（position > 0）说明中间被回收过，
        // 状态已丢失 → 打 warn（结果会错，只能提高 max_num_seqs 避免）。
        let mut has_init = vec![1i32; n];
        let mut restarts = Vec::new();
        for i in 0..n {
            let p0 = plan.cu_seqlens[i] as usize;
            let pos0 = plan.positions.get(p0).copied().unwrap_or(0);
            let fresh = fresh_slots.contains(&(state_idx[i] as usize));
            if pos0 == 0 || fresh {
                if fresh && pos0 != 0 {
                    tracing::warn!(
                        req = req_ids[i],
                        pos = pos0,
                        slot = state_idx[i],
                        "qwen35: GDN 状态 slot 被复用给带历史的请求，状态已丢失（提高 max_num_seqs / 调大 slot 表）"
                    );
                }
                has_init[i] = 0;
                restarts.push(state_idx[i] as usize);
            }
        }
        if !restarts.is_empty() {
            // 图外、重放之前（同流有序）：清零从头开始的请求的 GDN 状态
            self.model.reset_slots(&self.stream, &restarts)?;
        }

        // 本步可否走图：纯 decode（每请求 1 token、1 行）且采样只需固定形状 kernel（惩罚 / logprobs 要 D2H → eager）
        let sb = &plan.sample;
        let graph_eligible = plan.mode == BatchMode::Decode
            && t == n
            && nrows == n
            && !sb.any_penalty
            && sb.max_logprobs.is_none()
            && !sb.any_mask;
        if graph_eligible {
            self.ensure_graph(n)?;
        }
        let use_graph = graph_eligible && self.graph_for(n).is_some();
        if use_graph {
            let nb = self.graph_for(n).unwrap().batch;
            self.upload_plan(plan, &state_idx, &has_init, nb)?;
            self.upload_sampling(sb, nb)?;
            self.graph_for(n).unwrap().graph.launch().context("launch decode graph")?;
            self.last_logprobs = vec![Vec::new(); n];
            if self.graph_steps == 0 || self.model.trace {
                tracing::info!(n, bucket = nb, "qwen35: decode step via cuda graph");
            }
            self.graph_steps += 1;
        } else {
            self.upload_plan(plan, &state_idx, &has_init, n)?;
            let max_seqlen = plan.num_tokens.iter().copied().max().unwrap_or(0).max(1) as usize;
            self.forward_device(plan, &req_ids, t, n, nrows, max_seqlen)?;
            self.sample_eager(sb, nrows)?;
            if self.model.trace {
                tracing::info!(?plan.mode, t, n, nrows, "qwen35: step eager");
            }
            self.eager_steps += 1;
        }
        self.last_rows = nrows;
        self.steps += 1;
        Ok(nrows)
    }

    /// 把 plan 写进固定地址缓冲；batch 维补到 `nb`（padding 行：token 0、pos 0、slot=-1 不写 KV、seq_len=1、
    /// block 0、GDN slot = 废纸篓）。非图路径 `nb == n`。
    fn upload_plan(&mut self, plan: &BatchPlan, state_idx: &[i32], has_init: &[i32], nb: usize) -> Result<()> {
        let s = &self.stream;
        let n = plan.num_reqs();
        let t = plan.total_tokens();
        let pad = nb - n;
        let tt = t + pad;
        self.buf.up.reset_step();
        let mut ids = plan.input_ids.clone();
        let mut pos = plan.positions.clone();
        let mut slots = plan.slot_mapping.clone();
        ids.resize(tt, 0);
        pos.resize(tt, 0);
        slots.resize(tt, -1);
        self.buf.up.push(s, &ids, &mut self.buf.ids.data)?;
        self.buf.up.push(s, &pos, &mut self.buf.pos.data)?;
        self.buf.up.push(s, &slots, &mut self.buf.slots.data)?;
        let mut rows = plan.logits_rows.clone();
        for i in 0..pad {
            rows.push((t + i) as i32);
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
        let mb = self.buf.max_blocks;
        let mut bt = vec![0i32; nb * mb];
        for r in 0..n {
            for j in 0..plan.max_blocks {
                bt[r * mb + j] = plan.block_table[r * plan.max_blocks + j];
            }
        }
        self.buf.up.push(s, &bt, &mut self.buf.bt.data)?;
        let mut si = state_idx.to_vec();
        si.resize(nb, self.scratch_slot());
        self.buf.up.push(s, &si, &mut self.buf.state_idx.data)?;
        let mut hi = has_init.to_vec();
        hi.resize(nb, 1);
        self.buf.up.push(s, &hi, &mut self.buf.has_init.data)?;
        self.buf.up.flush(s)?;
        Ok(())
    }

    /// forward（固定缓冲、显式形状）。decode 纯批时可被 capture：期间不分配 / 不 D2H / 不同步（trace 关掉）。
    fn forward_device(&mut self, plan: &BatchPlan, req_ids: &[u64], t: usize, n: usize, nrows: usize, max_seqlen: usize) -> Result<()> {
        let s = &self.stream;
        // GDN chunk prefill 工作区：只有 prefill 步（有请求本步 > 1 token）才建表；GDN_PREFILL=recurrent 回退。
        let gdn_chunk = if t > n && self.gdn_chunk_enabled {
            let mut ws = self.gdn_chunk.take().unwrap_or_default();
            ws.prepare(s, &plan.cu_seqlens, &self.model.cfg)?;
            Some(std::cell::RefCell::new(ws))
        } else {
            None
        };
        let meta = StepMeta {
            mode: plan.mode,
            cu_seqlens: &self.buf.cu,
            seq_lens: &self.buf.seq,
            block_table: &self.buf.bt,
            state_indices: &self.buf.state_idx,
            has_initial: &self.buf.has_init,
            num_reqs: n,
            max_blocks: self.buf.max_blocks,
            max_seqlen,
            gdn_chunk: gdn_chunk.as_ref(),
        };
        let f = Forward {
            input_ids: &self.buf.ids,
            positions: &self.buf.pos,
            slot_mapping: &self.buf.slots,
            meta,
            logits_rows: &self.buf.rows,
            t,
            rows: nrows,
            req_ids,
        };
        self.model.forward(s, &self.gemm, &f, &mut self.logits)?;
        if let Some(ws) = gdn_chunk {
            self.gdn_chunk = Some(ws.into_inner());
        }
        Ok(())
    }

    /// 图内采样：logits → work，再一个融合 kernel（温度 / min_p / top-k / top-p / 多项式；温度 0 行 = argmax）。
    /// 全按 nb 行、固定地址，参数来自 `upload_sampling` 的设备缓冲。
    fn sample_device_fixed(&mut self, nb: usize) -> Result<()> {
        let s = &self.stream;
        let v = self.model.cfg.vocab_size as usize;
        s.dtod_copy_n(&self.logits.data, &mut self.work.data, nb * v)?;
        let StepBuffers { chosen, temperature, min_p, top_k, top_p, temp01, seeds, offsets, .. } = &mut self.buf;
        let mut wv = self.work.data.slice_mut(..nb * v);
        let mut ch = chosen.data.slice_mut(..nb);
        k::sample_fused_f32(
            s, &mut wv, &temperature.data, &min_p.data, &top_k.data, &top_p.data, &temp01.data, &seeds.data,
            &offsets.data, &mut ch, nb, v,
        )
    }

    // ---------------------------------------------------------------- CUDA graph

    /// 捕获用的假 decode plan（slot=-1 不写 KV、seq_len=1、温度 0；GDN slot 全部指废纸篓）。
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

    /// 捕获一个 batch 桶的 decode 图：先 eager 预热一次（cuBLASLt 启发式 / 模块加载在 capture 外），再捕获。
    fn capture_bucket(&mut self, nb: usize) -> Result<()> {
        let t0 = std::time::Instant::now();
        let p = Self::capture_plan(nb);
        let si = vec![self.scratch_slot(); nb];
        let hi = vec![1i32; nb];
        let trace = std::mem::replace(&mut self.model.trace, false);
        let r = (|| -> Result<Graph> {
            self.upload_plan(&p, &si, &hi, nb)?;
            self.upload_sampling(&p.sample, nb)?;
            self.forward_device(&p, &p.req_ids, nb, nb, nb, 1)?;
            self.sample_device_fixed(nb)?;
            self.stream.synchronize()?;
            self.stream.begin_capture()?;
            let r = self
                .forward_device(&p, &p.req_ids, nb, nb, nb, 1)
                .context("capture: forward")
                .and_then(|_| self.sample_device_fixed(nb).context("capture: sample"));
            let g = self.stream.end_capture();
            r?;
            g.with_context(|| format!("capture decode graph for batch {nb}"))
        })();
        self.model.trace = trace;
        let graph = r?;
        self.graphs.push(DecodeGraph { batch: nb, graph });
        self.graphs.sort_by_key(|g| g.batch);
        self.stream.synchronize()?;
        tracing::info!(bucket = nb, elapsed_ms = t0.elapsed().as_millis() as u64, "qwen35: cuda graph captured (decode)");
        Ok(())
    }

    /// 启动时一次捕获全部桶（GRAPH_CAPTURE=eager）。
    fn capture_decode_graphs(&mut self) -> Result<()> {
        let t0 = std::time::Instant::now();
        let buckets = graph_buckets(self.model.max_logits_rows);
        for &nb in &buckets {
            self.capture_bucket(nb)?;
        }
        tracing::info!(buckets = ?buckets, elapsed_ms = t0.elapsed().as_millis() as u64, "qwen35: cuda graphs captured for decode");
        Ok(())
    }

    /// 懒捕获：n 行 decode 需要的桶若还没有图，现在捕获。
    fn ensure_graph(&mut self, n: usize) -> Result<()> {
        if !self.graphs_enabled || self.graph_for(n).is_some() {
            return Ok(());
        }
        let Some(&nb) = graph_buckets(self.model.max_logits_rows).iter().find(|&&b| b >= n) else { return Ok(()) };
        self.capture_bucket(nb)
    }

    fn graph_for(&self, n: usize) -> Option<&DecodeGraph> {
        self.graphs.iter().find(|g| g.batch >= n)
    }

    fn collect(&mut self, n: usize) -> Result<Vec<i32>> {
        self.stream.dtoh_sync_copy_view(&self.buf.chosen.data.slice(..n))
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
        let t01: Vec<f32> = (0..nb).map(|i| if i < n && sb.temperature[i] > 1e-5 { 1.0 } else { 0.0 }).collect();
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
                ho.push(*ho.last().unwrap_or(&0));
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

    /// 采样流水（与 Qwen3 `sample_eager` 同序；不含 guided mask / prompt_logprobs）。
    fn sample_eager(&mut self, sb: &SampleBatch, nrows: usize) -> Result<()> {
        let v = self.model.cfg.vocab_size as usize;
        let fast = !sb.any_random && !sb.any_penalty && sb.max_logprobs.is_none();
        if fast {
            let lg = self.logits.data.slice(..nrows * v);
            let mut ch = self.buf.chosen.data.slice_mut(..nrows);
            k::argmax_f32_n(&self.stream, &lg, &mut ch, nrows, v)?;
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
                k::topk_topp_mask_f32(s, &mut wv, &b.top_k.data.slice(..nrows), &b.top_p.data.slice(..nrows), v)?;
            }
        }
        {
            let work = &self.work;
            let b = &mut self.buf;
            let wv = work.data.slice(..nrows * v);
            let mut ch = b.chosen.data.slice_mut(..nrows);
            if sb.any_random {
                k::sample_f32(
                    s,
                    &wv,
                    &b.temp01.data.slice(..nrows),
                    &b.seeds.data.slice(..nrows),
                    &b.offsets.data.slice(..nrows),
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
        Ok(())
    }
}

impl ModelExecutor for Qwen35Executor {
    fn forward_sample(
        &mut self,
        plan: &BatchPlan,
        generated: &mut Vec<Vec<u32>>,
        drafts: &mut Vec<Vec<u32>>,
    ) -> Result<()> {
        let t0 = std::time::Instant::now();
        // 引擎线程 != 构建线程：cudarc 的流/上下文对象不会自动把 ctx 绑到新线程，首个 kernel/cuBLASLt 调用会报
        // CUDA_ERROR_INVALID_CONTEXT（B300-50 真权重首请求复现；qwen3 走 record_timing 顺带绑上）。每步绑一次很便宜。
        self.stream.bind_ctx()?;
        let nrows = self.enqueue(plan).context("qwen35 enqueue")?;
        let chosen = self.collect(nrows)?;
        push_tokens(generated, drafts, &chosen);
        tracing::trace!(rows = nrows, ms = t0.elapsed().as_millis() as u64, "qwen35 step");
        Ok(())
    }

    fn take_logprobs(&mut self, num_reqs: usize) -> Vec<Vec<TokenLogprobs>> {
        let mut v = std::mem::take(&mut self.last_logprobs);
        v.resize(num_reqs, Vec::new());
        v
    }
}

/// 采样结果 → 每请求一个 `generated`（无 draft：长度 1）。
fn push_tokens(generated: &mut Vec<Vec<u32>>, drafts: &mut Vec<Vec<u32>>, chosen: &[i32]) {
    generated.clear();
    drafts.clear();
    for &c in chosen {
        generated.push(vec![c as u32]);
        drafts.push(Vec::new());
    }
}
