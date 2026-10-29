//! `KaonModel` 的 `ModelExecutor` 实现。
//!
//! 照 glm53 / qwen35 的 executor：每步把 `BatchPlan` 上传成设备张量 → `KaonModel::forward`（全模型前向，
//! 输出 `[tokens, vocab]` bf16 logits，softcap 已施加）→ gather 要取的 logits 行 → bf16→f32 → argmax → D2H。
//! kaon3 的 KV cache 由模型内部逐层按 `head_dim` 分配（`alloc_kv`），scheduler 只管逻辑块号；所有层共用同一套
//! block_size / num_blocks，block_table 的物理块号直接喂进每层的 paged attention。
//!
//! Gemma4 无递归状态（不像 qwen35 的 GDN），所以不需要 state slot / 快照；KV 按位置寻址，被抢占重算直接重写。
//!
//! **decode 步走 CUDA graph**（B4，照抄 qwen35 的桶捕获）：`plan.mode == Decode` 且无 guided mask 的步按 batch 桶
//! （`graph_buckets`）捕获一次，之后只换固定地址缓冲（`PlanBuffers`）的内容再 `Graph::launch`。图内 =
//! `forward`（30 层）+ gather + bf16→f32 + argmax 到固定 `argmax` 缓冲；argmax 的 D2H 在图外（launch 后）做。
//! prefill / 混批 / 带 guided mask 的步走 eager。`NO_GRAPH` 环境变量强制全程 eager（A/B 用）。
//! **前置：`MOE_GROUPED=1`**——默认 host 路由 MoE 支路每层有 D2H + host 决策，整层 forward 无法被捕获（见 R42/R43）；
//! grouped 路径固定 launch 次数、零 D2H，才能进图。grouped 关时图自动退化为 eager（`graphs_eligible` 检查）。
use crate::layer::{moe_grouped_enabled, ActBuf, LayerScratch};
use crate::model::{KaonModel, StepMeta};
use anyhow::{ensure, Context, Result};
use dinfer_core::runner::{BatchMode, BatchPlan, ModelExecutor};
use dinfer_kernels::{self as k, bf16};
use dinfer_layers::exec::{graph_capture_eager, CapturePhase, DecodeGraphs, PlanBuffers};
use dinfer_layers::{Gemm, PinnedUploader, Stream, Tensor};

pub use dinfer_layers::exec::graph_buckets;

pub struct KaonExecutor {
    pub stream: Stream,
    pub gemm: Gemm,
    pub model: KaonModel,
    scratch: LayerScratch,
    acts: ActBuf,
    /// 固定地址的 plan 列（ids / pos / slot / rows / cu / seq / block_table）——图内 kernel 只读这些地址。
    plan: PlanBuffers,
    /// 每步小输入打包上传器。
    up: PinnedUploader,
    /// 全 batch logits（模型 forward 对每个 token 都算 lm_head）：[max_tokens, vocab] bf16。
    logits: Tensor<bf16>,
    /// forward 的 hidden dump 缓冲（本版不 dump，但 forward 签名要求）。
    hidden: Tensor<bf16>,
    /// 采样行 gather 后的 logits：[max_rows, vocab] bf16 → f32。
    gathered: Tensor<bf16>,
    gathered_f32: Tensor<f32>,
    argmax: Tensor<i32>,
    /// decode 图（按 batch 桶，懒/全捕获；公共件）。
    graphs: DecodeGraphs,
    max_tokens: usize,
    max_rows: usize,
    pub graph_steps: u64,
    pub eager_steps: u64,
    /// B29：本步是插队 prefill（cut-in）→ 采样结果不写 prev_chosen（在飞 decode 行的 gather 要上一 decode 步的）。
    skip_prev_chosen: bool,
    /// B35：跨流 prefill/decode 并行的副套资源（`XSTREAM=0` 关 → None → runner 不拆 plan）。
    alt: Option<AltSet>,
    /// 主流上一步结束事件：副流 prefill 开始前等它（KV 块复用 / 残差缓冲安全）。
    main_done: Option<dinfer_kernels::Event>,
    pub pair_steps: u64,
}

/// B35（loop kaon3-decode-opt）：副套 = 一条流 + 该流上的全部可变缓冲（与 qwen3 `AltSet` 同构）。模型权重与 KV cache 共享：
/// 两步写不同 slot、decode 只读历史、prefill 只写自己的 slot，无数据冲突。
struct AltSet {
    stream: Stream,
    gemm: Gemm,
    plan: PlanBuffers,
    up: PinnedUploader,
    logits: Tensor<bf16>,
    hidden: Tensor<bf16>,
    gathered: Tensor<bf16>,
    gathered_f32: Tensor<f32>,
    argmax: Tensor<i32>,
    scratch: LayerScratch,
    acts: ActBuf,
}

/// B29 开关：异步调度在飞行 token 的设备侧 gather + save_chosen。`INFLIGHT_GATHER=1` 开（默认关，见函数体注释）。
/// B46：lm_head 只算采样行（默认开；`LM_HEAD_ROWS_ONLY=0` 回退全 tokens 行，A/B 用）。
fn lm_head_rows_only() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("LM_HEAD_ROWS_ONLY").map(|v| v != "0").unwrap_or(true))
}

fn inflight_gather_enabled() -> bool {
    static IG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // b29 定稿：同步 runner 下两轮 c8 TPOT 9.80 → 10.05/10.06（+2.5%）、吞吐 -2.5%，c1/TTFT 不变 → **默认关**，`INFLIGHT_GATHER=1` 开；
    // kaon3 实现 supports_async（在飞行 token 真出现）时必须打开。
    *IG.get_or_init(|| std::env::var("INFLIGHT_GATHER").map_or(false, |v| v == "1"))
}

impl KaonExecutor {
    /// 默认开 decode 图（`NO_GRAPH` 环境变量 / `enable_graphs=false` 关）。
    pub fn new(
        stream: Stream,
        gemm: Gemm,
        model: KaonModel,
        max_tokens: usize,
        max_rows: usize,
        max_blocks: usize,
        enable_graphs: bool,
    ) -> Result<Self> {
        let v = model.cfg.vocab_size;
        let h = model.cfg.hidden;
        // CUDA graph 捕获要求非默认流（cudarc 的跨流事件跟踪在捕获期被驱动拒绝），且全程单流顺序执行。
        // 权重 / KV / scratch 都在旧流上分配并同步完；cuBLASLt 句柄绑流，在新流上重建（旧 gemm 丢弃）。
        let _ = gemm;
        stream.synchronize()?;
        let dev = stream.device();
        dev.disable_event_tracking();
        let stream = dev.new_stream()?;
        let gemm = Gemm::new(&stream).context("cublasLt on executor stream")?;
        // GEMM 自调优（方案2，kaon3-mega R44）：新 stream 的 gemm 必须在 CUDA graph 捕获前 autotune，
        // 否则 lm_head(唯一 cuBLASLt bf16 GEMM) 走启发式——graph 捕获烧进一个 algo、eager fallback 另选一个
        // → logits 末位差 → 近并列 token 翻转。autotune 定死 algo 进 default_algos 缓存，graph/eager 共享。
        if gemm.tune_enabled() {
            match gemm.autotune(&model.gemm_tune_entries()) {
                Ok(r) => tracing::info!("kaon3 gemm autotune: {:.0} ms", r.elapsed_ms),
                Err(e) => tracing::warn!("kaon3 gemm autotune 失败（回退启发式）: {e}"),
            }
        } else {
            tracing::info!("kaon3 gemm autotune: off (GEMM_TUNE=off)");
        }

        let scratch = LayerScratch::new_with_rows(&stream, &model.cfg, max_tokens, max_rows)?;
        let acts = ActBuf::new(&stream, &model.cfg)?;
        // grouped MoE 关时图无法捕获（host 路由有 D2H）：禁用图并打一行说明。
        let grouped = moe_grouped_enabled();
        let graphs_on = enable_graphs && grouped && std::env::var("NO_GRAPH").is_err();
        let graphs = DecodeGraphs::new(&stream, graphs_on, max_rows);
        let up = PinnedUploader::new(&stream, PlanBuffers::uploader_bytes(max_tokens, max_rows, max_blocks, 1 << 20))?;
        let mut ex = Self {
            plan: PlanBuffers::alloc(&stream, max_tokens, max_rows, max_blocks)?,
            logits: Tensor::uninit(&stream, &[max_tokens, v])?,
            hidden: Tensor::uninit(&stream, &[max_tokens, h])?,
            gathered: Tensor::uninit(&stream, &[max_rows, v])?,
            gathered_f32: Tensor::zeros(&stream, &[max_rows, v])?,
            argmax: Tensor::zeros(&stream, &[max_rows])?,
            scratch,
            acts,
            up,
            graphs,
            max_tokens,
            max_rows,
            graph_steps: 0,
            skip_prev_chosen: false,
            eager_steps: 0,
            stream,
            gemm,
            model,
            alt: None,
            main_done: None,
            pair_steps: 0,
        };
        // B35：副套（跨流 prefill/decode 并行）。`XSTREAM=0` 关；分配失败只告警（退回单流）。
        if std::env::var("XSTREAM").map(|v| v != "0").unwrap_or(true) {
            let t0 = std::time::Instant::now();
            match AltSet::new(&dev, &ex.model, max_tokens, max_rows, max_blocks) {
                Ok(a) => {
                    tracing::info!(alloc_ms = t0.elapsed().as_millis() as u64, "kaon3 xstream: alt set ready");
                    ex.alt = Some(a);
                }
                Err(e) => tracing::warn!(err = %e, "kaon3 xstream: alt set 分配失败，退回单流"),
            }
        }
        if !grouped && enable_graphs && std::env::var("NO_GRAPH").is_err() {
            tracing::info!("kaon3: decode graphs disabled（MOE_GROUPED 未开：host 路由 MoE 有 D2H，整层 forward 不可捕获）");
        }
        // 预热一次 max_tokens prefill：把各 QLinear 的量化暂存 / cuBLASLt 启发式 / 模块加载都做掉。
        // 关键是**把 FP8 量化暂存一次性撑到 max_tokens 行**——否则后面小桶捕获锁住地址、大桶 warmup 再 realloc
        // 会让已捕获的图读到已释放的缓冲（use-after-free）。`WARMUP=0` 关。
        if std::env::var("WARMUP").as_deref() != Ok("0") {
            if let Err(e) = ex.warmup_prefill() {
                tracing::warn!(err = %e, "kaon3: warmup prefill failed (ignored)");
            }
        }
        // decode 图：默认启动时全桶捕获；`GRAPH_CAPTURE=lazy` 回到懒捕获。
        if ex.graphs.enabled() && (graph_capture_eager() || std::env::var("GRAPH_CAPTURE").as_deref() != Ok("lazy")) {
            ex.capture_decode_graphs()?;
        }
        ex.graphs.report_startup(Some(if grouped { "--no-cuda-graph / NO_GRAPH" } else { "MOE_GROUPED 未开" }));
        Ok(ex)
    }

    /// B35：主套 ↔ 副套整体互换（流、GEMM、plan 列、上传器、logits/gather/argmax、scratch、acts）。
    fn swap_alt(&mut self) {
        let mut a = self.alt.take().expect("alt set");
        std::mem::swap(&mut self.stream, &mut a.stream);
        std::mem::swap(&mut self.gemm, &mut a.gemm);
        std::mem::swap(&mut self.plan, &mut a.plan);
        std::mem::swap(&mut self.up, &mut a.up);
        std::mem::swap(&mut self.logits, &mut a.logits);
        std::mem::swap(&mut self.hidden, &mut a.hidden);
        std::mem::swap(&mut self.gathered, &mut a.gathered);
        std::mem::swap(&mut self.gathered_f32, &mut a.gathered_f32);
        std::mem::swap(&mut self.argmax, &mut a.argmax);
        std::mem::swap(&mut self.scratch, &mut a.scratch);
        std::mem::swap(&mut self.acts, &mut a.acts);
        self.alt = Some(a);
    }
    /// 上传 + 入队（forward + gather + argmax），不等结果；返回要 collect 的行数。图只在主套上（副套永远 eager）。
    fn enqueue(&mut self, plan: &BatchPlan, allow_graph: bool) -> Result<usize> {
        let n = plan.num_reqs();
        let t = plan.total_tokens();
        let nrows = plan.logits_rows.len();
        ensure!(t <= self.max_tokens, "batch of {t} tokens > max_tokens {}", self.max_tokens);
        ensure!(nrows <= self.max_rows, "rows {nrows} > max_rows {}", self.max_rows);
        ensure!(plan.max_blocks <= self.plan.max_blocks, "block_table {} > buffer {}", plan.max_blocks, self.plan.max_blocks);
        ensure!(!plan.sample.any_mask, "kaon3 executor: guided decoding 未接（第一版不做）");
        self.skip_prev_chosen = plan.skip_prev_chosen;
        let graph_eligible = allow_graph && plan.mode == BatchMode::Decode && t == n && nrows == n && !plan.sample.any_mask;
        if graph_eligible {
            self.ensure_graph(n)?;
        }
        if graph_eligible && self.graphs.get(n).is_some() {
            let nb = self.graphs.get(n).unwrap().0;
            self.upload_plan(plan, nb)?;
            self.graphs.get(n).unwrap().1.launch().context("launch decode graph")?;
            self.graph_steps += 1;
            Ok(n)
        } else {
            self.upload_plan(plan, n)?;
            let max_seqlen = plan.num_tokens.iter().copied().max().unwrap_or(0).max(1) as usize;
            self.forward_device_mode(plan.mode, t, n, nrows, max_seqlen)?;
            self.eager_steps += 1;
            Ok(nrows)
        }
    }
    /// 启动预热：一条 max_tokens 长的假 prefill 请求跑一次 forward（slot=-1 不写 KV、结果丢弃），
    /// 把所有按形状懒分配的缓冲（FP8 量化暂存等）一次撑到上限，并建好 MoE grouped 的 per-layer 设备表。
    fn warmup_prefill(&mut self) -> Result<()> {
        let t = self.max_tokens;
        if t == 0 {
            return Ok(());
        }
        self.stream.bind_ctx()?;
        let mut p = BatchPlan::default();
        p.req_ids = vec![u64::MAX - 1];
        p.num_tokens = vec![t as u32];
        p.input_ids = vec![0; t];
        p.positions = (0..t as i32).collect();
        p.slot_mapping = vec![-1; t];
        p.cu_seqlens = vec![0, t as i32];
        p.seq_lens = vec![t as i32];
        p.block_table = vec![0; self.plan.max_blocks];
        p.max_blocks = self.plan.max_blocks;
        p.logits_rows = vec![t as i32 - 1];
        p.logits_row_start = vec![0, 1];
        p.spec_tokens = vec![Vec::new()];
        p.next_known = vec![-1];
        p.mode = BatchMode::Prefill;
        self.upload_plan(&p, 1)?;
        self.forward_device_mode(BatchMode::Prefill, t, 1, 1, t)?;
        self.stream.synchronize()?;
        tracing::info!(tokens = t, "kaon3: warmup prefill done");
        Ok(())
    }

    /// 把 plan 写进固定地址缓冲；batch 维补到 `nb`（图路径）或 `n`（eager）。
    fn upload_plan(&mut self, plan: &BatchPlan, nb: usize) -> Result<()> {
        let s = &self.stream;
        self.plan.upload(s, &mut self.up, plan, nb)?;
        self.up.flush(s)?;
        Ok(())
    }

    /// forward（固定缓冲、显式形状、显式 mode）+ gather + f32 + argmax。
    /// decode 纯批时可被 capture：期间不分配 / 不 D2H / 不同步。
    /// `t` = token 数（含 padding），`n` = 请求数（含 padding），`nrows` = logits 行数，`max_seqlen` = 本步每请求最大 query 长。
    fn forward_device_mode(&mut self, mode: BatchMode, t: usize, n: usize, nrows: usize, max_seqlen: usize) -> Result<()> {
        let s = &self.stream;
        let v = self.model.cfg.vocab_size;
        // B29（loop kaon3-decode-opt D109）：异步调度下在飞行的 token 行 `ids[i] = prev_chosen[gather_idx[i]]`（设备侧）。
        // 之前 kaon3 只 upload 了 gather_idx 却没 gather、也没 save_chosen → 在飞行 input id = 0，c≥2 每步必错。
        // 非异步 / 图捕获假 plan：gather_idx 全 -1，空操作（图内安全）。
        // `INFLIGHT_GATHER=0` 关（同步 runner 下 gather_idx 永远全 -1，这两步只有开销；kaon3 实现 supports_async 前可关）。
        let inflight_gather = inflight_gather_enabled();
        if inflight_gather {
            self.plan.gather_inflight(s, t)?;
        }
        let meta = StepMeta {
            mode,
            cu_seqlens: &self.plan.cu,
            seq_lens: &self.plan.seq,
            block_table: &self.plan.bt,
            positions: &self.plan.pos,
            slot_mapping: &self.plan.slots,
            num_reqs: n,
            max_blocks: self.plan.max_blocks,
            max_seqlen,
        };
        self.model.forward(
            s,
            &self.gemm,
            &dinfer_layers::DefaultOps,
            &self.plan.ids,
            &meta,
            &mut self.scratch,
            &mut self.acts,
            &mut self.logits,
            &mut self.hidden,
            t,
            None,
            None,
            None,
            // B46：混批 / prefill（nrows < t）只算采样行，logits 前 nrows 行即结果；纯 decode（nrows == t）照旧算全部再 gather（图内路径不变）。
            if nrows < t && lm_head_rows_only() { Some((&self.plan.rows.data, nrows)) } else { None },
        )?;
        if nrows < t && lm_head_rows_only() {
            s.dtod_copy_n(&self.logits.data, &mut self.gathered.data, nrows * v)?;
        } else {
            k::ops::gather_rows_bf16(s, &self.logits.data, &mut self.gathered.data, &self.plan.rows.data, nrows, v)?;
        }
        k::ops::bf16_to_f32_n(s, &self.gathered.data, &mut self.gathered_f32.data, nrows * v)?;
        k::argmax_f32_n(s, &self.gathered_f32.data, &mut self.argmax.data, nrows, v)?;
        // B29：本步采样存 prev_chosen 供下一步在飞行 gather（图内 dtod；cut-in 步不写）。无投机 → nrows == n，行 i = 请求 i。
        if inflight_gather && !self.skip_prev_chosen {
            self.plan.save_chosen(s, &self.argmax.data, nrows)?;
        }
        Ok(())
    }

    /// argmax 的 D2H（图外）：取前 `k` 行。
    fn collect(&mut self, k: usize) -> Result<Vec<i32>> {
        self.stream.dtoh_sync_copy_view(&self.argmax.data.slice(..k))
    }

    // ---------------------------------------------------------------- CUDA graph

    /// 捕获体：`Prepare` 上传假 decode plan（capture 外），`Body` = forward + gather + argmax（进图）。
    fn capture_body(&mut self, nb: usize, phase: CapturePhase) -> Result<()> {
        match phase {
            CapturePhase::Prepare => {
                let p = dinfer_layers::exec::decode_capture_plan(nb);
                self.upload_plan(&p, nb)
            }
            CapturePhase::Body => {
                // 纯 decode：每请求 1 token、1 行，max_seqlen = 1。
                self.forward_device_mode(BatchMode::Decode, nb, nb, nb, 1)
            }
        }
    }

    /// 启动时一次捕获全部桶。
    fn capture_decode_graphs(&mut self) -> Result<()> {
        let mut graphs = std::mem::replace(&mut self.graphs, DecodeGraphs::new(&self.stream, false, 0));
        let r = graphs.capture_all(|nb, ph| self.capture_body(nb, ph));
        self.graphs = graphs;
        r
    }

    /// 懒捕获：n 行 decode 需要的桶若还没有图，现在捕获。
    fn ensure_graph(&mut self, n: usize) -> Result<()> {
        let mut graphs = std::mem::replace(&mut self.graphs, DecodeGraphs::new(&self.stream, false, 0));
        let r = graphs.ensure(n, |nb, ph| self.capture_body(nb, ph));
        self.graphs = graphs;
        r
    }
}

impl ModelExecutor for KaonExecutor {
    fn supports_pair(&self) -> bool {
        self.alt.is_some()
    }
    /// B35：decode 批走主流（graph），prefill 批走副流（eager），两流只在"主流上一步结束 → 副流开始"与
    /// "副流 prefill 结束 → 主流 collect 后"两处用事件同步；KV cache 共享（写不同 slot）。
    fn forward_sample_pair(
        &mut self,
        decode: &BatchPlan,
        prefill: &BatchPlan,
        gen_d: &mut Vec<Vec<u32>>,
        dr_d: &mut Vec<Vec<u32>>,
        gen_p: &mut Vec<Vec<u32>>,
        dr_p: &mut Vec<Vec<u32>>,
    ) -> Result<dinfer_core::runner::PairLogprobs> {
        ensure!(self.alt.is_some(), "forward_sample_pair without alt set");
        self.stream.bind_ctx()?;
        let rows_d = self.enqueue(decode, true)?;
        if let Some(prev) = &self.main_done {
            self.alt.as_ref().unwrap().stream.wait_event(prev)?;
        }
        // 副流：prefill
        self.swap_alt();
        let r = (|| -> Result<(Vec<i32>, dinfer_kernels::Event)> {
            self.stream.bind_ctx()?;
            let rows_p = self.enqueue(prefill, false)?;
            let done = self.stream.record_event()?;
            Ok((self.collect(rows_p)?, done))
        })();
        self.swap_alt();
        let (am_p, prefill_done) = r?;
        self.stream.wait_event(&prefill_done)?;
        let am_d = self.collect(rows_d)?;
        self.main_done = Some(self.stream.record_event()?);
        self.pair_steps += 1;
        let fill = |plan: &BatchPlan, am: &[i32], gen: &mut Vec<Vec<u32>>, dr: &mut Vec<Vec<u32>>| {
            gen.clear();
            dr.clear();
            for i in 0..plan.num_reqs() {
                let r0 = plan.logits_row_start[i];
                gen.push(vec![am[r0] as u32]);
                dr.push(Vec::new());
            }
        };
        fill(decode, &am_d, gen_d, dr_d);
        fill(prefill, &am_p, gen_p, dr_p);
        Ok((vec![Vec::new(); decode.num_reqs()], vec![Vec::new(); prefill.num_reqs()]))
    }
    fn forward_sample(
        &mut self,
        plan: &BatchPlan,
        generated: &mut Vec<Vec<u32>>,
        drafts: &mut Vec<Vec<u32>>,
    ) -> Result<()> {
        // 引擎线程 != 构建线程：绑一次 ctx，否则首个 kernel 报 CUDA_ERROR_INVALID_CONTEXT。
        self.stream.bind_ctx()?;
        let n = plan.num_reqs();
        let t = plan.total_tokens();
        let nrows = plan.logits_rows.len();
        let _fs0 = std::time::Instant::now();
        tracing::trace!(mode = ?plan.mode, t, n, nrows, "fwd-sample enter");  // B17a：原无条件 eprintln 每步一次 stderr syscall
        ensure!(t <= self.max_tokens, "batch of {t} tokens > max_tokens {}", self.max_tokens);
        ensure!(nrows <= self.max_rows, "rows {nrows} > max_rows {}", self.max_rows);
        ensure!(plan.max_blocks <= self.plan.max_blocks, "block_table {} > buffer {}", plan.max_blocks, self.plan.max_blocks);
        ensure!(!plan.sample.any_mask, "kaon3 executor: guided decoding 未接（第一版不做）");
        self.skip_prev_chosen = plan.skip_prev_chosen;

        // 本步可否走图：纯 decode（每请求 1 token、1 行）且无 guided mask。
        let graph_eligible = plan.mode == BatchMode::Decode && t == n && nrows == n && !plan.sample.any_mask;
        if graph_eligible {
            self.ensure_graph(n)?;
        }
        let use_graph = graph_eligible && self.graphs.get(n).is_some();

        // HOST_TIME=1：host 各段墙钟计时（纯 CPU Instant，不碰 GPU，graph/eager 都准）。定 13ms host 哪段。
        let ht = std::env::var("HOST_TIME").ok().as_deref() == Some("1");
        let t_bind = std::time::Instant::now();

        let collect_rows;
        let (mut t_up, mut t_launch) = (t_bind, t_bind);
        if use_graph {
            let nb = self.graphs.get(n).unwrap().0;
            self.upload_plan(plan, nb)?;
            t_up = std::time::Instant::now();
            self.graphs.get(n).unwrap().1.launch().context("launch decode graph")?;
            t_launch = std::time::Instant::now();
            if self.graph_steps == 0 {
                tracing::info!(n, bucket = nb, "kaon3: decode step via cuda graph");
            }
            self.graph_steps += 1;
            collect_rows = n; // 图里算了 nb 行，只取前 n 行（logits_row_start[i] == i）
        } else {
            self.upload_plan(plan, n)?;
            t_up = std::time::Instant::now();
            let max_seqlen = plan.num_tokens.iter().copied().max().unwrap_or(0).max(1) as usize;
            self.forward_device_mode(plan.mode, t, n, nrows, max_seqlen)?;
            t_launch = std::time::Instant::now();
            self.eager_steps += 1;
            collect_rows = nrows;
        }

        let am = self.collect(collect_rows)?;
        if ht {
            let t_col = std::time::Instant::now();
            let us = |a: std::time::Instant, b: std::time::Instant| b.duration_since(a).as_micros();
            eprintln!("[kaon3-host] upload={}us launch(submit)={}us collect(dtoh_sync)={}us",
                us(t_bind, t_up), us(t_up, t_launch), us(t_launch, t_col));
        }
        // 每请求本步的 logits 行区间（无投机 → 恰 1 行）：取该行 argmax 为下一个 token。
        generated.clear();
        drafts.clear();
        for i in 0..n {
            let r0 = plan.logits_row_start[i];
            generated.push(vec![am[r0] as u32]);
            drafts.push(Vec::new());
        }
        if self.alt.is_some() {
            self.main_done = Some(self.stream.record_event()?);
        }
        eprintln!("[fwd-sample] EXIT mode={:?} t={t} total={:.0}us", plan.mode, _fs0.elapsed().as_secs_f64()*1e6);
        Ok(())
    }
}

impl AltSet {
    fn new(dev: &dinfer_kernels::Device, model: &KaonModel, max_tokens: usize, max_rows: usize, max_blocks: usize) -> Result<Self> {
        let stream = dev.new_stream()?;
        let gemm = Gemm::new(&stream).context("cublasLt on alt stream")?;
        if gemm.tune_enabled() {
            if let Err(e) = gemm.autotune(&model.gemm_tune_entries()) {
                tracing::warn!("kaon3 alt gemm autotune 失败（回退启发式）: {e}");
            }
        }
        let v = model.cfg.vocab_size;
        let h = model.cfg.hidden;
        let up = PinnedUploader::new(&stream, PlanBuffers::uploader_bytes(max_tokens, max_rows, max_blocks, 1 << 20))?;
        Ok(Self {
            plan: PlanBuffers::alloc(&stream, max_tokens, max_rows, max_blocks)?,
            up,
            logits: Tensor::uninit(&stream, &[max_tokens, v])?,
            hidden: Tensor::uninit(&stream, &[max_tokens, h])?,
            gathered: Tensor::uninit(&stream, &[max_rows, v])?,
            gathered_f32: Tensor::zeros(&stream, &[max_rows, v])?,
            argmax: Tensor::zeros(&stream, &[max_rows])?,
            scratch: LayerScratch::new_with_rows(&stream, &model.cfg, max_tokens, max_rows)?,
            acts: ActBuf::new(&stream, &model.cfg)?,
            gemm,
            stream,
        })
    }
}
