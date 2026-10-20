//! Kimi-K3 文本主干：`embed_tokens` → N × `KimiDecoderLayer`（块残差列表按 HF `KimiLinearModel.forward`）→
//! `output_attn_res`（AttnRes，候选 = 全部块 ∪ {最终 prefix_sum}）→ `norm` → `lm_head`。
//!
//! 块残差语义（HF）：`block_residual` 起始为空；每层返回 `(prefix_sum, block_residual)`；写块层（`layer % 12 == 0`）
//! 把**本层输入** `x` 追加进列表（`layer.rs` 的 `writes_block`）。所以第 k 个块 = 第 `12k` 层的输入（k=0 是 embedding 输出）。
//!
//! 只加载前 N 层（`load(.., n_layers)`）：对拍 / 显存受限时用；N < 93 时 `output_attn_res + norm + lm_head` 仍然接在第 N-1 层之后
//! （与 torch 参考里 `num_hidden_layers = N` 的截断模型同口径）。
//! 装载按层打开分片（`weights::shards_for_layer`，每层恰好一片 16 GB），顶层张量在 `shards_for_top`。
use crate::config::KimiLinearConfig;
use crate::scratch::ActScratch;
use crate::layer::{Attn, AttnRes, KimiDecoderLayer, LayerBudget, StepMeta};
use crate::weights::{shards_for_layer, shards_for_top, KdaShapes, Shard, Table, LM_HEAD};
use anyhow::{ensure, Result};
use dinfer_kernels::ops as k;
use dinfer_layers::{bf16, Gemm, Linear, RmsNorm, Stream, Tensor, WeightSource};
use std::path::Path;

pub struct KimiK3Model {
    pub layers: Vec<KimiDecoderLayer>,
    /// 全模型共享的一套激活缓冲（`scratch.rs`），forward 时 `&mut` 借给每层 / output_res。
    pub act: ActScratch,
    embed: Tensor<bf16>, // [vocab, hidden]
    norm: RmsNorm,
    output_res: Option<AttnRes>,
    pub lm_head: Linear, // [vocab, hidden]
    hidden: usize,
    vocab: usize,
    max_tokens: usize,
    /// `normed` / `logits_bf16` 的行数（executor 只对采样行取 logits；`load` 时 = max_tokens）。
    max_rows: usize,
    kda_shapes: KdaShapes,
    // 激活
    x: Tensor<bf16>,           // 当前 prefix_sum（层输入）
    y: Tensor<bf16>,           // 层输出
    blocks: Vec<Tensor<bf16>>, // 块残差（写块层的输入拷贝），容量 max_blocks
    n_blocks: usize,
    res_out: Tensor<bf16>,
    normed: Tensor<bf16>,
    logits_bf16: Tensor<bf16>,
    /// `trace` 开时每层输出的拷贝（对拍用，`[T, hidden]` × N）。
    pub trace: bool,
    pub layer_out: Vec<Tensor<bf16>>,
}

impl KimiK3Model {
    /// 装载前 `n_layers` 层 + 顶层张量。`budget.max_tokens` 同时决定 logits 缓冲行数（对拍用：每行都要 logits）。
    pub fn load(
        stream: &Stream,
        dir: &Path,
        cfg: &KimiLinearConfig,
        n_layers: usize,
        budget: LayerBudget,
    ) -> Result<Self> {
        Self::load_with_rows(stream, dir, cfg, n_layers, budget, budget.max_tokens)
    }

    /// 同 `load`，但 logits 缓冲只留 `max_rows` 行（服务用：每请求只采样 1 行，`forward_rows` 只对这些行做 norm + lm_head；
    /// 全 token 行的 bf16 logits 在 16384 token 预算下是 5.3 GB，f32 再翻倍，没必要）。
    pub fn load_with_rows(
        stream: &Stream,
        dir: &Path,
        cfg: &KimiLinearConfig,
        n_layers: usize,
        budget: LayerBudget,
        max_rows: usize,
    ) -> Result<Self> {
        Self::load_tp(stream, dir, cfg, n_layers, budget, max_rows, Shard::SINGLE)
    }

    /// TP/EP 分片装载（design-kimik3 §2.4）：每层按 `shard` 切；embed / norm / output_attn_res / lm_head 复制。
    /// 通信由调用方经 `StepMeta.comm` 传入 forward（模型本身不持 Comm）。`shard.world == 1` 与 `load_with_rows` 逐位相同。
    pub fn load_tp(
        stream: &Stream,
        dir: &Path,
        cfg: &KimiLinearConfig,
        n_layers: usize,
        budget: LayerBudget,
        max_rows: usize,
        shard: Shard,
    ) -> Result<Self> {
        ensure!(n_layers >= 1 && n_layers <= cfg.num_hidden_layers as usize, "n_layers {n_layers}");
        let t_load = std::time::Instant::now();
        if !shard.is_single() {
            tracing::info!(rank = shard.rank, world = shard.world, n_layers, "kimik3: TP/EP 分片装载开始");
        }
        if n_layers < cfg.num_hidden_layers as usize {
            tracing::warn!(
                n_layers,
                total = cfg.num_hidden_layers,
                "kimik3: TRUNCATED MODEL（只装前 {n_layers} 层 + 顶层；输出无语义，只用于流程冒烟 / 对拍）"
            );
        }
        let hidden = cfg.hidden_size as usize;
        let vocab = cfg.vocab_size as usize;
        let tm = budget.max_tokens;
        let max_rows = max_rows.clamp(1, tm);
        let max_blocks = match cfg.attn_res_block_size {
            Some(bs) => (cfg.num_hidden_layers as usize).div_ceil(bs as usize),
            None => 0,
        };

        // 顶层
        let top = WeightSource::open_files(&shards_for_top(dir)?)?;
        let tt = Table::from_names(&top.names().collect::<Vec<_>>())?;
        let embed = tt.load_bf16(stream, &top, "embed_tokens.weight")?;
        ensure!(embed.shape == [vocab, hidden], "embed_tokens shape {:?}", embed.shape);
        let norm = tt.rmsnorm(stream, &top, "norm.weight", cfg.rms_norm_eps)?;
        let lm_head = Linear::new(top.load_bf16(stream, LM_HEAD)?, None)?;
        ensure!(lm_head.w.shape == [vocab, hidden], "lm_head shape {:?}", lm_head.w.shape);
        let output_res = if cfg.attn_res_block_size.is_some() {
            Some(AttnRes::load_output(stream, &top, &tt, cfg)?)
        } else {
            None
        };
        drop(top);
        tracing::info!(
            rank = shard.rank,
            secs = format!("{:.1}", t_load.elapsed().as_secs_f32()),
            "kimik3 top loaded (embed / norm / lm_head / output_res)"
        );

        // 全模型共享一套 per-token 激活缓冲（原先每层各一套，预算 16384 时一层 ≈ 13 GB，93 层 tp8 必 OOM，见 scratch.rs / k3-serve 3b）。
        let act = ActScratch::new(stream, cfg, tm, shard)?;
        let act_bytes = ActScratch::bytes(cfg, tm, shard);
        tracing::info!(
            gib = format!("{:.2}", act_bytes as f64 / (1u64 << 30) as f64),
            "kimik3: activation scratch bytes={act_bytes} (max_tokens={tm}, world={})",
            shard.world
        );
        // 逐层（每层一片，装完即释放 mmap）
        let mut layers = Vec::with_capacity(n_layers);
        let t_layers = std::time::Instant::now();
        let mut layers_bytes: u64 = 0;
        for l in 0..n_layers {
            let t0 = std::time::Instant::now();
            let (free0, _) = stream.mem_info()?;
            let src = WeightSource::open_files(&shards_for_layer(dir, l)?)?;
            let table = Table::from_names(&src.names().collect::<Vec<_>>())?;
            let layer = KimiDecoderLayer::load_tp(stream, &src, &table, l, cfg, budget, shard)?;
            stream.synchronize()?;
            let (free1, _) = stream.mem_info()?;
            let secs = t0.elapsed().as_secs_f64();
            let bytes = free0.saturating_sub(free1) as u64;
            layers_bytes += bytes;
            // gb = 该层占的显存（含激活外的权重 + 缓冲），gbs = 装载速率（显存增量 / 秒；mmap 读盘 + H2D + 分片切片的合速）
            tracing::info!(
                layer = l,
                kind = if cfg.is_kda_layer(l as u32) { "kda" } else { "mla" },
                ffn = if cfg.is_moe_layer(l as u32) { "moe" } else { "dense" },
                secs = format!("{:.1}", secs),
                gb = format!("{:.2}", bytes as f64 / 1e9),
                gbs = format!("{:.2}", bytes as f64 / 1e9 / secs.max(1e-6)),
                done = format!("{}/{n_layers}", l + 1),
                "kimik3 layer loaded"
            );
            layers.push(layer);
        }
        {
            let ls = t_layers.elapsed().as_secs_f64();
            let total = t_load.elapsed().as_secs_f64();
            tracing::info!(
                rank = shard.rank,
                world = shard.world,
                n_layers,
                layers_secs = format!("{:.1}", ls),
                layers_gb = format!("{:.2}", layers_bytes as f64 / 1e9),
                avg_gbs = format!("{:.2}", layers_bytes as f64 / 1e9 / ls.max(1e-6)),
                total_secs = format!("{:.1}", total),
                "kimik3 model loaded（整体装载耗时；total 含顶层 + 激活缓冲，不含 KV / 引擎启动）"
            );
        }
        let z = |s: &Stream| Tensor::<bf16>::zeros(s, &[tm, hidden]);
        let mut blocks = Vec::with_capacity(max_blocks);
        for _ in 0..max_blocks {
            blocks.push(z(stream)?);
        }
        Ok(Self {
            layers,
            act,
            embed,
            norm,
            output_res,
            lm_head,
            hidden,
            vocab,
            max_tokens: tm,
            max_rows,
            kda_shapes: KdaShapes::with_shard(cfg, shard),
            x: z(stream)?,
            y: z(stream)?,
            blocks,
            n_blocks: 0,
            res_out: z(stream)?,
            normed: Tensor::zeros(stream, &[max_rows, hidden])?,
            logits_bf16: Tensor::zeros(stream, &[max_rows, vocab])?,
            trace: false,
            layer_out: Vec::new(),
        })
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }
    pub fn max_rows(&self) -> usize {
        self.max_rows
    }
    pub fn vocab(&self) -> usize {
        self.vocab
    }
    pub fn kda_shapes(&self) -> &KdaShapes {
        &self.kda_shapes
    }
    /// 全部 KDA 层的状态槽数（装载时 `LayerBudget::num_slots`）；没有 KDA 层时 0。
    pub fn num_slots(&self) -> usize {
        self.layers
            .iter()
            .find_map(|l| match &l.attn {
                Attn::Kda(k) => Some(k.state.rec.shape[0]),
                Attn::Mla(_) => None,
            })
            .unwrap_or(0)
    }
    /// 清零这些 slot 在**全部 KDA 层**的 conv + 递推状态（新请求 / 抢占重算）。图外、同流有序。
    pub fn reset_slots(&mut self, stream: &Stream, slots: &[usize]) -> Result<()> {
        for l in self.layers.iter_mut() {
            if let Attn::Kda(k) = &mut l.attn {
                for &s in slots {
                    k.reset_slot(stream, s)?;
                }
            }
        }
        Ok(())
    }

    /// `ids[:t]` → `logits[:t, vocab]`（f32）：每个 token 行都出 logits（对拍 / prompt_logprobs 用）。要求 `max_rows ≥ t`。
    pub fn forward(
        &mut self,
        stream: &Stream,
        gemm: &Gemm,
        t: usize,
        ids: &Tensor<i32>,
        meta: &StepMeta<'_>,
        logits: &mut Tensor<f32>,
    ) -> Result<()> {
        ensure!(t <= self.max_rows, "model::forward: {t} 行 > max_rows {}（用 forward_rows）", self.max_rows);
        let rows = Tensor::from_vec(stream, (0..t as i32).collect(), &[t])?;
        self.forward_rows(stream, gemm, t, ids, meta, &rows, t, logits)
    }

    /// `ids[:t]` → `logits[:nrows, vocab]`（f32），只对 `rows[:nrows]`（hidden 行号）做 norm + lm_head。
    /// `meta` 由调用方按本步 batch 构造（KDA slot / MLA paged 两套都要给）。`self.normed` 是这些行 norm 之后的 `[nrows, hidden]`。
    #[allow(clippy::too_many_arguments)]
    pub fn forward_rows(
        &mut self,
        stream: &Stream,
        gemm: &Gemm,
        t: usize,
        ids: &Tensor<i32>,
        meta: &StepMeta<'_>,
        rows: &Tensor<i32>,
        nrows: usize,
        logits: &mut Tensor<f32>,
    ) -> Result<()> {
        let h = self.hidden;
        ensure!(t <= self.max_tokens, "model: {t} > max_tokens {}", self.max_tokens);
        ensure!(nrows <= self.max_rows && rows.numel() >= nrows, "model: {nrows} rows > max_rows {}", self.max_rows);
        ensure!(ids.numel() >= t && logits.numel() >= nrows * self.vocab, "model: ids / logits");
        let n = t * h;
        k::embedding_bf16_n(stream, &ids.data, &self.embed.data, &mut self.x.data, t, h)?;
        self.n_blocks = 0;
        if self.trace {
            self.layer_out.clear();
        }
        for i in 0..self.layers.len() {
            let Self { layers, act, x, y, blocks, n_blocks, .. } = self;
            let layer = &mut layers[i];
            let writer = layer.writes_block();
            {
                let bl: Vec<&Tensor<bf16>> = blocks[..*n_blocks].iter().collect();
                layer.forward(stream, gemm, t, x, &bl, meta, act, y)?;
            }
            if writer {
                ensure!(*n_blocks < blocks.len(), "块残差超过 {}", blocks.len());
                stream.dtod_copy_n(&x.data, &mut blocks[*n_blocks].data, n)?;
                *n_blocks += 1;
            }
            std::mem::swap(x, y);
            if self.trace {
                let mut c = Tensor::<bf16>::zeros(stream, &[t, h])?;
                stream.dtod_copy_n(&self.x.data, &mut c.data, n)?;
                self.layer_out.push(c);
            }
        }
        // output_attn_res：候选 = 全部块 ∪ {x}
        let final_in = if let Some(res) = self.output_res.as_ref() {
            let mut cands: Vec<&Tensor<bf16>> = self.blocks[..self.n_blocks].iter().collect();
            cands.push(&self.x);
            let rs = self.act.res.as_mut().ok_or_else(|| anyhow::anyhow!("model: output_attn_res 但 scratch 无 res"))?;
            res.apply(stream, gemm, t, &cands, 0, rs, &mut self.act.acc, &mut self.res_out)?;
            &self.res_out
        } else {
            &self.x
        };
        k::gather_rmsnorm_bf16(
            stream,
            &final_in.data,
            &rows.data,
            &self.norm.w.data,
            &mut self.normed.data,
            nrows,
            h,
            self.norm.eps,
        )?;
        self.lm_head.forward_rows(gemm, &self.normed, &mut self.logits_bf16, nrows)?;
        k::bf16_to_f32_n(stream, &self.logits_bf16.data, &mut logits.data, nrows * self.vocab)
    }

    /// 上一步最后一层之后、output_attn_res 之前的 prefix_sum（对拍 HF `hidden_states[-1]`）。
    pub fn last_prefix_sum(&self) -> &Tensor<bf16> {
        &self.x
    }
    /// 上一步 `output_attn_res` 的输出（无 AttnRes 时等于 `last_prefix_sum`）。
    pub fn last_res_out(&self) -> &Tensor<bf16> {
        if self.output_res.is_some() {
            &self.res_out
        } else {
            &self.x
        }
    }
    /// 上一步 norm 之后的隐状态。
    pub fn last_normed(&self) -> &Tensor<bf16> {
        &self.normed
    }
    pub fn num_blocks(&self) -> usize {
        self.n_blocks
    }
}
