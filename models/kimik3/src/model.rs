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
use crate::layer::{AttnRes, KimiDecoderLayer, LayerBudget, StepMeta};
use crate::weights::{shards_for_layer, shards_for_top, Table, LM_HEAD};
use anyhow::{ensure, Result};
use dinfer_kernels::ops as k;
use dinfer_layers::{bf16, Gemm, Linear, RmsNorm, Stream, Tensor, WeightSource};
use std::path::Path;

pub struct KimiK3Model {
    pub layers: Vec<KimiDecoderLayer>,
    embed: Tensor<bf16>, // [vocab, hidden]
    norm: RmsNorm,
    output_res: Option<AttnRes>,
    pub lm_head: Linear, // [vocab, hidden]
    hidden: usize,
    vocab: usize,
    max_tokens: usize,
    // 激活
    x: Tensor<bf16>,  // 当前 prefix_sum（层输入）
    y: Tensor<bf16>,  // 层输出
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
    /// 装载前 `n_layers` 层 + 顶层张量。`budget.max_tokens` 同时决定 logits 缓冲行数。
    pub fn load(stream: &Stream, dir: &Path, cfg: &KimiLinearConfig, n_layers: usize, budget: LayerBudget) -> Result<Self> {
        ensure!(n_layers >= 1 && n_layers <= cfg.num_hidden_layers as usize, "n_layers {n_layers}");
        let hidden = cfg.hidden_size as usize;
        let vocab = cfg.vocab_size as usize;
        let tm = budget.max_tokens;
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
        let output_res = if cfg.attn_res_block_size.is_some() { Some(AttnRes::load_output(stream, &top, &tt, cfg, tm, max_blocks)?) } else { None };
        drop(top);

        // 逐层（每层一片，装完即释放 mmap）
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let t0 = std::time::Instant::now();
            let (free0, _) = stream.mem_info()?;
            let src = WeightSource::open_files(&shards_for_layer(dir, l)?)?;
            let table = Table::from_names(&src.names().collect::<Vec<_>>())?;
            let layer = KimiDecoderLayer::load(stream, &src, &table, l, cfg, budget)?;
            stream.synchronize()?;
            let (free1, _) = stream.mem_info()?;
            tracing::info!(
                layer = l,
                kind = if cfg.is_kda_layer(l as u32) { "kda" } else { "mla" },
                ffn = if cfg.is_moe_layer(l as u32) { "moe" } else { "dense" },
                secs = format!("{:.1}", t0.elapsed().as_secs_f32()),
                gb = format!("{:.2}", free0.saturating_sub(free1) as f64 / 1e9),
                "kimik3 layer loaded"
            );
            layers.push(layer);
        }
        let z = |s: &Stream| Tensor::<bf16>::zeros(s, &[tm, hidden]);
        let mut blocks = Vec::with_capacity(max_blocks);
        for _ in 0..max_blocks {
            blocks.push(z(stream)?);
        }
        Ok(Self {
            layers,
            embed,
            norm,
            output_res,
            lm_head,
            hidden,
            vocab,
            max_tokens: tm,
            x: z(stream)?,
            y: z(stream)?,
            blocks,
            n_blocks: 0,
            res_out: z(stream)?,
            normed: z(stream)?,
            logits_bf16: Tensor::zeros(stream, &[tm, vocab])?,
            trace: false,
            layer_out: Vec::new(),
        })
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// `ids[:t]` → `logits[:t, vocab]`（f32）。`meta` 由调用方按本步 batch 构造（KDA slot / MLA paged 两套都要给）。
    /// 返回的 `hidden_final`（`self.normed`）是 norm 之后、lm_head 之前的 `[t, hidden]`。
    pub fn forward(&mut self, stream: &Stream, gemm: &Gemm, t: usize, ids: &Tensor<i32>, meta: &StepMeta<'_>, logits: &mut Tensor<f32>) -> Result<()> {
        let h = self.hidden;
        ensure!(t <= self.max_tokens, "model: {t} > max_tokens {}", self.max_tokens);
        ensure!(ids.numel() >= t && logits.numel() >= t * self.vocab, "model: ids / logits");
        let n = t * h;
        k::embedding_bf16_n(stream, &ids.data, &self.embed.data, &mut self.x.data, t, h)?;
        self.n_blocks = 0;
        if self.trace {
            self.layer_out.clear();
        }
        for i in 0..self.layers.len() {
            let Self { layers, x, y, blocks, n_blocks, .. } = self;
            let layer = &mut layers[i];
            let writer = layer.writes_block();
            {
                let bl: Vec<&Tensor<bf16>> = blocks[..*n_blocks].iter().collect();
                layer.forward(stream, gemm, t, x, &bl, meta, y)?;
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
        let final_in = if let Some(res) = self.output_res.as_mut() {
            let mut cands: Vec<&Tensor<bf16>> = self.blocks[..self.n_blocks].iter().collect();
            cands.push(&self.x);
            res.apply(stream, gemm, t, &cands, 0, &mut self.res_out)?;
            &self.res_out
        } else {
            &self.x
        };
        k::rmsnorm_bf16_n(stream, &final_in.data, &self.norm.w.data, &mut self.normed.data, t, h, self.norm.eps)?;
        self.lm_head.forward_rows(gemm, &self.normed, &mut self.logits_bf16, t)?;
        k::bf16_to_f32_n(stream, &self.logits_bf16.data, &mut logits.data, t * self.vocab)
    }

    /// 上一步最后一层之后、output_attn_res 之前的 prefix_sum（对拍 HF `hidden_states[-1]`）。
    pub fn last_prefix_sum(&self) -> &Tensor<bf16> {
        &self.x
    }
    /// 上一步 `output_attn_res` 的输出（无 AttnRes 时等于 `last_prefix_sum`）。
    pub fn last_res_out(&self) -> &Tensor<bf16> {
        if self.output_res.is_some() { &self.res_out } else { &self.x }
    }
    /// 上一步 norm 之后的隐状态。
    pub fn last_normed(&self) -> &Tensor<bf16> {
        &self.normed
    }
    pub fn num_blocks(&self) -> usize {
        self.n_blocks
    }
}
