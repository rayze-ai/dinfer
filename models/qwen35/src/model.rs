//! Qwen3.5 主干：embed → 64 层（48 GDN + 16 全注意力，按 `layer_types`）→ norm → lm_head（**不 tie**）。
//!
//! 与 Qwen3 的骨架相同（残差 + `fused_add_rmsnorm` 省一次 add kernel），两处不同：
//! - 层是异构的：全注意力层有 paged KV，GDN 层有自己的 conv / 递推状态（`GdnLayer::state`，按 slot 自管）；
//! - 输出门 / partial RoPE 在 `full_attn.rs` 里处理，本文件只做主干与 logits。
//!
//! `lm_head.weight` 独立（`tie_word_embeddings false`）。
use crate::config::Qwen35Config;
use crate::full_attn::PagedKv;
use crate::layer::{Attn, DecoderLayer, StepMeta};
use crate::weights::Table;
use anyhow::{ensure, Context, Result};
use dinfer_core::runner::BatchMode;
use dinfer_kernels as k;
use dinfer_layers::{bf16, Embedding, Gemm, Linear, OpSet, RmsNorm, Rope, RopeStyle, Stream, Tensor, WeightSource};
use dinfer_layers::Norm as _;

/// 一步 forward 的描述（executor 上传后的设备张量）。
pub struct Forward<'a> {
    pub input_ids: &'a Tensor<i32>,
    pub positions: &'a Tensor<i32>,
    pub slot_mapping: &'a Tensor<i64>,
    pub meta: StepMeta<'a>,
    pub logits_rows: &'a Tensor<i32>,
    pub t: usize,
    pub rows: usize,
    /// 本步每请求的 GDN 状态 slot（按 batch 顺序；`meta.state_indices` 是它的设备副本）。
    pub req_ids: &'a [u64],
}

/// 每层可换的一套激活缓冲（跨流并行用；GDN 层不参与）。
pub struct LayerScratch {
    pub qkv_buf: Tensor<bf16>,
    pub gate_buf: Tensor<bf16>,
    pub attn: Tensor<bf16>,
    pub gate_up: Tensor<bf16>,
    pub act: Tensor<bf16>,
}

pub struct ModelScratch {
    pub hidden: Tensor<bf16>,
    pub normed: Tensor<bf16>,
    pub attn_out: Tensor<bf16>,
    pub mlp_out: Tensor<bf16>,
    pub final_normed: Tensor<bf16>,
    pub logits_bf16: Tensor<bf16>,
    pub decode_scratch: Tensor<f32>,
    pub layers: Vec<LayerScratch>,
}

pub struct Qwen35Model<O: OpSet = dinfer_layers::DefaultOps> {
    pub ops: O,
    pub cfg: Qwen35Config,
    pub embed: Embedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: RmsNorm,
    pub lm_head: Linear,
    pub rope: Rope,
    /// 只有全注意力层有 KV；GDN 层是 `None`。
    pub kv: Vec<Option<PagedKv>>,
    pub max_tokens: usize,
    pub max_logits_rows: usize,
    pub tp: (usize, usize),
    pub num_slots: usize,
    hidden: Tensor<bf16>,
    normed: Tensor<bf16>,
    attn_out: Tensor<bf16>,
    mlp_out: Tensor<bf16>,
    final_normed: Tensor<bf16>,
    logits_bf16: Tensor<bf16>,
    decode_scratch: Tensor<f32>,
}

impl<O: OpSet> Qwen35Model<O> {
    /// `num_slots`：GDN 状态最多容纳多少并发请求（= `max_num_seqs`）。
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        stream: &Stream,
        src: &WeightSource,
        cfg: Qwen35Config,
        max_tokens: usize,
        num_blocks: usize,
        block_size: usize,
        num_slots: usize,
    ) -> Result<Self>
    where
        O: Default,
    {
        Self::load_with_budget(stream, src, cfg, max_tokens, max_tokens, num_blocks, block_size, num_slots, (0, 1))
    }

    /// 全注意力层每个 KV block 的字节数（只数全注意力层）：`n_full × block_size × kv_heads × 2 × D × 2`。
    pub fn kv_bytes_per_block(cfg: &Qwen35Config, block_size: usize) -> usize {
        let n_full = cfg.layer_types.iter().filter(|t| **t == crate::config::LayerType::FullAttention).count();
        n_full
            * block_size
            * cfg.num_key_value_heads as usize
            * 2
            * cfg.head_dim as usize
            * 2
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_with_budget(
        stream: &Stream,
        src: &WeightSource,
        cfg: Qwen35Config,
        max_tokens: usize,
        max_logits_rows: usize,
        num_blocks: usize,
        block_size: usize,
        num_slots: usize,
        tp: (usize, usize),
    ) -> Result<Self>
    where
        O: Default,
    {
        let (rank, world) = tp;
        ensure!(world >= 1 && rank < world, "bad tp shard {rank}/{world}");
        ensure!(num_blocks > 0 && num_slots > 0, "num_blocks / num_slots must be > 0");
        let mut cfg = cfg;
        if world > 1 {
            ensure!(cfg.num_attention_heads as usize % world == 0, "num_attention_heads % tp != 0");
            ensure!(cfg.num_key_value_heads as usize % world == 0, "num_key_value_heads % tp != 0");
            ensure!(cfg.intermediate_size as usize % world == 0, "intermediate_size % tp != 0");
            cfg.num_attention_heads /= world as u32;
            cfg.num_key_value_heads /= world as u32;
            cfg.intermediate_size /= world as u32;
            tracing::info!(rank, world, heads = cfg.num_attention_heads, kv_heads = cfg.num_key_value_heads, "tp: local shard config");
        }
        let t_load0 = std::time::Instant::now();
        let table = Table::open(src)?;
        let hidden = cfg.hidden_size as usize;
        let (h, hk, d) = (
            cfg.num_attention_heads as usize,
            cfg.num_key_value_heads as usize,
            cfg.head_dim as usize,
        );
        let embed = table.embedding(stream, src, "embed_tokens.weight")?;
        let n = cfg.num_hidden_layers as usize;
        ensure!(cfg.layer_types.len() == n, "layer_types {} != num_hidden_layers {n}", cfg.layer_types.len());
        let mut layers = Vec::with_capacity(n);
        for l in 0..n {
            layers.push(
                DecoderLayer::load(stream, src, &table, l, &cfg, max_tokens, num_slots, tp)
                    .with_context(|| format!("layer {l}"))?,
            );
        }
        tracing::info!(layers = n, ms = t_load0.elapsed().as_millis() as u64, "load timing: decoder layers");
        let norm = table.rmsnorm(stream, src, "norm.weight", cfg.rms_norm_eps)?;
        let lm_head = Linear::new(table.load_bf16(stream, src, "lm_head.weight")?, None)?;
        ensure!(
            lm_head.out_features == cfg.vocab_size as usize && lm_head.in_features == hidden,
            "lm_head shape {:?}（期望 [{}, {hidden}]）；tie_word_embeddings = {}",
            lm_head.w.shape,
            cfg.vocab_size,
            cfg.tie_word_embeddings
        );
        let rope = Rope::new(stream, cfg.rope_dim() as usize, cfg.max_position_embeddings as usize, cfg.rope_theta(), RopeStyle::Neox)?;
        stream.synchronize()?;

        let v = cfg.vocab_size as usize;
        let mut kv = Vec::with_capacity(n);
        for l in 0..n {
            kv.push(if layers[l].attn.is_full() {
                Some(PagedKv {
                    cache: Tensor::uninit(stream, &[num_blocks, block_size, hk, 2, d])?,
                    num_blocks,
                    block_size,
                })
            } else {
                None
            });
        }
        let hidden_t = Tensor::zeros(stream, &[max_tokens, hidden])?;
        let normed = Tensor::zeros(stream, &[max_tokens, hidden])?;
        let attn_out = Tensor::zeros(stream, &[max_tokens, hidden])?;
        let mlp_out = Tensor::zeros(stream, &[max_tokens, hidden])?;
        let final_normed = Tensor::zeros(stream, &[max_logits_rows, hidden])?;
        let logits_bf16 = Tensor::zeros(stream, &[max_logits_rows, v])?;
        let decode_scratch =
            Tensor::zeros(stream, &[k::gqa_decode_scratch_len(max_logits_rows, h, d)])?;
        stream.synchronize()?;
        tracing::info!(
            num_blocks,
            block_size,
            tokens = num_blocks * block_size,
            kv_gib = (num_blocks * Self::kv_bytes_per_block(&cfg, block_size)) as f64 / (1u64 << 30) as f64,
            num_slots,
            "qwen35: kv / gdn slots"
        );
        Ok(Self {
            ops: O::default(),
            cfg,
            embed,
            layers,
            norm,
            lm_head,
            rope,
            kv,
            max_tokens,
            max_logits_rows,
            tp,
            num_slots,
            hidden: hidden_t,
            normed,
            attn_out,
            mlp_out,
            final_normed,
            logits_bf16,
            decode_scratch,
        })
    }

    pub fn last_hidden(&self) -> &Tensor<bf16> {
        &self.hidden
    }

    /// 新请求首次出现时清它分到的 GDN slot（每层各一份状态）。
    pub fn reset_slots(&mut self, stream: &Stream, slots: &[usize]) -> Result<()> {
        if slots.is_empty() {
            return Ok(());
        }
        for l in self.layers.iter_mut() {
            if let Attn::Gdn(g) = &mut l.attn {
                for &s in slots {
                    g.reset_slot(stream, s, &self.cfg)?;
                }
            }
        }
        Ok(())
    }

    pub fn alloc_scratch(&self, stream: &Stream) -> Result<ModelScratch> {
        let cfg = &self.cfg;
        let (h, hk, d) = (
            cfg.num_attention_heads as usize,
            cfg.num_key_value_heads as usize,
            cfg.head_dim as usize,
        );
        let hidden = cfg.hidden_size as usize;
        let (t, rows, v) = (self.max_tokens, self.max_logits_rows, cfg.vocab_size as usize);
        let mut layers = Vec::with_capacity(self.layers.len());
        for l in &self.layers {
            let inter = l.mlp.inter;
            layers.push(LayerScratch {
                qkv_buf: Tensor::zeros(stream, &[t, (h + 2 * hk) * d])?,
                gate_buf: Tensor::zeros(stream, &[t, h * d])?,
                attn: Tensor::zeros(stream, &[t, h * d])?,
                gate_up: Tensor::zeros(stream, &[t, 2 * inter])?,
                act: Tensor::zeros(stream, &[t, inter])?,
            });
        }
        Ok(ModelScratch {
            hidden: Tensor::zeros(stream, &[t, hidden])?,
            normed: Tensor::zeros(stream, &[t, hidden])?,
            attn_out: Tensor::zeros(stream, &[t, hidden])?,
            mlp_out: Tensor::zeros(stream, &[t, hidden])?,
            final_normed: Tensor::zeros(stream, &[rows, hidden])?,
            logits_bf16: Tensor::zeros(stream, &[rows, v])?,
            decode_scratch: Tensor::zeros(stream, &[k::gqa_decode_scratch_len(rows, h, d)])?,
            layers,
        })
    }

    pub fn swap_scratch(&mut self, s: &mut ModelScratch) {
        std::mem::swap(&mut self.hidden, &mut s.hidden);
        std::mem::swap(&mut self.normed, &mut s.normed);
        std::mem::swap(&mut self.attn_out, &mut s.attn_out);
        std::mem::swap(&mut self.mlp_out, &mut s.mlp_out);
        std::mem::swap(&mut self.final_normed, &mut s.final_normed);
        std::mem::swap(&mut self.logits_bf16, &mut s.logits_bf16);
        std::mem::swap(&mut self.decode_scratch, &mut s.decode_scratch);
        for (l, ls) in self.layers.iter_mut().zip(s.layers.iter_mut()) {
            l.attn.swap_bufs(&mut ls.qkv_buf, &mut ls.gate_buf, &mut ls.attn);
            l.mlp.swap_bufs(&mut ls.gate_up, &mut ls.act);
        }
    }

    /// 诊断：捕获第 `l` 层 attention 后的 attn_out 与 MLP 后的 hidden（对拍用）。
    pub fn capture(&self, stream: &Stream, t: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let n = t * self.cfg.hidden_size as usize;
        Ok((self.attn_out.to_vec_f32(stream)?[..n].to_vec(), self.hidden.to_vec_f32(stream)?[..n].to_vec()))
    }

    pub fn forward(&mut self, stream: &Stream, gemm: &Gemm, f: &Forward<'_>, logits: &mut Tensor<f32>) -> Result<()> {
        let t = f.t;
        ensure!(t <= self.max_tokens && t <= f.input_ids.numel(), "forward: {t} tokens > max {}", self.max_tokens);
        let h = self.cfg.hidden_size as usize;
        let eps = self.cfg.rms_norm_eps;
        let world = self.tp.1;
        let (h_attn, hk, d) = (
            self.cfg.num_attention_heads as usize / world,
            self.cfg.num_key_value_heads as usize / world,
            self.cfg.head_dim as usize,
        );
        let _ = (h_attn, hk, d);
        k::embedding_bf16_n(stream, &f.input_ids.data, &self.embed.table.data, &mut self.hidden.data, t, h)?;
        // 每层：input_norm → attn → (res += attn_out) + post_norm → mlp → 下一层开头 res += mlp_out
        //（最后一层循环后补一次 add）。与 Qwen3 的 `pending_mlp` 写法一致。
        let mut pending_mlp = false;
        for (li, layer) in self.layers.iter_mut().enumerate() {
            if pending_mlp {
                self.ops.norm().fused_add_rmsnorm(
                    stream,
                    &self.mlp_out.data,
                    &mut self.hidden.data,
                    &layer.input_norm.w.data,
                    &mut self.normed.data,
                    t,
                    h,
                    eps,
                )?;
            } else {
                self.ops.norm().rmsnorm(
                    stream,
                    &self.hidden.data,
                    &layer.input_norm.w.data,
                    &mut self.normed.data,
                    t,
                    h,
                    eps,
                )?;
            }
            layer.forward(
                stream,
                gemm,
                &self.ops,
                &self.cfg,
                &self.rope,
                t,
                &self.normed,
                f.positions,
                f.slot_mapping,
                &mut self.kv[li],
                &f.meta,
                &mut self.decode_scratch,
                &mut self.attn_out,
            )?;
            self.ops.norm().fused_add_rmsnorm(
                stream,
                &self.attn_out.data,
                &mut self.hidden.data,
                &layer.post_norm.w.data,
                &mut self.normed.data,
                t,
                h,
                eps,
            )?;
            layer.mlp.forward(stream, gemm, self.ops.act(), t, &self.normed, &mut self.mlp_out)?;
            pending_mlp = true;
        }
        if pending_mlp {
            self.ops.norm().add_inplace(stream, &self.mlp_out.data, &mut self.hidden.data, t * h)?;
        }
        let rows = f.rows;
        ensure!(
            rows <= self.max_logits_rows && rows <= f.logits_rows.numel(),
            "{rows} logits rows > max_logits_rows {}",
            self.max_logits_rows
        );
        self.ops.norm().gather_rmsnorm(
            stream,
            &self.hidden.data,
            &f.logits_rows.data,
            &self.norm.w.data,
            &mut self.final_normed.data,
            rows,
            h,
            eps,
        )?;
        let v = self.cfg.vocab_size as usize;
        gemm.forward_bf16(&self.final_normed.data, &self.lm_head.w.data, None, &mut self.logits_bf16.data, rows, v, h)?;
        k::bf16_to_f32_n(stream, &self.logits_bf16.data, &mut logits.data, rows * v)
    }

    /// 只对给定行取 logits（对拍 / prompt_logprobs）；`rows` 是 hidden 行号。
    pub fn logits_for_rows(&mut self, stream: &Stream, gemm: &Gemm, rows: &[i32]) -> Result<Tensor<f32>> {
        let h = self.cfg.hidden_size as usize;
        let v = self.cfg.vocab_size as usize;
        let n = rows.len();
        let mut gathered = Tensor::<bf16>::zeros(stream, &[n, h])?;
        let idx = Tensor::from_vec(stream, rows.to_vec(), &[n])?;
        self.ops.norm().gather_rmsnorm(
            stream,
            &self.hidden.data,
            &idx.data,
            &self.norm.w.data,
            &mut gathered.data,
            n,
            h,
            self.cfg.rms_norm_eps,
        )?;
        let mut lb = Tensor::<bf16>::zeros(stream, &[n, v])?;
        gemm.forward_bf16(&gathered.data, &self.lm_head.w.data, None, &mut lb.data, n, v, h)?;
        let mut out = Tensor::<f32>::zeros(stream, &[n, v])?;
        k::bf16_to_f32_n(stream, &lb.data, &mut out.data, n * v)?;
        Ok(out)
    }
}

/// 本步是否纯 decode（GDN 用 update 路径的判据与 Qwen3 的 attention 分流同源）。
pub fn is_pure_decode(mode: BatchMode, num_reqs: usize, t: usize) -> bool {
    mode == BatchMode::Decode && t == num_reqs
}
