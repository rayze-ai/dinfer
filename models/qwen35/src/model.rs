//! Qwen3.5 主干：embed → 64 层（48 GDN + 16 全注意力，按 `layer_types`）→ norm → lm_head（**不 tie**）。
//!
//! 与 Qwen3 的骨架相同（残差 + `fused_add_rmsnorm` 省一次 add kernel），两处不同：
//! - 层是异构的：全注意力层有 paged KV，GDN 层有自己的 conv / 递推状态（`GdnLayer::state`，按 slot 自管）；
//! - 输出门 / partial RoPE 在 `full_attn.rs` 里处理，本文件只做主干与 logits。
//!
//! `lm_head.weight` 独立（`tie_word_embeddings false`）。
use crate::config::Qwen35Config;
use crate::full_attn::PagedKv;
use crate::scratch::ActScratch;
use crate::layer::{Attn, DecoderLayer, StepMeta};
use crate::weights::Table;
use anyhow::{ensure, Context, Result};
use dinfer_core::runner::BatchMode;
use dinfer_kernels as k;
use dinfer_layers::{bf16, Embedding, Gemm, Linear, OpSet, RmsNorm, Rope, RopeStyle, Stream, Tensor, WeightSource};
use dinfer_layers::Norm as _;
use dinfer_layers::LoadProgress;

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

pub struct ModelScratch {
    pub hidden: Tensor<bf16>,
    pub normed: Tensor<bf16>,
    pub attn_out: Tensor<bf16>,
    pub mlp_out: Tensor<bf16>,
    pub final_normed: Tensor<bf16>,
    pub logits_bf16: Tensor<bf16>,
    pub decode_scratch: Tensor<f32>,
    /// 全模型共享的 per-token 激活缓冲（GDN / 全注意力 / Mlp 轮流借用）。
    pub act: ActScratch,
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
    /// 递归状态快照槽数（design §15.7；0 = 无快照区）。
    pub num_snapshots: usize,
    hidden: Tensor<bf16>,
    normed: Tensor<bf16>,
    act: ActScratch,
    attn_out: Tensor<bf16>,
    mlp_out: Tensor<bf16>,
    final_normed: Tensor<bf16>,
    logits_bf16: Tensor<bf16>,
    decode_scratch: Tensor<f32>,
    /// `QWEN35_TRACE`：逐层 D2H 打范数（启动时读一次；executor 在 CUDA graph 捕获期间临时关掉，捕获期不能 D2H）。
    pub trace: bool,
}

impl<O: OpSet> Qwen35Model<O> {
    /// GEMM 自调优的形状表：按 (N, K, 有无 bias) 去重，每个形状取第一层的真权重（各层形状相同，只调一次）。
    pub fn gemm_tune_entries(&self) -> Vec<dinfer_kernels::GemmTuneEntry<'_>> {
        let mut all: Vec<(&str, &Linear)> = Vec::new();
        for l in &self.layers {
            match &l.attn {
                crate::layer::Attn::Gdn(g) => all.extend(g.linears()),
                crate::layer::Attn::Full(f) => all.extend(f.linears()),
            }
            all.push(("mlp.gate_up", &l.mlp.gate_up));
            all.push(("mlp.down", &l.mlp.down));
        }
        all.push(("lm_head", &self.lm_head));
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for (name, l) in all {
            if seen.insert((l.out_features, l.in_features, l.b.is_some())) {
                out.push(dinfer_kernels::GemmTuneEntry {
                    name: name.to_string(),
                    n: l.out_features,
                    k: l.in_features,
                    w: &l.w.data,
                    bias: l.b.as_ref().map(|b| &b.data),
                });
            }
        }
        out
    }

    /// `num_slots`：GDN 状态最多容纳多少并发请求（= `max_num_seqs`）。
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        stream: &Stream,
        src: &WeightSource,
        cfg: Qwen35Config,
        max_tokens: usize,
        num_blocks: Option<usize>,
        block_size: usize,
        num_slots: usize,
    ) -> Result<Self>
    where
        O: Default,
    {
        Self::load_with_budget(stream, src, cfg, max_tokens, max_tokens, num_blocks, block_size, num_slots, (0, 1), 0.9, 512 << 20, Some(0))
    }

    /// 每层每个请求的 GDN 状态字节数 × 层数：conv `[conv_dim, width-1]` bf16 + 递推 `[HV, V, K]` f32。
    /// 真 27B ≈ 152 MB / slot（48 层），`max_num_seqs` 每加 1 就多吃这么多显存——自动 KV 预算里要算进去。
    pub fn gdn_state_bytes_per_slot(cfg: &Qwen35Config) -> usize {
        let n_lin = cfg.layer_types.iter().filter(|t| **t == crate::config::LayerType::LinearAttention).count();
        let conv_dim = 2 * cfg.linear_num_key_heads as usize * cfg.linear_key_head_dim as usize
            + cfg.linear_num_value_heads as usize * cfg.linear_value_head_dim as usize;
        let conv = conv_dim * (cfg.linear_conv_kernel_dim as usize - 1) * 2;
        let rec = cfg.linear_num_value_heads as usize
            * cfg.linear_value_head_dim as usize
            * cfg.linear_key_head_dim as usize
            * 4;
        n_lin * (conv + rec)
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

    /// `num_blocks = None`：权重 + 激活 scratch 分配完之后读剩余显存，按 `gpu_memory_utilization` 算 KV 块数
    /// （等价 vLLM 的 profile run，但不用真跑一次 forward；K3/GLM/Qwen3 同法）。
    #[allow(clippy::too_many_arguments)]
    pub fn load_with_budget(
        stream: &Stream,
        src: &WeightSource,
        cfg: Qwen35Config,
        max_tokens: usize,
        max_logits_rows: usize,
        num_blocks: Option<usize>,
        block_size: usize,
        num_slots: usize,
        tp: (usize, usize),
        gpu_memory_utilization: f32,
        reserve_bytes: usize,
        snapshot_bytes: Option<usize>,
    ) -> Result<Self>
    where
        O: Default,
    {
        let (rank, world) = tp;
        // 里程碑 4 不做 TP 通信（executor 里没有 all-reduce）；分片装载本身是对的，但跑起来会少一半结果，
        // 所以直接拒绝，而不是给一个静默错的路径。接 TP 时：加 comm、把 cfg 按 world 切、kernels 侧不用改。
        ensure!(world == 1 && rank == 0, "qwen35: TP（world {world}）尚未实现；请用单卡（design-qwen35 §3）");
        ensure!(num_slots > 0, "num_slots must be > 0");
        if let Some(n) = num_blocks {
            ensure!(n > 0, "num_blocks must be > 0");
        }
        let mut cfg = cfg;
        let _ = &mut cfg;
        let mut prog = LoadProgress::new(tp);
        let table = Table::open(src)?;
        let hidden = cfg.hidden_size as usize;
        let (h, hk, d) = (
            cfg.num_attention_heads as usize,
            cfg.num_key_value_heads as usize,
            cfg.head_dim as usize,
        );
        let embed = prog.timed("embed_tokens", || table.embedding(stream, src, "embed_tokens.weight"))?;
        let n = cfg.num_hidden_layers as usize;
        ensure!(cfg.layer_types.len() == n, "layer_types {} != num_hidden_layers {n}", cfg.layer_types.len());
        let mut layers = prog.load_layers(stream, n, |l| {
            let layer = DecoderLayer::load(stream, src, &table, l, &cfg, max_tokens, num_slots, tp)
                .with_context(|| format!("layer {l}"))?;
            let kind = if layer.attn.is_full() { "full_attn" } else { "gdn" };
            Ok((layer, kind))
        })?;
        let norm = prog.timed("norm", || table.rmsnorm(stream, src, "norm.weight", cfg.rms_norm_eps))?;
        let lm_head = prog.timed("lm_head", || Linear::new(table.load_bf16(stream, src, "lm_head.weight")?, None))?;
        ensure!(
            lm_head.out_features == cfg.vocab_size as usize && lm_head.in_features == hidden,
            "lm_head shape {:?}（期望 [{}, {hidden}]）；tie_word_embeddings = {}",
            lm_head.w.shape,
            cfg.vocab_size,
            cfg.tie_word_embeddings
        );
        let rope = Rope::new(stream, cfg.rope_dim() as usize, cfg.max_position_embeddings as usize, cfg.rope_theta(), RopeStyle::Neox)?;
        stream.synchronize()?;
        prog.finish();

        let v = cfg.vocab_size as usize;
        let hidden_t = Tensor::zeros(stream, &[max_tokens, hidden])?;
        let normed = Tensor::zeros(stream, &[max_tokens, hidden])?;
        let attn_out = Tensor::zeros(stream, &[max_tokens, hidden])?;
        let mlp_out = Tensor::zeros(stream, &[max_tokens, hidden])?;
        let final_normed = Tensor::zeros(stream, &[max_logits_rows, hidden])?;
        let logits_bf16 = Tensor::zeros(stream, &[max_logits_rows, v])?;
        let decode_scratch =
            Tensor::zeros(stream, &[k::gqa_decode_scratch_len(max_logits_rows, h, d)])?;
        // 全模型共享一套 per-token 激活缓冲（原先每层各一套，预算 16384 时 64 层 ≈ 180 GB，见 scratch.rs）。
        let act = ActScratch::new(stream, &cfg, max_tokens)?;
        stream.synchronize()?;
        let act_bytes = ActScratch::bytes(&cfg, max_tokens);
        tracing::info!(
            gib = act_bytes as f64 / (1u64 << 30) as f64,
            "qwen35: activation scratch bytes={act_bytes} (max_tokens={max_tokens})"
        );
        // KV 最后分配：此时 free 已扣掉权重与 scratch（等价 vLLM profile run 的作用，但不真跑 forward）。
        let per_block = Self::kv_bytes_per_block(&cfg, block_size).max(1);
        let exec_scratch = 3 * max_logits_rows * v * 4; // executor 的 fp32 logits / 采样工作区 / logprobs 副本
        // 递归状态快照区（design §15.7）：`snapshot_bytes` None = 自动拿 KV 预算的 10%；Some(0) = 关；Some(b) = 指定字节。
        // 先于 KV 块数计算：快照区从同一预算里扣（`--num-blocks` 指定时 None 按 10% × 块数×块字节折算）。
        let snap_per = Self::gdn_state_bytes_per_slot(&cfg).max(1);
        let (free0, total0) = stream.mem_info()?;
        let budget0 = {
            let used = total0.saturating_sub(free0);
            let cap = (total0 as f64 * gpu_memory_utilization as f64) as usize;
            free0.min(cap.saturating_sub(used)).saturating_sub(reserve_bytes + exec_scratch)
        };
        let snapshot_bytes = match (snapshot_bytes, num_blocks) {
            (Some(b), _) => b,
            (None, Some(nb)) => nb * per_block / 10,
            (None, None) => budget0 / 10,
        };
        let num_snapshots = (snapshot_bytes / snap_per).min(budget0 / snap_per);
        if num_snapshots > 0 {
            for l in layers.iter_mut() {
                if let Attn::Gdn(g) = &mut l.attn {
                    g.alloc_snapshots(stream, num_snapshots)?;
                }
            }
            stream.synchronize()?;
        }
        tracing::info!(
            num_snapshots,
            snapshot_gib = (num_snapshots * snap_per) as f64 / (1u64 << 30) as f64,
            per_snapshot_mib = snap_per as f64 / (1u64 << 20) as f64,
            "qwen35: GDN state snapshot pool (design §15.7)"
        );
        let num_blocks = match num_blocks {
            Some(n) => n,
            None => {
                let (free, total) = stream.mem_info()?;
                let used = total.saturating_sub(free);
                let cap = (total as f64 * gpu_memory_utilization as f64) as usize;
                let budget = free.min(cap.saturating_sub(used)).saturating_sub(reserve_bytes + exec_scratch);
                let nb = budget / per_block;
                ensure!(
                    nb >= 1,
                    "not enough GPU memory for KV cache: free {:.2} GiB, cap {:.0}% of {:.2} GiB, reserve {:.2} GiB",
                    free as f64 / (1u64 << 30) as f64,
                    gpu_memory_utilization * 100.0,
                    total as f64 / (1u64 << 30) as f64,
                    (reserve_bytes + exec_scratch) as f64 / (1u64 << 30) as f64
                );
                nb
            }
        };
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
        stream.synchronize()?;
        // GDN 状态显存（真 27B：每 slot ≈ 152 MB = 48 层 × (48·128·128 f32 + 10240·3 bf16)），随 max_num_seqs 线性增长。
        let gdn_bytes_per_slot = Self::gdn_state_bytes_per_slot(&cfg);
        tracing::info!(
            gdn_slots = num_slots,
            gdn_state_gib = (gdn_bytes_per_slot * num_slots) as f64 / (1u64 << 30) as f64,
            "qwen35: GDN state buffers"
        );
        tracing::info!(
            num_blocks,
            block_size,
            tokens = num_blocks * block_size,
            kv_gib = (num_blocks * per_block) as f64 / (1u64 << 30) as f64,
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
            num_snapshots,
            hidden: hidden_t,
            normed,
            act,
            attn_out,
            mlp_out,
            final_normed,
            logits_bf16,
            decode_scratch,
            trace: std::env::var("QWEN35_TRACE").is_ok(),
        })
    }

    /// 全注意力层的 KV 块数。
    pub fn num_blocks(&self) -> usize {
        self.kv.iter().flatten().next().map_or(0, |k| k.num_blocks)
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

    /// 递归状态按块缓存（design §15.7）：活槽 → 快照槽（prefill 步 forward 之后）。
    pub fn save_snapshots(&mut self, stream: &Stream, pairs: &[(usize, usize)]) -> Result<()> {
        if pairs.is_empty() {
            return Ok(());
        }
        for l in self.layers.iter_mut() {
            if let Attn::Gdn(g) = &mut l.attn {
                for &(slot, snap) in pairs {
                    g.state.save_snapshot(stream, slot, snap)?;
                }
            }
        }
        Ok(())
    }

    /// 快照槽 → 活槽（forward 之前；之后该请求 `has_initial = 1`）。
    pub fn restore_snapshots(&mut self, stream: &Stream, pairs: &[(usize, usize)]) -> Result<()> {
        if pairs.is_empty() {
            return Ok(());
        }
        for l in self.layers.iter_mut() {
            if let Attn::Gdn(g) = &mut l.attn {
                for &(snap, slot) in pairs {
                    g.state.restore_snapshot(stream, snap, slot)?;
                }
            }
        }
        Ok(())
    }

    /// 第二套激活缓冲（跨流并行用；形状同主套）。
    pub fn alloc_scratch(&self, stream: &Stream) -> Result<ModelScratch> {
        let cfg = &self.cfg;
        let (h, d) = (cfg.num_attention_heads as usize, cfg.head_dim as usize);
        let hidden = cfg.hidden_size as usize;
        let (t, rows, v) = (self.max_tokens, self.max_logits_rows, cfg.vocab_size as usize);
        Ok(ModelScratch {
            hidden: Tensor::zeros(stream, &[t, hidden])?,
            normed: Tensor::zeros(stream, &[t, hidden])?,
            attn_out: Tensor::zeros(stream, &[t, hidden])?,
            mlp_out: Tensor::zeros(stream, &[t, hidden])?,
            final_normed: Tensor::zeros(stream, &[rows, hidden])?,
            logits_bf16: Tensor::zeros(stream, &[rows, v])?,
            decode_scratch: Tensor::zeros(stream, &[k::gqa_decode_scratch_len(rows, h, d)])?,
            act: ActScratch::new(stream, cfg, t)?,
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
        std::mem::swap(&mut self.act, &mut s.act);
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
        k::embedding_bf16_n(stream, &f.input_ids.data, &self.embed.table.data, &mut self.hidden.data, t, h)?;
        // 每层：input_norm → attn → (res += attn_out) + post_norm → mlp → 下一层开头 res += mlp_out
        //（最后一层循环后补一次 add）。与 Qwen3 的 `pending_mlp` 写法一致。
        let mut pending_mlp = false;
        let trace = self.trace;
        if trace {
            let n = trace_norm(stream, &self.hidden, &self.hidden, t, h)?;
            tracing::info!(norm = n.0, "qwen35 trace embed");
        }
        // `LAYER_PROFILE=1`：分层分阶段墙钟计时（诊断 prefill 缺口用；只在 t 大时开，避免 decode 噪声）。
        let lprof = std::env::var("LAYER_PROFILE").ok().as_deref() == Some("1") && t >= 256;
        let (mut p_norm, mut p_attn, mut p_mlp, mut p_post) = (0f64, 0f64, 0f64, 0f64);
        let mut tp0 = std::time::Instant::now();
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
            if lprof { p_norm += tp0.elapsed().as_secs_f64() * 1e3; tp0 = std::time::Instant::now(); }
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
                &mut self.act,
                &mut self.attn_out,
            )?;
            if lprof { p_attn += tp0.elapsed().as_secs_f64() * 1e3; tp0 = std::time::Instant::now(); }
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
            if lprof { p_post += tp0.elapsed().as_secs_f64() * 1e3; tp0 = std::time::Instant::now(); }
            layer.mlp.forward_with_scratch(stream, gemm, self.ops.act(), t, &self.normed, &mut self.mlp_out, &mut self.act.gate_up, &mut self.act.act)?;
            if lprof { p_mlp += tp0.elapsed().as_secs_f64() * 1e3; tp0 = std::time::Instant::now(); }
            pending_mlp = true;
            if trace {
                // 调试：最后一个 token 的残差（含本层 mlp 输出）范数，与 HF `output_hidden_states` 逐层对拍
                let n = trace_norm(stream, &self.hidden, &self.mlp_out, t, h)?;
                tracing::info!(layer = li, kind = ?layer.kind(), norm = n.0, attn_out = n.1, "qwen35 trace");
            }
        }
        if lprof {
            let tot = p_norm + p_attn + p_post + p_mlp;
            eprintln!("[layer-prof] t={t} layers={} total={tot:.2}ms | norm={p_norm:.1} attn={p_attn:.1} postnorm={p_post:.1} mlp={p_mlp:.1}",
                self.layers.len());
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

/// 调试：取最后一个 token 行，返回 (|hidden + add| , |add|) 的 L2 范数（同步 D2H，只在 QWEN35_TRACE 下调用）。
fn trace_norm(stream: &Stream, hidden: &Tensor<bf16>, add: &Tensor<bf16>, t: usize, h: usize) -> Result<(f32, f32)> {
    let a = stream.dtoh_sync_copy_view(&hidden.data.slice((t - 1) * h..t * h))?;
    let b = stream.dtoh_sync_copy_view(&add.data.slice((t - 1) * h..t * h))?;
    let same = std::ptr::eq(hidden, add);
    let (mut s1, mut s2) = (0f64, 0f64);
    for i in 0..h {
        let x = a[i].to_f32() as f64;
        let y = if same { 0.0 } else { b[i].to_f32() as f64 };
        s1 += (x + y) * (x + y);
        s2 += y * y;
    }
    Ok((s1.sqrt() as f32, s2.sqrt() as f32))
}
