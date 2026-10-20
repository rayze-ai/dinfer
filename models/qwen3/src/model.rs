//! Qwen3 主干：embed → N × (RMSNorm → GQA attention → +res → RMSNorm → SwiGLU → +res) → norm → lm_head（tied）。
//! attention：q/k/v 投影 → per-head q_norm / k_norm（RMSNorm over head_dim）→ neox RoPE（整个 head_dim）→
//! 写 paged K/V cache → GQA attention（prefill 展开 / paged）→ o_proj。对照 HF `modeling_qwen3.py` 与 vLLM `qwen3.py`。
//!
//! 权重名（真 Qwen3-0.6B safetensors）：`model.embed_tokens`、`model.layers.{l}.{input_layernorm, self_attn.{q,k,v,o}_proj,
//! self_attn.{q,k}_norm, post_attention_layernorm, mlp.{gate,up,down}_proj}`、`model.norm`、`lm_head`（tied 时文件里也有一份）。

use crate::config::Qwen3Config;
use crate::scratch::ActScratch;
use anyhow::{ensure, Context, Result};
use dinfer_core::runner::BatchMode;
use dinfer_kernels as k;
use dinfer_layers::{
    bf16, Embedding, Gemm, Linear, Mlp, RmsNorm, Rope, RopeStyle, Stream, Tensor, WeightSource, DefaultOps, OpSet,
};
// 算子 trait 方法要在作用域内才能通过 `self.ops.norm().xxx()` 调用；匿名引入避免与本文件的 Attention 结构体重名
use dinfer_layers::{Act as _, Attention as _, Norm as _, QkvPost as _};
use dinfer_layers::LoadProgress;

/// 一层的 paged K/V cache：[num_blocks, block_size, kv_heads, 2, head_dim] bf16。
pub struct PagedKv {
    pub cache: Tensor<bf16>,
    pub num_blocks: usize,
    pub block_size: usize,
}

pub struct Attention {
    pub qkv: Linear, // 融合 [q | k | v]：[(H + 2·Hkv)·D, hidden]
    pub o_proj: Linear,
    pub q_norm: RmsNorm, // over head_dim
    pub k_norm: RmsNorm,
    // 激活缓冲不再每层各持一套：qkv GEMM 输出 / attention 输出借全模型共享的 `ActScratch`
    //（每层一套在大预算下把 KV 池挤爆，见 scratch.rs）。
}

pub struct DecoderLayer {
    pub input_norm: RmsNorm,
    pub attn: Attention,
    pub post_norm: RmsNorm,
    pub mlp: Mlp,
}

/// 一步 attention 的 batch 描述（runner 的 BatchPlan 上传后的设备张量）。
pub struct AttnMeta<'a> {
    pub mode: BatchMode,
    pub cu_seqlens: &'a Tensor<i32>, // [B+1]，本步每请求 query 数前缀和
    pub seq_lens: &'a Tensor<i32>,   // [B]，本步后每请求总长
    pub block_table: &'a Tensor<i32>,
    pub num_reqs: usize,
    pub max_blocks: usize,
    /// 本步单请求最大 query 数（prefill flash kernel 的 grid 上界）。
    pub max_seqlen: usize,
}

pub struct Forward<'a> {
    pub input_ids: &'a Tensor<i32>,
    pub positions: &'a Tensor<i32>,
    pub slot_mapping: &'a Tensor<i64>,
    pub meta: AttnMeta<'a>,
    pub logits_rows: &'a Tensor<i32>,
    /// 本步 token 数（缓冲可以比它大：CUDA graph 的固定地址缓冲按上限分配）。
    pub t: usize,
    /// 本步 logits 行数。
    pub rows: usize,
}

/// 调试捕获：(layer_idx, attn_out [t*h] f32, 本层输出 hidden [t*h] f32)。
pub type LayerCapture = (usize, Vec<f32>, Vec<f32>);

pub struct Qwen3Model<O: OpSet = DefaultOps> {
    /// 算子实现（design §10.1）；默认 `DefaultOps`，启动期选择。
    pub ops: O,
    pub cfg: Qwen3Config,
    pub embed: Embedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: RmsNorm,
    pub lm_head: Linear,
    pub rope: Rope,
    pub kv: Vec<PagedKv>,
    pub max_tokens: usize,
    /// 一步最多取多少行 logits（logits / final_normed 缓冲按它分配）。
    pub max_logits_rows: usize,
    /// 调试：若非空，每层 forward 后记录 (layer, attn_out[:t*h] f32, hidden_after_layer[:t*h] f32)。
    pub capture_layers: Option<Vec<LayerCapture>>,
    /// TP 组通信器（tp > 1 时由 executor 在自己的流上 init 后挂上）；o_proj / down 之后原地 all-reduce。
    pub comm: Option<k::comm::Comm>,
    /// (rank, world)；单卡 (0, 1)。
    pub tp: (usize, usize),
    /// lm_head 按 vocab（输出维）切分（`LMHEAD_COLSPLIT=1`，world > 1 且文件有独立 `lm_head.weight` 时生效）：
    /// 本 rank 只算 `vocab_local` 列 logits，采样前 all_gather 成全 vocab（见 `lm_head_logits`）。默认关，复制。
    pub lm_head_colsplit: bool,
    /// 本 rank 的 lm_head 输出列数（列切 = vocab / world；否则 = vocab）。
    pub vocab_local: usize,
    hidden: Tensor<bf16>,
    normed: Tensor<bf16>,
    attn_out: Tensor<bf16>,
    mlp_out: Tensor<bf16>,
    final_normed: Tensor<bf16>,
    logits_bf16: Tensor<bf16>,
    /// 列切时本 rank 局部 logits [max_logits_rows, vocab_local] bf16（all_gather 的 send 端）；非列切时 1 个元素占位。
    logits_local: Tensor<bf16>,
    /// decode split-K attention 的 partial 缓冲（全层共享）：[max_logits_rows × H × 8 × (D+2)] f32。
    decode_scratch: Tensor<f32>,
    /// 全模型共享的一套 per-token 激活缓冲（28 层轮流借用；每层一套在大预算下把 KV 池挤爆，见 scratch.rs）。
    pub act: ActScratch,
}


/// 启动分段（线程局部累计 µs）：Attention::load / Mlp::load 内各阶段，28 层求和后打一行日志。
mod load_phase {
    use std::cell::RefCell;
    thread_local! { static ACC: RefCell<[u64; 8]> = const { RefCell::new([0; 8]) }; }
    pub const QKV_LOAD: usize = 0;
    pub const QKV_CONCAT: usize = 1;
    pub const O_PROJ: usize = 2;
    pub const NORMS: usize = 3;
    pub const ATTN_ZEROS: usize = 4;
    pub const MLP: usize = 5;
    pub const LAYER_NORMS: usize = 6;
    pub const LAYER_TOTAL: usize = 7;
    pub fn add(i: usize, t: std::time::Instant) {
        ACC.with(|a| a.borrow_mut()[i] += t.elapsed().as_micros() as u64);
    }
    pub fn take() -> [u64; 8] {
        ACC.with(|a| std::mem::take(&mut *a.borrow_mut()))
    }
}

impl Attention {
    fn load(
        stream: &Stream,
        src: &WeightSource,
        p: &str,
        cfg: &Qwen3Config,
        (rank, world): (usize, usize),
    ) -> Result<Self> {
        // TP（design §13.2）：cfg 已是本 rank 的本地配置（头数 / tp）；q/k/v 按头行切，o_proj 按输入列切。
        let (h, hk, d) =
            (cfg.num_attention_heads as usize, cfg.num_key_value_heads as usize, cfg.head_dim as usize);
        let hidden = cfg.hidden_size as usize;
        // q/k/v 三个权重拼成一个 [(H+2Hkv)·D, hidden] 的 qkv 权重：bf16 直接上传后在设备上拼接
        // （原先走 f32 host 拷贝再转回 bf16，每层 ~30 ms，28 层占了启动的 ~0.8 s；见 loops/q3-startup.md）
        let t = std::time::Instant::now();
        let q = src.load_bf16_shard(stream, &format!("{p}.q_proj.weight"), 0, rank, world)?;
        let kw = src.load_bf16_shard(stream, &format!("{p}.k_proj.weight"), 0, rank, world)?;
        let vw = src.load_bf16_shard(stream, &format!("{p}.v_proj.weight"), 0, rank, world)?;
        ensure!(
            q.shape == [h * d, hidden] && kw.shape == [hk * d, hidden] && vw.shape == kw.shape,
            "qkv shapes {:?} {:?} {:?}",
            q.shape,
            kw.shape,
            vw.shape
        );
        load_phase::add(load_phase::QKV_LOAD, t);
        let t = std::time::Instant::now();
        let mut qkv_w = Tensor::<bf16>::zeros(stream, &[(h + 2 * hk) * d, hidden])?;
        let mut off = 0;
        for part in [&q, &kw, &vw] {
            let n = part.numel();
            let mut dst = qkv_w.data.slice_mut(off..off + n);
            stream.inner_arc().memcpy_dtod(&part.data, &mut dst)?;
            off += n;
        }
        let qkv = Linear::new(qkv_w, None)?;
        load_phase::add(load_phase::QKV_CONCAT, t);
        let t = std::time::Instant::now();
        let o_proj = Linear::load_shard(stream, src, &format!("{p}.o_proj"), 1, rank, world)?;
        load_phase::add(load_phase::O_PROJ, t);
        ensure!(o_proj.in_features == h * d && o_proj.out_features == hidden, "o_proj shape");
        let tn = std::time::Instant::now();
        let q_norm = RmsNorm::load(stream, src, &format!("{p}.q_norm"), cfg.rms_norm_eps)?;
        let k_norm = RmsNorm::load(stream, src, &format!("{p}.k_norm"), cfg.rms_norm_eps)?;
        load_phase::add(load_phase::NORMS, tn);
        Ok(Self { qkv, o_proj, q_norm, k_norm })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward<O: OpSet>(
        &mut self,
        stream: &Stream,
        gemm: &Gemm,
        ops: &O,
        cfg: &Qwen3Config,
        rope: &Rope,
        t: usize,
        x: &Tensor<bf16>,
        positions: &Tensor<i32>,
        slot_mapping: &Tensor<i64>,
        kv: &mut PagedKv,
        meta: &AttnMeta<'_>,
        decode_scratch: &mut Tensor<f32>,
        sc: &mut ActScratch,
        out: &mut Tensor<bf16>,
    ) -> Result<()> {
        let (h, hk, d) =
            (cfg.num_attention_heads as usize, cfg.num_key_value_heads as usize, cfg.head_dim as usize);
        let hidden = cfg.hidden_size as usize;
        let eps = cfg.rms_norm_eps;
        // 本层逻辑行宽：GEMM 的 ld、qkv_post 的逐行写入、attention 的 q_stride / kv_stride 三者必须都是它。
        // 共享缓冲比本层宽是安全的（只用前缀），但**不能**从 `numel / t` 反推行宽。
        let qkv_w = (h + 2 * hk) * d;
        ensure!(
            t <= sc.max_tokens && sc.qkv.numel() >= t * qkv_w && sc.attn.numel() >= t * h * d,
            "attn: t={t} 超过共享激活缓冲 (max_tokens={}, qkv={} < {} / attn={} < {})",
            sc.max_tokens,
            sc.qkv.numel(),
            t * qkv_w,
            sc.attn.numel(),
            t * h * d
        );
        // 1. qkv = x · W^T → [t, (H+2Hkv)·D]（q | k | v）
        gemm.forward_bf16(&x.data, &self.qkv.w.data, None, &mut sc.qkv.data, t, qkv_w, hidden)?;
        // 2–4. 一个 kernel：q/k per-head RMSNorm + neox RoPE（原地）+ K/V 写 paged cache（Q3-e 融合；D 必须 128）
        ensure!(d == 128 && rope.rope_dim == d, "qkv_post needs head_dim == rope_dim == 128");
        ops.qkv_post().apply(
            stream,
            &mut sc.qkv.data,
            &self.q_norm.w.data,
            &self.k_norm.w.data,
            &positions.data,
            &rope.cos_sin.data,
            &mut kv.cache.data,
            &slot_mapping.data,
            t,
            h,
            hk,
            d,
            eps,
            kv.block_size,
        )?;
        // 5. attention：q 从 qkv_buf 第 0 列起、k 从 H·D 列起、v 从 (H+Hkv)·D 列起，行跨度 qkv_w
        let scale = (d as f32).powf(-0.5);
        let q_view = sc.qkv.data.slice(..);
        let k_view = sc.qkv.data.slice(h * d..);
        let v_view = sc.qkv.data.slice((h + hk) * d..);
        match meta.mode {
            BatchMode::Prefill => ops.attn().prefill(
                stream,
                &q_view,
                &k_view,
                &v_view,
                &mut sc.attn.data,
                &meta.cu_seqlens.data,
                meta.num_reqs,
                meta.max_seqlen,
                h,
                hk,
                d,
                scale,
                qkv_w,
                qkv_w,
            )?,
            // 纯 decode 且 D=128：split-K kernel（reqs × Hkv × 8 块并行）；否则 flash paged kernel
            BatchMode::Decode if d == 128 => ops.attn().decode(
                stream,
                &q_view,
                &kv.cache.data,
                &meta.block_table.data,
                &meta.seq_lens.data,
                &mut sc.attn.data,
                &mut decode_scratch.data,
                meta.num_reqs,
                h,
                hk,
                d,
                kv.block_size,
                meta.max_blocks,
                scale,
                qkv_w,
                None,
            )?,
            // 混批（prefill 行 + decode 行）且 D=128：decode 行走 split-K kernel，其余走 paged tensor-core kernel（Q3 第 34 轮）
            BatchMode::Paged if d == 128 => ops.attn().mixed(
                stream,
                &q_view,
                &kv.cache.data,
                &meta.block_table.data,
                &meta.seq_lens.data,
                &meta.cu_seqlens.data,
                &mut sc.attn.data,
                &mut decode_scratch.data,
                meta.num_reqs,
                meta.max_seqlen,
                h,
                hk,
                d,
                kv.block_size,
                meta.max_blocks,
                scale,
                qkv_w,
            )?,
            BatchMode::Decode | BatchMode::Paged => ops.attn().paged(
                stream,
                &q_view,
                &kv.cache.data,
                &meta.block_table.data,
                &meta.seq_lens.data,
                &meta.cu_seqlens.data,
                &mut sc.attn.data,
                meta.num_reqs,
                meta.max_seqlen,
                h,
                hk,
                d,
                kv.block_size,
                meta.max_blocks,
                scale,
                qkv_w,
                false,
            )?,
        }
        // 6. o_proj
        gemm.forward_bf16(&sc.attn.data, &self.o_proj.w.data, None, &mut out.data, t, hidden, h * d)
    }
}

/// 第二套 per-step 激活缓冲（跨流并行时副流用；`Qwen3Model::swap_scratch` 整体换入 / 换出）。
/// 每层各持有的一套（原 `LayerScratch`）已并入 `ActScratch`：层串行执行，一套够用。
pub struct ModelScratch {
    pub hidden: Tensor<bf16>,
    pub normed: Tensor<bf16>,
    pub attn_out: Tensor<bf16>,
    pub mlp_out: Tensor<bf16>,
    pub final_normed: Tensor<bf16>,
    pub logits_bf16: Tensor<bf16>,
    pub logits_local: Tensor<bf16>,
    pub decode_scratch: Tensor<f32>,
    pub act: ActScratch,
}

impl<O: OpSet> Qwen3Model<O> {
    pub fn load(
        stream: &Stream,
        src: &WeightSource,
        cfg: Qwen3Config,
        max_tokens: usize,
        num_blocks: Option<usize>,
        block_size: usize,
    ) -> Result<Self>
    where
        O: Default,
    {
        // 测试 / 对拍：logits 行数上限 = max_tokens（允许取全部位置的 logits）
        Self::load_with_budget(
            stream,
            src,
            cfg,
            max_tokens,
            max_tokens,
            num_blocks,
            block_size,
            0.9,
            512 << 20,
        )
    }

    /// 每个 KV block 的字节数（全部层）：`layers × block_size × kv_heads × 2(K,V) × head_dim × 2(bf16)`。
    pub fn kv_bytes_per_block(cfg: &Qwen3Config, block_size: usize) -> usize {
        cfg.num_hidden_layers as usize
            * block_size
            * cfg.num_key_value_heads as usize
            * 2
            * cfg.head_dim as usize
            * 2
    }

    /// `max_tokens`：一步最多多少 token（激活 scratch 按它分配）；`max_logits_rows`：一步最多取多少行 logits
    /// （服务 = max_num_seqs：每请求只要最后一行；对拍 = max_tokens）。logits 按 vocab 151936 宽，不能按 max_tokens 分。
    /// `num_blocks = None` 时按显存算：权重 + 激活 scratch 分配完后读 `free`，
    /// `budget = min(free, total × gpu_memory_utilization − 已用) − reserve − executor 的 fp32 logits 副本`，
    /// `num_blocks = budget / bytes_per_block`。KV 最后分配，所以 free 已经扣掉了权重与 scratch
    /// （等价 vLLM profile run 的作用，但不用真跑一次 forward）。
    #[allow(clippy::too_many_arguments)]
    pub fn load_with_budget(
        stream: &Stream,
        src: &WeightSource,
        cfg: Qwen3Config,
        max_tokens: usize,
        max_logits_rows: usize,
        num_blocks: Option<usize>,
        block_size: usize,
        gpu_memory_utilization: f32,
        reserve_bytes: usize,
    ) -> Result<Self>
    where
        O: Default,
    {
        Self::load_with_budget_tp(stream, src, cfg, max_tokens, max_logits_rows, num_blocks, block_size, gpu_memory_utilization, reserve_bytes, (0, 1))
    }

    /// TP 版：`tp = (rank, world)`。本地 cfg 的 `num_attention_heads / num_key_value_heads / intermediate_size`
    /// 除以 world；embed / norm / lm_head 复制（design §13.2）。KV 池只存本 rank 的 kv 头。
    #[allow(clippy::too_many_arguments)]
    pub fn load_with_budget_tp(
        stream: &Stream,
        src: &WeightSource,
        cfg: Qwen3Config,
        max_tokens: usize,
        max_logits_rows: usize,
        num_blocks: Option<usize>,
        block_size: usize,
        gpu_memory_utilization: f32,
        reserve_bytes: usize,
        tp: (usize, usize),
    ) -> Result<Self>
    where
        O: Default,
    {
        let (rank, world) = tp;
        ensure!(world >= 1 && rank < world, "bad tp shard {rank}/{world}");
        let mut cfg = cfg;
        if world > 1 {
            ensure!(cfg.num_attention_heads as usize % world == 0, "num_attention_heads {} % tp {world} != 0", cfg.num_attention_heads);
            ensure!(cfg.num_key_value_heads as usize % world == 0, "num_key_value_heads {} % tp {world} != 0 (kv-head replication unsupported)", cfg.num_key_value_heads);
            ensure!(cfg.intermediate_size as usize % world == 0, "intermediate_size {} % tp {world} != 0", cfg.intermediate_size);
            cfg.num_attention_heads /= world as u32;
            cfg.num_key_value_heads /= world as u32;
            cfg.intermediate_size /= world as u32;
            tracing::info!(rank, world, heads = cfg.num_attention_heads, kv_heads = cfg.num_key_value_heads, inter = cfg.intermediate_size, "tp: local shard config");
        }
        let hidden = cfg.hidden_size as usize;
        let (hk, d) = (cfg.num_key_value_heads as usize, cfg.head_dim as usize);
        let t_load0 = std::time::Instant::now();
        let wall = |what: &str| tracing::info!(at_ms = t_load0.elapsed().as_millis() as u64, "load timing: wall after {what}");
        let mut prog = LoadProgress::new(tp);
        // 首个张量含懒 CUDA context / 模块加载，单独一行（`loaded what=embed_tokens`）
        let embed = prog.timed("embed_tokens", || Embedding::load(stream, src, "model.embed_tokens"))?;
        let layers = prog.load_layers(stream, cfg.num_hidden_layers as usize, |l| {
            let tl = std::time::Instant::now();
            let p = format!("model.layers.{l}");
            let layer = DecoderLayer {
                input_norm: RmsNorm::load(stream, src, &format!("{p}.input_layernorm"), cfg.rms_norm_eps)?,
                attn: Attention::load(stream, src, &format!("{p}.self_attn"), &cfg, tp)
                    .with_context(|| format!("layer {l} attn"))?,
                post_norm: RmsNorm::load(
                    stream,
                    src,
                    &format!("{p}.post_attention_layernorm"),
                    cfg.rms_norm_eps,
                )?,
                mlp: {
                    let tm = std::time::Instant::now();
                    // max_tokens = 0：Mlp 不自带激活缓冲，走全模型共享的 `ActScratch`（`forward_with_scratch`）
                    let m = Mlp::load_tp(stream, src, &format!("{p}.mlp"), 0, rank, world)
                        .with_context(|| format!("layer {l} mlp"))?;
                    load_phase::add(load_phase::MLP, tm);
                    m
                },
            };
            load_phase::add(load_phase::LAYER_TOTAL, tl);
            Ok((layer, "attn/dense"))
        })?;
        wall("layers loop");
        {
            let a = load_phase::take();
            let ms = |i: usize| a[i] / 1000;
            tracing::info!(
                qkv_load_ms = ms(load_phase::QKV_LOAD),
                qkv_concat_ms = ms(load_phase::QKV_CONCAT),
                o_proj_ms = ms(load_phase::O_PROJ),
                qk_norm_ms = ms(load_phase::NORMS),
                attn_zeros_ms = ms(load_phase::ATTN_ZEROS),
                mlp_ms = ms(load_phase::MLP),
                layer_total_ms = ms(load_phase::LAYER_TOTAL),
                "load timing: per-layer phases (sum over layers)"
            );
        }
        let norm = prog.timed("norm", || RmsNorm::load(stream, src, "model.norm", cfg.rms_norm_eps))?;
        // lm_head：默认复制（design §13.2）。开关 `LMHEAD_COLSPLIT=1`（loop b-nvlink B2）：world > 1 且文件里有独立
        // `lm_head.weight` 时按 vocab（输出维，dim 0）切，各 rank 只算自己那段 logits，采样前 all_gather（`lm_head_logits`）。
        // 文件无 lm_head.weight（纯 tied）时不切：embed 表要整份做 gather，再切一份 lm_head 等于多占 1/world 的 embed
        // 显存，先不做，打 info 说明。Qwen3-0.6B 虽 tie_word_embeddings=true 但文件带 lm_head.weight，列切可生效。
        let v = cfg.vocab_size as usize;
        let want_colsplit = std::env::var("LMHEAD_COLSPLIT").is_ok_and(|x| x == "1");
        let has_lm_head = src.contains("lm_head.weight");
        let lm_head_colsplit = want_colsplit && world > 1 && has_lm_head && v % world == 0;
        if want_colsplit && !lm_head_colsplit {
            let reason = if world <= 1 {
                "world == 1"
            } else if !has_lm_head {
                "no standalone lm_head.weight (tied to embed)"
            } else {
                "vocab % world != 0"
            };
            tracing::info!(rank, world, vocab = v, reason, "lm_head colsplit requested but NOT enabled");
        }
        let lm_head = if has_lm_head {
            if lm_head_colsplit {
                Linear::load_shard(stream, src, "lm_head", 0, rank, world)?
            } else {
                Linear::load(stream, src, "lm_head")?
            }
        } else {
            ensure!(cfg.tie_word_embeddings, "no lm_head.weight and not tied");
            Linear::load(stream, src, "model.embed_tokens")?
        };
        let vocab_local = lm_head.out_features;
        if lm_head_colsplit {
            tracing::info!(rank, world, vocab = v, vocab_local, "lm_head colsplit enabled (LMHEAD_COLSPLIT=1)");
        }
        wall("norm + lm_head");
        prog.finish();
        let t_rope = std::time::Instant::now();
        let rope =
            Rope::new(stream, d, cfg.max_position_embeddings as usize, cfg.rope_theta, RopeStyle::Neox)?;
        tracing::info!(rope_ms = t_rope.elapsed().as_millis() as u64, max_pos = cfg.max_position_embeddings, "load timing: rope cos/sin table");
        wall("rope");
        // 激活 scratch 先分配，KV 最后：这样 free 已含权重 + scratch 的占用
        let t_sync = std::time::Instant::now();
        stream.staged_finish()?; // pinned staging 的 DMA 全部落地
        stream.synchronize()?; // 权重 H2D 全部入队后同步一次（load_bf16 不再逐张量同步）
        let weights_sync_ms = t_sync.elapsed().as_millis() as u64;
        let ls = dinfer_layers::take_load_stats();
        let (z_us, z_bytes, z_n) = dinfer_layers::take_zeros_stats();
        tracing::info!(
            zeros_n = z_n,
            zeros_mib = z_bytes / (1 << 20),
            zeros_ms = z_us / 1000,
            "load timing: per-layer activation buffers (should be 0: 共享 ActScratch，见下一条)"
        );
        tracing::info!(
            tensors = ls.tensors,
            mib = ls.bytes / (1 << 20),
            mmap_touch_ms = ls.touch_us / 1000,
            host_convert_ms = ls.host_us / 1000,
            h2d_enqueue_ms = ls.h2d_us / 1000,
            final_sync_ms = weights_sync_ms,
            h2d_gib_s = if ls.h2d_us > 0 { ls.bytes as f64 / (1u64 << 30) as f64 / (ls.h2d_us as f64 * 1e-6) } else { 0.0 },
            "load timing: weights (mmap→host Vec → H2D)"
        );
        let t_scratch = std::time::Instant::now();
        let hidden_t = Tensor::zeros(stream, &[max_tokens, hidden])?;
        let normed = Tensor::zeros(stream, &[max_tokens, hidden])?;
        let attn_out = Tensor::zeros(stream, &[max_tokens, hidden])?;
        let mlp_out = Tensor::zeros(stream, &[max_tokens, hidden])?;
        let final_normed = Tensor::zeros(stream, &[max_logits_rows, hidden])?;
        let logits_bf16 = Tensor::zeros(stream, &[max_logits_rows, v])?;
        let logits_local = Tensor::zeros(stream, &[if lm_head_colsplit { max_logits_rows * vocab_local } else { 1 }])?;
        let decode_scratch = Tensor::zeros(
            stream,
            &[k::gqa_decode_scratch_len(max_logits_rows, cfg.num_attention_heads as usize, d)],
        )?;
        // 全模型共享一套 per-token 激活缓冲（原先 Attention::load / Mlp::load_tp 每层各一套，
        // 预算 16384 时 28 层 ≈ 14 GB 白占，见 scratch.rs）。必须在读 free 算 KV 预算之前分配。
        let act_bytes = ActScratch::bytes(&cfg, max_tokens);
        let act = ActScratch::new(stream, &cfg, max_tokens)?;
        // executor 还会分配 fp32 logits [max_logits_rows, vocab] + 采样工作区 + logprobs 原始副本（各 ≤ 同大小）
        let exec_scratch = 3 * max_logits_rows * v * 4;

        stream.synchronize()?;
        let (z_us, z_bytes, z_n) = dinfer_layers::take_zeros_stats();
        tracing::info!(
            scratch_ms = t_scratch.elapsed().as_millis() as u64,
            zeros_n = z_n,
            zeros_mib = z_bytes / (1 << 20),
            zeros_ms = z_us / 1000,
            act_scratch_rows = max_tokens,
            act_scratch_bytes = act_bytes,
            act_scratch_mib = act_bytes / (1 << 20),
            "load timing: activation scratch (含全模型共享的 ActScratch：qkv/attn/gate_up/act 各一套)"
        );
        let t_kv = std::time::Instant::now();
        let per_block = Self::kv_bytes_per_block(&cfg, block_size);
        let num_blocks = match num_blocks {
            Some(n) => {
                ensure!(n > 0, "num_blocks must be > 0");
                tracing::info!(
                    num_blocks = n,
                    tokens = n * block_size,
                    kv_gib = n as f64 * per_block as f64 / (1u64 << 30) as f64,
                    "kv_cache: fixed"
                );
                n
            }
            None => {
                let t_mi = std::time::Instant::now();
                stream.synchronize()?;
                let (free, total) = stream.mem_info()?;
                tracing::info!(mem_info_ms = t_mi.elapsed().as_millis() as u64, "load timing: kv budget mem_info");
                let used = total.saturating_sub(free);
                let cap = (total as f64 * gpu_memory_utilization as f64) as usize;
                let budget = free.min(cap.saturating_sub(used)).saturating_sub(reserve_bytes + exec_scratch);
                let n = budget / per_block;
                ensure!(
                    n >= 1,
                    "not enough GPU memory for KV cache: free {:.2} GiB, cap {:.0}% of {:.2} GiB, reserve {:.2} GiB",
                    free as f64 / (1u64 << 30) as f64,
                    gpu_memory_utilization * 100.0,
                    total as f64 / (1u64 << 30) as f64,
                    (reserve_bytes + exec_scratch) as f64 / (1u64 << 30) as f64
                );
                tracing::info!(
                    num_blocks = n,
                    tokens = n * block_size,
                    kv_gib = n as f64 * per_block as f64 / (1u64 << 30) as f64,
                    available_gib = free as f64 / (1u64 << 30) as f64,
                    total_gib = total as f64 / (1u64 << 30) as f64,
                    utilization = gpu_memory_utilization,
                    "kv_cache: auto"
                );
                n
            }
        };
        let mut kv = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for _ in 0..cfg.num_hidden_layers {
            kv.push(PagedKv {
                // 不清零：块在写入（qkv_post / cache_write）之前不会被任何 block_table 引用
                cache: Tensor::uninit(stream, &[num_blocks, block_size, hk, 2, d])?,
                num_blocks,
                block_size,
            });
        }
        stream.synchronize()?;
        tracing::info!(kv_ms = t_kv.elapsed().as_millis() as u64, "load timing: kv cache alloc");
        wall("kv (end of load_with_budget)");
        Ok(Self {
            comm: None,
            tp,
            lm_head_colsplit,
            vocab_local,
            ops: O::default(),
            embed,
            layers,
            norm,
            lm_head,
            rope,
            kv,
            max_tokens,
            max_logits_rows,
            capture_layers: None,
            hidden: hidden_t,
            normed,
            attn_out,
            mlp_out,
            final_normed,
            logits_bf16,
            logits_local,
            decode_scratch,
            act,
            cfg,
        })
    }

    /// 实际分配的 KV block 数。
    pub fn num_blocks(&self) -> usize {
        self.kv.first().map_or(0, |k| k.num_blocks)
    }

    /// 上一次 forward 后最后一层输出（norm 前）[t, hidden]。
    pub fn last_hidden(&self) -> &Tensor<bf16> {
        &self.hidden
    }

    /// 分配第二套激活缓冲（形状同主套：max_tokens / max_logits_rows）。
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
            logits_local: Tensor::zeros(stream, &[if self.lm_head_colsplit { rows * self.vocab_local } else { 1 }])?,
            decode_scratch: Tensor::zeros(stream, &[k::gqa_decode_scratch_len(rows, h, d)])?,
            act: ActScratch::new(stream, cfg, t)?,
        })
    }
    /// 整体互换当前激活缓冲与 `s`（O(层数) 次指针交换，无拷贝）。
    pub fn swap_scratch(&mut self, s: &mut ModelScratch) {
        std::mem::swap(&mut self.hidden, &mut s.hidden);
        std::mem::swap(&mut self.normed, &mut s.normed);
        std::mem::swap(&mut self.attn_out, &mut s.attn_out);
        std::mem::swap(&mut self.mlp_out, &mut s.mlp_out);
        std::mem::swap(&mut self.final_normed, &mut s.final_normed);
        std::mem::swap(&mut self.logits_bf16, &mut s.logits_bf16);
        std::mem::swap(&mut self.logits_local, &mut s.logits_local);
        std::mem::swap(&mut self.decode_scratch, &mut s.decode_scratch);
        std::mem::swap(&mut self.act, &mut s.act);
    }
    pub fn forward(
        &mut self,
        stream: &Stream,
        gemm: &Gemm,
        f: &Forward<'_>,
        logits: &mut Tensor<f32>,
    ) -> Result<()> {
        let t = f.t;
        ensure!(
            t <= self.max_tokens && t <= f.input_ids.numel(),
            "forward: {t} tokens > max {}",
            self.max_tokens
        );
        let h = self.cfg.hidden_size as usize;
        let eps = self.cfg.rms_norm_eps;
        k::embedding_bf16_n(stream, &f.input_ids.data, &self.embed.table.data, &mut self.hidden.data, t, h)?;
        // 残差：hidden 是主干；第 l 层 MLP 的输出不单独加回，而是在第 l+1 层的 input_norm 里用 fused_add 一次完成
        //（最后一层之后再补一次加）。省掉每层一次 add kernel。
        // TP（design §13.6 T3）：两处 all-reduce 都与紧随的 fused_add_rmsnorm 合成一个 kernel
        //（`all_reduce_add_rmsnorm_bf16`：小消息 one-shot 直接读 peer 镜像区求和，和不落回 attn_out / mlp_out；
        // 大消息或无 P2P 时内部退化为 NCCL + fused_add_rmsnorm）。pending_mlp 在 TP 下表示 "上一层 mlp_out 还是
        // **本 rank 的部分和**"，在下一层开头的融合 kernel 里一并 all-reduce。
        let mut pending_mlp = false;
        let comm = self.comm.as_ref();
        for (li, layer) in self.layers.iter_mut().enumerate() {
            if pending_mlp {
                if let Some(c) = comm {
                    k::comm::all_reduce_add_rmsnorm_bf16(
                        c,
                        &mut self.mlp_out.data,
                        &mut self.hidden.data,
                        &layer.input_norm.w.data,
                        &mut self.normed.data,
                        t,
                        h,
                        eps,
                    )?; // TP all-reduce ②（上一层 down 部分和）+ 残差 + input_norm
                } else {
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
                }
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
            layer.attn.forward(
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
            if let Some(c) = comm {
                k::comm::all_reduce_add_rmsnorm_bf16(
                    c,
                    &mut self.attn_out.data,
                    &mut self.hidden.data,
                    &layer.post_norm.w.data,
                    &mut self.normed.data,
                    t,
                    h,
                    eps,
                )?; // TP all-reduce ①（o_proj 部分和）+ 残差 + post_norm
            } else {
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
            }
            // mlp 不再自带缓冲（load 时 max_tokens = 0），借全模型共享的一套
            layer.mlp.forward_with_scratch(
                stream,
                gemm,
                self.ops.act(),
                t,
                &self.normed,
                &mut self.mlp_out,
                &mut self.act.gate_up,
                &mut self.act.act,
            )?;
            // TP all-reduce ②（down 部分和）推迟到下一层开头 / 循环后的补加里与 fused_add 合成
            pending_mlp = true;
            if let Some(cap) = self.capture_layers.as_mut() {
                // 诊断捕获需要本层完整输出：这里显式 all-reduce + 加一次（只在捕获模式下，保留旧行为）
                if let Some(c) = comm {
                    k::comm::all_reduce_bf16(c, &mut self.mlp_out.data, t * h)?;
                }
                self.ops.norm().add_inplace(stream, &self.mlp_out.data, &mut self.hidden.data, t * h)?;
                pending_mlp = false;
                let a = self.attn_out.to_vec_f32(stream)?[..t * h].to_vec();
                let hd = self.hidden.to_vec_f32(stream)?[..t * h].to_vec();
                cap.push((li, a, hd));
            }
        }
        if pending_mlp {
            if let Some(c) = comm {
                k::comm::all_reduce_bf16(c, &mut self.mlp_out.data, t * h)?; // 最后一层的 all-reduce ②（one-shot / NCCL）
            }
            self.ops.norm().add_inplace(stream, &self.mlp_out.data, &mut self.hidden.data, t * h)?;
        }
        let rows = f.rows;
        ensure!(
            rows <= self.max_logits_rows && rows <= f.logits_rows.numel(),
            "{rows} logits rows > max_logits_rows {}",
            self.max_logits_rows
        );
        // 取每请求最后一行 + 末层 RMSNorm 一个 kernel（原 gather_rows + rmsnorm 两次）
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
        self.lm_head_logits(stream, gemm, rows, &mut logits.data)
    }

    /// `final_normed[..rows] · lm_head^T` → fp32 `logits[..rows*vocab]`。
    /// 列切（`lm_head_colsplit`）：GEMM 只算本 rank 的 `vocab_local` 列到 `logits_local`，再 bf16 all_gather 进 `logits_bf16`
    ///（NCCL 出来是 `[world][rows][vocab_local]` 块布局），最后 `gather_blocks_bf16_to_f32` 一趟修正布局 + 升 f32。
    /// 选这条而不是"GEMM 直接写转置布局"（要改 GEMM 的 ldc / 子矩阵接口）或"gather 后 transpose + bf16_to_f32 两趟"
    ///（多一次 rows×vocab 读写）；传 bf16 而非 f32 让 NVLink 字节减半（decode t=32：9.7 MB vs 19 MB）。
    /// 非列切路径与原来逐位相同（GEMM 全 vocab + bf16_to_f32）。都在 CUDA graph 内（NCCL 可捕获）。
    /// TODO(B2 方案 B)：各 rank 局部 top-k / argmax 后归并，免传全 vocab logits；需改采样流水，bench 说话再做。
    fn lm_head_logits(
        &mut self,
        stream: &Stream,
        gemm: &Gemm,
        rows: usize,
        logits: &mut k::cudarc::driver::CudaSlice<f32>,
    ) -> Result<()> {
        let h = self.cfg.hidden_size as usize;
        let v = self.cfg.vocab_size as usize;
        ensure!(logits.len() >= rows * v, "lm_head_logits: logits buffer {} < {}", logits.len(), rows * v);
        if !self.lm_head_colsplit {
            gemm.forward_bf16(&self.final_normed.data, &self.lm_head.w.data, None, &mut self.logits_bf16.data, rows, v, h)?;
            return k::bf16_to_f32_n(stream, &self.logits_bf16.data, logits, rows * v);
        }
        let comm = self.comm.as_ref().ok_or_else(|| anyhow::anyhow!("lm_head colsplit: comm not set (init_tp before forward)"))?;
        let world = self.tp.1;
        let vl = self.vocab_local;
        ensure!(comm.world() == world && vl * world == v, "lm_head colsplit: comm world {} / tp {world} / vocab_local {vl} × world != {v}", comm.world());
        gemm.forward_bf16(&self.final_normed.data, &self.lm_head.w.data, None, &mut self.logits_local.data, rows, vl, h)?;
        k::comm::all_gather_bf16(comm, &self.logits_local.data, &mut self.logits_bf16.data, rows * vl)?;
        k::gather_blocks_bf16_to_f32(stream, &self.logits_bf16.data, logits, world, rows, vl)
    }

    /// `prompt_logprobs` 用：对上一次 `forward` 留在 `hidden` 里的任意 `rows[..n]` 行（n ≤ max_logits_rows）再做一次
    /// 末层 RMSNorm + lm_head，fp32 logits 写进 `logits[..n*vocab]`。只在 prefill 步、请求要了 prompt_logprobs 时调用
    ///（forward 本身只取每请求最后一行，不为此多算）。
    pub fn logits_for_rows(
        &mut self,
        stream: &Stream,
        gemm: &Gemm,
        rows: &k::cudarc::driver::CudaSlice<i32>,
        n: usize,
        logits: &mut k::cudarc::driver::CudaSlice<f32>,
    ) -> Result<()> {
        ensure!(n <= self.max_logits_rows && rows.len() >= n, "logits_for_rows: {n} > max_logits_rows");
        let h = self.cfg.hidden_size as usize;
        let v = self.cfg.vocab_size as usize;
        ensure!(logits.len() >= n * v, "logits_for_rows: logits buffer");
        self.ops.norm().gather_rmsnorm(
            stream,
            &self.hidden.data,
            rows,
            &self.norm.w.data,
            &mut self.final_normed.data,
            n,
            h,
            self.cfg.rms_norm_eps,
        )?;
        self.lm_head_logits(stream, gemm, n, logits)
    }
}
