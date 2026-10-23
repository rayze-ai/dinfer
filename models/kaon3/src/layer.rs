//! Gemma4 解码层（Kaon-V3）。
//!
//! 结构（vLLM `gemma4.py` 的 `Gemma4DecoderLayer.forward`，见 `docs/loops/kaon-v3.md`
//! 「Gemma4 前向语义」）：
//!
//! ```text
//! residual = x
//! h = input_layernorm(residual)
//! h = self_attn(h)                     # 内部：qkv → q_norm/k_norm(逐头，带权重) → rope → v_norm(无权重) → gqa(窗口) → o_proj
//! h = post_attention_layernorm(h)
//! h = h + residual                     # 残差加在 attn 之后
//! residual = h
//! h1 = post_ff_layernorm_1( mlp(pre_ff_layernorm(h)) )      # dense MLP，无条件执行
//! h2 = post_ff_layernorm_2( moe(pre_ff_layernorm_2(residual)) )  # MoE 与 dense **并联**
//! h  = post_ff_layernorm(h1 + h2)
//! h  = h + residual
//! out = h * layer_scalar                # 每层一个标量
//! ```
//!
//! 与 Qwen 系的最大差别：**dense MLP 和 MoE 不是二选一，是两条并行支路相加**；
//! 残差加在 norm **之后**（Gemma 式）。照着 vLLM 抄，不要套 Qwen 的模板。

use crate::config::KaonConfig;
use crate::weights::{self, QWeight};
use anyhow::{ensure, Context, Result};
use dinfer_kernels::{self as k, bf16, Gemm, Stream};
use dinfer_layers::{Act as _, Attention as _, Norm as _, OpSet, Rope, RopeStyle, Tensor, WeightSource};

/// 一张量化线性层（权重常驻压缩态，GEMM 内反量化）。
pub enum QLinear {
    /// BF16 直读（router、小矩阵）。
    Bf16 { w: Tensor<bf16>, n: usize, k: usize },
    /// FP8 per-tensor e4m3。
    /// `a_scale` = 静态激活 scale（`input_scale`），`scale` = 权重 per-tensor scale（合并组已换成组内最大值）。
    ///
    /// `quant` 是走 cuBLASLt 路径时的**激活量化暂存**（bf16 → e4m3，按需扩容）。
    /// 为什么默认走 cuBLASLt：实测 vLLM 的 fp8 GEMM 真实命中 cuBLAS，自写 kernel 与它差
    /// 8/196608 个 1-ulp（累加序不同），而 cuBLASLt 的启发式 algo[2] 与它**逐位一致**
    /// （`test/kaon3/2026-10-23/D-16-REBASE.md` §28）。`FP8_SELF_KERNEL=1` 可切回自写 kernel 做 A/B。
    Fp8 {
        w: dinfer_kernels::cudarc::driver::CudaSlice<u8>,
        scale: f32,
        a_scale: f32,
        n: usize,
        k: usize,
        /// 量化暂存（按 `max_tokens × k` 容量），`Mutex` 只为内部可变性，无跨线程竞争。
        quant: std::sync::Mutex<Option<dinfer_kernels::cudarc::driver::CudaSlice<u8>>>,
        /// 两个 per-tensor scale 的设备副本（cuBLASLt 要设备指针），首次调用时建一次。
        scache: std::sync::Mutex<
            Option<(
                dinfer_kernels::cudarc::driver::CudaSlice<f32>,
                dinfer_kernels::cudarc::driver::CudaSlice<f32>,
            )>,
        >,
    },
    /// NVFP4：packed `[n, k/2]` + block scale `[n, k/16]` + 全局 scale。
    NvFp4 {
        packed: dinfer_kernels::cudarc::driver::CudaSlice<u8>,
        scale: dinfer_kernels::cudarc::driver::CudaSlice<u8>,
        global: f32,
        /// 激活 input_scale：A 按 gs = 1/a_scale 的网格量化，输出乘 global·a_scale（vLLM 的 alpha）。
        a_scale: f32,
        n: usize,
        k: usize,
    },
}

impl QLinear {
    /// 按 `weights::load_linear` 的分派结果建层。
    pub fn from_qw(qw: QWeight) -> Self {
        match qw {
            QWeight::Bf16 { w, n, k } => {
                // QWeight 里存的是 u16 位模式；转成 bf16 张量（同布局，直接复用缓冲区）。
                let data = unsafe {
                    // SAFETY: bf16 与 u16 同尺寸同对齐，位模式一一对应；所有权随之转移。
                    std::mem::transmute::<
                        dinfer_kernels::cudarc::driver::CudaSlice<u16>,
                        dinfer_kernels::cudarc::driver::CudaSlice<bf16>,
                    >(w)
                };
                Self::Bf16 { w: Tensor { data, shape: vec![n, k] }, n, k }
            }
            QWeight::Fp8 { w, scale, a_scale, n, k } => {
                Self::Fp8 {
                    w, scale, a_scale, n, k,
                    quant: std::sync::Mutex::new(None),
                    scache: std::sync::Mutex::new(None),
                }
            }
            QWeight::NvFp4 { packed, scale, global, a_scale, n, k } => Self::NvFp4 { packed, scale, global, a_scale, n, k },
        }
    }

    pub fn load(stream: &Stream, src: &WeightSource, name: &str) -> Result<Self> {
        Ok(Self::from_qw(weights::load_linear(src, stream, name)?))
    }

    /// 一组会被 vLLM 合并成一个线性层的权重（q/k/v、gate/up），按 `weights::load_fp8_group` 的口径重量化。
    pub fn load_group(stream: &Stream, src: &WeightSource, names: &[String]) -> Result<Vec<Self>> {
        Ok(weights::load_fp8_group(src, stream, names)?.into_iter().map(Self::from_qw).collect())
    }

    /// 取 BF16 权重（router 这类必须用 bf16 路由算子的场合）；其它格式报错。
    pub fn as_bf16(&self) -> Result<&Tensor<bf16>> {
        match self {
            Self::Bf16 { w, .. } => Ok(w),
            _ => anyhow::bail!("该线性层不是 BF16 权重，无法用于 bf16 路由算子"),
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        match self {
            Self::Bf16 { n, k, .. } | Self::Fp8 { n, k, .. } | Self::NvFp4 { n, k, .. } => (*n, *k),
        }
    }

    /// 本层激活的静态 scale（只有 FP8 才有）；调用方据此决定能否把 norm 与量化融合。
    pub fn a_scale(&self) -> Option<f32> {
        match self {
            Self::Fp8 { a_scale, .. } => Some(*a_scale),
            _ => None,
        }
    }

    /// A 已经量化成 e4m3（`[tokens, kk]`）时直接做 GEMM。
    ///
    /// 与 [`Self::forward`] 的唯一区别是跳过 `x` 的量化——用于把 RMSNorm 与量化**融合**成一次 kernel
    /// （vLLM compile 口径；见 `docs/loops/kaon3-parity.md` 第 20 轮）。`a` 必须是用本层 `a_scale` 量化出来的。
    pub fn forward_preq(
        &self,
        gemm: &Gemm,
        stream: &Stream,
        a: &dinfer_kernels::cudarc::driver::CudaSlice<u8>,
        out: &mut Tensor<bf16>,
        tokens: usize,
    ) -> Result<()> {
        let (n, kk) = self.shape();
        ensure!(out.numel() >= tokens * n, "QLinear: out 太小");
        match self {
            Self::Fp8 { w, scale, a_scale, scache, .. } => {
                ensure!(a.len() >= tokens * kk, "QLinear::forward_preq: a 太小");
                let mut sc = scache.lock().unwrap();
                if sc.is_none() {
                    *sc = Some((stream.htod_copy(vec![*a_scale])?, stream.htod_copy(vec![*scale])?));
                }
                let (a_s, w_s) = sc.as_ref().unwrap();
                gemm.forward_fp8_cublaslt(a, a_s, w, w_s, &mut out.data, tokens, n, kk)
            }
            // 非 FP8 权重没有可复用的量化结果，退回普通路径
            _ => anyhow::bail!("forward_preq 只用于 FP8 权重"),
        }
    }

    /// `out[..tokens, ..] = x[..tokens, :] · Wᵀ`；`x`/`out` 允许按 max_tokens 预分配。
    pub fn forward(
        &self,
        gemm: &Gemm,
        stream: &Stream,
        x: &Tensor<bf16>,
        out: &mut Tensor<bf16>,
        tokens: usize,
    ) -> Result<()> {
        let (n, kk) = self.shape();
        ensure!(x.numel() >= tokens * kk, "QLinear: x 太小");
        ensure!(out.numel() >= tokens * n, "QLinear: out 太小");
        match self {
            Self::Bf16 { w, .. } => gemm.forward_bf16(&x.data, &w.data, None, &mut out.data, tokens, n, kk),
            Self::Fp8 { w, scale, a_scale, quant, scache, .. } => {
                // 自写 kernel 做 A/B 用（`FP8_SELF_KERNEL=1`）；默认走 cuBLASLt，与 vLLM 逐位一致。
                let self_kernel = std::env::var("FP8_SELF_KERNEL").map(|v| v == "1").unwrap_or(false);
                if self_kernel {
                    return gemm.forward_fp8(&x.data, *a_scale, w, *scale, &mut out.data, tokens, n, kk);
                }
                let mut g = quant.lock().unwrap();
                let need = tokens * kk;
                let buf = match g.as_mut() {
                    Some(b) if b.len() >= need => b,
                    _ => {
                        *g = Some(stream.alloc_zeros::<u8>(need)?);
                        g.as_mut().unwrap()
                    }
                };
                gemm.quant_fp8(&x.data, *a_scale, buf, need)?;
                // 两个 per-tensor scale 以设备标量形式传给 cuBLASLt；按 shape 缓存一次即可（每层不同）。
                let mut sc = scache.lock().unwrap();
                if sc.is_none() {
                    *sc = Some((stream.htod_copy(vec![*a_scale])?, stream.htod_copy(vec![*scale])?));
                }
                let (a_s, w_s) = sc.as_ref().unwrap();
                gemm.forward_fp8_cublaslt(buf, a_s, w, w_s, &mut out.data, tokens, n, kk)
            }
            Self::NvFp4 { packed, scale, global, a_scale, .. } => {
                gemm.forward_nvfp4(&x.data, 1.0 / *a_scale, packed, scale, *global * *a_scale, &mut out.data, tokens, n, kk)
            }
        }
    }
}

/// 一层里的一个专家（gate/up/down 三个 NVFP4 矩阵）。
pub struct Expert {
    pub gate: QLinear,
    pub up: QLinear,
    pub down: QLinear,
}

/// MoE 的 router（`proj` 是 BF16，另有 `scale` 与 `per_expert_scale`）。
pub struct Router {
    pub proj: QLinear,
    /// `router.scale` [hidden]：路由输入预处理 `rmsnorm_no_weight(x) * hidden^-0.5 * scale`。
    pub scale: Tensor<bf16>,
    /// `router.per_expert_scale` [E]：折进路由权重。
    pub per_expert: Tensor<bf16>,
}

impl Router {
    /// 路由 logits 用 bf16 GEMM（`moe_gate_f32` 只吃 bf16 权重）。
    pub fn bf16_weight(&self) -> Result<&Tensor<bf16>> {
        self.proj.as_bf16()
    }
}

pub struct Gemma4Layer {
    pub input_norm: Tensor<bf16>,
    pub post_attn_norm: Tensor<bf16>,
    pub pre_ff_norm: Tensor<bf16>,
    pub pre_ff_norm_2: Tensor<bf16>,
    pub post_ff_norm: Tensor<bf16>,
    pub post_ff_norm_1: Tensor<bf16>,
    pub post_ff_norm_2: Tensor<bf16>,
    pub q_proj: QLinear,
    pub k_proj: QLinear,
    /// full 层 `attention_k_eq_v` 时为空（K 权重兼作 V）。
    pub v_proj: Option<QLinear>,
    pub o_proj: QLinear,
    /// 逐头 RMSNorm（带权重），长度 = head_dim。
    pub q_norm: Tensor<bf16>,
    pub k_norm: Tensor<bf16>,
    pub dense_gate: QLinear,
    pub dense_up: QLinear,
    pub dense_down: QLinear,
    pub router: Router,
    pub experts: Vec<Expert>,
    /// `per_expert_scale`（host 副本）；`MOE_COMBINE_F32=1` 时不折进 down，在合并时乘进路由权重。
    pub pes: Vec<f32>,
    /// `layer_scalar`，BF16 单元素。
    pub layer_scalar: f32,
    /// 本层注意力形状：head_dim / kv 头数 / 窗口 / rope。
    pub head_dim: usize,
    pub nkv: usize,
    pub window: usize,
    /// FP8 KV 层:attention 的 P 也以 e4m3 参与 PV(vLLM Triton FP8 路径),传给 gqa 的 p_fp8。
    pub p_fp8: bool,
    pub rope_dim: usize,
    pub k_eq_v: bool,
}

fn load_vec(stream: &Stream, src: &WeightSource, name: &str) -> Result<Tensor<bf16>> {
    src.load_bf16(stream, name)
}

/// 读一个标量权重（`layer_scalar` 这种单元素张量）到 host f32。
fn load_scalar(src: &WeightSource, name: &str) -> Result<f32> {
    let (shape, v) = src.load_f32_host(name)?;
    ensure!(v.len() == 1, "{name}: 期望标量，实际形状 {shape:?}");
    Ok(v[0])
}

impl Gemma4Layer {
    pub fn load(
        stream: &Stream,
        src: &WeightSource,
        cfg: &KaonConfig,
        l: usize,
    ) -> Result<Self> {
        let p = weights::layer_prefix(l);
        let (head_dim, nkv, _rope) = cfg.attn_shape(l);
        let k_eq_v = cfg.k_eq_v_at(l);
        let attn = format!("{p}.self_attn");
        // vLLM 的 qkv_proj 是合并层：k_eq_v 时 K 权重填 K、V 两个槽，组 scale 只由 q/k 决定
        let mut qk = QLinear::load_group(
            stream,
            src,
            &if k_eq_v {
                vec![format!("{attn}.q_proj.weight"), format!("{attn}.k_proj.weight")]
            } else {
                vec![format!("{attn}.q_proj.weight"), format!("{attn}.k_proj.weight"), format!("{attn}.v_proj.weight")]
            },
        )?
        .into_iter();
        let q_proj = qk.next().context("q_proj")?;
        let k_proj = qk.next().context("k_proj")?;
        let v_proj = if k_eq_v {
            None
        } else {
            Some(qk.next().context("v_proj")?)
        };
        let o_proj = QLinear::load(stream, src, &format!("{attn}.o_proj.weight"))?;
        ensure!(q_proj.shape() == (cfg.num_heads * head_dim, cfg.hidden), "第 {l} 层 q_proj {:?}", q_proj.shape());
        ensure!(k_proj.shape() == (nkv * head_dim, cfg.hidden), "第 {l} 层 k_proj {:?}", k_proj.shape());
        ensure!(o_proj.shape() == (cfg.hidden, cfg.num_heads * head_dim), "第 {l} 层 o_proj {:?}", o_proj.shape());

        let dense = format!("{p}.mlp");
        // gate_up_proj 同为合并层
        let mut gate_up = QLinear::load_group(
            stream,
            src,
            &[format!("{dense}.gate_proj.weight"), format!("{dense}.up_proj.weight")],
        )?
        .into_iter();
        let router_p = format!("{p}.router");
        // per_expert_scale 每层一份 [E]；读一次，逐专家把该专家的系数折进 down 的全局 scale。
        let pes = weights::per_expert_scales(src, &p)?;
        ensure!(pes.len() == cfg.num_experts, "per_expert_scale 长度 {} != 专家数 {}", pes.len(), cfg.num_experts);
        let mut experts = Vec::with_capacity(cfg.num_experts);
        for e in 0..cfg.num_experts {
            let ep = format!("{p}.experts.{e}");
            let mut down = QLinear::load(stream, src, &format!("{ep}.down_proj.weight"))?;
            // 折法精确：非线性在 gate/up 之后，down 是线性 → down(x)·s == down_with_global_s(x)
            if moe_combine_f32() {
                if let QLinear::NvFp4 { global, .. } = &mut down {
                    *global *= pes[e];
                }
            }
            let mut gate = QLinear::load(stream, src, &format!("{ep}.gate_proj.weight"))?;
            let mut up = QLinear::load(stream, src, &format!("{ep}.up_proj.weight"))?;
            // vLLM 的 w13（gate∥up）共用一个激活全局 scale：取两分片 input_scale 的最大值
            if let (QLinear::NvFp4 { a_scale: ag, .. }, QLinear::NvFp4 { a_scale: au, .. }) = (&mut gate, &mut up) {
                let m = ag.max(*au);
                *ag = m;
                *au = m;
            }
            experts.push(Expert { gate, up, down });
        }

        Ok(Self {
            input_norm: load_vec(stream, src, &format!("{p}.input_layernorm.weight"))?,
            post_attn_norm: load_vec(stream, src, &format!("{p}.post_attention_layernorm.weight"))?,
            pre_ff_norm: load_vec(stream, src, &format!("{p}.pre_feedforward_layernorm.weight"))?,
            pre_ff_norm_2: load_vec(stream, src, &format!("{p}.pre_feedforward_layernorm_2.weight"))?,
            post_ff_norm: load_vec(stream, src, &format!("{p}.post_feedforward_layernorm.weight"))?,
            post_ff_norm_1: load_vec(stream, src, &format!("{p}.post_feedforward_layernorm_1.weight"))?,
            post_ff_norm_2: load_vec(stream, src, &format!("{p}.post_feedforward_layernorm_2.weight"))?,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: load_vec(stream, src, &format!("{attn}.q_norm.weight"))?,
            k_norm: load_vec(stream, src, &format!("{attn}.k_norm.weight"))?,
            dense_gate: gate_up.next().context("gate_proj")?,
            dense_up: gate_up.next().context("up_proj")?,
            dense_down: QLinear::load(stream, src, &format!("{dense}.down_proj.weight"))?,
            router: Router {
                proj: QLinear::load(stream, src, &format!("{router_p}.proj.weight"))?,
                scale: load_vec(stream, src, &format!("{router_p}.scale"))?,
                per_expert: load_vec(stream, src, &format!("{router_p}.per_expert_scale"))?,
            },
            experts,
            pes,
            layer_scalar: load_scalar(src, &format!("{p}.layer_scalar"))?,
            head_dim,
            nkv,
            window: cfg.window_at(l),
            p_fp8: cfg.kv_fp8,
            rope_dim: cfg.attn_shape(l).2.rope_dim,
            k_eq_v,
        })
    }
}

/// 一层 forward 需要的设备侧元数据（由 executor 上传）。
pub struct LayerMeta<'a> {
    pub mode: dinfer_core::runner::BatchMode,
    pub cu_seqlens: &'a Tensor<i32>,
    pub seq_lens: &'a Tensor<i32>,
    pub block_table: &'a Tensor<i32>,
    pub num_reqs: usize,
    pub max_blocks: usize,
    pub max_seqlen: usize,
    pub positions: &'a Tensor<i32>,
}

/// RoPE 表（两种：sliding 与 full 各一套）。
pub struct Ropes {
    pub sliding: Rope,
    pub full: Rope,
}

impl Ropes {
    pub fn new(stream: &Stream, cfg: &KaonConfig) -> Result<Self> {
        Ok(Self {
            // **bf16 表**（2026-10-22 逐层对拍定案）：vLLM `RotaryEmbeddingBase.__init__` 里
            // `cache = cache.to(dtype)`，dtype = 模型 bf16；`forward_cuda` 再 `_match_cos_sin_cache_dtype`。
            // CUDA kernel 把 bf16 表 `static_cast<float>` 后做 fp32 乘加。此前用 f32 表，
            // `q_rope` 相对误差 1.83e-3（= bf16 尾数分辨率），现已改。
            sliding: Rope::with_denom(
                stream,
                cfg.rope_sliding.rope_dim,
                cfg.rope_sliding.denom,
                cfg.max_pos,
                cfg.rope_sliding.theta,
                RopeStyle::Neox,
            )?
            .with_span(cfg.rope_sliding.span)?
            .with_bf16_cache(stream)?,
            full: Rope::with_denom(
                stream,
                cfg.rope_full.rope_dim,
                cfg.rope_full.denom,
                cfg.max_pos,
                cfg.rope_full.theta,
                RopeStyle::Neox,
            )?
            .with_span(cfg.rope_full.span)?
            .with_bf16_cache(stream)?,
        })
    }
}

/// 逐头 RMSNorm（q_norm / k_norm 带权重，v_norm 无权重）。
///
/// `x` 形状 `[tokens, heads, head_dim]`，每 `head_dim` 个元素归一化一次 —— 所以直接按
/// `hidden = head_dim` 调 `rmsnorm_*`，不用额外的 gather/scatter。
fn head_rmsnorm(
    stream: &Stream,
    ops: &impl OpSet,
    x: &mut Tensor<bf16>,
    w: Option<&Tensor<bf16>>,
    head_dim: usize,
    eps: f32,
) -> Result<()> {
    let tokens = x.numel() / head_dim;
    let mut out = Tensor::uninit(stream, &x.shape)?;
    match w {
        Some(w) => ops
            .norm()
            .rmsnorm(stream, &x.data, &w.data, &mut out.data, tokens, head_dim, eps)?,
        None => ops.norm().rmsnorm_noweight(stream, &x.data, &mut out.data, tokens, head_dim, eps)?,
    }
    x.data = out.data;
    Ok(())
}

/// 无权重 RMSNorm 到新缓冲（不覆盖输入）：`y = x / rms(x)`。
fn rmsnorm_noweight_into(
    stream: &Stream,
    ops: &impl OpSet,
    x: &Tensor<bf16>,
    y: &mut Tensor<bf16>,
    hidden: usize,
    tokens: usize,
    eps: f32,
) -> Result<()> {
    ensure!(x.numel() >= tokens * hidden && y.numel() >= tokens * hidden, "rmsnorm: 缓冲太小");
    ops.norm().rmsnorm_noweight(stream, &x.data, &mut y.data, tokens, hidden, eps)
}

/// 带权重 RMSNorm 到新缓冲。
/// 数值口径：1 = eager BI；0 / 2 = compile（vLLM 未开 `enforce_eager` 时的实现），
/// 两者差别在 `x·inv` 后是否先舍 bf16（见 `crates/kernels/src/ops.rs::norm_mode`）。
/// 与 `dinfer_kernels` 内部的 `norm_mode()` 读同一个 `RMSNORM_MODE`，两边必须一致。
fn norm_mode() -> i32 {
    static V: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("RMSNORM_MODE").ok().and_then(|s| s.trim().parse::<i32>().ok()).unwrap_or(1))
}

/// norm 与 fp8 量化是否融合（`NORM_FUSE_QUANT=1` 开启）。**默认关**：
/// 实测在 Kaon-V3 L1 上融合后 top5 |Δlp| 0.0102→0.0108、实际 token 1.4898→1.6856，**反而更差**，
/// 说明 inductor 融合链的真实语义还没对上（见 `docs/loops/kaon3-parity.md` 第 20 轮）。留着供继续排查。
fn fuse_norm_quant() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NORM_FUSE_QUANT").map(|v| v == "1").unwrap_or(false))
}

fn rmsnorm_into(
    stream: &Stream,
    ops: &impl OpSet,
    x: &Tensor<bf16>,
    w: &Tensor<bf16>,
    y: &mut Tensor<bf16>,
    hidden: usize,
    tokens: usize,
    eps: f32,
) -> Result<()> {
    ensure!(x.numel() >= tokens * hidden && y.numel() >= tokens * hidden, "rmsnorm: 缓冲太小");
    ops.norm().rmsnorm(stream, &x.data, &w.data, &mut y.data, tokens, hidden, eps)
}

/// 一层 forward 需要的共享激活缓冲（全模型一套，避免逐层分配）。
pub struct ActBuf {
    pub h: Tensor<bf16>,
    pub norm: Tensor<bf16>,
    pub qkv: Tensor<bf16>,
    pub attn_out: Tensor<bf16>,
    pub o: Tensor<bf16>,
    pub routed: Tensor<bf16>,
    pub acc: Tensor<f32>,
    pub mlp_gate_up: Tensor<bf16>,
    pub mlp_act: Tensor<bf16>,
    pub dense_out: Tensor<bf16>,
    pub moe_out: Tensor<bf16>,
    pub h2: Tensor<bf16>,
    pub logits: Tensor<f32>,
}

impl ActBuf {
    pub fn new(stream: &Stream, cfg: &KaonConfig) -> Result<Self> {
        let h = cfg.hidden;
        Ok(Self {
            h: Tensor::uninit(stream, &[h])?,
            norm: Tensor::uninit(stream, &[h])?,
            qkv: Tensor::uninit(stream, &[cfg.num_heads * cfg.head_dim_full + 2 * cfg.nkv_full * cfg.head_dim_full])?,
            attn_out: Tensor::uninit(stream, &[cfg.num_heads * cfg.head_dim_full])?,
            o: Tensor::uninit(stream, &[h])?,
            routed: Tensor::uninit(stream, &[h])?,
            acc: Tensor::uninit(stream, &[h])?,
            mlp_gate_up: Tensor::uninit(stream, &[2 * cfg.dense_inter])?,
            mlp_act: Tensor::uninit(stream, &[cfg.dense_inter])?,
            dense_out: Tensor::uninit(stream, &[h])?,
            moe_out: Tensor::uninit(stream, &[h])?,
            h2: Tensor::uninit(stream, &[h])?,
            logits: Tensor::uninit(stream, &[cfg.num_experts.max(1)])?,
        })
    }
}

/// 一层 forward 的工作区（按 `max_tokens` 预分配，全模型一套复用）。
pub struct LayerScratch {
    pub x: Tensor<bf16>,        // 残差主干 [tokens, hidden]
    pub norm: Tensor<bf16>,     // norm 输出 [tokens, hidden]
    /// q/k/v/attn 按**层型**分两套（滑窗 D=256 / 全注意力 D=512），
    /// 因为 rope 与 attention 都按精确长度校验，不能共用一个大缓冲。
    pub q_s: Tensor<bf16>,
    pub k_s: Tensor<bf16>,
    pub v_s: Tensor<bf16>,
    pub attn_s: Tensor<bf16>,
    pub q_f: Tensor<bf16>,
    pub k_f: Tensor<bf16>,
    pub v_f: Tensor<bf16>,
    pub attn_f: Tensor<bf16>,
    pub o: Tensor<bf16>,        // [tokens, hidden]
    pub gate_up: Tensor<bf16>,  // [tokens, 2·dense_inter]
    pub act: Tensor<bf16>,      // [tokens, max(dense_inter, moe_inter)] 专家 up 复用
    pub gelu: Tensor<bf16>,     // [tokens, max(dense_inter, moe_inter)] gelu 输出
    pub dense: Tensor<bf16>,    // [tokens, hidden]
    pub routed: Tensor<bf16>,   // [tokens, hidden]（router 预处理后的输入）
    pub acc: Tensor<f32>,      // [tokens, hidden] MoE 加权和
    pub y: Tensor<bf16>,        // 单个专家的输出 [tokens, hidden]
    pub xg: Tensor<bf16>,       // 分桶后 gather 的行 [tokens·topk, hidden]
    pub yg: Tensor<bf16>,       // 分桶后专家输出 [tokens·topk, hidden]
    pub h2: Tensor<bf16>,       // [tokens, hidden]
    pub logits: Tensor<f32>,    // [tokens, E]
    pub logits_bf16: Tensor<bf16>, // [tokens, E] router GEMM 的 bf16 输出（vLLM 口径，再 cast 到 logits）
    pub topk_ids: Tensor<i32>,  // [tokens, topk]
    pub topk_w: Tensor<f32>,    // [tokens, topk]
    pub buckets: k::ops::MoeBuckets,
    /// compile 口径下 input_norm 的**融合量化**结果（e4m3 `[max_tokens, hidden]`），q/k/v 三个投影共用。
    pub preq: std::sync::Mutex<Option<dinfer_kernels::cudarc::driver::CudaSlice<u8>>>,
}

impl LayerScratch {
    pub fn new(stream: &Stream, cfg: &KaonConfig, max_tokens: usize) -> Result<Self> {
        let (h, d, nkv) = (cfg.hidden, cfg.head_dim_full, cfg.nkv_full);
        let _ = (d, nkv);
        Ok(Self {
            x: Tensor::uninit(stream, &[max_tokens, cfg.hidden])?,
            norm: Tensor::uninit(stream, &[max_tokens, cfg.hidden])?,
            q_s: Tensor::uninit(stream, &[max_tokens, cfg.num_heads * cfg.head_dim_sliding])?,
            k_s: Tensor::uninit(stream, &[max_tokens, cfg.nkv_sliding * cfg.head_dim_sliding])?,
            v_s: Tensor::uninit(stream, &[max_tokens, cfg.nkv_sliding * cfg.head_dim_sliding])?,
            attn_s: Tensor::uninit(stream, &[max_tokens, cfg.num_heads * cfg.head_dim_sliding])?,
            q_f: Tensor::uninit(stream, &[max_tokens, cfg.num_heads * cfg.head_dim_full])?,
            k_f: Tensor::uninit(stream, &[max_tokens, cfg.nkv_full * cfg.head_dim_full])?,
            v_f: Tensor::uninit(stream, &[max_tokens, cfg.nkv_full * cfg.head_dim_full])?,
            attn_f: Tensor::uninit(stream, &[max_tokens, cfg.num_heads * cfg.head_dim_full])?,
            o: Tensor::uninit(stream, &[max_tokens, h])?,
            gate_up: Tensor::uninit(stream, &[max_tokens, 2 * cfg.dense_inter])?,
            act: Tensor::uninit(stream, &[max_tokens, cfg.dense_inter.max(cfg.moe_inter)])?,
            gelu: Tensor::uninit(stream, &[max_tokens, cfg.dense_inter.max(cfg.moe_inter)])?,
            dense: Tensor::uninit(stream, &[max_tokens, h])?,
            routed: Tensor::uninit(stream, &[max_tokens, h])?,
            acc: Tensor::uninit(stream, &[max_tokens, h])?,
            y: Tensor::uninit(stream, &[max_tokens, h])?,
            xg: Tensor::uninit(stream, &[max_tokens * cfg.top_k, h])?,
            yg: Tensor::uninit(stream, &[max_tokens * cfg.top_k, h])?,
            h2: Tensor::uninit(stream, &[max_tokens, h])?,
            logits: Tensor::uninit(stream, &[max_tokens, cfg.num_experts])?,
            logits_bf16: Tensor::uninit(stream, &[max_tokens, cfg.num_experts])?,
            topk_ids: Tensor::uninit(stream, &[max_tokens * cfg.top_k])?,
            topk_w: Tensor::uninit(stream, &[max_tokens * cfg.top_k])?,
            buckets: k::ops::MoeBuckets::new(stream, cfg.num_experts, max_tokens, cfg.top_k)?,
            preq: std::sync::Mutex::new(None),
        })
    }
}

/// 一步里每层都要的设备张量（executor 上传一次，30 层共用）。
pub struct StepCtx<'a> {
    pub mode: dinfer_core::runner::BatchMode,
    pub cu_seqlens: &'a Tensor<i32>,
    pub seq_lens: &'a Tensor<i32>,
    pub block_table: &'a Tensor<i32>,
    pub positions: &'a Tensor<i32>,
    pub num_reqs: usize,
    pub max_blocks: usize,
    pub max_seqlen: usize,
    pub block_size: usize,
    /// 本步 token 的 cache 槽位（写 KV 用）。
    pub slot_mapping: &'a Tensor<i64>,
    /// 本层 paged KV cache：[num_blocks, block_size, nkv, 2, D]（`D` = 本层 head_dim）。
    /// 写 KV 与 attention 读同一块，顺序执行；只保留可变引用避免 Rust 的别名冲突。
    pub kv_mut: &'a mut dinfer_kernels::cudarc::driver::CudaSlice<bf16>,
}


/// layer 0 子模块对拍的写盘助手：`sub_out` 为 `Some((dir, 0))` 时把 `buf` 前 `tokens*h` 个 bf16 写
/// `<dir>/<name>.bin`（f32 LE，[T,H]），文件名与 `scripts/dev/kaon3_vllm_sublayer.py` 对齐。
fn sub_dump_op(stream: &Stream, sub_out: Option<(&std::path::Path, usize)>, name: &str,
               buf: &Tensor<bf16>, tokens: usize, h: usize) -> Result<()> {
    sub_dump(stream, sub_out, name, buf, tokens, h)
}

fn sub_dump(stream: &Stream, sub_out: Option<(&std::path::Path, usize)>, name: &str,
            buf: &Tensor<bf16>, tokens: usize, h: usize) -> Result<()> {
    let Some((dir, _l)) = sub_out else { return Ok(()) };
    let _ = stream;
    let v = buf.to_vec_f32(stream)?;
    let bytes: Vec<u8> = v.iter().take(tokens * h).flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(dir.join(format!("{name}.bin")), bytes)
        .map_err(|e| anyhow::anyhow!("写 {name}.bin 失败: {e}"))?;
    Ok(())
}

impl Gemma4Layer {
    /// 一层的完整前向。`tokens` = 本步 token 数（prefill 多行 / decode 每请求 1 行）。
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        stream: &Stream,
        gemm: &Gemm,
        ops: &impl OpSet,
        cfg: &KaonConfig,
        ropes: &Ropes,
        sc: &mut LayerScratch,
        ctx: &mut StepCtx<'_>,
        tokens: usize,
        // `Some((dir, layer))` 时把本层各子模块输出写盘，供 layer 0 逐项对拍
        // （与 `scripts/dev/kaon3_vllm_sublayer.py` 抓的同名 `.bin` 一一对应）。
        sub_out: Option<(&std::path::Path, usize)>,
    ) -> Result<()> {
        let (h, d, nkv) = (cfg.hidden, self.head_dim, self.nkv);
        let heads = cfg.num_heads;
        let eps = cfg.eps;
        ensure!(self.rope_dim <= d, "rope_dim {} > head_dim {d}", self.rope_dim);

        // ---- 1. input_norm ----
        // vLLM **compile 口径**（默认）：inductor 把 RMSNorm 与下游 fp8 量化融成一个 kernel，中间不落 bf16。
        // q/k/v 共用同一个 input_norm 且共用同一个 `input_scale`（实测 L0/L5 三者 scale 相同），
        // 所以这里量化一次、三个投影复用。eager 口径（`RMSNORM_MODE=1`）走原来的两步。
        let mut qkv_preq = false;
        if norm_mode() != 1 && fuse_norm_quant() {
            // k_eq_v 的层没有 v_proj，此时 v 复用 k 的权重与 scale。
            let v_sc = self.v_proj.as_ref().map_or(self.k_proj.a_scale(), |p| p.a_scale());
            if let (Some(qs), Some(ks), Some(vs)) = (self.q_proj.a_scale(), self.k_proj.a_scale(), v_sc) {
                if qs == ks && qs == vs {
                    let need = tokens * h;
                    let mut g = sc.preq.lock().unwrap();
                    let buf = match g.as_mut() {
                        Some(b) if b.len() >= need => b,
                        _ => {
                            *g = Some(stream.alloc_zeros::<u8>(need)?);
                            g.as_mut().unwrap()
                        }
                    };
                    k::ops::rmsnorm_quant_fp8(stream, &sc.x.data, &self.input_norm.data, buf, tokens, h, eps, qs)?;
                    qkv_preq = true;
                }
            }
        }
        if !qkv_preq {
            rmsnorm_into(stream, ops, &sc.x, &self.input_norm, &mut sc.norm, h, tokens, eps)?;
        }
        sub_dump(stream, sub_out, "input_norm", &sc.norm, tokens, h)?;

        // ---- 2. q/k/v 投影（K 权重兼作 V 的层只算一次）----
        // 两种层型的 head_dim 不同（256 / 512），缓冲按层型取。
        let (q, kk, v, attn) = if self.window > 0 {
            (&mut sc.q_s, &mut sc.k_s, &mut sc.v_s, &mut sc.attn_s)
        } else {
            (&mut sc.q_f, &mut sc.k_f, &mut sc.v_f, &mut sc.attn_f)
        };
        if qkv_preq {
            let g = sc.preq.lock().unwrap();
            let buf = g.as_ref().unwrap();
            self.q_proj.forward_preq(gemm, stream, buf, q, tokens)?;
            self.k_proj.forward_preq(gemm, stream, buf, kk, tokens)?;
            match &self.v_proj {
                Some(vp) => vp.forward_preq(gemm, stream, buf, v, tokens)?,
                None => {
                    stream.dtod_copy_n(&kk.data, &mut v.data, tokens * nkv * d)?;
                }
            }
        } else {
            self.q_proj.forward(gemm, stream, &sc.norm, q, tokens)?;
            self.k_proj.forward(gemm, stream, &sc.norm, kk, tokens)?;
            match &self.v_proj {
                Some(vp) => vp.forward(gemm, stream, &sc.norm, v, tokens)?,
                None => {
                    // k_eq_v：K 权重兼作 V（只对 full 层成立）
                    stream.dtod_copy_n(&kk.data, &mut v.data, tokens * nkv * d)?;
                }
            }
        }

        // 子 loop D：q 投影的**原始输出**（norm 之前），用于区分"GEMM 累加"与"norm"引入的 1 ulp。
        sub_dump(stream, sub_out, "q_proj_raw", q, tokens, heads * d)?;
        // v_norm 定位（§53）：k/v 投影的原始输出，与 vLLM `qkv_raw` 的 k/v 段逐位比。
        sub_dump(stream, sub_out, "k_proj_raw", kk, tokens, nkv * d)?;
        sub_dump(stream, sub_out, "v_proj_raw", v, tokens, nkv * d)?;
        // ---- 3. 逐头 RMSNorm：q_norm / k_norm 带权重，v_norm 无权重 ----
        head_rmsnorm(stream, ops, q, Some(&self.q_norm), d, eps)?;
        head_rmsnorm(stream, ops, kk, Some(&self.k_norm), d, eps)?;
        head_rmsnorm(stream, ops, v, None, d, eps)?;
        if let Some((dir, _)) = sub_out {
            for (nm, buf, rows) in [("q_norm", &*q, heads), ("k_norm", &*kk, nkv), ("v_norm", &*v, nkv)] {
                let vv = buf.to_vec_f32(stream)?;
                let bytes: Vec<u8> = vv.iter().take(tokens * rows * d).flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(dir.join(format!("{nm}.bin")), bytes)
                    .map_err(|err| anyhow::anyhow!("写 {nm}.bin 失败: {err}"))?;
            }
        }

        // ---- 4. RoPE（只转前 rope_dim 维；proportional 的分母已由 Ropes 处理）----
        let rope = if self.window > 0 { &ropes.sliding } else { &ropes.full };
        rope.apply(stream, q, Some(kk), ctx.positions, heads, nkv, d)?;
        if let Some((dir, _)) = sub_out {
            for (nm, buf, rows) in [("q_rope", &*q, heads), ("k_rope", &*kk, nkv)] {
                let vv = buf.to_vec_f32(stream)?;
                let bytes: Vec<u8> = vv.iter().take(tokens * rows * d).flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(dir.join(format!("{nm}.bin")), bytes)
                    .map_err(|err| anyhow::anyhow!("写 {nm}.bin 失败: {err}"))?;
            }
        }

        // ---- 4a. FP8 KV（vLLM `kv_cache_dtype=fp8_e4m3`，scale 1.0）：attention 读的是量化后的 K/V ----
        if cfg.kv_fp8 {
            k::ops::fp8_e4m3_roundtrip_bf16_n(stream, &mut kk.data, tokens * nkv * d, 1.0)?;
            k::ops::fp8_e4m3_roundtrip_bf16_n(stream, &mut v.data, tokens * nkv * d, 1.0)?;
            // q 也要量化成 e4m3（vLLM Triton 后端 `supports_quant_query_input = is_cuda()`，
            // 上层会把 query 先量化再交给 kernel，`q_descale` 再把它反量化回去）。
            // **实测依据（2026-10-22 A/B）**：关掉这一步后 ΔNLL 从 −0.0018 恶化到 **+0.0438**、
            // 长文本从 −0.0464 到 +0.0194 —— 明显变差，说明 vLLM 确实量化 q。
            // （曾据 `triton_attn.py:659` 的 `query.dtype == fp8` 条件推断"不量化"，**该推断是错的**：
            //  那个条件是 kernel 内部"已经是 fp8 才反量化"，而量化发生在上层调用之前。）
            k::ops::fp8_e4m3_roundtrip_bf16_n(stream, &mut q.data, tokens * heads * d, 1.0)?;
        }

        // ---- 4b. 写本层 paged KV（每层 head_dim / nkv 不同）----
        k::ops::gqa_cache_write_bf16(
            stream,
            &kk.data,
            &v.data,
            &mut *ctx.kv_mut,
            &ctx.slot_mapping.data,
            tokens,
            nkv,
            d,
            ctx.block_size,
        )?;

        // ---- 5. attention（滑窗层传 window）----
        let scale = 1.0f32; // Gemma4：scaling = 1.0，由 QK-norm 隐式承担
        match ctx.mode {
            dinfer_core::runner::BatchMode::Prefill => ops.attn().prefill_win(
                stream,
                &q.data,
                &kk.data,
                &v.data,
                &mut attn.data,
                &ctx.cu_seqlens.data,
                ctx.num_reqs,
                ctx.max_seqlen,
                heads,
                nkv,
                d,
                scale,
                0,
                0,
                self.window,
                self.p_fp8,
            )?,
            _ => ops.attn().paged_win(
                stream,
                &q.data,
                &*ctx.kv_mut,
                &ctx.block_table.data,
                &ctx.seq_lens.data,
                &ctx.cu_seqlens.data,
                &mut attn.data,
                ctx.num_reqs,
                ctx.max_seqlen,
                heads,
                nkv,
                d,
                ctx.block_size,
                ctx.max_blocks,
                scale,
                0,
                // 混批（Paged 且同一步里也有 prefill 行）时跳过单行请求；纯 decode 不跳。
                matches!(ctx.mode, dinfer_core::runner::BatchMode::Paged),
                self.window,
                self.p_fp8,
            )?,
        }

        // ---- 6. o_proj + post_attention_norm + 残差 ----
        if let Some((dir, _)) = sub_out {
            let vv = attn.to_vec_f32(stream)?;
            let bytes: Vec<u8> = vv.iter().take(tokens * heads * d).flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(dir.join("attn_out.bin"), bytes)
                .map_err(|err| anyhow::anyhow!("写 attn_out.bin 失败: {err}"))?;
        }
        self.o_proj.forward(gemm, stream, attn, &mut sc.o, tokens)?;
        sub_dump(stream, sub_out, "o_proj", &sc.o, tokens, h)?;
        rmsnorm_into(stream, ops, &sc.o, &self.post_attn_norm, &mut sc.norm, h, tokens, eps)?;
        sub_dump(stream, sub_out, "post_attn_norm", &sc.norm, tokens, h)?;
        k::ops::add_inplace_bf16_n(stream, &sc.norm.data, &mut sc.x.data, tokens * h)?;

        // ---- 7. dense MLP 支路（无条件执行）----
        rmsnorm_into(stream, ops, &sc.x, &self.pre_ff_norm, &mut sc.norm, h, tokens, eps)?;
        sub_dump(stream, sub_out, "pre_ff_norm", &sc.norm, tokens, h)?;
        self.dense_gate.forward(gemm, stream, &sc.norm, &mut sc.gate_up, tokens)?;
        self.dense_up.forward(gemm, stream, &sc.norm, &mut sc.act, tokens)?;
        if let Some((dir, _)) = sub_out {
            // mlp_gate_up = [gate | up] 拼起来（vLLM 的 MergedColumnParallelLinear 输出布局）
            let (g, u) = (sc.gate_up.to_vec_f32(stream)?, sc.act.to_vec_f32(stream)?);
            let di = cfg.dense_inter;
            let mut outv = Vec::with_capacity(tokens * 2 * di);
            for t in 0..tokens {
                outv.extend_from_slice(&g[t * di..(t + 1) * di]);
                outv.extend_from_slice(&u[t * di..(t + 1) * di]);
            }
            let bytes: Vec<u8> = outv.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(dir.join("mlp_gate_up.bin"), bytes)
                .map_err(|err| anyhow::anyhow!("写 mlp_gate_up.bin 失败: {err}"))?;
        }
        gelu_mul(stream, ops, &sc.gate_up.data, &sc.act.data, &mut sc.gelu.data, tokens, cfg.dense_inter)?;
        sub_dump(stream, sub_out, "mlp_gelu", &sc.gelu, tokens, cfg.dense_inter)?;
        self.dense_down.forward(gemm, stream, &sc.gelu, &mut sc.dense, tokens)?;
        sub_dump(stream, sub_out, "dense", &sc.dense, tokens, h)?;
        rmsnorm_into(stream, ops, &sc.dense, &self.post_ff_norm_1, &mut sc.norm, h, tokens, eps)?;
        sub_dump(stream, sub_out, "ff1", &sc.norm, tokens, h)?;

        // ---- 8. MoE 支路（与 dense 并联）----
        self.moe_branch(stream, gemm, ops, cfg, sc, tokens, eps, sub_out)?;

        // ---- 9. 两条支路相加 → post_ff_norm → 残差 → layer_scalar ----
        k::ops::add_inplace_bf16_n(stream, &sc.h2.data, &mut sc.norm.data, tokens * h)?; // h2 = post_ff_norm_2(moe)
        sub_dump(stream, sub_out, "post_ff_in", &sc.norm, tokens, h)?;
        rmsnorm_into(stream, ops, &sc.norm, &self.post_ff_norm, &mut sc.h2, h, tokens, eps)?;
        sub_dump(stream, sub_out, "post_ff_norm", &sc.h2, tokens, h)?;
        k::ops::add_inplace_bf16_n(stream, &sc.h2.data, &mut sc.x.data, tokens * h)?;
        if self.layer_scalar != 1.0 {
            k::ops::scale_bf16_n(stream, &mut sc.x.data, tokens * h, self.layer_scalar)?;
        }
        Ok(())
    }

    /// MoE 支路：router 预处理 → top-k → 分桶 → 逐专家 → 加权和。
    #[allow(clippy::too_many_arguments)]
    fn moe_branch(
        &self,
        stream: &Stream,
        gemm: &Gemm,
        ops: &impl OpSet,
        cfg: &KaonConfig,
        sc: &mut LayerScratch,
        tokens: usize,
        eps: f32,
        sub_out: Option<(&std::path::Path, usize)>,
    ) -> Result<()> {
        let h = cfg.hidden;
        // router 输入：rmsnorm_no_weight(x) · root_size · scale[hidden]
        //
        // vLLM `Gemma4Router.forward`：
        //   x = self.norm(x); x = x * self.root_size.to(x.dtype); x = x * self.scale.to(x.dtype)
        // `root_size` 是 **bf16 buffer**（`register_buffer(..., persistent=False)`，dtype 跟随模型），
        // 所以 `root_size.to(bf16)` 会先把 hidden^-0.5 **舍入到 bf16**，再与 bf16 的 x 相乘、
        // 结果再舍入一次。我们原来用 f32 的 `(h as f32).powf(-0.5)`（scale_bf16_n 内部做 f32 乘、
        // 舍入一次），少了那一步 bf16 舍入 → `router_logits` 有 3.3e-3 的偏差。
        // 2026-10-22 逐层对拍定案：这里必须用 **bf16 舍入后的 root_size**。
        //
        // **compile 口径（第 23 轮实测）**：inductor 把 `norm → ×root_size → ×scale` 融成一条 fp32 链，
        // 只在写回时舍入一次。分三步做会多两次 bf16 舍入（实测差 17299/67584）。所以默认走融合算子；
        // `RMSNORM_MODE=1`（eager BI）才退回三步。
        let root_size = f32::from(bf16::from_f32((h as f32).powf(-0.5)));
        if norm_mode() != 1 {
            k::ops::rmsnorm_scale2_bf16(
                stream, &sc.x.data, &self.router.scale.data, &mut sc.routed.data, tokens, h, eps, root_size,
            )?;
        } else {
            ops.norm().rmsnorm_noweight(stream, &sc.x.data, &mut sc.routed.data, tokens, h, eps)?;
            k::ops::scale_bf16_n(stream, &mut sc.routed.data, tokens * h, root_size)?;
            k::ops::mul_bf16_n(stream, &mut sc.routed.data, &self.router.scale.data, tokens, h)?;
        }

        sub_dump_op(stream, sub_out, "router_in", &sc.routed, tokens, h)?;
        let e = cfg.num_experts;
        let kk = cfg.top_k;
        // vLLM（容器 0.26.1rc1，sm_120）`GateLinear` 落到 Tier 6：`F.linear`（cuBLAS，bf16 输出）→ `.to(fp32)`
        // （sm_120 不满足 `allow_specialized_router_gemm`，Tier 5 的 fp32 epilogue 不开）。
        // 所以必须走 cuBLASLt bf16 GEMM（`run_lt_default` 钉 `id=21 tile=11 splitk=8`）再 cast；
        // 原来的自写 `moe_gate_f32` + 舍入累加序不同，差 4/3072（D-16-REBASE §51）。
        if router_bf16_ab() {
            gemm.forward_bf16(&sc.routed.data, &self.router.bf16_weight()?.data, None, &mut sc.logits_bf16.data, tokens, e, h)?;
            k::ops::bf16_to_f32_n(stream, &sc.logits_bf16.data, &mut sc.logits.data, tokens * e)?;
        } else {
            k::ops::moe_gate_f32(stream, &sc.routed.data, &self.router.bf16_weight()?.data, &mut sc.logits.data, tokens, e, h)?;
        }
        // Gemma4 路由（vLLM `gemma4.py`）：按**原始 logits** 取 top-k → 全专家 softmax →
        // 只在 top-k 上重新归一化。**不是 sigmoid**（那是 DeepSeekV3/GLM/K3 的 `moe_route_f32` 口径，
        // 2026-10-22 发现此前误用，是短 prompt 翻转率超标的主因之一）。
        // per_expert_scale 已折进各专家 down 的 global scale（线性等价，见 weights/load），此处不再乘。
        if let Some((dir, _)) = sub_out {
            let v = sc.logits.to_vec(stream)?;
            let bytes: Vec<u8> = v.iter().take(tokens * e).flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(dir.join("router_logits.bin"), bytes)
                .map_err(|err| anyhow::anyhow!("写 router_logits.bin 失败: {err}"))?;
        }
        // per_expert_scale 在这里折进路由权重（vLLM `_gemma4_routing_kernel` 同位置；默认口径）；
        // `MOE_COMBINE_F32=1` 时改在 load 期折进 down 的全局 scale，这里传 None。
        let pes_route: Option<&k::cudarc::driver::CudaSlice<bf16>> = if moe_combine_f32() {
            None
        } else {
            Some(&self.router.per_expert.data)
        };
        k::ops::moe_route_softmax_f32(
            stream,
            &sc.logits.data,
            pes_route,
            &mut sc.topk_ids.data,
            &mut sc.topk_w.data,
            tokens,
            e,
            kk,
        )?;
        k::ops::moe_bucket(stream, &sc.topk_ids.data, &mut sc.buckets, tokens, kk)?;
        let offsets = stream.dtoh_sync_copy(&sc.buckets.offsets)?;

        // 专家输入 = pre_feedforward_layernorm_2(residual)（router 用自己的无权重 norm，两者不同）；
        // 借 sc.h2 存，循环结束后才被 post_ff_norm_2 的输出覆盖
        rmsnorm_into(stream, ops, &sc.x, &self.pre_ff_norm_2, &mut sc.h2, h, tokens, eps)?;
        sub_dump(stream, sub_out, "pre_ff2_norm", &sc.h2, tokens, h)?;
        stream.memset_zeros_n(&mut sc.acc.data, tokens * h)?;
        for (eid, start, len) in k::ops::moe_expert_ranges(&offsets) {
            if len == 0 {
                continue;
            }
            k::ops::moe_gather_bf16(stream, &sc.h2.data, &mut sc.xg.data, &sc.buckets.sorted_tok, start, len, h)?;
            let ex = &self.experts[eid];
            ex.gate.forward(gemm, stream, &sc.xg, &mut sc.gate_up, len)?;
            ex.up.forward(gemm, stream, &sc.xg, &mut sc.act, len)?;
            gelu_mul(stream, ops, &sc.gate_up.data, &sc.act.data, &mut sc.gelu.data, len, cfg.moe_inter)?;
            let g = Tensor { data: sc.gelu.data.clone(), shape: vec![len.max(1), cfg.moe_inter] };
            ex.down.forward(gemm, stream, &g, &mut sc.yg, len)?;
            // 子 loop C-5：逐专家中间量（gate / up / 激活 / down 输出 + 该专家的 token 列表），按专家 id 与 vLLM 对齐比较
            if let Some((dir, _)) = sub_out {
                let m = cfg.moe_inter;
                sub_dump(stream, sub_out, &format!("moe_e{eid}_gate"), &sc.gate_up, len, m)?;
                sub_dump(stream, sub_out, &format!("moe_e{eid}_up"), &sc.act, len, m)?;
                sub_dump(stream, sub_out, &format!("moe_e{eid}_act"), &g, len, m)?;
                sub_dump(stream, sub_out, &format!("moe_e{eid}_down"), &sc.yg, len, h)?;
                let st = stream.dtoh_sync_copy(&sc.buckets.sorted_tok)?;
                let toks: Vec<String> = st[start..start + len].iter().map(|t| t.to_string()).collect();
                std::fs::write(dir.join(format!("moe_e{eid}_tok.txt")), toks.join(" "))?;
            }
            // 合并口径（2026-10-23 用户定：**默认为 vLLM 口径**）：w = bf16(topk_w·per_expert_scale)、每项 bf16(w·y)。
            // `MOE_COMBINE_F32=1` 退回旧口径（per_expert_scale 折进 down 全局 scale、f32 权重与乘积）。
            if !moe_combine_f32() {
                k::ops::moe_scatter_add_bf16w_f32(
                    stream,
                    &sc.yg.data,
                    &mut sc.acc.data,
                    &sc.buckets.sorted_tok,
                    &sc.buckets.sorted_k,
                    &sc.topk_w.data,
                    None,
                    start,
                    len,
                    kk,
                    h,
                    1.0,
                )?;
                continue;
            }
            k::ops::moe_scatter_add_f32(
                stream,
                &sc.yg.data,
                &mut sc.acc.data,
                &sc.buckets.sorted_tok,
                &sc.buckets.sorted_k,
                &sc.topk_w.data,
                start,
                len,
                kk,
                h,
            )?;
        }
        // acc（fp32）→ bf16，再做 post_feedforward_layernorm_2
        stream.memset_zeros_n(&mut sc.y.data, tokens * h)?;
        k::ops::f32_add_to_bf16_n(stream, &sc.acc.data, &mut sc.y.data, tokens * h)?;
        sub_dump(stream, sub_out, "moe_y", &sc.y, tokens, h)?;
        rmsnorm_into(stream, ops, &sc.y, &self.post_ff_norm_2, &mut sc.h2, h, tokens, eps)?;
        sub_dump(stream, sub_out, "ff2", &sc.h2, tokens, h)?;
        Ok(())
    }
}

/// router logits 是否舍入到 bf16。**默认开**（vLLM 口径）；`ROUTER_F32=1` 保留 f32。
fn router_bf16_ab() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        // 默认开：vLLM 在 sm_120 上 router 走 F.linear → bf16 输出再 cast fp32；`ROUTER_F32=1` 退回精确口径。
        let on = std::env::var("ROUTER_F32").map(|v| v.trim() != "1").unwrap_or(true);
        if !on {
            eprintln!("[kaon3] ROUTER_F32=1：router logits 保持 f32（偏离 vLLM）");
        }
        on
    })
}

/// `MOE_COMBINE_F32=1`：MoE 合并退回"f32 权重 + f32 乘积 + per_expert_scale 折进 down"的旧口径。
/// **默认（不设）= vLLM cutlass_moe 口径**（2026-10-23 用户定：先与 vLLM 对齐，精确性另议）。
fn moe_combine_f32() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let on = std::env::var("MOE_COMBINE_F32").map(|v| v.trim() == "1").unwrap_or(false);
        if on {
            eprintln!("[kaon3] MOE_COMBINE_F32=1：MoE 合并用旧 f32 口径（偏离 vLLM）");
        }
        on
    })
}

/// gelu_tanh(gate)·up。**默认按 vLLM 口径**（gelu 先舍入 bf16 再乘 up）；`GELU_F32=1` 用一次舍入的精确版。
fn gelu_mul(stream: &Stream, ops: &impl OpSet, gate: &k::cudarc::driver::CudaSlice<bf16>, up: &k::cudarc::driver::CudaSlice<bf16>,
            out: &mut k::cudarc::driver::CudaSlice<bf16>,
            rows: usize, inter: usize) -> Result<()> {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let r2 = *V.get_or_init(|| {
        // 默认开：vLLM `act_and_mul_kernel` 的 ACT_FN 返回 bf16（gelu 先舍入一次再乘 up）；`GELU_F32=1` 退回一次舍入。
        let on = std::env::var("GELU_F32").map(|v| v.trim() != "1").unwrap_or(true);
        if !on {
            eprintln!("[kaon3] GELU_F32=1：gelu 全 f32 一次舍入（偏离 vLLM）");
        }
        on
    });
    if r2 {
        k::ops::gelu_tanh_mul_round2_bf16(stream, gate, up, out, rows, inter)
    } else {
        ops.act().gelu_tanh_mul(stream, gate, up, out, rows, inter)
    }
}
