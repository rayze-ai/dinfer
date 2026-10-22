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
    Fp8 { w: dinfer_kernels::cudarc::driver::CudaSlice<u8>, scale: f32, a_scale: f32, n: usize, k: usize },
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
            QWeight::Fp8 { w, scale, a_scale, n, k } => Self::Fp8 { w, scale, a_scale, n, k },
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
            Self::Fp8 { w, scale, a_scale, .. } => {
                gemm.forward_fp8(&x.data, *a_scale, w, *scale, &mut out.data, tokens, n, kk)
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
            if let QLinear::NvFp4 { global, .. } = &mut down {
                *global *= pes[e];
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
            // f32 表（第 8 轮 A/B：bf16 表让 long_zh ΔNLL 从 +0.03 恶化到 +0.48；vLLM 的 rope 实际按 fp32 算）
            sliding: Rope::with_denom(
                stream,
                cfg.rope_sliding.rope_dim,
                cfg.rope_sliding.denom,
                cfg.max_pos,
                cfg.rope_sliding.theta,
                RopeStyle::Neox,
            )?
            .with_span(cfg.rope_sliding.span)?,
            full: Rope::with_denom(
                stream,
                cfg.rope_full.rope_dim,
                cfg.rope_full.denom,
                cfg.max_pos,
                cfg.rope_full.theta,
                RopeStyle::Neox,
            )?
            .with_span(cfg.rope_full.span)?,
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
    pub topk_ids: Tensor<i32>,  // [tokens, topk]
    pub topk_w: Tensor<f32>,    // [tokens, topk]
    pub buckets: k::ops::MoeBuckets,
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
            topk_ids: Tensor::uninit(stream, &[max_tokens * cfg.top_k])?,
            topk_w: Tensor::uninit(stream, &[max_tokens * cfg.top_k])?,
            buckets: k::ops::MoeBuckets::new(stream, cfg.num_experts, max_tokens, cfg.top_k)?,
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
    ) -> Result<()> {
        let (h, d, nkv) = (cfg.hidden, self.head_dim, self.nkv);
        let heads = cfg.num_heads;
        let eps = cfg.eps;
        ensure!(self.rope_dim <= d, "rope_dim {} > head_dim {d}", self.rope_dim);

        // ---- 1. input_norm ----
        rmsnorm_into(stream, ops, &sc.x, &self.input_norm, &mut sc.norm, h, tokens, eps)?;

        // ---- 2. q/k/v 投影（K 权重兼作 V 的层只算一次）----
        // 两种层型的 head_dim 不同（256 / 512），缓冲按层型取。
        let (q, kk, v, attn) = if self.window > 0 {
            (&mut sc.q_s, &mut sc.k_s, &mut sc.v_s, &mut sc.attn_s)
        } else {
            (&mut sc.q_f, &mut sc.k_f, &mut sc.v_f, &mut sc.attn_f)
        };
        self.q_proj.forward(gemm, stream, &sc.norm, q, tokens)?;
        self.k_proj.forward(gemm, stream, &sc.norm, kk, tokens)?;
        match &self.v_proj {
            Some(vp) => vp.forward(gemm, stream, &sc.norm, v, tokens)?,
            None => {
                // k_eq_v：K 权重兼作 V（只对 full 层成立）
                stream.dtod_copy_n(&kk.data, &mut v.data, tokens * nkv * d)?;
            }
        }

        // ---- 3. 逐头 RMSNorm：q_norm / k_norm 带权重，v_norm 无权重 ----
        head_rmsnorm(stream, ops, q, Some(&self.q_norm), d, eps)?;
        head_rmsnorm(stream, ops, kk, Some(&self.k_norm), d, eps)?;
        head_rmsnorm(stream, ops, v, None, d, eps)?;

        // ---- 4. RoPE（只转前 rope_dim 维；proportional 的分母已由 Ropes 处理）----
        let rope = if self.window > 0 { &ropes.sliding } else { &ropes.full };
        rope.apply(stream, q, Some(kk), ctx.positions, heads, nkv, d)?;

        // ---- 4a. FP8 KV（vLLM `kv_cache_dtype=fp8_e4m3`，scale 1.0）：attention 读的是量化后的 K/V ----
        if cfg.kv_fp8 {
            k::ops::fp8_e4m3_roundtrip_bf16_n(stream, &mut kk.data, tokens * nkv * d, 1.0)?;
            k::ops::fp8_e4m3_roundtrip_bf16_n(stream, &mut v.data, tokens * nkv * d, 1.0)?;
            // vLLM Triton 后端在 FP8 KV 下也把 query 量化成 e4m3（`Attention.query_quant`，q_scale 缺省 = k_scale = 1.0）
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
        self.o_proj.forward(gemm, stream, attn, &mut sc.o, tokens)?;
        rmsnorm_into(stream, ops, &sc.o, &self.post_attn_norm, &mut sc.norm, h, tokens, eps)?;
        k::ops::add_inplace_bf16_n(stream, &sc.norm.data, &mut sc.x.data, tokens * h)?;

        // ---- 7. dense MLP 支路（无条件执行）----
        rmsnorm_into(stream, ops, &sc.x, &self.pre_ff_norm, &mut sc.norm, h, tokens, eps)?;
        self.dense_gate.forward(gemm, stream, &sc.norm, &mut sc.gate_up, tokens)?;
        self.dense_up.forward(gemm, stream, &sc.norm, &mut sc.act, tokens)?;
        ops.act().gelu_tanh_mul(stream, &sc.gate_up.data, &sc.act.data, &mut sc.gelu.data, tokens, cfg.dense_inter)?;
        self.dense_down.forward(gemm, stream, &sc.gelu, &mut sc.dense, tokens)?;
        rmsnorm_into(stream, ops, &sc.dense, &self.post_ff_norm_1, &mut sc.norm, h, tokens, eps)?;

        // ---- 8. MoE 支路（与 dense 并联）----
        self.moe_branch(stream, gemm, ops, cfg, sc, tokens, eps)?;

        // ---- 9. 两条支路相加 → post_ff_norm → 残差 → layer_scalar ----
        k::ops::add_inplace_bf16_n(stream, &sc.h2.data, &mut sc.norm.data, tokens * h)?; // h2 = post_ff_norm_2(moe)
        rmsnorm_into(stream, ops, &sc.norm, &self.post_ff_norm, &mut sc.h2, h, tokens, eps)?;
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
    ) -> Result<()> {
        let h = cfg.hidden;
        // router 输入：rmsnorm_no_weight(x) · hidden^-0.5 · scale[hidden]
        ops.norm().rmsnorm_noweight(stream, &sc.x.data, &mut sc.routed.data, tokens, h, eps)?;
        k::ops::scale_bf16_n(stream, &mut sc.routed.data, tokens * h, (h as f32).powf(-0.5))?;
        k::ops::mul_bf16_n(stream, &mut sc.routed.data, &self.router.scale.data, tokens, h)?;

        let e = cfg.num_experts;
        let kk = cfg.top_k;
        k::ops::moe_gate_f32(
            stream,
            &sc.routed.data,
            &self.router.bf16_weight()?.data,
            &mut sc.logits.data,
            tokens,
            e,
            h,
        )?;
        // Gemma4 路由（vLLM `gemma4.py`）：按**原始 logits** 取 top-k → 全专家 softmax →
        // 只在 top-k 上重新归一化。**不是 sigmoid**（那是 DeepSeekV3/GLM/K3 的 `moe_route_f32` 口径，
        // 2026-10-22 发现此前误用，是短 prompt 翻转率超标的主因之一）。
        // per_expert_scale 已折进各专家 down 的 global scale（线性等价，见 weights/load），此处不再乘。
        k::ops::moe_route_softmax_f32(
            stream,
            &sc.logits.data,
            None,
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
        stream.memset_zeros_n(&mut sc.acc.data, tokens * h)?;
        for (eid, start, len) in k::ops::moe_expert_ranges(&offsets) {
            if len == 0 {
                continue;
            }
            k::ops::moe_gather_bf16(stream, &sc.h2.data, &mut sc.xg.data, &sc.buckets.sorted_tok, start, len, h)?;
            let ex = &self.experts[eid];
            ex.gate.forward(gemm, stream, &sc.xg, &mut sc.gate_up, len)?;
            ex.up.forward(gemm, stream, &sc.xg, &mut sc.act, len)?;
            ops.act().gelu_tanh_mul(stream, &sc.gate_up.data, &sc.act.data, &mut sc.gelu.data, len, cfg.moe_inter)?;
            let g = Tensor { data: sc.gelu.data.clone(), shape: vec![len.max(1), cfg.moe_inter] };
            ex.down.forward(gemm, stream, &g, &mut sc.yg, len)?;
            // per_expert_scale 已折进 down_proj 的全局 scale（见 weights::load），这里只按路由权重累加
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
        rmsnorm_into(stream, ops, &sc.y, &self.post_ff_norm_2, &mut sc.h2, h, tokens, eps)?;
        Ok(())
    }
}
