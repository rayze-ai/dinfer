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
use anyhow::{ensure, Result};
use dinfer_kernels::{self as k, bf16, Gemm, Stream};
use dinfer_layers::{OpSet, Rope, RopeStyle, Tensor, WeightSource};

/// 一张量化线性层（权重常驻压缩态，GEMM 内反量化）。
pub enum QLinear {
    /// BF16 直读（router、小矩阵）。
    Bf16 { w: Tensor<bf16>, n: usize, k: usize },
    /// FP8 per-tensor e4m3。
    Fp8 { w: dinfer_kernels::cudarc::driver::CudaSlice<u8>, scale: f32, n: usize, k: usize },
    /// NVFP4：packed `[n, k/2]` + block scale `[n, k/16]` + 全局 scale。
    NvFp4 {
        packed: dinfer_kernels::cudarc::driver::CudaSlice<u8>,
        scale: dinfer_kernels::cudarc::driver::CudaSlice<u8>,
        global: f32,
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
            QWeight::Fp8 { w, scale, n, k } => Self::Fp8 { w, scale, n, k },
            QWeight::NvFp4 { packed, scale, global, n, k } => Self::NvFp4 { packed, scale, global, n, k },
        }
    }

    pub fn load(stream: &Stream, src: &WeightSource, name: &str) -> Result<Self> {
        Ok(Self::from_qw(weights::load_linear(src, stream, name)?))
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
            Self::Fp8 { w, scale, .. } => gemm.forward_fp8(&x.data, w, *scale, &mut out.data, tokens, n, kk),
            Self::NvFp4 { packed, scale, global, .. } => {
                gemm.forward_nvfp4(&x.data, packed, scale, *global, &mut out.data, tokens, n, kk)
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
        let q_proj = QLinear::load(stream, src, &format!("{attn}.q_proj.weight"))?;
        let k_proj = QLinear::load(stream, src, &format!("{attn}.k_proj.weight"))?;
        let v_proj = if k_eq_v {
            None
        } else {
            Some(QLinear::load(stream, src, &format!("{attn}.v_proj.weight"))?)
        };
        let o_proj = QLinear::load(stream, src, &format!("{attn}.o_proj.weight"))?;
        ensure!(q_proj.shape() == (cfg.num_heads * head_dim, cfg.hidden), "第 {l} 层 q_proj {:?}", q_proj.shape());
        ensure!(k_proj.shape() == (nkv * head_dim, cfg.hidden), "第 {l} 层 k_proj {:?}", k_proj.shape());
        ensure!(o_proj.shape() == (cfg.hidden, cfg.num_heads * head_dim), "第 {l} 层 o_proj {:?}", o_proj.shape());

        let dense = format!("{p}.mlp");
        let router_p = format!("{p}.router");
        let mut experts = Vec::with_capacity(cfg.num_experts);
        for e in 0..cfg.num_experts {
            let ep = format!("{p}.experts.{e}");
            experts.push(Expert {
                gate: QLinear::load(stream, src, &format!("{ep}.gate_proj.weight"))?,
                up: QLinear::load(stream, src, &format!("{ep}.up_proj.weight"))?,
                down: QLinear::load(stream, src, &format!("{ep}.down_proj.weight"))?,
            });
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
            dense_gate: QLinear::load(stream, src, &format!("{dense}.gate_proj.weight"))?,
            dense_up: QLinear::load(stream, src, &format!("{dense}.up_proj.weight"))?,
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
            sliding: Rope::with_denom(
                stream,
                cfg.rope_sliding.rope_dim,
                cfg.rope_sliding.denom,
                cfg.max_pos,
                cfg.rope_sliding.theta,
                RopeStyle::Neox,
            )?,
            full: Rope::with_denom(
                stream,
                cfg.rope_full.rope_dim,
                cfg.rope_full.denom,
                cfg.max_pos,
                cfg.rope_full.theta,
                RopeStyle::Neox,
            )?,
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
    pub acc: Tensor<bf16>,
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
