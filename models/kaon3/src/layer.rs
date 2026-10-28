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

    /// 一组会被 vLLM 合并成**一个 GEMM** 的 fp8 权重（q/k/v、gate/up），重量化后拼成单块 `[ΣN,K]`。
    /// 返回合并线性层 + 各分片 N 宽度（供前向切列段）。见 `weights::load_fp8_group_merged`。
    pub fn load_group_merged(stream: &Stream, src: &WeightSource, names: &[String]) -> Result<(Self, Vec<usize>)> {
        let (qw, widths) = weights::load_fp8_group_merged(src, stream, names)?;
        Ok((Self::from_qw(qw), widths))
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

    /// FP8 权重与两个 per-tensor scale（megakernel 直接吃原始指针，绕过 `forward` 的量化暂存）。
    /// 返回 `(w, 权重 scale, 激活 scale, n, k)`；非 FP8 权重返回 `None`。
    pub fn fp8_parts(&self) -> Option<(&dinfer_kernels::cudarc::driver::CudaSlice<u8>, f32, f32, usize, usize)> {
        match self {
            Self::Fp8 { w, scale, a_scale, n, k, .. } => Some((w, *scale, *a_scale, *n, *k)),
            _ => None,
        }
    }

    /// NVFP4 权重各件（B3 grouped GEMM 建 device 指针表用）：`(packed, scale, global, a_scale, n, k)`。
    pub fn nvfp4_parts(
        &self,
    ) -> Option<(
        &dinfer_kernels::cudarc::driver::CudaSlice<u8>,
        &dinfer_kernels::cudarc::driver::CudaSlice<u8>,
        f32,
        f32,
        usize,
        usize,
    )> {
        match self {
            Self::NvFp4 { packed, scale, global, a_scale, n, k } => Some((packed, scale, *global, *a_scale, *n, *k)),
            _ => None,
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
                // B6（loop kaon3-decode-opt）：decode 小 M 走带宽型 fp8 GEMV（`mlp_fp8_gemv`，与 dense MLP 同 kernel），
                // 条件：tokens ≤ 16、K % 16 == 0、A 的 bf16 驻留 M_TILE·K·2 ≤ 24 KB（≥ 4 block/SM 才能把权重流读的延迟藏住；
                // D62 实测 45 KB（m=8,K=2816）时 c8 反而比 cuBLASLt 慢 5%）。K=2816 → m ≤ 4；K=8192 → m = 1。`FP8_GEMV=1` 开。
                if fp8_gemv_enabled() && tokens <= 16 && kk % 16 == 0 && fp8_gemv_m_tile(tokens) * kk * 2 <= 24 * 1024 {
                    return k::ops::mlp_fp8_gemv(stream, buf, w, &mut out.data, tokens, n, kk, *scale * *a_scale);
                }
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

/// B3 MoE device-grouped 的 per-layer 常驻设备表（专家权重指针 + 标量，一次建，30 层各一份）。
/// 让 grouped GEMM 在 device 上按 offsets 选专家权重，去掉 host 逐专家 launch。
pub struct MoeGrouped {
    /// 各专家 packed/scale 的设备地址（u64）：gate/up/down 各 [E]。
    pub gate_packed: k::cudarc::driver::CudaSlice<u64>,
    pub gate_scale: k::cudarc::driver::CudaSlice<u64>,
    pub up_packed: k::cudarc::driver::CudaSlice<u64>,
    pub up_scale: k::cudarc::driver::CudaSlice<u64>,
    pub down_packed: k::cudarc::driver::CudaSlice<u64>,
    pub down_scale: k::cudarc::driver::CudaSlice<u64>,
    /// 激活量化网格 gs = 1/a_scale（gate/up 共用一个，down 自己的）。
    pub agscale_gu: k::cudarc::driver::CudaSlice<f32>,
    pub agscale_dn: k::cudarc::driver::CudaSlice<f32>,
    /// 输出乘子 alpha = global · a_scale（= 单专家 forward_nvfp4 的 `global * a_scale`）。
    pub alpha_gate: k::cudarc::driver::CudaSlice<f32>,
    pub alpha_up: k::cudarc::driver::CudaSlice<f32>,
    pub alpha_down: k::cudarc::driver::CudaSlice<f32>,
    pub n_gate: usize,
    pub k_gate: usize,
    pub n_down: usize,
    pub k_down: usize,
}

impl MoeGrouped {
    pub fn build(stream: &Stream, experts: &[Expert]) -> Result<Self> {
        let e = experts.len();
        let (mut gp, mut gs, mut up_, mut us, mut dp, mut ds) =
            (vec![], vec![], vec![], vec![], vec![], vec![]);
        let (mut ag_gu, mut ag_dn) = (vec![], vec![]);
        let (mut al_g, mut al_u, mut al_d) = (vec![], vec![], vec![]);
        let (mut shape_gate, mut shape_down) = (None, None);
        for ex in experts {
            let (gpk, gsc, gglobal, ga, gn, gk) =
                ex.gate.nvfp4_parts().context("grouped MoE 需要 NVFP4 gate 权重")?;
            let (upk, usc, uglobal, ua, _un, _uk) =
                ex.up.nvfp4_parts().context("grouped MoE 需要 NVFP4 up 权重")?;
            let (dpk, dsc, dglobal, da, dn, dk) =
                ex.down.nvfp4_parts().context("grouped MoE 需要 NVFP4 down 权重")?;
            // gate/up 的 a_scale 在 load 时已取 max 置相等（见 Layer::load），这里断言一致。
            ensure!((ga - ua).abs() < 1e-12, "grouped MoE: gate/up a_scale 不一致 {ga} {ua}");
            gp.push(stream.ptr(gpk));
            gs.push(stream.ptr(gsc));
            up_.push(stream.ptr(upk));
            us.push(stream.ptr(usc));
            dp.push(stream.ptr(dpk));
            ds.push(stream.ptr(dsc));
            ag_gu.push(1.0f32 / ga);
            ag_dn.push(1.0f32 / da);
            al_g.push(gglobal * ga);
            al_u.push(uglobal * ua);
            al_d.push(dglobal * da);
            shape_gate.get_or_insert((gn, gk));
            shape_down.get_or_insert((dn, dk));
        }
        ensure!(gp.len() == e, "grouped MoE: 专家数 {}", e);
        let (n_gate, k_gate) = shape_gate.context("无专家")?;
        let (n_down, k_down) = shape_down.unwrap();
        Ok(Self {
            gate_packed: stream.htod_copy(gp)?,
            gate_scale: stream.htod_copy(gs)?,
            up_packed: stream.htod_copy(up_)?,
            up_scale: stream.htod_copy(us)?,
            down_packed: stream.htod_copy(dp)?,
            down_scale: stream.htod_copy(ds)?,
            agscale_gu: stream.htod_copy(ag_gu)?,
            agscale_dn: stream.htod_copy(ag_dn)?,
            alpha_gate: stream.htod_copy(al_g)?,
            alpha_up: stream.htod_copy(al_u)?,
            alpha_down: stream.htod_copy(al_d)?,
            n_gate,
            k_gate,
            n_down,
            k_down,
        })
    }
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
    /// q/k/v 合并成**一次 N=ΣN 的 GEMM**（vLLM `QKVParallelLinear`，loop `kaon3-layer1` 第 7 轮根因）。
    /// 前向一次算完再按 `qkv_widths` 切列段。k_eq_v 的 full 层只合并 q+k，v 复用 k 段。
    pub qkv: QLinear,
    /// 合并 GEMM 各列段宽度：`[q_n, k_n]`（k_eq_v）或 `[q_n, k_n, v_n]`。
    pub qkv_widths: Vec<usize>,
    pub o_proj: QLinear,
    /// 逐头 RMSNorm（带权重），长度 = head_dim。
    pub q_norm: Tensor<bf16>,
    pub k_norm: Tensor<bf16>,
    pub dense_gate: QLinear,
    pub dense_up: QLinear,
    pub dense_down: QLinear,
    pub router: Router,
    pub experts: Vec<Expert>,
    /// B3 MoE device-grouped 的 per-layer 设备指针表（`MOE_GROUPED=1` 时首用懒建）。
    pub moe_grouped: std::sync::OnceLock<MoeGrouped>,
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
        // vLLM 的 qkv_proj 是**合并成一次 N=ΣN GEMM** 的层（loop `kaon3-layer1` 第 7 轮：分开三次差 13 万，
        // 合并一次只差 13）。k_eq_v 的 full 层 K 权重兼作 V，只合并 q+k，组 scale 只由 q/k 决定。
        let (qkv, qkv_widths) = QLinear::load_group_merged(
            stream,
            src,
            &if k_eq_v {
                vec![format!("{attn}.q_proj.weight"), format!("{attn}.k_proj.weight")]
            } else {
                vec![format!("{attn}.q_proj.weight"), format!("{attn}.k_proj.weight"), format!("{attn}.v_proj.weight")]
            },
        )?;
        let o_proj = QLinear::load(stream, src, &format!("{attn}.o_proj.weight"))?;
        ensure!(qkv_widths[0] == cfg.num_heads * head_dim, "第 {l} 层 q 宽 {}", qkv_widths[0]);
        ensure!(qkv_widths[1] == nkv * head_dim, "第 {l} 层 k 宽 {}", qkv_widths[1]);
        if !k_eq_v {
            ensure!(qkv_widths[2] == nkv * head_dim, "第 {l} 层 v 宽 {}", qkv_widths[2]);
        }
        ensure!(qkv.shape() == (qkv_widths.iter().sum::<usize>(), cfg.hidden), "第 {l} 层 qkv {:?}", qkv.shape());
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
            qkv,
            qkv_widths,
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
            moe_grouped: std::sync::OnceLock::new(),
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
    rows: usize,
) -> Result<()> {
    let _ = ops;
    // 原位 RMSNorm：不分配、不换缓冲身份（CUDA graph 捕获期禁止两者）。逐位等价于旧的
    // "写新缓冲再换 x.data"（kernel 先把整行读进寄存器再写回，见 `rmsnorm_bf16_inplace`）。
    // rows = tokens × heads（实际行数）。缓冲是 max_tokens 满长，按 numel 跑会把 decode 的 1 行放大成
    // 4160 行（loop kaon3-decode-opt D54：三个逐头 norm 每步白跑 2.3 ms）。
    ensure!(rows * head_dim <= x.numel(), "head_rmsnorm: rows {rows} × {head_dim} > numel {}", x.numel());
    k::ops::rmsnorm_bf16_inplace(stream, &mut x.data, w.map(|w| &w.data), rows, head_dim, eps)
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

/// dense MLP megakernel 一次 launch 的最大行数（kernel host 侧断言 `m <= 8`）。
/// B5：decode（tokens ≤ 16）dense MLP 走带宽型 fp8 GEMV（`mlp_fp8_decode.cu`），不走 persistent mega。
/// `DENSE_DECODE_OLD=1` 回退（A/B 用）。
pub const DENSE_GEMV_ROWS: usize = 16;
/// B11b：MoE 输出合并融合（默认开；`MOE_COMBINE_OLD=1` 回退 scatter_add 链）。
fn moe_combine_fused_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("MOE_COMBINE_OLD").ok().as_deref() != Some("1"))
}
/// B10：MoE 路由前处理融合（默认开；`MOE_ROUTE_FUSED=0` 回退 7 次 launch 的旧链）。
fn moe_route_fused_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("MOE_ROUTE_FUSED").ok().as_deref() != Some("0"))
}
/// B7：MoE decode grouped NVFP4 GEMV（默认关，`MOE_DECODE_GEMV=1` 开）。
fn moe_decode_gemv_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("MOE_DECODE_GEMV").ok().as_deref() == Some("1"))
}
/// B6 回滚中（D62：输出坏）：默认关，`FP8_GEMV=1` 才开（排查用）。
fn fp8_gemv_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("FP8_GEMV").ok().as_deref() == Some("1"))
}
/// 与 `mlp_fp8_decode.cu` 的 M_TILE 档位同式（{1,2,4,8,16}），用于算共享内存是否放得下。
fn fp8_gemv_m_tile(m: usize) -> usize {
    match m { 0 | 1 => 1, 2 => 2, 3 | 4 => 4, 5..=8 => 8, _ => 16 }
}
fn dense_gemv_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("DENSE_DECODE_OLD").ok().as_deref() != Some("1"))
}
pub const MEGA_ROWS: usize = 8;
/// megakernel 的阶段数（`S_COUNT`，见 `mega_mlp_sm120.cu`）；`ready` 按它清零。
const MEGA_STAGES: usize = 6;

/// `KAON3_MEGA=1` 时 dense MLP 支路走 persistent megakernel（一次 launch），默认关。
pub fn mega_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("KAON3_MEGA").map(|v| v.trim() == "1").unwrap_or(false))
}

/// B3：MoE device-grouped（去 host 路由 + 去 D2H）。默认关（A/B）；`MOE_GROUPED=1` 开。
pub fn moe_grouped_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("MOE_GROUPED").map(|v| v.trim() == "1").unwrap_or(false))
}

/// megakernel 的中间缓冲与阶段表。**只按 [`MEGA_ROWS`] 行分配**（prefill 走原路径）；
/// 全模型一套复用（逐层串行）。`ready` 每次 launch 前必须清零。
pub struct MegaBufs {
    pub norm1: Tensor<bf16>, // [8, H]  pre_ff_norm 的输出（对拍用）
    pub norm1_q: k::cudarc::driver::CudaSlice<u8>, // [8, H]  喂 gate/up 的 fp8 激活
    pub gate_buf: Tensor<bf16>, // [8, I]
    pub up_buf: Tensor<bf16>,   // [8, I]
    pub act_bf: Tensor<bf16>,   // [8, I]  gelu·mul（bf16）
    pub act_q: k::cudarc::driver::CudaSlice<u8>, // [8, I]  喂 down 的 fp8 激活
    pub down_out: Tensor<bf16>, // [8, H]
    pub dep: k::cudarc::driver::CudaSlice<i32>,
    pub count: k::cudarc::driver::CudaSlice<i32>,
    pub start: k::cudarc::driver::CudaSlice<i32>,
    /// kernel 的全局阶段计数器；**每次 launch 前必须清零**。
    pub ready: k::cudarc::driver::CudaSlice<i32>,
    /// 第二前驱表（ACT 等 gate+up）；serve 侧也要传真 buf（R27 后 ffi 签名需要）。
    pub dep2: k::cudarc::driver::CudaSlice<i32>,
    /// split-K（R29）：gate/up 份数 sp_gu、down 份数 sp_dn（env `MEGA_SPGU`/`MEGA_SPDN`，默认 1）。
    pub sp_gu: i32,
    pub sp_dn: i32,
    /// 每 SM 常驻 CTA 数（R4，env `MEGA_CTAS`，默认 1）；≥2 用 OCC=2 kernel 变体。
    pub ctas: i32,
    pub part_gate: k::cudarc::driver::CudaSlice<f32>, // [sp_gu, 8, I]
    pub part_up: k::cudarc::driver::CudaSlice<f32>,   // [sp_gu, 8, I]
    pub part_dn: k::cudarc::driver::CudaSlice<f32>,   // [sp_dn, 8, H]
    pub sem_gate: k::cudarc::driver::CudaSlice<i32>,  // [ceil(I/128)]
    pub sem_up: k::cudarc::driver::CudaSlice<i32>,
    /// KAON3_DENSE_TIME 分段计时缓冲（6 stage exec/wait cycle，device 累加）。
    pub wait_cyc: k::cudarc::driver::CudaSlice<u64>,
    pub exec_cyc: k::cudarc::driver::CudaSlice<u64>,
    pub sem_dn: k::cudarc::driver::CudaSlice<i32>,    // [ceil(H/128)]
}

impl MegaBufs {
    pub fn new(stream: &Stream, cfg: &KaonConfig) -> Result<Self> {
        let (h, i) = (cfg.hidden, cfg.dense_inter);
        let sp_gu = std::env::var("MEGA_SPGU").ok().and_then(|v| v.parse().ok()).unwrap_or(1i32).max(1);
        let sp_dn = std::env::var("MEGA_SPDN").ok().and_then(|v| v.parse().ok()).unwrap_or(1i32).max(1);
        let ctas = std::env::var("MEGA_CTAS").ok().and_then(|v| v.parse().ok()).unwrap_or(1i32).max(1);
        let n_gu = (i.div_ceil(128)) as i32;
        let n_dn = (h.div_ceil(128)) as i32;
        // dep/count/start/dep2 表**在构造时（非 capture 期）算好并 htod 一次**（内容只依赖形状/split，层内每步不变）。
        // 原先在 C ABI 里每次调用从**栈数组** H2D memcpy，graph capture 录下栈地址 → replay 读失效栈 → dep 表乱 → 乱码
        // （kaon3-mega R45 根因）。这里改成常驻 device 表，kernel launch 前不再 memcpy，graph 内零 H2D。
        // stage 枚举顺序须与 mega_mlp_sm120.cu 的 Stage 一致：NORM=0 GATE=1 UP=2 ACT=3 DOWN=4 POST=5。
        let count_v: Vec<i32> = vec![1, n_gu * sp_gu, n_gu * sp_gu, 1, n_dn * sp_dn, 1];
        let dep_v: Vec<i32> = vec![-1, 0, 0, 1, 3, 4]; // NORM 无依赖；GATE/UP 依赖 NORM；ACT 依赖 GATE(+dep2 UP)；DOWN 依赖 ACT；POST 依赖 DOWN
        let dep2_v: Vec<i32> = vec![-1, -1, -1, 2, -1, -1]; // 仅 ACT 有第二前驱 = UP(2)
        let mut start_v = vec![0i32; MEGA_STAGES];
        let mut acc = 0i32;
        for s in 0..MEGA_STAGES {
            start_v[s] = acc;
            acc += count_v[s];
        }
        Ok(Self {
            norm1: Tensor::uninit(stream, &[MEGA_ROWS, h])?,
            norm1_q: stream.alloc_zeros::<u8>(MEGA_ROWS * h)?,
            gate_buf: Tensor::uninit(stream, &[MEGA_ROWS, i])?,
            up_buf: Tensor::uninit(stream, &[MEGA_ROWS, i])?,
            act_bf: Tensor::uninit(stream, &[MEGA_ROWS, i])?,
            act_q: stream.alloc_zeros::<u8>(MEGA_ROWS * i)?,
            down_out: Tensor::uninit(stream, &[MEGA_ROWS, h])?,
            dep: stream.htod_copy(dep_v)?,
            count: stream.htod_copy(count_v)?,
            start: stream.htod_copy(start_v)?,
            ready: stream.alloc_zeros::<i32>(MEGA_STAGES)?,
            dep2: stream.htod_copy(dep2_v)?,
            sp_gu,
            sp_dn,
            ctas,
            part_gate: stream.alloc_zeros::<f32>(sp_gu as usize * MEGA_ROWS * i)?,
            part_up: stream.alloc_zeros::<f32>(sp_gu as usize * MEGA_ROWS * i)?,
            part_dn: stream.alloc_zeros::<f32>(sp_dn as usize * MEGA_ROWS * h)?,
            sem_gate: stream.alloc_zeros::<i32>(n_gu as usize)?,
            sem_up: stream.alloc_zeros::<i32>(n_gu as usize)?,
            wait_cyc: stream.alloc_zeros::<u64>(MEGA_STAGES)?,
            exec_cyc: stream.alloc_zeros::<u64>(MEGA_STAGES)?,
            sem_dn: stream.alloc_zeros::<i32>(n_dn as usize)?,
        })
    }
}

/// 一层 forward 的工作区（按 `max_tokens` 预分配，全模型一套复用）。
pub struct LayerScratch {
    pub x: Tensor<bf16>,        // 残差主干 [tokens, hidden]
    pub norm: Tensor<bf16>,     // norm 输出 [tokens, hidden]
    /// final norm（lm_head 前）的固定缓冲 [max_tokens, hidden]；CUDA graph 捕获期不能每步 `uninit`。
    pub final_normed: Tensor<bf16>,
    /// q/k/v/attn 按**层型**分两套（滑窗 D=256 / 全注意力 D=512），
    /// 因为 rope 与 attention 都按精确长度校验，不能共用一个大缓冲。
    pub q_s: Tensor<bf16>,
    pub k_s: Tensor<bf16>,
    pub v_s: Tensor<bf16>,
    pub attn_s: Tensor<bf16>,
    /// 合并 qkv GEMM 的输出 `[tokens, ΣN]`，之后切列段到 q/k/v。按最大层型容量分配。
    pub qkv: Tensor<bf16>,
    pub q_f: Tensor<bf16>,
    pub k_f: Tensor<bf16>,
    pub v_f: Tensor<bf16>,
    pub attn_f: Tensor<bf16>,
    /// 纯 decode split-K attention 的 partial 缓冲（B3）：按最大 head_dim 与 max_tokens 行分配。
    pub decode_scratch: Tensor<f32>,
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
    /// `KAON3_MEGA=1` 时 dense MLP 支路的中间缓冲；关时为空（不占显存）。
    pub mega: Option<MegaBufs>,
    /// `MOE_GROUPED=1` 时 device-grouped MoE 的中间缓冲（R=max_tokens·top_k 行）；关时为空。
    pub grouped: Option<GroupedBufs>,
    /// B5 decode dense MLP fp8 GEMV 路径的小 fp8 缓冲（[16, hidden] / [16, dense_inter]）。
    pub dq16: dinfer_kernels::cudarc::driver::CudaSlice<u8>,
    pub act_q16: dinfer_kernels::cudarc::driver::CudaSlice<u8>,
}

/// B3 device-grouped MoE 的中间缓冲（`MOE_GROUPED=1`）。R = max_tokens·top_k。
pub struct GroupedBufs {
    pub row2e: k::cudarc::driver::CudaSlice<i32>,     // [R]
    pub aq_gu: k::cudarc::driver::CudaSlice<u8>,      // [R, k_gate/2]
    pub as_gu: k::cudarc::driver::CudaSlice<u8>,      // [R, k_gate/16]
    pub aq_dn: k::cudarc::driver::CudaSlice<u8>,      // [R, k_down/2]
    pub as_dn: k::cudarc::driver::CudaSlice<u8>,      // [R, k_down/16]
    pub gate_o: Tensor<bf16>,                         // [R, moe_inter]
    pub up_o: Tensor<bf16>,                           // [R, moe_inter]
    pub gelu_o: Tensor<bf16>,                         // [R, moe_inter]
}

impl GroupedBufs {
    pub fn new(stream: &Stream, cfg: &KaonConfig, max_tokens: usize) -> Result<Self> {
        let r = max_tokens * cfg.top_k;
        let (kg, kd, mi) = (cfg.hidden, cfg.moe_inter, cfg.moe_inter);
        Ok(Self {
            row2e: stream.alloc_zeros::<i32>(r)?,
            aq_gu: stream.alloc_zeros::<u8>(r * (kg / 2))?,
            as_gu: stream.alloc_zeros::<u8>(r * (kg / 16))?,
            aq_dn: stream.alloc_zeros::<u8>(r * (kd / 2))?,
            as_dn: stream.alloc_zeros::<u8>(r * (kd / 16))?,
            gate_o: Tensor::uninit(stream, &[r, mi])?,
            up_o: Tensor::uninit(stream, &[r, mi])?,
            gelu_o: Tensor::uninit(stream, &[r, mi])?,
        })
    }
}

impl LayerScratch {
    pub fn new(stream: &Stream, cfg: &KaonConfig, max_tokens: usize) -> Result<Self> {
        Self::new_with_rows(stream, cfg, max_tokens, max_tokens.min(64))
    }
    /// `max_decode_rows`：纯 decode 一步最多多少请求（= max_num_seqs），只决定 decode_scratch 大小
    /// （B×heads×32×(D+2) f32；按 max_tokens 分配会是 4.4 GB → OOM，B3 第一次验证踩过）。
    pub fn new_with_rows(stream: &Stream, cfg: &KaonConfig, max_tokens: usize, max_decode_rows: usize) -> Result<Self> {
        let (h, d, nkv) = (cfg.hidden, cfg.head_dim_full, cfg.nkv_full);
        let _ = (d, nkv);
        Ok(Self {
            x: Tensor::uninit(stream, &[max_tokens, cfg.hidden])?,
            norm: Tensor::uninit(stream, &[max_tokens, cfg.hidden])?,
            final_normed: Tensor::uninit(stream, &[max_tokens, cfg.hidden])?,
            q_s: Tensor::uninit(stream, &[max_tokens, cfg.num_heads * cfg.head_dim_sliding])?,
            k_s: Tensor::uninit(stream, &[max_tokens, cfg.nkv_sliding * cfg.head_dim_sliding])?,
            v_s: Tensor::uninit(stream, &[max_tokens, cfg.nkv_sliding * cfg.head_dim_sliding])?,
            attn_s: Tensor::uninit(stream, &[max_tokens, cfg.num_heads * cfg.head_dim_sliding])?,
            q_f: Tensor::uninit(stream, &[max_tokens, cfg.num_heads * cfg.head_dim_full])?,
            k_f: Tensor::uninit(stream, &[max_tokens, cfg.nkv_full * cfg.head_dim_full])?,
            v_f: Tensor::uninit(stream, &[max_tokens, cfg.nkv_full * cfg.head_dim_full])?,
            attn_f: Tensor::uninit(stream, &[max_tokens, cfg.num_heads * cfg.head_dim_full])?,
            decode_scratch: Tensor::zeros(
                stream,
                &[k::gqa_decode_scratch_len(max_decode_rows.max(1), cfg.num_heads, cfg.head_dim_full.max(cfg.head_dim_sliding))],
            )?,
            // 合并 qkv 输出的上界：两层型 q+k+v 的较大者（full: num_heads·D + 2·nkv·D）。
            qkv: Tensor::uninit(stream, &[max_tokens,
                (cfg.num_heads * cfg.head_dim_sliding + 2 * cfg.nkv_sliding * cfg.head_dim_sliding)
                    .max(cfg.num_heads * cfg.head_dim_full + 2 * cfg.nkv_full * cfg.head_dim_full)])?,
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
            mega: if mega_enabled() { Some(MegaBufs::new(stream, cfg)?) } else { None },
            grouped: if moe_grouped_enabled() { Some(GroupedBufs::new(stream, cfg, max_tokens)?) } else { None },
            dq16: stream.alloc_zeros::<u8>(DENSE_GEMV_ROWS * cfg.hidden)?,
            act_q16: stream.alloc_zeros::<u8>(DENSE_GEMV_ROWS * cfg.dense_inter)?,
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
    /// dense MLP 支路走 megakernel（`KAON3_MEGA=1` 且 `tokens <= MEGA_ROWS` 且三个投影都是 FP8）。
    ///
    /// 成功时把 `post_ff_norm_1` 的结果写进 `sc.norm` 并返回 `Ok(true)`；任何前置条件不满足
    /// （开关关 / prefill 大 batch / 权重不是 FP8）返回 `Ok(false)`，由调用方走逐 op 路径。
    ///
    /// 两处硬约束：① kernel host 侧断言 `m <= 8`（更大行数只会在 prefill 出现）；
    /// ② `ready` 是 kernel 的全局阶段计数器，**每次 launch 前必须清零**，否则第二次 launch 会直接越过等待。
    /// B5：decode dense MLP 带宽型 fp8 路径（norm→fp8 / gate‖up GEMV+gelu·mul→fp8 / down GEMV），写 `sc.dense`
    /// （未做 post_ff_norm_1，由调用方 `rmsnorm_into` 到 `sc.norm`）。返回 false = 不适用（走 mega / 通用路径）。
    fn dense_decode_gemv(
        &self,
        stream: &Stream,
        cfg: &KaonConfig,
        sc: &mut LayerScratch,
        tokens: usize,
        eps: f32,
    ) -> Result<bool> {
        if !dense_gemv_enabled() || tokens == 0 || tokens > DENSE_GEMV_ROWS {
            return Ok(false);
        }
        let (Some((gw, gw_s, gw_a, gn, gk)), Some((uw, uw_s, uw_a, un, uk)), Some((dw, dw_s, dw_a, dn, dk))) =
            (self.dense_gate.fp8_parts(), self.dense_up.fp8_parts(), self.dense_down.fp8_parts())
        else {
            return Ok(false);
        };
        let (h, inter) = (cfg.hidden, cfg.dense_inter);
        if h % 16 != 0 || inter % 16 != 0 {
            return Ok(false);
        }
        ensure!(gn == inter && un == inter && gk == h && uk == h && dn == h && dk == inter, "dense gemv: 形状");
        ensure!((gw_a - uw_a).abs() < 1e-12, "dense gemv: gate/up a_scale 不一致 {gw_a} {uw_a}");
        k::ops::mlp_fp8_norm_quant(stream, &sc.x.data, &self.pre_ff_norm.data, &mut sc.dq16, tokens, h, eps, 1.0 / gw_a)?;
        k::ops::mlp_fp8_gateup_act(stream, &sc.dq16, gw, uw, &mut sc.act_q16, tokens, inter, h, gw_s * gw_a, uw_s * uw_a, 1.0 / dw_a)?;
        k::ops::mlp_fp8_gemv(stream, &sc.act_q16, dw, &mut sc.dense.data, tokens, h, inter, dw_s * dw_a)?;
        Ok(true)
    }

    fn mega_dense_branch(
        &self,
        stream: &Stream,
        cfg: &KaonConfig,
        sc: &mut LayerScratch,
        tokens: usize,
        eps: f32,
    ) -> Result<bool> {
        if !mega_enabled() || tokens == 0 || tokens > MEGA_ROWS {
            return Ok(false);
        }
        let (Some((gw, gw_s, gw_a, gn, gk)), Some((uw, uw_s, uw_a, un, uk)), Some((dw, dw_s, dw_a, dn, dk))) =
            (self.dense_gate.fp8_parts(), self.dense_up.fp8_parts(), self.dense_down.fp8_parts())
        else {
            return Ok(false); // 非 FP8 权重：kernel 只支持 fp8 输入
        };
        let (h, inter) = (cfg.hidden, cfg.dense_inter);
        ensure!(gn == inter && un == inter && gk == h && uk == h, "mega: gate/up 形状 {gn}x{gk} / {un}x{uk} 不是 [{inter},{h}]");
        ensure!(dn == h && dk == inter, "mega: down 形状 {dn}x{dk} 不是 [{h},{inter}]");
        let Some(mb) = sc.mega.as_mut() else { return Ok(false) };

        // `ready` 必须从 0 开始（kernel 的 stage 计数器是累加的）。
        stream.memset_zeros_n(&mut mb.ready, MEGA_STAGES)?;
        // KAON3_DENSE_TIME=1：开 dense branch 6-stage exec/wait 计时（每次清零累加缓冲）。
        let dense_time = std::env::var("KAON3_DENSE_TIME").ok().as_deref() == Some("1");
        if dense_time {
            stream.memset_zeros_n(&mut mb.wait_cyc, MEGA_STAGES)?;
            stream.memset_zeros_n(&mut mb.exec_cyc, MEGA_STAGES)?;
        }

        let n_gu = (inter.div_ceil(128)) as i32; // kernel kBN = 128
        let n_dn = (h.div_ceil(128)) as i32;
        let rc = unsafe {
            dinfer_kernels::ffi::mega_mlp_dense_branch(
                stream.ptr(&sc.x.data) as *const std::ffi::c_void,
                stream.ptr_mut(&mut sc.norm.data) as *mut std::ffi::c_void,
                stream.ptr(&self.pre_ff_norm.data) as *const std::ffi::c_void,
                stream.ptr(&self.post_ff_norm_1.data) as *const std::ffi::c_void,
                stream.ptr(gw) as *const std::ffi::c_void,
                gw_s,
                1.0 / gw_a,
                stream.ptr(uw) as *const std::ffi::c_void,
                uw_s,
                1.0 / uw_a,
                stream.ptr(dw) as *const std::ffi::c_void,
                dw_s,
                1.0 / dw_a,
                stream.ptr_mut(&mut mb.norm1.data) as *mut std::ffi::c_void,
                stream.ptr_mut(&mut mb.norm1_q) as *mut std::ffi::c_void,
                stream.ptr_mut(&mut mb.gate_buf.data) as *mut std::ffi::c_void,
                stream.ptr_mut(&mut mb.up_buf.data) as *mut std::ffi::c_void,
                stream.ptr_mut(&mut mb.act_bf.data) as *mut std::ffi::c_void,
                stream.ptr_mut(&mut mb.act_q) as *mut std::ffi::c_void,
                stream.ptr_mut(&mut mb.down_out.data) as *mut std::ffi::c_void,
                tokens as i32,
                h as i32,
                inter as i32,
                eps,
                n_gu,
                n_dn,
                stream.ptr_mut(&mut mb.dep) as *mut i32,
                stream.ptr_mut(&mut mb.count) as *mut i32,
                stream.ptr_mut(&mut mb.start) as *mut i32,
                stream.ptr_mut(&mut mb.ready) as *mut i32,
                stream.ptr_mut(&mut mb.dep2) as *mut i32,
                2, // pipe
                dense_time as i32, // prof（KAON3_DENSE_TIME=1 开）
                if dense_time { stream.ptr_mut(&mut mb.wait_cyc) as *mut u64 } else { std::ptr::null_mut() },
                if dense_time { stream.ptr_mut(&mut mb.exec_cyc) as *mut u64 } else { std::ptr::null_mut() },
                mb.sp_gu,
                mb.sp_dn,
                stream.ptr_mut(&mut mb.part_gate) as *mut std::ffi::c_void,
                stream.ptr_mut(&mut mb.part_up) as *mut std::ffi::c_void,
                stream.ptr_mut(&mut mb.part_dn) as *mut std::ffi::c_void,
                stream.ptr_mut(&mut mb.sem_gate) as *mut i32,
                stream.ptr_mut(&mut mb.sem_up) as *mut i32,
                stream.ptr_mut(&mut mb.sem_dn) as *mut i32,
                mb.ctas,
                stream.raw() as _,
            )
        };
        if dense_time {
            stream.synchronize()?;
            let wc = stream.dtoh_sync_copy(&mb.wait_cyc)?;
            let ec = stream.dtoh_sync_copy(&mb.exec_cyc)?;
            let sn = ["NORM", "GATE", "UP", "ACT", "DOWN", "POST"];
            let et: u64 = ec.iter().sum();
            let wt: u64 = wc.iter().sum();
            let parts: Vec<String> = (0..MEGA_STAGES.min(6))
                .map(|i| format!("{}=e{} w{}", sn[i], ec[i], wc[i]))
                .collect();
            tracing::info!("[dense-time] tok={tokens} exec合计={et} wait合计={wt} | {}", parts.join(" "));
        }
        ensure!(rc == 0, "mega_mlp_dense_branch 返回 {rc}（非 sm120 构建为 801）");
        Ok(true)
    }

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

        // 分段计时（`KAON3_STAGE_TIME=1`，默认关零开销）：量 serve 真实 decode 各段 GPU 时间。
        // 段:0入口 1=attn完(含qkv/norm/rope/paged) 2=oproj+残差完 3=dense完 4=moe完。
        let stage_time = std::env::var("KAON3_STAGE_TIME").ok().as_deref() == Some("1");
        let st_ev = if stage_time { Some(stream.new_timing_events(5)?) } else { None };
        if let Some(e) = &st_ev { stream.record_timing(&e[0])?; }

        // ---- 1. input_norm ----
        // vLLM **compile 口径**（默认）：inductor 把 RMSNorm 与下游 fp8 量化融成一个 kernel，中间不落 bf16。
        // q/k/v 共用同一个 input_norm 且共用同一个 `input_scale`（实测 L0/L5 三者 scale 相同），
        // 所以这里量化一次、三个投影复用。eager 口径（`RMSNORM_MODE=1`）走原来的两步。
        // q/k/v 合并成一次 GEMM，共用一个 `input_scale`（= 合并组 input_scale 的 max，见 `load_fp8_group_merged`）。
        let mut qkv_preq = false;
        if norm_mode() != 1 && fuse_norm_quant() {
            if let Some(qs) = self.qkv.a_scale() {
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
        // ---- 2a. 一次合并 GEMM 到 sc.qkv[tokens, ΣN]（vLLM 口径：分开三次差 13 万，合并只差 13）----
        // ATTN_SKIP_QKV=1（纯计时,结果错）：跳过 qkv GEMM,测其 graph 态占比。
        let attn_skip_qkv = std::env::var("ATTN_SKIP_QKV").ok().as_deref() == Some("1");
        if attn_skip_qkv {
            // 不跑 GEMM,sc.qkv 留旧值(数值错,纯测时间)
        } else if qkv_preq {
            let g = sc.preq.lock().unwrap();
            let buf = g.as_ref().unwrap();
            self.qkv.forward_preq(gemm, stream, buf, &mut sc.qkv, tokens)?;
        } else {
            self.qkv.forward(gemm, stream, &sc.norm, &mut sc.qkv, tokens)?;
        }
        // ---- 2b. 切列段：q=[0,q_n) k=[q_n,q_n+k_n) v=... ；k_eq_v 时 v 复用 k 段 ----
        let merged_n = self.qkv_widths.iter().sum::<usize>();
        let q_n = self.qkv_widths[0];
        let k_n = self.qkv_widths[1];
        let col_bytes = |cols: usize| cols * std::mem::size_of::<bf16>();
        let src_base = stream.ptr(&sc.qkv.data);
        let src_pitch = col_bytes(merged_n);
        {
            let dst = stream.ptr(&q.data);
            stream.copy2d(dst, false, col_bytes(q_n), src_base, false, src_pitch, col_bytes(q_n), tokens)?;
        }
        {
            let dst = stream.ptr(&kk.data);
            stream.copy2d(dst, false, col_bytes(k_n), src_base + col_bytes(q_n) as u64, false, src_pitch, col_bytes(k_n), tokens)?;
        }
        if self.k_eq_v {
            // K 权重兼作 V（只对 full 层成立）：k 段逐位复制给 v
            stream.dtod_copy_n(&kk.data, &mut v.data, tokens * nkv * d)?;
        } else {
            let v_n = self.qkv_widths[2];
            let dst = stream.ptr(&v.data);
            stream.copy2d(dst, false, col_bytes(v_n), src_base + col_bytes(q_n + k_n) as u64, false, src_pitch, col_bytes(v_n), tokens)?;
        }

        // 子 loop D：q 投影的**原始输出**（norm 之前），用于区分"GEMM 累加"与"norm"引入的 1 ulp。
        sub_dump(stream, sub_out, "q_proj_raw", q, tokens, heads * d)?;
        // v_norm 定位（§53）：k/v 投影的原始输出，与 vLLM `qkv_raw` 的 k/v 段逐位比。
        sub_dump(stream, sub_out, "k_proj_raw", kk, tokens, nkv * d)?;
        sub_dump(stream, sub_out, "v_proj_raw", v, tokens, nkv * d)?;
        // ---- 3. 逐头 RMSNorm：q_norm / k_norm 带权重，v_norm 无权重 ----
        // ATTN_SKIP_NORM=1（纯计时,结果错）：跳过 3 个逐头 RMSNorm,测其 graph 态占比。
        if std::env::var("ATTN_SKIP_NORM").ok().as_deref() != Some("1") {
            head_rmsnorm(stream, ops, q, Some(&self.q_norm), d, eps, tokens * heads)?;
            head_rmsnorm(stream, ops, kk, Some(&self.k_norm), d, eps, tokens * nkv)?;
            head_rmsnorm(stream, ops, v, None, d, eps, tokens * nkv)?;
        }
        if let Some((dir, _)) = sub_out {
            for (nm, buf, rows) in [("q_norm", &*q, heads), ("k_norm", &*kk, nkv), ("v_norm", &*v, nkv)] {
                let vv = buf.to_vec_f32(stream)?;
                let bytes: Vec<u8> = vv.iter().take(tokens * rows * d).flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(dir.join(format!("{nm}.bin")), bytes)
                    .map_err(|err| anyhow::anyhow!("写 {nm}.bin 失败: {err}"))?;
            }
        }

        // ---- 4. RoPE（只转前 rope_dim 维；proportional 的分母已由 Ropes 处理）----
        // ATTN_SKIP_ROPE=1（纯计时,结果错）：跳过 rope,测其 graph 态占比。
        let rope = if self.window > 0 { &ropes.sliding } else { &ropes.full };
        if std::env::var("ATTN_SKIP_ROPE").ok().as_deref() != Some("1") {
            rope.apply(stream, q, Some(kk), ctx.positions, tokens, heads, nkv, d)?;
        }
        if let Some((dir, _)) = sub_out {
            for (nm, buf, rows) in [("q_rope", &*q, heads), ("k_rope", &*kk, nkv)] {
                let vv = buf.to_vec_f32(stream)?;
                let bytes: Vec<u8> = vv.iter().take(tokens * rows * d).flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(dir.join(format!("{nm}.bin")), bytes)
                    .map_err(|err| anyhow::anyhow!("写 {nm}.bin 失败: {err}"))?;
            }
        }

        // ---- 4a. FP8 KV（vLLM `kv_cache_dtype=fp8_e4m3`，scale 1.0）：attention 读的是量化后的 K/V ----
        // ATTN_SKIP_KVFP8=1（纯计时,结果错）：跳过 3 个 fp8 roundtrip,测其 graph 态占比。
        if cfg.kv_fp8 && std::env::var("ATTN_SKIP_KVFP8").ok().as_deref() != Some("1") {
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
        // ATTN_SKIP_PAGED=1（纯计时,结果错）：跳过 paged/prefill attention kernel。
        let scale = 1.0f32; // Gemma4：scaling = 1.0，由 QK-norm 隐式承担
        if std::env::var("ATTN_SKIP_PAGED").ok().as_deref() == Some("1") {
            // 不跑 attention,attn 留旧值
        } else {
        match ctx.mode {
            // 纯 decode（每请求 1 行）：split-K flash-decode（v3 / d512，带滑窗、GQA 组共享 K/V）。
            // 之前走 paged_win → 滑窗层 v5 每头 1 block（50 us）、D=512 层通用慢路径（369 us）（B3，nsys D54）。
            // ATTN_DECODE_OLD=1 回退旧路径（A/B 用）。
            dinfer_core::runner::BatchMode::Decode if std::env::var("ATTN_DECODE_OLD").ok().as_deref() != Some("1") => {
                k::ops::gqa_decode_win_bf16(
                    stream,
                    &q.data,
                    &*ctx.kv_mut,
                    &ctx.block_table.data,
                    &ctx.seq_lens.data,
                    &mut attn.data,
                    &mut sc.decode_scratch.data,
                    ctx.num_reqs,
                    heads,
                    nkv,
                    d,
                    ctx.block_size,
                    ctx.max_blocks,
                    scale,
                    0,
                    None,
                    self.window,
                )?
            }
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
        } // end !ATTN_SKIP_PAGED

        // ---- 6. o_proj + post_attention_norm + 残差 ----
        if let Some((dir, _)) = sub_out {
            let vv = attn.to_vec_f32(stream)?;
            let bytes: Vec<u8> = vv.iter().take(tokens * heads * d).flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(dir.join("attn_out.bin"), bytes)
                .map_err(|err| anyhow::anyhow!("写 attn_out.bin 失败: {err}"))?;
        }
        if let Some(e) = &st_ev { stream.record_timing(&e[1])?; }  // 段1完=attn(qkv/norm/rope/paged)
        // ATTN_SKIP_OPROJ=1（纯计时,结果错）：跳过 o_proj GEMM。
        if std::env::var("ATTN_SKIP_OPROJ").ok().as_deref() != Some("1") {
            self.o_proj.forward(gemm, stream, attn, &mut sc.o, tokens)?;
        }
        sub_dump(stream, sub_out, "o_proj", &sc.o, tokens, h)?;
        rmsnorm_into(stream, ops, &sc.o, &self.post_attn_norm, &mut sc.norm, h, tokens, eps)?;
        sub_dump(stream, sub_out, "post_attn_norm", &sc.norm, tokens, h)?;
        k::ops::add_inplace_bf16_n(stream, &sc.norm.data, &mut sc.x.data, tokens * h)?;
        if let Some(e) = &st_ev { stream.record_timing(&e[2])?; }  // 段2完=oproj+post_attn_norm+残差

        // ---- 7. dense MLP 支路（无条件执行）----
        // `KAON3_MEGA=1` 且 decode（tokens ≤ 8）时，整条支路走 persistent megakernel 一次 launch：
        // pre_ff_norm → gate ‖ up → gelu·mul(round2) → down → post_ff_norm_1，输出落在 `sc.norm`。
        // 与下面逐 op 路径逐位一致（`crates/kernels/tests/mega_mlp_parity.rs`）。prefill / 非 FP8 回退。
        // DENSE_SKIP=1（纯计时，结果错）：整条 dense 支路不跑，测 graph 态 dense 省掉后 TPOT 掉多少 = dense 真占比。
        let dense_skip = std::env::var("DENSE_SKIP").ok().as_deref() == Some("1");
        let gemv_done = if dense_skip { false } else { self.dense_decode_gemv(stream, cfg, sc, tokens, eps)? };
        if gemv_done {
            rmsnorm_into(stream, ops, &sc.dense, &self.post_ff_norm_1, &mut sc.norm, h, tokens, eps)?;
        }
        let mega_done = if dense_skip || gemv_done { true } else { self.mega_dense_branch(stream, cfg, sc, tokens, eps)? };
        if !mega_done {
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
        }
        sub_dump(stream, sub_out, "ff1", &sc.norm, tokens, h)?;
        if let Some(e) = &st_ev { stream.record_timing(&e[3])?; }  // 段3完=dense MLP支路

        // ---- 8. MoE 支路（与 dense 并联）----
        self.moe_branch(stream, gemm, ops, cfg, sc, tokens, eps, sub_out)?;
        if let Some(e) = &st_ev { stream.record_timing(&e[4])?; }  // 段4完=MoE支路

        // ---- 9. 两条支路相加 → post_ff_norm → 残差 → layer_scalar ----
        k::ops::add_inplace_bf16_n(stream, &sc.h2.data, &mut sc.norm.data, tokens * h)?; // h2 = post_ff_norm_2(moe)
        sub_dump(stream, sub_out, "post_ff_in", &sc.norm, tokens, h)?;
        rmsnorm_into(stream, ops, &sc.norm, &self.post_ff_norm, &mut sc.h2, h, tokens, eps)?;
        sub_dump(stream, sub_out, "post_ff_norm", &sc.h2, tokens, h)?;
        k::ops::add_inplace_bf16_n(stream, &sc.h2.data, &mut sc.x.data, tokens * h)?;
        if self.layer_scalar != 1.0 {
            k::ops::scale_bf16_n(stream, &mut sc.x.data, tokens * h, self.layer_scalar)?;
        }
        if let Some(e) = &st_ev {
            stream.synchronize()?;
            let ms = |a: usize, b: usize| k::TimingEvent::elapsed_ms(&e[a], &e[b]).unwrap_or(0.0);
            tracing::info!(
                "[stage-time] tok={tokens} attn={:.3} oproj={:.3} dense={:.3} moe={:.3} total={:.3} ms",
                ms(0, 1), ms(1, 2), ms(2, 3), ms(3, 4), ms(0, 4)
            );
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
        // MOE_SKIP=1（纯计时，结果错）：整条 MoE 支路不跑，测 graph 态 MoE 省掉后 TPOT 掉多少 = MoE 真占比。
        if std::env::var("MOE_SKIP").ok().as_deref() == Some("1") { return Ok(()); }
        let h = cfg.hidden;
        // 前处理计时（KAON3_MOE_TIME=1）：拆 router / softmax+bucket / norm+gather+quant。
        // 段:[0]起点 [1]router完 [2]softmax+bucket完 [3]norm+row2e+gather+quant完(=第一个gemm前)。
        let moe_pre_ev = if std::env::var("KAON3_MOE_TIME").ok().as_deref() == Some("1") {
            Some(stream.new_timing_events(4)?)
        } else {
            None
        };
        if let Some(e) = &moe_pre_ev { stream.record_timing(&e[0])?; }
        // --- MoE 隔离（MOE_ISO=<dir>，loop kaon3-moe）：喂 vLLM 真实 MoE 输入，纯测路由+专家 ---
        // 设了就从 dir 读 vllm_moe_logits.bin[T,e] f32 → sc.logits（跳过 router 预处理+GEMM），
        // 并记下 hidden 待 pre_ff2_norm 处用 vllm_moe_hidden.bin[T,h] 覆盖 sc.h2。比 vLLM moe_out。
        let moe_iso = std::env::var("MOE_ISO").ok();
        let rdf_iso = |name: &str| -> Result<Vec<f32>> {
            let b = std::fs::read(format!("{}/{name}", moe_iso.as_deref().unwrap()))?;
            Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        };
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
        let e = cfg.num_experts;
        let kk = cfg.top_k;
        let root_size = f32::from(bf16::from_f32((h as f32).powf(-0.5)));
        // MOE_ISO：直接灌 vLLM router logits，跳过 router 预处理 + GEMM（纯测专家）。
        if moe_iso.is_some() {
            let lg = rdf_iso("vllm_moe_logits.bin")?;
            ensure!(lg.len() >= tokens * e, "moe_iso logits 太小 {} < {}", lg.len(), tokens * e);
            stream.htod_into(&lg[..tokens * e].to_vec(), &mut sc.logits.data)?;
        } else {
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
        // vLLM（容器 0.26.1rc1，sm_120）`GateLinear` 落到 Tier 6：`F.linear`（cuBLAS，bf16 输出）→ `.to(fp32)`
        // （sm_120 不满足 `allow_specialized_router_gemm`，Tier 5 的 fp32 epilogue 不开）。
        // 所以必须走 cuBLASLt bf16 GEMM（`run_lt_default` 钉 `id=21 tile=11 splitk=8`）再 cast；
        // 原来的自写 `moe_gate_f32` + 舍入累加序不同，差 4/3072（D-16-REBASE §51）。
        if router_bf16_ab() {
            // **BI 口径**（`ROUTER_GEMM_BI=1`，默认）：vLLM 在 `VLLM_BATCH_INVARIANT=1` 下
            // `linear.py:209` 显式走 `linear_batch_invariant` → `matmul_persistent`（Triton），
            // **sm_120 也一样**（不装 dispatcher patch 但照样显式调用）。与 cuBLAS 在同形状上
            // 43% 元素不同 → 必须用同归约结构的 kernel（`docs/loops/kaon3-compile.md` 第 5 轮）。
            if router_gemm_bi() && h % 64 == 0 && e % 8 == 0 {
                {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| eprintln!("[kaon3] router GEMM = BI persistent matmul"));
                }
                // 诊断（`ROUTER_GEMM_DIAG=1`）：同一输入下再跑一遍 cuBLAS，比两者差异。
                if std::env::var("ROUTER_GEMM_DIAG").map(|v| v == "1").unwrap_or(false) {
                    let mut alt = Tensor::uninit(stream, &[tokens, e])?;
                    gemm.forward_bf16(&sc.routed.data, &self.router.bf16_weight()?.data, None, &mut alt.data, tokens, e, h)?;
                    let mut a32 = Tensor::uninit(stream, &[tokens, e])?;
                    k::ops::bf16_to_f32_n(stream, &alt.data, &mut a32.data, tokens * e)?;
                    let bi = k::ops::router_gemm_bi_f32(
                        stream, &sc.routed.data, &self.router.bf16_weight()?.data,
                        &mut sc.logits.data, tokens, e, h,
                    );
                    bi?;
                    let va = a32.to_vec(stream)?;
                    let vb = sc.logits.to_vec(stream)?;
                    let n = va.iter().zip(&vb).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                    let mx = va.iter().zip(&vb).map(|(x, y)| (*x - *y).abs()).fold(0f32, f32::max);
                    let md5 = |v: &[f32]| -> String {
                        let mut hsh: u64 = 0xcbf29ce484222325;
                        for x in v { for b in x.to_le_bytes() { hsh ^= b as u64; hsh = hsh.wrapping_mul(0x100000001b3); } }
                        format!("{hsh:016x}")
                    };
                    eprintln!("[kaon3] DIAG cuBLAS vs BI: ndiff={n}/{} max|d|={mx:.3e} md5={}/{}", tokens * e, md5(&va), md5(&vb));
                }
                k::ops::router_gemm_bi_f32(
                    stream,
                    &sc.routed.data,
                    &self.router.bf16_weight()?.data,
                    &mut sc.logits.data,
                    tokens,
                    e,
                    h,
                )?;
            } else {
                gemm.forward_bf16(&sc.routed.data, &self.router.bf16_weight()?.data, None, &mut sc.logits_bf16.data, tokens, e, h)?;
                k::ops::bf16_to_f32_n(stream, &sc.logits_bf16.data, &mut sc.logits.data, tokens * e)?;
            }
        } else {
            k::ops::moe_gate_f32(stream, &sc.routed.data, &self.router.bf16_weight()?.data, &mut sc.logits.data, tokens, e, h)?;
        }
        } // end !moe_iso（router 预处理+GEMM）
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
        if let Some(e) = &moe_pre_ev { stream.record_timing(&e[1])?; }  // router完
        // B10：grouped 路径 + tokens ≤ 64 → 路由前处理 7 次 launch 融合成 1 次（topk/权重/bucket/tile_off/row2e 一起）。
        // `MOE_ROUTE_FUSED=0` 回退旧链（A/B 用）。
        let route_fused = moe_grouped_enabled() && tokens <= 64 && moe_route_fused_enabled() && sc.grouped.is_some();
        if route_fused {
            let g = sc.grouped.as_mut().unwrap();
            k::ops::moe_route_bucket_small(
                stream, &sc.logits.data, pes_route, &mut sc.topk_ids.data, &mut sc.topk_w.data, &mut sc.buckets,
                &mut g.row2e, tokens, kk,
            )?;
        } else {
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
        }
        if let Some((dir, _)) = sub_out {
            let ids = stream.dtoh_sync_copy(&sc.topk_ids.data)?;
            let ws = stream.dtoh_sync_copy(&sc.topk_w.data)?;
            let ib: Vec<u8> = ids.iter().take(tokens * kk).flat_map(|x| x.to_le_bytes()).collect();
            let wb: Vec<u8> = ws.iter().take(tokens * kk).flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(dir.join("topk_ids.bin"), ib).ok();
            std::fs::write(dir.join("topk_w.bin"), wb).ok();
        }
        if !route_fused {
            k::ops::moe_bucket(stream, &sc.topk_ids.data, &mut sc.buckets, tokens, kk)?;
        }
        if let Some(e) = &moe_pre_ev { stream.record_timing(&e[2])?; }  // softmax+bucket完

        // 专家输入 = pre_feedforward_layernorm_2(residual)（router 用自己的无权重 norm，两者不同）；
        // 借 sc.h2 存，循环结束后才被 post_ff_norm_2 的输出覆盖
        rmsnorm_into(stream, ops, &sc.x, &self.pre_ff_norm_2, &mut sc.h2, h, tokens, eps)?;
        if moe_iso.is_some() {
            // 专家输入用 vLLM 真实 moe_hidden（覆盖我方 pre_ff2_norm），纯测专家 GEMM
            let hid = rdf_iso("vllm_moe_hidden.bin")?;
            ensure!(hid.len() >= tokens * h, "moe_iso hidden 太小");
            let hb: Vec<bf16> = hid[..tokens * h].iter().map(|&x| bf16::from_f32(x)).collect();
            stream.htod_into(&hb, &mut sc.h2.data)?;
        }
        sub_dump(stream, sub_out, "pre_ff2_norm", &sc.h2, tokens, h)?;
        stream.memset_zeros_n(&mut sc.acc.data, tokens * h)?;

        // === B3：device-grouped MoE（去 host 路由 + 去 D2H）。固定 launch 次数，offsets 留 device ===
        if moe_grouped_enabled() {
            let r = tokens * kk; // = offsets[E]，host 已知（每 token 选 kk 专家）
            let mi = cfg.moe_inter;
            let exact_rq = std::env::var("NQ4_EXACT_RCP").map(|v| v.trim() == "1").unwrap_or(false);
            let mg = match self.moe_grouped.get() {
                Some(m) => m,
                None => {
                    let m = MoeGrouped::build(stream, &self.experts)?;
                    let _ = self.moe_grouped.set(m);
                    self.moe_grouped.get().unwrap()
                }
            };
            let g = sc.grouped.as_mut().expect("MOE_GROUPED=1 但 GroupedBufs 未分配");
            // 1) row2e[R]：device offsets → 每行专家 id（E blocks，无 D2H）
            if !route_fused {
                k::ops::moe_row2expert(stream, &sc.buckets.offsets, &mut g.row2e, e, r)?;
            }
            // 2) gather 已融进量化（B11a：量化直接按 sorted_tok 读 h2 行，不再写 xg）
            // MoE 内部细分计时（KAON3_MOE_TIME=1）：量 3×grouped_gemm vs 其余。
            let moe_time = std::env::var("KAON3_MOE_TIME").ok().as_deref() == Some("1");
            let mt_ev = if moe_time { Some(stream.new_timing_events(3)?) } else { None };
            // 3) gate/up 共用一次量化（a_scale 相等），再两次 grouped GEMM
            k::ops::moe_grouped_quant(
                stream, &sc.h2.data, &g.row2e, Some(&sc.buckets.sorted_tok), &mg.agscale_gu, &mut g.aq_gu, &mut g.as_gu, r, mg.k_gate, exact_rq,
            )?;
            if let Some(ev) = &mt_ev { stream.record_timing(&ev[0])?; }  // 量化后=3gemm前
            if let Some(e) = &moe_pre_ev { stream.record_timing(&e[3])?; }  // norm+row2e+gather+quant完
            // MOE_GEMM_SKIP=gu|dn|all（纯计时，结果错）：只跳 grouped GEMM 本身，路由 / gather / quant / gelu / combine
            // 照跑，graph 态下看 GEMM 在 MoE 段里的真占比（loop kaon3-decode-opt D51）。
            let gemm_skip = std::env::var("MOE_GEMM_SKIP").unwrap_or_default();
            let (skip_gu, skip_dn) = (gemm_skip == "gu" || gemm_skip == "all", gemm_skip == "dn" || gemm_skip == "all");
            // 第二刀（MOE_GEMM_DUAL=1）：gate+up 融合单 launch（grid.z=2E 填满 SM）；逐位同分开两次。默认分开。
            // B7（loop kaon3-decode-opt）：decode 小 R 走带宽型 NVFP4 grouped GEMV（gate‖up+gelu 融合 / down），
            // 默认关（验证中），`MOE_DECODE_GEMV=1` 开；R ≤ 128（tokens ≤ 16）才走。
            let moe_gemv = moe_decode_gemv_enabled() && r <= 128 && !skip_gu && !skip_dn;
            if moe_gemv {
                k::ops::moe_fp4_gateup_act(
                    stream, &g.aq_gu, &g.as_gu, &mg.gate_packed, &mg.gate_scale, &mg.alpha_gate,
                    &mg.up_packed, &mg.up_scale, &mg.alpha_up, &sc.buckets.offsets, &mut g.gelu_o.data,
                    r, mg.n_gate, mg.k_gate, e,
                )?;
            } else if skip_gu {
            } else if std::env::var("MOE_GEMM_DUAL").ok().as_deref() == Some("1") {
                let (gate_o, up_o) = (&mut g.gate_o, &mut g.up_o);
                k::ops::moe_grouped_gemm_dual(
                    stream, &g.aq_gu, &g.as_gu,
                    &mg.gate_packed, &mg.gate_scale, &mg.alpha_gate, &mut gate_o.data,
                    &mg.up_packed, &mg.up_scale, &mg.alpha_up, &mut up_o.data,
                    &sc.buckets.offsets, &sc.buckets.tile_off, r, mg.n_gate, mg.k_gate, e,
                )?;
            } else {
                k::ops::moe_grouped_gemm(
                    stream, &g.aq_gu, &g.as_gu, &mg.gate_packed, &mg.gate_scale, &mg.alpha_gate, &sc.buckets.offsets, &sc.buckets.tile_off,
                    &mut g.gate_o.data, r, mg.n_gate, mg.k_gate, e,
                )?;
                k::ops::moe_grouped_gemm(
                    stream, &g.aq_gu, &g.as_gu, &mg.up_packed, &mg.up_scale, &mg.alpha_up, &sc.buckets.offsets, &sc.buckets.tile_off,
                    &mut g.up_o.data, r, mg.n_gate, mg.k_gate, e,
                )?;
            }
            // 4) gelu·mul（全 R 行一次）
            if !moe_gemv {
                gelu_mul(stream, ops, &g.gate_o.data, &g.up_o.data, &mut g.gelu_o.data, r, mi)?;
            }
            // 5) down：量化 + grouped GEMM
            k::ops::moe_grouped_quant(
                stream, &g.gelu_o.data, &g.row2e, None, &mg.agscale_dn, &mut g.aq_dn, &mut g.as_dn, r, mg.k_down, exact_rq,
            )?;
            if moe_gemv {
                k::ops::moe_fp4_gemv(
                    stream, &g.aq_dn, &g.as_dn, &mg.down_packed, &mg.down_scale, &mg.alpha_down, &sc.buckets.offsets,
                    &mut sc.yg.data, r, mg.n_down, mg.k_down, e,
                )?;
            } else if !skip_dn {
                k::ops::moe_grouped_gemm(
                    stream, &g.aq_dn, &g.as_dn, &mg.down_packed, &mg.down_scale, &mg.alpha_down, &sc.buckets.offsets, &sc.buckets.tile_off,
                    &mut sc.yg.data, r, mg.n_down, mg.k_down, e,
                )?;
            }
            if let Some(ev) = &mt_ev { stream.record_timing(&ev[1])?; }  // 3×gemm完
            // 6) scatter_add（全 R 行一次，start=0）——累加序与 host 版（专家升序）由 atomic 决定，逐位须实测
            // B11b：一次 combine 替代 memset(acc)+scatter_add(atomic)+memset(y)+f32_add_to_bf16（`MOE_COMBINE_OLD=1` 回退）。
            if moe_combine_fused_enabled() {
                k::ops::moe_combine_bf16(
                    stream, &sc.yg.data, &sc.topk_w.data, &sc.buckets.inv, None, &sc.topk_ids.data, 1.0, &mut sc.y.data,
                    tokens, kk, h, if moe_combine_f32() { 0 } else { 1 },
                )?;
            } else {
                if !moe_combine_f32() {
                    k::ops::moe_scatter_add_bf16w_f32(
                        stream, &sc.yg.data, &mut sc.acc.data, &sc.buckets.sorted_tok, &sc.buckets.sorted_k,
                        &sc.topk_w.data, None, 0, r, kk, h, 1.0,
                    )?;
                } else {
                    k::ops::moe_scatter_add_f32(
                        stream, &sc.yg.data, &mut sc.acc.data, &sc.buckets.sorted_tok, &sc.buckets.sorted_k,
                        &sc.topk_w.data, 0, r, kk, h,
                    )?;
                }
                stream.memset_zeros_n(&mut sc.y.data, tokens * h)?;
                k::ops::f32_add_to_bf16_n(stream, &sc.acc.data, &mut sc.y.data, tokens * h)?;
            }
            sub_dump(stream, sub_out, "moe_y", &sc.y, tokens, h)?;
            if let Some(ev) = &mt_ev { stream.record_timing(&ev[2])?; }  // scatter+合并完
            rmsnorm_into(stream, ops, &sc.y, &self.post_ff_norm_2, &mut sc.h2, h, tokens, eps)?;
            sub_dump(stream, sub_out, "ff2", &sc.h2, tokens, h)?;
            if let Some(ev) = &mt_ev {
                stream.synchronize()?;
                let ms = |a: usize, b: usize| k::TimingEvent::elapsed_ms(&ev[a], &ev[b]).unwrap_or(0.0);
                let pm = |a: usize, b: usize| {
                    moe_pre_ev.as_ref().map(|pe| k::TimingEvent::elapsed_ms(&pe[a], &pe[b]).unwrap_or(0.0)).unwrap_or(0.0)
                };
                tracing::info!(
                    "[moe-time] r={r} router={:.3} sm+bucket={:.3} norm+gather+quant={:.3} | gemm3={:.3} scatter={:.3} ms",
                    pm(0, 1), pm(1, 2), pm(2, 3), ms(0, 1), ms(1, 2)
                );
            }
            return Ok(());
        }

        // === host 路由版（默认 / A-B 基准）：D2H offsets + host 逐专家 launch ===
        let offsets = stream.dtoh_sync_copy(&sc.buckets.offsets)?;
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

/// router GEMM 是否走 **BI persistent matmul**（复刻参照）。**默认开**：vLLM 在 `VLLM_BATCH_INVARIANT=1`
/// 下 `linear.py:209` 显式调 `linear_batch_invariant`，与 cuBLAS 同形状 43% 元素不同
/// （`docs/loops/kaon3-compile.md` 第 5 轮）。`ROUTER_GEMM_BI=0` 退回 cuBLASLt，仅供 A/B。
fn router_gemm_bi() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("ROUTER_GEMM_BI").map(|v| v.trim() != "0").unwrap_or(true))
}

/// MoE 合并口径：**f32 权重 + f32 乘积 + per_expert_scale 折进 down**（默认，`=1`）；
/// `MOE_COMBINE_F32=0` 退回"权重先落 bf16、每项 bf16(w·y)"的旧假设口径。
///
/// **2026-10-24 改正**：原来默认是 `0`（当时判断为 vLLM cutlass_moe 口径）。p02 单例 A/B：
/// `=0` → top-1 翻转 **4/22**、top1 |Δlp| max 2.04；`=1` → **0/22**、max **0.52**。
/// 参照实际跑的是 Triton/inductor 融合路径，combine 不做那次 bf16 舍入。见 `docs/loops/kaon3-compile.md` 第 6 轮。
fn moe_combine_f32() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        // 默认 **false = vLLM 口径**（w=bf16(topk_w·pes)、每项 bf16(w·y)、bf16 累加，与 cutlass_moe
        // topk_weight_and_reduce 一致）。loop kaon3-moe 第 25-26 轮：combine_f32=0 时 MoE 与 vLLM 逐位
        // （p08 0/101376、p03 0/87296）；旧默认 true（f32 累加+pes折down）差 0.43%。
        // `MOE_COMBINE_F32=1` 显式退回旧口径（A/B）。
        let on = std::env::var("MOE_COMBINE_F32").map(|v| v.trim() == "1").unwrap_or(false);
        if on {
            eprintln!("[kaon3] MOE_COMBINE_F32=1：MoE 合并用 f32 累加+pes折down 旧口径（A/B 用，非 vLLM）");
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
