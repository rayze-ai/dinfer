//! 基础层：Linear / RmsNorm / Rope / Embedding / Mlp。全部 bf16，fp32 累加。

use crate::tensor::Tensor;
use crate::weights::WeightSource;
use anyhow::{ensure, Result};
use dinfer_kernels::{self as k, Gemm, RopeStyle, Stream};
use half::bf16;

/// 把 f32 按 bf16 精度就近偶数舍入后还原成 f32（复现 `torch.to(bfloat16).float()`）。
#[inline]
fn round_to_bf16(x: f32) -> f32 {
    f32::from(bf16::from_f32(x))
}

/// `y = x · W^T (+ b)`，W: [out, in]（PyTorch 布局）。
pub struct Linear {
    pub w: Tensor<bf16>,
    pub b: Option<Tensor<bf16>>,
    pub out_features: usize,
    pub in_features: usize,
}

impl Linear {
    pub fn new(w: Tensor<bf16>, b: Option<Tensor<bf16>>) -> Result<Self> {
        ensure!(w.shape.len() == 2, "Linear weight must be 2-D, got {:?}", w.shape);
        let (out_features, in_features) = (w.shape[0], w.shape[1]);
        if let Some(b) = &b {
            ensure!(b.numel() == out_features, "bias shape");
        }
        Ok(Self { w, b, out_features, in_features })
    }
    pub fn load(stream: &Stream, src: &WeightSource, prefix: &str) -> Result<Self> {
        let w = src.load_bf16(stream, &format!("{prefix}.weight"))?;
        let b = if src.contains(&format!("{prefix}.bias")) {
            Some(src.load_bf16(stream, &format!("{prefix}.bias"))?)
        } else {
            None
        };
        Self::new(w, b)
    }
    /// TP 分片装载：`dim` 0 = 按输出行切（列并行，如 qkv / gate_up），1 = 按输入列切（行并行，如 o_proj / down）。
    /// bias 只在 dim 0 时切；dim 1 的 bias 需在 all-reduce 后只加一次，这里不支持（Qwen3 / GLM 无 bias）。
    pub fn load_shard(stream: &Stream, src: &WeightSource, prefix: &str, dim: usize, rank: usize, world: usize) -> Result<Self> {
        let w = src.load_bf16_shard(stream, &format!("{prefix}.weight"), dim, rank, world)?;
        let bname = format!("{prefix}.bias");
        let b = if src.contains(&bname) {
            ensure!(dim == 0 || world == 1, "{prefix}: bias with input-dim shard unsupported");
            Some(src.load_bf16_shard(stream, &bname, 0, rank, world)?)
        } else {
            None
        };
        Self::new(w, b)
    }
    /// x: [tokens, in] → out: [tokens, out]
    pub fn forward(&self, gemm: &Gemm, x: &Tensor<bf16>, out: &mut Tensor<bf16>) -> Result<()> {
        let tokens = x.numel() / self.in_features;
        ensure!(x.numel() == tokens * self.in_features, "Linear: x shape {:?}", x.shape);
        ensure!(out.numel() == tokens * self.out_features, "Linear: out shape {:?}", out.shape);
        self.forward_rows(gemm, x, out, tokens)
    }

    /// 只算前 `tokens` 行：`x` / `out` 允许是按 max_tokens 预分配的更大缓冲（decode 时 t ≪ max_tokens）。
    pub fn forward_rows(&self, gemm: &Gemm, x: &Tensor<bf16>, out: &mut Tensor<bf16>, tokens: usize) -> Result<()> {
        ensure!(x.numel() >= tokens * self.in_features, "Linear: x {:?} < {tokens} 行", x.shape);
        ensure!(out.numel() >= tokens * self.out_features, "Linear: out {:?} < {tokens} 行", out.shape);
        gemm.forward_bf16(
            &x.data,
            &self.w.data,
            self.b.as_ref().map(|b| &b.data),
            &mut out.data,
            tokens,
            self.out_features,
            self.in_features,
        )
    }
}

pub struct RmsNorm {
    pub w: Tensor<bf16>,
    pub eps: f32,
    pub hidden: usize,
}

impl RmsNorm {
    pub fn load(stream: &Stream, src: &WeightSource, name: &str, eps: f32) -> Result<Self> {
        let w = src.load_bf16(stream, &format!("{name}.weight"))?;
        let hidden = w.numel();
        Ok(Self { w, eps, hidden })
    }
    pub fn forward(&self, stream: &Stream, x: &Tensor<bf16>, out: &mut Tensor<bf16>) -> Result<()> {
        k::rmsnorm_bf16(stream, &x.data, &self.w.data, &mut out.data, self.hidden, self.eps)
    }
    /// res += x; out = norm(res)
    pub fn forward_fused_add(
        &self,
        stream: &Stream,
        x: &Tensor<bf16>,
        res: &mut Tensor<bf16>,
        out: &mut Tensor<bf16>,
    ) -> Result<()> {
        k::fused_add_rmsnorm_bf16(
            stream,
            &x.data,
            &mut res.data,
            &self.w.data,
            &mut out.data,
            self.hidden,
            self.eps,
        )
    }
}

/// RoPE 表：[max_pos, rope_dim] fp32，前半 cos 后半 sin。
pub struct Rope {
    pub cos_sin: Tensor<f32>,
    pub rope_dim: usize,
    /// neox 配对跨度：`(i, i + span)`。默认 `rope_dim/2`；见 [`Rope::with_span`]。
    pub span: usize,
    pub style: RopeStyle,
    /// 表是否已按 bf16 精度舍入（复现 vLLM `cos_sin_cache.to(dtype)`）。只影响解释，不影响存储类型。
    pub bf16_cache: bool,
}

impl Rope {
    /// `theta` 为 rope_theta（GLM-5.3 8e6）。与 HF `_compute_default_rope_parameters` 一致：
    /// inv_freq[i] = theta^(-2i/d)，freqs = pos · inv_freq。
    pub fn new(
        stream: &Stream,
        rope_dim: usize,
        max_pos: usize,
        theta: f64,
        style: RopeStyle,
    ) -> Result<Self> {
        Self::with_denom(stream, rope_dim, rope_dim, max_pos, theta, style)
    }
    /// `rope_dim` 维被旋转，但 inv_freq 用 `denom` 做分母。
    ///
    /// Gemma4 的 `rope_type=proportional`（`full_attention` 层）是这种情况：只旋转
    /// `head_dim * partial_rotary_factor` = 128 维，而频率分母是 **head_dim** = 512
    /// （vLLM `rotary_embedding/gemma4_rope.py`，HF `_compute_proportional_rope_parameters`）。
    /// `denom == rope_dim` 时与 `new` 完全等价。
    pub fn with_denom(
        stream: &Stream,
        rope_dim: usize,
        denom: usize,
        max_pos: usize,
        theta: f64,
        style: RopeStyle,
    ) -> Result<Self> {
        ensure!(rope_dim % 2 == 0, "rope_dim must be even");
        ensure!(denom >= rope_dim && denom % 2 == 0, "rope denom {denom} 必须 >= rope_dim 且为偶数");
        let half = rope_dim / 2;
        // 默认存 fp32 表、fp32 旋转。**注意**：vLLM 在 `RotaryEmbeddingBase.__init__` 里
        // `cache = cache.to(dtype)`，dtype 即模型 dtype（bf16），所以它用的 cos/sin **是 bf16**；
        // 需要逐位对拍的模型（如 Kaon-V3）用 `with_bf16_cache()` 复现这一点。
        // 40960 × 128 的 cos/sin 在 host 单线程要 ~120 ms → 按位置分片多线程算（启动 loop 第 6 轮）。
        let inv_freq: Vec<f64> =
            (0..half).map(|i| 1.0 / theta.powf(2.0 * i as f64 / denom as f64)).collect();
        let mut table = vec![0f32; max_pos * rope_dim];
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 16);
        let rows_per = max_pos.div_ceil(threads).max(1);
        std::thread::scope(|sc| {
            for (chunk_idx, chunk) in table.chunks_mut(rows_per * rope_dim).enumerate() {
                let inv_freq = &inv_freq;
                sc.spawn(move || {
                    let p0 = chunk_idx * rows_per;
                    for (r, row) in chunk.chunks_mut(rope_dim).enumerate() {
                        let p = (p0 + r) as f64;
                        for i in 0..half {
                            let f = p * inv_freq[i];
                            row[i] = f.cos() as f32;
                            row[half + i] = f.sin() as f32;
                        }
                    }
                });
            }
        });
        Ok(Self {
            cos_sin: Tensor::from_vec(stream, table, &[max_pos, rope_dim])?,
            rope_dim,
            span: rope_dim / 2,
            style,
            bf16_cache: false,
        })
    }
    /// 把 cos/sin 表按 **bf16 精度**舍入（仍是 f32 存储，kernel 不变）。
    ///
    /// vLLM `RotaryEmbeddingBase.__init__` 做完 `_compute_cos_sin_cache()` 后立刻
    /// `cache = cache.to(dtype)`，`dtype` 是模型 dtype（bf16）；`forward_cuda` 里
    /// `_match_cos_sin_cache_dtype(query)` 再对齐一次。CUDA kernel 把它 `static_cast<float>`
    /// 后参与 fp32 乘加。所以"bf16 表 + fp32 旋转"才是 vLLM 的实际数值路径，
    /// 用 f32 表会带来 ~2e-3 的相对差（bf16 尾数 8 位）。Kaon-V3 对拍实测正是这个量级。
    pub fn with_bf16_cache(mut self, stream: &Stream) -> Result<Self> {
        if self.bf16_cache {
            return Ok(self);
        }
        let v = self.cos_sin.to_vec(stream)?;
        let r: Vec<f32> = v.into_iter().map(round_to_bf16).collect();
        self.cos_sin = Tensor::from_vec(stream, r, &self.cos_sin.shape)?;
        self.bf16_cache = true;
        Ok(self)
    }
    /// 改 neox 配对跨度。Gemma4 proportional RoPE 在 vLLM / HF 里是"inv_freq 零频补齐到 head_dim/2 对、
    /// 整头 neox 旋转"，非零频只有前 `rope_dim/2` 对，配对是 `(i, i + head_dim/2)` —— 不是只转前
    /// `rope_dim` 维的 `(i, i + rope_dim/2)`。等价实现：表仍只存 `rope_dim` 个频率，`span = head_dim/2`。
    pub fn with_span(mut self, span: usize) -> Result<Self> {
        ensure!(span >= self.rope_dim / 2, "rope span {span} < rope_dim/2 {}", self.rope_dim / 2);
        self.span = span;
        Ok(self)
    }
    /// 就地旋转 q[tokens, q_heads, head_dim] 与可选 k[tokens, k_heads, head_dim] 的前 rope_dim 维。
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &self,
        stream: &Stream,
        q: &mut Tensor<bf16>,
        k: Option<&mut Tensor<bf16>>,
        positions: &Tensor<i32>,
        q_heads: usize,
        k_heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        k::rope_span_bf16(
            stream,
            &mut q.data,
            k.map(|t| &mut t.data),
            &positions.data,
            &self.cos_sin.data,
            q_heads,
            k_heads,
            head_dim,
            self.rope_dim,
            self.span,
            self.style,
        )
    }
}

pub struct Embedding {
    pub table: Tensor<bf16>,
    pub vocab: usize,
    pub hidden: usize,
}

impl Embedding {
    pub fn load(stream: &Stream, src: &WeightSource, name: &str) -> Result<Self> {
        let table = src.load_bf16(stream, &format!("{name}.weight"))?;
        ensure!(table.shape.len() == 2, "embedding shape");
        let (vocab, hidden) = (table.shape[0], table.shape[1]);
        Ok(Self { table, vocab, hidden })
    }
    pub fn forward(&self, stream: &Stream, ids: &Tensor<i32>, out: &mut Tensor<bf16>) -> Result<()> {
        k::embedding_bf16(stream, &ids.data, &self.table.data, &mut out.data, self.hidden)
    }
}

/// gate/up 合并为一个 [2I, H] 的 GEMM，再 silu_and_mul，再 down。
pub struct Mlp {
    pub gate_up: Linear,
    pub down: Linear,
    pub inter: usize,
    /// 自带的两块激活缓冲（`[max_tokens, 2I]` / `[max_tokens, I]`）。`load*` 的 `max_tokens == 0` 时不分配，
    /// 调用方必须走 `forward_with_scratch` 借全模型共享的一套（每层各持一套在大预算下会爆显存，见 pitfalls）。
    gate_up_buf: Option<Tensor<bf16>>,
    act_buf: Option<Tensor<bf16>>,
}

impl Mlp {
    /// 从 HF 的 gate_proj / up_proj / down_proj 装载并把 gate|up 纵向拼接。
    pub fn load(stream: &Stream, src: &WeightSource, prefix: &str, max_tokens: usize) -> Result<Self> {
        Self::load_tp(stream, src, prefix, max_tokens, 0, 1)
    }
    /// TP：gate/up 按 inter 行切（列并行），down 按 inter 列切（行并行）；down 的输出是部分和，调用方 all-reduce。
    pub fn load_tp(stream: &Stream, src: &WeightSource, prefix: &str, max_tokens: usize, rank: usize, world: usize) -> Result<Self> {
        // gate / up 以 bf16 直接上传，再在设备上拼成 [2·inter, hidden]（原先 f32 host 往返，28 层 ~1.2 s 启动时间）
        let g = src.load_bf16_shard(stream, &format!("{prefix}.gate_proj.weight"), 0, rank, world)?;
        let u = src.load_bf16_shard(stream, &format!("{prefix}.up_proj.weight"), 0, rank, world)?;
        ensure!(g.shape == u.shape && g.shape.len() == 2, "gate/up shape mismatch {:?} {:?}", g.shape, u.shape);
        let (inter, hidden) = (g.shape[0], g.shape[1]);
        let mut cat = Tensor::<bf16>::zeros(stream, &[2 * inter, hidden])?;
        {
            let n = g.numel();
            let mut top = cat.data.slice_mut(..n);
            stream.inner_arc().memcpy_dtod(&g.data, &mut top)?;
            let mut bottom = cat.data.slice_mut(n..2 * n);
            stream.inner_arc().memcpy_dtod(&u.data, &mut bottom)?;
        }
        let gate_up = Linear::new(cat, None)?;
        let down = Linear::load_shard(stream, src, &format!("{prefix}.down_proj"), 1, rank, world)?;
        Ok(Self {
            gate_up,
            down,
            inter,
            gate_up_buf: if max_tokens > 0 { Some(Tensor::zeros(stream, &[max_tokens, 2 * inter])?) } else { None },
            act_buf: if max_tokens > 0 { Some(Tensor::zeros(stream, &[max_tokens, inter])?) } else { None },
        })
    }
    /// `tokens`：本步实际 token 数；x / out 缓冲只用前 tokens 行。
    /// 跨流并行：把本层的两块激活缓冲与外部一套互换（副流用自己的 scratch）。形状必须一致；只对自带缓冲的 Mlp 有效。
    pub fn swap_bufs(&mut self, gate_up: &mut Tensor<bf16>, act: &mut Tensor<bf16>) {
        std::mem::swap(self.gate_up_buf.as_mut().expect("Mlp::swap_bufs: Mlp 未自带缓冲"), gate_up);
        std::mem::swap(self.act_buf.as_mut().expect("Mlp::swap_bufs: Mlp 未自带缓冲"), act);
    }
    /// 用自带缓冲算（`load*` 时 `max_tokens > 0`）。
    pub fn forward(
        &mut self,
        stream: &Stream,
        gemm: &Gemm,
        act: &impl crate::Act,
        tokens: usize,
        x: &Tensor<bf16>,
        out: &mut Tensor<bf16>,
    ) -> Result<()> {
        let (Some(gu), Some(ab)) = (self.gate_up_buf.as_mut(), self.act_buf.as_mut()) else {
            anyhow::bail!("Mlp::forward: 未自带激活缓冲（max_tokens = 0），请用 forward_with_scratch");
        };
        mlp_forward(&self.gate_up, &self.down, self.inter, stream, gemm, act, tokens, x, out, gu, ab)
    }
    /// 用调用方借来的缓冲算：`gate_up` ≥ `[tokens, 2I]`，`act_buf` ≥ `[tokens, I]`（全模型共享一套时用）。
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_scratch(
        &self,
        stream: &Stream,
        gemm: &Gemm,
        act: &impl crate::Act,
        tokens: usize,
        x: &Tensor<bf16>,
        out: &mut Tensor<bf16>,
        gate_up: &mut Tensor<bf16>,
        act_buf: &mut Tensor<bf16>,
    ) -> Result<()> {
        mlp_forward(&self.gate_up, &self.down, self.inter, stream, gemm, act, tokens, x, out, gate_up, act_buf)
    }
}

#[allow(clippy::too_many_arguments)]
fn mlp_forward(
    gate_up: &Linear,
    down: &Linear,
    inter: usize,
    stream: &Stream,
    gemm: &Gemm,
    act: &impl crate::Act,
    tokens: usize,
    x: &Tensor<bf16>,
    out: &mut Tensor<bf16>,
    gate_up_buf: &mut Tensor<bf16>,
    act_buf: &mut Tensor<bf16>,
) -> Result<()> {
    let hidden = gate_up.in_features;
    ensure!(x.numel() >= tokens * hidden && out.numel() >= tokens * hidden, "Mlp: buffer too small");
    ensure!(gate_up_buf.numel() >= tokens * 2 * inter && act_buf.numel() >= tokens * inter, "Mlp: tokens > max_tokens");
    // `MLP_PROFILE=1`：拆 gate_up GEMM / silu_and_mul / down GEMM 三段（GPU 事件）。
    let mp = std::env::var("MLP_PROFILE").ok().as_deref() == Some("1") && tokens >= 256;
    let mev = if mp { Some(stream.new_timing_events(4)?) } else { None };
    if let Some(e) = &mev {
        stream.record_timing(&e[0])?;
    }
    gemm.forward_bf16(&x.data, &gate_up.w.data, None, &mut gate_up_buf.data, tokens, 2 * inter, hidden)?;
    if let Some(e) = &mev {
        stream.record_timing(&e[1])?;
    }
    act.silu_and_mul(stream, &gate_up_buf.data, &mut act_buf.data, tokens, inter)?;
    if let Some(e) = &mev {
        stream.record_timing(&e[2])?;
    }
    gemm.forward_bf16(&act_buf.data, &down.w.data, None, &mut out.data, tokens, hidden, inter)?;
    if let Some(e) = &mev {
        stream.record_timing(&e[3])?;
        stream.synchronize()?;
        let ms = |a: usize, b: usize| k::TimingEvent::elapsed_ms(&e[a], &e[b]).unwrap_or(0.0);
        eprintln!("[mlp-prof] t={tokens} gate_up={:.3} silu={:.3} down={:.3} total={:.3} ms",
            ms(0,1), ms(1,2), ms(2,3), ms(0,3));
    }
    Ok(())
}
