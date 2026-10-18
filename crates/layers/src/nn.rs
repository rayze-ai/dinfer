//! 基础层：Linear / RmsNorm / Rope / Embedding / Mlp。全部 bf16，fp32 累加。

use crate::tensor::Tensor;
use crate::weights::WeightSource;
use anyhow::{ensure, Result};
use dinfer_kernels::{self as k, Gemm, RopeStyle, Stream};
use half::bf16;

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
    pub style: RopeStyle,
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
        ensure!(rope_dim % 2 == 0, "rope_dim must be even");
        let half = rope_dim / 2;
        // 注：HF 在 bf16 模型里把 cos/sin cast 到 bf16 再逐项乘加；这里存 fp32 表、fp32 旋转。
        // 实测对 logits 的影响 ≤ 1 ULP（P3/P4/P5 对拍均用此表），不改。
        // 40960 × 128 的 cos/sin 在 host 单线程要 ~120 ms → 按位置分片多线程算（启动 loop 第 6 轮）。
        let inv_freq: Vec<f64> = (0..half).map(|i| 1.0 / theta.powf(2.0 * i as f64 / rope_dim as f64)).collect();
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
        Ok(Self { cos_sin: Tensor::from_vec(stream, table, &[max_pos, rope_dim])?, rope_dim, style })
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
        k::rope_bf16(
            stream,
            &mut q.data,
            k.map(|t| &mut t.data),
            &positions.data,
            &self.cos_sin.data,
            q_heads,
            k_heads,
            head_dim,
            self.rope_dim,
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
    gate_up_buf: Tensor<bf16>,
    act_buf: Tensor<bf16>,
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
            gate_up_buf: Tensor::zeros(stream, &[max_tokens, 2 * inter])?,
            act_buf: Tensor::zeros(stream, &[max_tokens, inter])?,
        })
    }
    /// `tokens`：本步实际 token 数；x / out 缓冲只用前 tokens 行。
    /// 跨流并行：把本层的两块激活缓冲与外部一套互换（副流用自己的 scratch）。形状必须一致。
    pub fn swap_bufs(&mut self, gate_up: &mut Tensor<bf16>, act: &mut Tensor<bf16>) {
        std::mem::swap(&mut self.gate_up_buf, gate_up);
        std::mem::swap(&mut self.act_buf, act);
    }
    pub fn forward(
        &mut self,
        stream: &Stream,
        gemm: &Gemm,
        act: &impl crate::Act,
        tokens: usize,
        x: &Tensor<bf16>,
        out: &mut Tensor<bf16>,
    ) -> Result<()> {
        let hidden = self.gate_up.in_features;
        ensure!(x.numel() >= tokens * hidden && out.numel() >= tokens * hidden, "Mlp: buffer too small");
        ensure!(self.gate_up_buf.numel() >= tokens * 2 * self.inter, "Mlp: tokens > max_tokens");
        gemm.forward_bf16(
            &x.data,
            &self.gate_up.w.data,
            None,
            &mut self.gate_up_buf.data,
            tokens,
            2 * self.inter,
            hidden,
        )?;
        act.silu_and_mul(stream, &self.gate_up_buf.data, &mut self.act_buf.data, tokens, self.inter)?;
        gemm.forward_bf16(
            &self.act_buf.data,
            &self.down.w.data,
            None,
            &mut out.data,
            tokens,
            hidden,
            self.inter,
        )
    }
}
