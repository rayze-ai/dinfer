//! 全模型共享的一套 per-token 激活缓冲（按 `max_tokens` 预分配一次，64 层轮流借用）。
//!
//! 为什么是一套而不是每层一套：层在同一条流上串行执行，层 i 的中间量在层 i+1 开始前已经消费完，
//! 不存在同时存活；每层各持一套时，预算 16384 下真 27B 合计 ≈ 180 GB（GDN 五块 ×10240/6144 列 +
//! Mlp 两块 ×34816/17408 列 × 64 层），把 KV 挤到只剩 ~20 GB，utilization 0.6 直接起不来（loop 5H）。
//! 共享后每 token ≈ 180 KB，16448 token ≈ 2.9 GB。跨流第二套（`ModelScratch.act`）再各开一份。
//!
//! 列数按两种层的最大值开：qkv/conv 同时给 GDN（conv_dim）与全注意力（(H+2Hkv)·D）用；
//! `z`/`o`/`n` 给 GDN（HV·V）用，全注意力借 `z` 做 gate、`o` 做 attention 输出（H·D）。
//! kernel 只按 `t` 行 × 本层列数读写前缀，多出的列是尾部死区，不影响正确性。
use crate::config::Qwen35Config;
use anyhow::Result;
use dinfer_layers::{bf16, Stream, Tensor};

pub struct ActScratch {
    /// GDN：in_proj_qkv 输出 `[t, conv_dim]`；全注意力：qkv GEMM 输出 `[t, (H+2Hkv)·D]`。
    pub qkv: Tensor<bf16>,
    /// GDN：causal conv 输出 `[t, conv_dim]`。
    pub conv: Tensor<bf16>,
    /// GDN：z 门 `[t, HV·V]`；全注意力：gate `[t, H·D]`。
    pub z: Tensor<bf16>,
    /// GDN：递推输出 `[t, HV·V]`；全注意力：attention 输出 `[t, H·D]`。
    pub o: Tensor<bf16>,
    /// GDN：gated RMSNorm 输出 `[t, HV·V]`。
    pub n: Tensor<bf16>,
    /// GDN：in_proj_a / in_proj_b 输出 `[t, HV]`。
    pub a: Tensor<bf16>,
    pub b: Tensor<bf16>,
    /// GDN：g / beta `[t, HV]` f32。
    pub g: Tensor<f32>,
    pub beta: Tensor<f32>,
    /// Mlp：gate|up GEMM 输出 `[t, 2I]`、silu·mul 输出 `[t, I]`。
    pub gate_up: Tensor<bf16>,
    pub act: Tensor<bf16>,
    pub max_tokens: usize,
}

impl ActScratch {
    /// 各缓冲的列数（bf16 列数 ×2 + f32 列数 ×4 = 每 token 字节）。
    fn cols(cfg: &Qwen35Config) -> (usize, usize, usize, usize) {
        let (h, hk, d) = (cfg.num_attention_heads as usize, cfg.num_key_value_heads as usize, cfg.head_dim as usize);
        let (lh, lhv) = (cfg.linear_num_key_heads as usize, cfg.linear_num_value_heads as usize);
        let (kd, vd) = (cfg.linear_key_head_dim as usize, cfg.linear_value_head_dim as usize);
        let conv_dim = 2 * lh * kd + lhv * vd;
        let qkv = conv_dim.max((h + 2 * hk) * d);
        let hvv = (lhv * vd).max(h * d);
        (qkv, hvv, lhv, cfg.intermediate_size as usize)
    }

    /// 一套的总字节数（分配前就能算，日志 / 预算用）。
    pub fn bytes(cfg: &Qwen35Config, max_tokens: usize) -> usize {
        let (qkv, hvv, hv, inter) = Self::cols(cfg);
        let bf16_cols = 2 * qkv + 3 * hvv + 2 * hv + 3 * inter;
        let f32_cols = 2 * hv;
        max_tokens * (bf16_cols * 2 + f32_cols * 4)
    }

    pub fn new(stream: &Stream, cfg: &Qwen35Config, max_tokens: usize) -> Result<Self> {
        let (qkv, hvv, hv, inter) = Self::cols(cfg);
        let t = max_tokens;
        Ok(Self {
            qkv: Tensor::zeros(stream, &[t, qkv])?,
            conv: Tensor::zeros(stream, &[t, qkv])?,
            z: Tensor::zeros(stream, &[t, hvv])?,
            o: Tensor::zeros(stream, &[t, hvv])?,
            n: Tensor::zeros(stream, &[t, hvv])?,
            a: Tensor::zeros(stream, &[t, hv])?,
            b: Tensor::zeros(stream, &[t, hv])?,
            g: Tensor::zeros(stream, &[t, hv])?,
            beta: Tensor::zeros(stream, &[t, hv])?,
            gate_up: Tensor::zeros(stream, &[t, 2 * inter])?,
            act: Tensor::zeros(stream, &[t, inter])?,
            max_tokens: t,
        })
    }
}
