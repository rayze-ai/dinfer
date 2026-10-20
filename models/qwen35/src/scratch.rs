//! 全模型共享的一套 per-token 激活缓冲（按 `max_tokens` 预分配一次，64 层轮流借用）。
//!
//! 为什么是一套而不是每层一套：层在同一条流上串行执行，层 i 的中间量在层 i+1 开始前已经消费完，
//! 不存在同时存活；每层各持一套时，预算 16384 下真 27B 合计 ≈ 180 GB（GDN 五块 ×10240/6144 列 +
//! Mlp 两块 ×34816/17408 列 × 64 层），把 KV 挤到只剩 ~20 GB，utilization 0.6 直接起不来（loop 5H）。
//! 共享后每 token ≈ 180 KB，16448 token ≈ 2.9 GB。跨流第二套（`ModelScratch.act`）再各开一份。
//!
//! 列数按两种层的最大值开：qkv 同时给 GDN（融合 in_proj 行宽）与全注意力（(H·qmul+2Hkv)·D）用，conv 只给 GDN（conv_dim）；
//! `o`/`n` 给 GDN（HV·V）用，全注意力借 `o` 做 attention 输出（H·D）；z / a / b 不再单独开（在融合 in_proj 行内）。
//! kernel 只按 `t` 行 × 本层列数读写前缀，多出的列是尾部死区，不影响正确性。
use crate::config::Qwen35Config;
use anyhow::Result;
use dinfer_layers::{bf16, Stream, Tensor};

pub struct ActScratch {
    /// GDN：融合 in_proj 输出 `[t, conv_dim + HV·V + 2HV]`（qkv | z | a | b）；全注意力：qkv GEMM 输出 `[t, (H+2Hkv)·D]`。
    pub qkv: Tensor<bf16>,
    /// GDN：causal conv 输出 `[t, conv_dim]`。
    pub conv: Tensor<bf16>,
    /// GDN：递推输出 `[t, HV·V]`；全注意力：attention 输出 `[t, H·D]`。
    pub o: Tensor<bf16>,
    /// GDN：gated RMSNorm 输出 `[t, HV·V]`。
    pub n: Tensor<bf16>,
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
    /// 返回 (qkv 行宽, conv_dim, HV·V / H·D 的大者, HV, inter)。
    fn cols(cfg: &Qwen35Config) -> (usize, usize, usize, usize, usize) {
        let (h, hk, d) = (cfg.num_attention_heads as usize, cfg.num_key_value_heads as usize, cfg.head_dim as usize);
        let (lh, lhv) = (cfg.linear_num_key_heads as usize, cfg.linear_num_value_heads as usize);
        let (kd, vd) = (cfg.linear_key_head_dim as usize, cfg.linear_value_head_dim as usize);
        let conv_dim = 2 * lh * kd + lhv * vd;
        // GDN 融合 in_proj 行宽 vs 全注意力 qkv 行宽，取大
        let qmul = if cfg.attn_output_gate { 2 } else { 1 };
        let qkv = (conv_dim + lhv * vd + 2 * lhv).max((h * qmul + 2 * hk) * d);
        let hvv = (lhv * vd).max(h * d);
        (qkv, conv_dim, hvv, lhv, cfg.intermediate_size as usize)
    }

    /// 一套的总字节数（分配前就能算，日志 / 预算用）。
    pub fn bytes(cfg: &Qwen35Config, max_tokens: usize) -> usize {
        let (qkv, conv_dim, hvv, hv, inter) = Self::cols(cfg);
        let bf16_cols = qkv + conv_dim + 2 * hvv + 3 * inter;
        let f32_cols = 2 * hv;
        max_tokens * (bf16_cols * 2 + f32_cols * 4)
    }

    pub fn new(stream: &Stream, cfg: &Qwen35Config, max_tokens: usize) -> Result<Self> {
        let (qkv, conv_dim, hvv, hv, inter) = Self::cols(cfg);
        let t = max_tokens;
        Ok(Self {
            qkv: Tensor::zeros(stream, &[t, qkv])?,
            conv: Tensor::zeros(stream, &[t, conv_dim])?,
            o: Tensor::zeros(stream, &[t, hvv])?,
            n: Tensor::zeros(stream, &[t, hvv])?,
            g: Tensor::zeros(stream, &[t, hv])?,
            beta: Tensor::zeros(stream, &[t, hv])?,
            gate_up: Tensor::zeros(stream, &[t, 2 * inter])?,
            act: Tensor::zeros(stream, &[t, inter])?,
            max_tokens: t,
        })
    }
}
