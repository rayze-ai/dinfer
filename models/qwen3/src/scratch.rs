//! 全模型共享的一套 per-token 激活缓冲（按 `max_tokens` 预分配一次，28 层轮流借用）。
//!
//! 为什么是一套而不是每层一套：层在同一条流上串行执行，层 i 的中间量在层 i+1 开始前已经消费完，
//! 不存在同时存活。每层各持一套时（第四例：qwen35 / kimik3 已改，glm53 同构未改），预算 16384 下
//! 0.6B 合计 28 层 × 15360 列 × 2 B ≈ 14 GB 白占，把 KV 池挤到起不来（pitfalls 2026-10-20 两条例）。
//! 共享后每 token 30 KB，16384 token ≈ 0.5 GB；跨流第二套（`ModelScratch.act`）再开一份。
//!
//! 列宽规则（照 qwen35 `scratch.rs`）：按各层形状取最大列数开一块，kernel 只按 `t` 行 × 本层列数读写前缀，
//! 多出的列是尾部死区。Qwen3 各层同构，所以四块恰好等于单层宽度，没有死区；这里仍从 cfg 推、不写死数字，
//! 保留"将来合并列宽"的余地。
//!
//! **行跨度必须显式传本层行宽，不能从 `numel / 行数` 反推**：GEMM 按 `ld = n`（= 本层行宽）写，`qkv_post`
//! kernel 按 `(H·qmul+2Hkv)·D` 逐行原地读写，attention 收 `q_stride` / `kv_stride` 参数——三者必须都等于
//! 本层逻辑行宽。缓冲比本层行宽**宽**是安全的（只要求 `numel ≥ t × 行宽`），比它**窄**则是错的。
use crate::config::Qwen3Config;
use anyhow::Result;
use dinfer_layers::{bf16, Stream, Tensor};

pub struct ActScratch {
    /// qkv GEMM 输出 `[t, (H+2Hkv)·D]`（q | k | v；q/k 经 `qkv_post` 原地 norm + RoPE）。
    pub qkv: Tensor<bf16>,
    /// attention 输出 `[t, H·D]`（o_proj 的输入）。
    pub attn: Tensor<bf16>,
    /// Mlp：gate|up GEMM 输出 `[t, 2I]`。
    pub gate_up: Tensor<bf16>,
    /// Mlp：silu·mul 输出 `[t, I]`。
    pub act: Tensor<bf16>,
    pub max_tokens: usize,
}

impl ActScratch {
    /// 各缓冲的列数（本 rank 的 cfg：TP 后头数 / inter 已除）。
    /// 返回 (qkv 行宽, attn 行宽, gate_up 行宽, act 行宽)，顺序同 `bytes` 的求和顺序。
    fn cols(cfg: &Qwen3Config) -> (usize, usize, usize, usize) {
        let (h, hk, d) = (cfg.num_attention_heads as usize, cfg.num_key_value_heads as usize, cfg.head_dim as usize);
        let inter = cfg.intermediate_size as usize;
        ((h + 2 * hk) * d, h * d, 2 * inter, inter)
    }

    /// 一套的总字节数（分配前就能算，日志 / 预算用）。
    pub fn bytes(cfg: &Qwen3Config, max_tokens: usize) -> usize {
        let (qkv, attn, gate_up, act) = Self::cols(cfg);
        max_tokens * (qkv + attn + gate_up + act) * std::mem::size_of::<bf16>()
    }

    pub fn new(stream: &Stream, cfg: &Qwen3Config, max_tokens: usize) -> Result<Self> {
        let (qkv, attn, gate_up, act) = Self::cols(cfg);
        let t = max_tokens;
        Ok(Self {
            qkv: Tensor::zeros(stream, &[t, qkv])?,
            attn: Tensor::zeros(stream, &[t, attn])?,
            gate_up: Tensor::zeros(stream, &[t, gate_up])?,
            act: Tensor::zeros(stream, &[t, act])?,
            max_tokens: t,
        })
    }
}
