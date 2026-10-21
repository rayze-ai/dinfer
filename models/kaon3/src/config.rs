//! Kaon-V3（Gemma4 MoE）`config.json`：顶层是多模态壳，只读 `text_config`。
//!
//! 结构事实全部来自真权重 config + `docs/loops/kaon-v3.md`「Gemma4 前向语义」：
//! - 30 层，`layer_types` 里 5 层 `full_attention`（下标 5/11/17/23/29）+ 25 层 `sliding_attention`（窗口 1024）
//! - full：`global_head_dim=512`、`num_global_key_value_heads=2`、proportional RoPE（theta 1e6，旋转 128 维但分母是 512）
//! - sliding：`head_dim=256`、`num_key_value_heads=8`、default RoPE（theta 1e4，全维旋转）
//! - `attention_k_eq_v=true`：**只有 full 层** K 权重兼作 V（`docs/loops/kaon-v3.md` 实测：sliding 层有独立 v_proj）
//! - MoE 与 dense MLP **并联**，不是二选一；`layer_scalar` 每层一个标量

use anyhow::{ensure, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    FullAttention,
    SlidingAttention,
}

/// 一种层类型对应的 RoPE 参数。
#[derive(Debug, Clone)]
pub struct RopeSpec {
    /// 实际被旋转的维数（`head_dim * partial_rotary_factor`）。
    pub rope_dim: usize,
    /// inv_freq 的分母（proportional 时 = head_dim，default 时 = rope_dim）。
    pub denom: usize,
    /// neox 配对跨度（proportional 时 = head_dim/2，default 时 = rope_dim/2），见 `Rope::with_span`。
    pub span: usize,
    pub theta: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct RopeEntry {
    rope_theta: f64,
    #[serde(default = "one")]
    partial_rotary_factor: f64,
    #[serde(default = "default_kind")]
    rope_type: String,
}

fn one() -> f64 {
    1.0
}
fn default_kind() -> String {
    "default".into()
}

#[derive(Debug, Clone, Deserialize)]
struct RawConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    global_head_dim: usize,
    num_global_key_value_heads: usize,
    moe_intermediate_size: usize,
    num_experts: usize,
    top_k_experts: usize,
    sliding_window: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    layer_types: Vec<LayerType>,
    rope_parameters: std::collections::HashMap<String, RopeEntry>,
    #[serde(default = "gelu")]
    hidden_activation: String,
    #[serde(default = "softcap")]
    final_logit_softcapping: f32,
    #[serde(default)]
    attention_k_eq_v: bool,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    hidden_size_per_layer_input: usize,
    #[serde(default)]
    num_kv_shared_layers: usize,
}

fn gelu() -> String {
    "gelu_pytorch_tanh".into()
}
fn softcap() -> f32 {
    30.0
}

/// 解析后的文本模型配置（只保留我们实际用到的字段）。
#[derive(Debug, Clone)]
pub struct KaonConfig {
    pub vocab_size: usize,
    pub hidden: usize,
    pub dense_inter: usize,
    pub moe_inter: usize,
    pub num_layers: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub layer_types: Vec<LayerType>,
    pub num_heads: usize,
    /// sliding 层的 kv 头数。
    pub nkv_sliding: usize,
    /// full 层的 kv 头数。
    pub nkv_full: usize,
    pub head_dim_sliding: usize,
    pub head_dim_full: usize,
    pub sliding_window: usize,
    pub max_pos: usize,
    pub eps: f32,
    pub softcap: f32,
    pub k_eq_v: bool,
    pub tie_embeddings: bool,
    pub rope_sliding: RopeSpec,
    pub rope_full: RopeSpec,
}

impl KaonConfig {
    /// 顶层壳（有 `text_config`）或裸 text_config 都接受。
    pub fn from_json(s: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(s)?;
        let text = v.get("text_config").cloned().unwrap_or(v);
        let raw: RawConfig = serde_json::from_value(text)?;
        Self::from_raw(raw)
    }

    fn from_raw(r: RawConfig) -> Result<Self> {
        ensure!(
            r.hidden_activation == "gelu_pytorch_tanh",
            "unsupported hidden_activation {}",
            r.hidden_activation
        );
        ensure!(r.hidden_size_per_layer_input == 0, "PLE (per-layer input) unsupported");
        ensure!(r.num_kv_shared_layers == 0, "kv sharing unsupported");
        ensure!(r.layer_types.len() == r.num_hidden_layers, "layer_types 长度不符");
        ensure!(
            r.num_attention_heads % r.num_key_value_heads == 0
                && r.num_attention_heads % r.num_global_key_value_heads == 0,
            "GQA 头数不整除"
        );
        ensure!(r.top_k_experts > 0 && r.top_k_experts <= r.num_experts, "top_k_experts 越界");

        let rope_sliding = rope_spec(&r, "sliding_attention", r.head_dim)?;
        let rope_full = rope_spec(&r, "full_attention", r.global_head_dim)?;
        Ok(Self {
            vocab_size: r.vocab_size,
            hidden: r.hidden_size,
            dense_inter: r.intermediate_size,
            moe_inter: r.moe_intermediate_size,
            num_layers: r.num_hidden_layers,
            num_experts: r.num_experts,
            top_k: r.top_k_experts,
            layer_types: r.layer_types.clone(),
            num_heads: r.num_attention_heads,
            nkv_sliding: r.num_key_value_heads,
            nkv_full: r.num_global_key_value_heads,
            head_dim_sliding: r.head_dim,
            head_dim_full: r.global_head_dim,
            sliding_window: r.sliding_window,
            max_pos: r.max_position_embeddings,
            eps: r.rms_norm_eps,
            softcap: r.final_logit_softcapping,
            k_eq_v: r.attention_k_eq_v,
            tie_embeddings: r.tie_word_embeddings,
            rope_sliding,
            rope_full,
        })
    }

    /// 第 `l` 层是否为滑窗层。
    pub fn is_sliding(&self, l: usize) -> bool {
        self.layer_types[l] == LayerType::SlidingAttention
    }

    /// 第 `l` 层的 (head_dim, kv 头数, rope)。
    pub fn attn_shape(&self, l: usize) -> (usize, usize, &RopeSpec) {
        if self.is_sliding(l) {
            (self.head_dim_sliding, self.nkv_sliding, &self.rope_sliding)
        } else {
            (self.head_dim_full, self.nkv_full, &self.rope_full)
        }
    }

    /// 该层 K 权重是否兼作 V（只有 full 层，且配置开了 `attention_k_eq_v`）。
    pub fn k_eq_v_at(&self, l: usize) -> bool {
        self.k_eq_v && !self.is_sliding(l)
    }

    /// 该层的注意力窗口：0 = 全注意力。
    pub fn window_at(&self, l: usize) -> usize {
        if self.is_sliding(l) {
            self.sliding_window
        } else {
            0
        }
    }
}

/// `rope_type` 决定分母：`proportional` 用 head_dim，`default` 用旋转维数。
fn rope_spec(r: &RawConfig, kind: &str, head_dim: usize) -> Result<RopeSpec> {
    let e = r
        .rope_parameters
        .get(kind)
        .or_else(|| r.rope_parameters.values().next())
        .ok_or_else(|| anyhow::anyhow!("config 缺 rope_parameters[{kind}]"))?;
    let rope_dim = ((head_dim as f64) * e.partial_rotary_factor).round() as usize;
    ensure!(rope_dim % 2 == 0 && rope_dim > 0 && rope_dim <= head_dim, "rope_dim {rope_dim} 非法");
    // proportional = vLLM `Gemma4RotaryEmbedding`：inv_freq 零频补齐到 head_dim/2 对、整头 neox 旋转，
    // 所以非零频的配对是 (i, i + head_dim/2)，不是 (i, i + rope_dim/2)。
    let (denom, span) = match e.rope_type.as_str() {
        "proportional" => (head_dim, head_dim / 2),
        "default" => (rope_dim, rope_dim / 2),
        other => anyhow::bail!("不支持的 rope_type {other}"),
    };
    Ok(RopeSpec { rope_dim, denom, span, theta: e.rope_theta })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proportional_rope_full_layer_uses_head_dim_denom() {
        let json = r#"{
            "text_config": {
                "vocab_size": 262144, "hidden_size": 2816, "intermediate_size": 2112,
                "num_hidden_layers": 2, "num_attention_heads": 16, "num_key_value_heads": 8,
                "head_dim": 256, "global_head_dim": 512, "num_global_key_value_heads": 2,
                "moe_intermediate_size": 704, "num_experts": 128, "top_k_experts": 8,
                "sliding_window": 1024, "max_position_embeddings": 262144,
                "rms_norm_eps": 1e-6, "layer_types": ["sliding_attention", "full_attention"],
                "attention_k_eq_v": true, "tie_word_embeddings": true,
                "rope_parameters": {
                    "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
                    "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
                }
            }
        }"#;
        let c = KaonConfig::from_json(json).unwrap();
        assert_eq!(c.rope_full.rope_dim, 128);
        assert_eq!(c.rope_full.denom, 512, "proportional 的分母是 head_dim，不是 rope_dim");
        assert_eq!(c.rope_full.span, 256, "proportional 是整头 neox：配对 (i, i+head_dim/2)");
        assert_eq!(c.rope_sliding.span, 128);
        assert_eq!(c.rope_sliding.rope_dim, 256);
        assert_eq!(c.rope_sliding.denom, 256);
        assert!(c.k_eq_v_at(1) && !c.k_eq_v_at(0), "K==V 只在 full 层");
        assert_eq!(c.window_at(0), 1024);
        assert_eq!(c.window_at(1), 0);
        assert_eq!(c.attn_shape(0).0, 256);
        assert_eq!(c.attn_shape(1).0, 512);
    }
}
