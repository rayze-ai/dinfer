//! Qwen35 `config.json`：顶层是多模态壳（`Qwen3_5ForConditionalGeneration`，含 `text_config` / `vision_config`），
//! 我们只读 `text_config`（HF `Qwen3_5TextConfig`）。`from_json` 接受顶层壳或裸 text_config 两种输入；`check()` 断言支持范围
//! （结构事实见 `docs/design-qwen35.md` §1，均来自 B300 真权重 config + safetensors header 核对）。
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35Config {
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_hidden_layers: u32,
    /// 每层类型；长度 == num_hidden_layers。Qwen3.5-27B：48 linear + 16 full，`full_attention_interval` 4。
    pub layer_types: Vec<LayerType>,
    #[serde(default)]
    pub full_attention_interval: Option<u32>,
    // ---- 全注意力层 ----
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    /// 只转 head_dim × partial_rotary_factor 维（27B：0.25 → 64 维）。
    pub partial_rotary_factor: f32,
    /// q_proj 输出 = heads × [q(D) | gate(D)]（per-head 交错，不能整段前后切）；attention 之后 `o *= sigmoid(gate)`。
    #[serde(default)]
    pub attn_output_gate: bool,
    #[serde(default)]
    pub attention_bias: bool,
    pub rope_parameters: serde_json::Value,
    // ---- Gated DeltaNet 层 ----
    pub linear_num_key_heads: u32,
    pub linear_num_value_heads: u32,
    pub linear_key_head_dim: u32,
    pub linear_value_head_dim: u32,
    pub linear_conv_kernel_dim: u32,
    #[serde(default = "f32_str")]
    pub mamba_ssm_dtype: String,
    // ---- 其他 ----
    pub rms_norm_eps: f32,
    pub max_position_embeddings: u32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "silu")]
    pub hidden_act: String,
    #[serde(default)]
    pub mtp_num_hidden_layers: u32,
    #[serde(default)]
    pub eos_token_id: serde_json::Value,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
}

fn silu() -> String {
    "silu".into()
}
fn f32_str() -> String {
    "float32".into()
}

impl Qwen35Config {
    /// 顶层壳（有 `text_config`）或裸 text_config 都接受。
    pub fn from_json(s: &str) -> anyhow::Result<Self> {
        let v: serde_json::Value = serde_json::from_str(s)?;
        let text = v.get("text_config").cloned().unwrap_or(v);
        let c: Self = serde_json::from_value(text)?;
        c.check()?;
        Ok(c)
    }

    /// 支持范围（里程碑 4 第一版）：两种层类型、silu、无 bias、GQA 整除、GDN 头数整除、head_dim 与 kernel 模板一致、不做 MTP。
    pub fn check(&self) -> anyhow::Result<()> {
        use anyhow::ensure;
        ensure!(self.hidden_act == "silu", "unsupported hidden_act {}", self.hidden_act);
        ensure!(!self.attention_bias, "attention_bias unsupported");
        ensure!(
            self.layer_types.len() == self.num_hidden_layers as usize,
            "layer_types.len {} != num_hidden_layers {}",
            self.layer_types.len(),
            self.num_hidden_layers
        );
        ensure!(self.num_attention_heads % self.num_key_value_heads == 0, "GQA heads not divisible");
        ensure!(
            matches!(self.head_dim, 128 | 256),
            "head_dim {} not in attention kernel templates (128 | 256)",
            self.head_dim
        );
        ensure!(
            matches!(self.num_attention_heads / self.num_key_value_heads, 1 | 2 | 4 | 6),
            "GQA group {} not in decode v3 templates (1|2|4|6)",
            self.num_attention_heads / self.num_key_value_heads
        );
        let rd = (self.head_dim as f32 * self.partial_rotary_factor).round() as u32;
        ensure!(rd % 2 == 0 && rd <= self.head_dim, "rope_dim {rd} invalid");
        ensure!(self.linear_num_value_heads % self.linear_num_key_heads == 0, "GDN v heads not divisible by k heads");
        ensure!(self.linear_key_head_dim == self.linear_value_head_dim, "GDN k/v head_dim differ: unsupported");
        ensure!(self.mamba_ssm_dtype == "float32", "mamba_ssm_dtype {} unsupported", self.mamba_ssm_dtype);
        ensure!(self.mtp_num_hidden_layers <= 1, "mtp layers {} unsupported", self.mtp_num_hidden_layers);
        Ok(())
    }

    pub fn gqa_group(&self) -> u32 {
        self.num_attention_heads / self.num_key_value_heads
    }
    pub fn rope_dim(&self) -> u32 {
        (self.head_dim as f32 * self.partial_rotary_factor).round() as u32
    }
    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters.get("rope_theta").and_then(|v| v.as_f64()).unwrap_or(10000.0)
    }
    pub fn gdn_group(&self) -> u32 {
        self.linear_num_value_heads / self.linear_num_key_heads
    }
    pub fn n_full_attention(&self) -> usize {
        self.layer_types.iter().filter(|t| **t == LayerType::FullAttention).count()
    }
    pub fn eos_ids(&self) -> Vec<u32> {
        match &self.eos_token_id {
            serde_json::Value::Number(n) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
            serde_json::Value::Array(a) => a.iter().filter_map(|v| v.as_u64().map(|x| x as u32)).collect(),
            _ => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 与 B300 `/data/models/Qwen/Qwen3.8-27B/config.json` text_config 一致的关键字段（2026-10-19 核对）。
    fn sample() -> String {
        let lt: Vec<&str> = (0..64)
            .map(|i| if (i + 1) % 4 == 0 { "full_attention" } else { "linear_attention" })
            .collect();
        serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": {
                "vocab_size": 248320, "hidden_size": 5120, "intermediate_size": 17408, "num_hidden_layers": 64,
                "layer_types": lt, "full_attention_interval": 4,
                "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256, "partial_rotary_factor": 0.25,
                "attn_output_gate": true, "attention_bias": false,
                "rope_parameters": {"rope_theta": 10000000.0, "mrope_section": [11, 11, 10], "mrope_interleaved": true},
                "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4, "mamba_ssm_dtype": "float32",
                "rms_norm_eps": 1e-6, "max_position_embeddings": 262144, "tie_word_embeddings": false,
                "hidden_act": "silu", "mtp_num_hidden_layers": 1, "eos_token_id": 248044, "bos_token_id": 248044
            }
        })
        .to_string()
    }

    #[test]
    fn parses_shell_and_derives() {
        let c = Qwen35Config::from_json(&sample()).unwrap();
        assert_eq!(c.n_full_attention(), 16);
        assert_eq!(c.gqa_group(), 6);
        assert_eq!(c.rope_dim(), 64);
        assert_eq!(c.gdn_group(), 3);
        assert_eq!(c.eos_ids(), vec![248044]);
        assert!(c.attn_output_gate && !c.tie_word_embeddings);
    }

    #[test]
    fn rejects_bad_head_dim() {
        let s = sample().replace("\"head_dim\":256", "\"head_dim\":192");
        assert!(Qwen35Config::from_json(&s).is_err());
    }
}
