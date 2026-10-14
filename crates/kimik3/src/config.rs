//! `config.json` 解析。字段对照 `/data/ref/models/kimik3/configuration_kimi_k3.py`。

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct KimiK3Config {
    pub text_config: KimiLinearConfig,
    pub vision_config: serde_json::Value,
    pub media_placeholder_token_id: u32,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KimiLinearConfig {
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub vocab_size: u32,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: u32,
    pub hidden_act: String,
    pub activation_situ_beta: Option<f32>,
    pub activation_situ_linear_beta: Option<f32>,

    // MLA
    pub q_lora_rank: Option<u32>,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,
    #[serde(default)]
    pub mla_use_nope: bool,
    #[serde(default)]
    pub mla_use_output_gate: bool,

    // MoE
    pub num_experts: u32,
    pub num_experts_per_token: u32,
    pub num_shared_experts: u32,
    pub moe_intermediate_size: u32,
    pub routed_expert_hidden_size: Option<u32>,
    #[serde(default)]
    pub latent_moe_use_norm: bool,
    pub first_k_dense_replace: u32,
    pub moe_layer_freq: u32,
    pub routed_scaling_factor: f32,
    pub moe_router_activation_func: String,
    pub topk_method: String,
    pub num_expert_group: u32,
    pub topk_group: u32,
    #[serde(default = "default_true")]
    pub moe_renormalize: bool,

    // AttnRes / KDA / MTP
    pub attn_res_block_size: Option<u32>,
    pub linear_attn_config: LinearAttnConfig,
    #[serde(default)]
    pub num_nextn_predict_layers: u32,

    pub quantization_config: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LinearAttnConfig {
    /// 1-based 层号。
    pub kda_layers: Vec<u32>,
    pub full_attn_layers: Vec<u32>,
    pub head_dim: u32,
    pub num_heads: u32,
    pub short_conv_kernel_size: u32,
    pub gate_lower_bound: Option<f32>,
    #[serde(default)]
    pub use_full_rank_gate: bool,
}

fn default_true() -> bool {
    true
}

impl KimiLinearConfig {
    /// 0-based 层号是否为 KDA 层（对照 `configuration_kimi_k3.py::is_kda_layer`）。
    pub fn is_kda_layer(&self, layer_idx: u32) -> bool {
        self.linear_attn_config.kda_layers.contains(&(layer_idx + 1))
    }

    pub fn is_moe_layer(&self, layer_idx: u32) -> bool {
        layer_idx >= self.first_k_dense_replace && layer_idx % self.moe_layer_freq == 0
    }
}

impl KimiK3Config {
    pub fn from_json(s: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(s)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_official_config() {
        let p = "/data/ref/models/kimik3/config.json";
        let Ok(s) = std::fs::read_to_string(p) else {
            eprintln!("skip: {p} not present");
            return;
        };
        let c = KimiK3Config::from_json(&s).unwrap();
        let t = &c.text_config;
        assert_eq!(t.num_hidden_layers, 93);
        assert_eq!(t.hidden_size, 7168);
        assert_eq!(t.num_experts, 896);
        assert_eq!(t.num_experts_per_token, 16);
        assert_eq!(t.routed_expert_hidden_size, Some(3584));
        assert!(t.mla_use_nope);
        assert_eq!(t.attn_res_block_size, Some(12));
        let n_mla = (0..93).filter(|&l| !t.is_kda_layer(l)).count();
        assert_eq!(n_mla, 24);
        assert!(!t.is_kda_layer(3) && t.is_kda_layer(0) && !t.is_kda_layer(92));
        assert!(!t.is_moe_layer(0) && t.is_moe_layer(1));
        assert_eq!(c.media_placeholder_token_id, 163605);
    }
}
