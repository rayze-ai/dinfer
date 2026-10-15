//! `config.json` 解析。字段对照 `/data/ref/models/glm53/config.json` 与 vLLM `deepseek_v32/attention.py`。

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct GlmMoeDsaConfig {
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub vocab_size: u32,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: u32,
    pub hidden_act: String,
    pub eos_token_id: EosIds,
    pub pad_token_id: Option<u32>,

    // MLA
    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,
    #[serde(default)]
    pub rope_interleave: bool,
    pub rope_parameters: RopeParameters,

    // DSA indexer
    pub index_n_heads: u32,
    pub index_head_dim: u32,
    pub index_topk: u32,
    #[serde(default = "one")]
    pub index_topk_freq: u32,
    #[serde(default = "two")]
    pub index_skip_topk_offset: u32,
    pub index_topk_pattern: Option<Vec<String>>,
    #[serde(default)]
    pub indexer_rope_interleave: bool,

    // MoE
    pub n_routed_experts: u32,
    pub num_experts_per_tok: u32,
    pub n_shared_experts: u32,
    pub moe_intermediate_size: u32,
    pub first_k_dense_replace: u32,
    #[serde(default = "one")]
    pub moe_layer_freq: u32,
    pub routed_scaling_factor: f32,
    pub scoring_func: String,
    pub topk_method: String,
    pub n_group: u32,
    pub topk_group: u32,
    #[serde(default)]
    pub norm_topk_prob: bool,

    #[serde(default)]
    pub num_nextn_predict_layers: u32,
    pub quantization_config: Option<serde_json::Value>,
}

impl GlmMoeDsaConfig {
    /// `quantization_config.weight_block_size`（fp8 块量化），无量化时 None。
    pub fn fp8_block_size(&self) -> Option<(usize, usize)> {
        let q = self.quantization_config.as_ref()?;
        if q.get("quant_method").and_then(|v| v.as_str()) != Some("fp8") {
            return None;
        }
        let b = q.get("weight_block_size")?.as_array()?;
        Some((b.first()?.as_u64()? as usize, b.get(1)?.as_u64()? as usize))
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EosIds {
    One(u32),
    Many(Vec<u32>),
}

impl EosIds {
    pub fn as_vec(&self) -> Vec<u32> {
        match self {
            EosIds::One(x) => vec![*x],
            EosIds::Many(v) => v.clone(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
    pub rope_type: String,
}

fn one() -> u32 {
    1
}
fn two() -> u32 {
    2
}

impl GlmMoeDsaConfig {
    pub fn from_json(s: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(s)?)
    }

    /// 该层是否自己算 top-k（否则复用上一个 full 层写入的索引）。
    /// 对照 vLLM `deepseek_v32/attention.py:166-177`。
    pub fn has_indexer(&self, layer_idx: u32) -> bool {
        if let Some(p) = &self.index_topk_pattern {
            return p.get(layer_idx as usize).is_none_or(|s| s != "S");
        }
        let shifted = (layer_idx + 1).saturating_sub(self.index_skip_topk_offset);
        shifted % self.index_topk_freq == 0
    }

    pub fn is_moe_layer(&self, layer_idx: u32) -> bool {
        layer_idx >= self.first_k_dense_replace && layer_idx % self.moe_layer_freq == 0
    }

    pub fn qk_head_dim(&self) -> u32 {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// MLA KV cache 每 token 维度。
    pub fn kv_latent_dim(&self) -> u32 {
        self.kv_lora_rank + self.qk_rope_head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_official_config() {
        let p = "/data/ref/models/glm53/config.json";
        let Ok(s) = std::fs::read_to_string(p) else {
            eprintln!("skip: {p} not present");
            return;
        };
        let c = GlmMoeDsaConfig::from_json(&s).unwrap();
        assert_eq!(c.num_hidden_layers, 78);
        assert_eq!(c.hidden_size, 6144);
        assert_eq!(c.n_routed_experts, 256);
        assert_eq!(c.num_experts_per_tok, 8);
        assert_eq!(c.kv_latent_dim(), 576);
        assert_eq!(c.qk_head_dim(), 256);
        assert_eq!(c.eos_token_id.as_vec(), vec![154820, 154827, 154829]);
        // indexer 层：0,1,2 及 6,10,...,74
        assert!(c.has_indexer(0) && c.has_indexer(1) && c.has_indexer(2));
        assert!(!c.has_indexer(3) && !c.has_indexer(4) && !c.has_indexer(5));
        assert!(c.has_indexer(6) && c.has_indexer(10) && c.has_indexer(74));
        assert!(!c.has_indexer(7) && !c.has_indexer(77));
        assert!(!c.is_moe_layer(2) && c.is_moe_layer(3));
    }
}
