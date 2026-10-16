//! Qwen3 `config.json`（HF `Qwen3Config`）。只保留推理要读的字段；`from_json` 后用 `check()` 断言结构假设。

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    /// Qwen3 显式给 head_dim（0.6B：128 = 2 × hidden/heads，不是 hidden/heads！）。
    pub head_dim: u32,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub max_position_embeddings: u32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub sliding_window: Option<u32>,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default = "silu")]
    pub hidden_act: String,
    /// 可为数或数组。
    #[serde(default)]
    pub eos_token_id: serde_json::Value,
}

fn silu() -> String {
    "silu".into()
}

impl Qwen3Config {
    pub fn from_json(s: &str) -> anyhow::Result<Self> {
        let c: Self = serde_json::from_str(s)?;
        c.check()?;
        Ok(c)
    }

    /// Q1 支持范围：dense、GQA、无 bias、无 sliding window、silu。
    pub fn check(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.hidden_act == "silu", "unsupported hidden_act {}", self.hidden_act);
        anyhow::ensure!(!self.attention_bias, "attention_bias unsupported");
        anyhow::ensure!(!self.use_sliding_window, "sliding window unsupported");
        anyhow::ensure!(
            self.num_attention_heads % self.num_key_value_heads == 0,
            "heads {} not divisible by kv heads {}",
            self.num_attention_heads,
            self.num_key_value_heads
        );
        Ok(())
    }

    pub fn eos_ids(&self) -> Vec<u32> {
        match &self.eos_token_id {
            serde_json::Value::Number(n) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
            serde_json::Value::Array(a) => a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect(),
            _ => Vec::new(),
        }
    }

    #[inline]
    pub fn q_size(&self) -> usize {
        (self.num_attention_heads * self.head_dim) as usize
    }
    #[inline]
    pub fn kv_size(&self) -> usize {
        (self.num_key_value_heads * self.head_dim) as usize
    }
    #[inline]
    pub fn group_size(&self) -> usize {
        (self.num_attention_heads / self.num_key_value_heads) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 真 config.json 的结构事实（docs/design-qwen3.md §1）。文件不在仓库里：没有就跳过。
    #[test]
    fn real_qwen3_0_6b_config() {
        let Ok(s) = std::fs::read_to_string("/data/ref/models/qwen3-0.6b/config.json") else { return };
        let c = Qwen3Config::from_json(&s).unwrap();
        assert_eq!(c.num_hidden_layers, 28);
        assert_eq!(c.hidden_size, 1024);
        assert_eq!(c.num_attention_heads, 16);
        assert_eq!(c.num_key_value_heads, 8);
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.intermediate_size, 3072);
        assert_eq!(c.vocab_size, 151936);
        assert!(c.tie_word_embeddings);
        assert_eq!(c.eos_ids(), vec![151645]);
        assert_eq!(c.group_size(), 2);
        assert_eq!(c.q_size(), 2048);
        assert_eq!(c.kv_size(), 1024);
    }
}
