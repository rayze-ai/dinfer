//! 每层 KV / 状态规格。对应 vLLM `v1/kv_cache_interface.py` 的子集。

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvCacheSpec {
    /// MLA：每 token 一个 latent 向量，`head_size = kv_lora_rank + rope_dim`。
    Mla { block_size: u32, head_size: u32, dtype_bytes: u32 },
    /// 标准 GQA/MHA：每 token K+V 各 `kv_heads × head_size`。
    FullAttention { block_size: u32, kv_heads: u32, head_size: u32, dtype_bytes: u32 },
    /// 递归状态（KDA / Mamba）：每请求固定若干 slot，不按 token 增长。
    Mamba { shapes: Vec<Vec<u32>>, dtype_bytes: Vec<u32>, block_size: u32 },
    /// DSA 索引器的 fp8 K cache：每 token 固定字节数。
    Indexer { block_size: u32, bytes_per_token: u32 },
}

impl KvCacheSpec {
    pub fn block_size(&self) -> u32 {
        match self {
            KvCacheSpec::Mla { block_size, .. }
            | KvCacheSpec::FullAttention { block_size, .. }
            | KvCacheSpec::Mamba { block_size, .. }
            | KvCacheSpec::Indexer { block_size, .. } => *block_size,
        }
    }

    /// 一个 block 占多少字节。
    pub fn page_size_bytes(&self) -> u64 {
        match self {
            KvCacheSpec::FullAttention { block_size, kv_heads, head_size, dtype_bytes } => {
                *block_size as u64 * 2 * *kv_heads as u64 * *head_size as u64 * *dtype_bytes as u64
            }
            KvCacheSpec::Mla { block_size, head_size, dtype_bytes } => {
                *block_size as u64 * *head_size as u64 * *dtype_bytes as u64
            }
            KvCacheSpec::Mamba { shapes, dtype_bytes, .. } => shapes
                .iter()
                .zip(dtype_bytes)
                .map(|(s, b)| s.iter().map(|&d| d as u64).product::<u64>() * *b as u64)
                .sum(),
            KvCacheSpec::Indexer { block_size, bytes_per_token } => {
                *block_size as u64 * *bytes_per_token as u64
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mla_page_size_glm53() {
        // GLM-5.3：kv_lora 512 + rope 64 = 576，bf16，block 16。
        let s = KvCacheSpec::Mla { block_size: 16, head_size: 576, dtype_bytes: 2 };
        assert_eq!(s.page_size_bytes(), 16 * 576 * 2);
    }

    #[test]
    fn kda_state_size_kimik3() {
        // Kimi-K3 KDA：conv [3*12288, 3] bf16 + recurrent [96,128,128] fp32。
        let s = KvCacheSpec::Mamba {
            shapes: vec![vec![3 * 12288, 3], vec![96, 128, 128]],
            dtype_bytes: vec![2, 4],
            block_size: 1,
        };
        assert_eq!(s.page_size_bytes(), 3 * 12288 * 3 * 2 + 96 * 128 * 128 * 4);
    }
}
