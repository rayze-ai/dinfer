//! 引擎配置。对应 vLLM 的 VllmConfig 子集（Model/Cache/Scheduler/Speculative/CUDAGraph）。
//! 只保留两目标模型需要的字段；默认值取自 vLLM `arg_utils.py` 的 ≥70 GB 显存档。

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub model: ModelPathConfig,
    pub cache: CacheConfig,
    pub scheduler: SchedulerConfig,
    pub spec: Option<SpecConfig>,
    pub cudagraph: CudaGraphConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPathConfig {
    /// HF 模型目录（含 config.json 与 safetensors）。
    pub path: String,
    /// 最大上下文长度；None 表示取 config.json 的 max_position_embeddings。
    pub max_model_len: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// 每个 KV block 的 token 数。
    pub block_size: u32,
    /// 显存占用上限比例。
    pub gpu_memory_utilization: f32,
    pub enable_prefix_caching: bool,
    /// 前缀缓存哈希粒度（token 数），必须整除 block_size。
    pub hash_block_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub max_num_seqs: u32,
    pub max_num_batched_tokens: u32,
    pub enable_chunked_prefill: bool,
    /// 单请求单 step 的 prefill token 上限；0 表示不限（受 max_num_batched_tokens 约束）。
    pub long_prefill_token_threshold: u32,
    /// decode 行（本步 1 token、已有历史）不占 `max_num_batched_tokens`：预算只约束 prefill token。
    /// 这样每步都能同时装下全部 decode 行 + 预算内的 prefill，跨流并行（Q3-j）才有机会；
    /// 代价：一步的总 token 数上限 = 预算 + max_num_seqs（executor 的 max_tokens 按此分配）。
    #[serde(default)]
    pub decode_exempt_from_budget: bool,
    /// B47（loop kaon3-decode-opt D206）：一步里**所有 prefill 行合计**的新 token 上限（0 = 不限 = 只受 max_num_batched_tokens）。
    /// 与 `long_prefill_token_threshold`（单请求上限）不同：这是整步 prefill 配额，把一波涌入的多请求 prefill 摊到多步，
    /// 让混批步别从 ~9 ms 跳到 98 ms（c16：一步 10 请求 2148 tok）。vLLM 把 prefill chunk 混进每个 decode 步只多 1–2 ms 就是这个效果。
    #[serde(default)]
    pub prefill_tokens_per_step: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecConfig {
    /// MTP 每 step 的 draft token 数。
    pub num_speculative_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaGraphConfig {
    pub enabled: bool,
    /// 捕获的 batch（token 数）档位；空表示按 vLLM 规则自动生成。
    pub capture_sizes: Vec<u32>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            gpu_memory_utilization: 0.95,
            enable_prefix_caching: true,
            hash_block_size: 16,
        }
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 1024,
            max_num_batched_tokens: 8192,
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 0,
            decode_exempt_from_budget: false,
            prefill_tokens_per_step: 0,
        }
    }
}

impl Default for CudaGraphConfig {
    fn default() -> Self {
        Self { enabled: true, capture_sizes: Vec::new() }
    }
}

impl CudaGraphConfig {
    /// vLLM `config/vllm.py:_set_cudagraph_sizes` 的规则：
    /// `[1,2,4] + 8..256 step 8 + 256..max step 16`，max = min(max_num_seqs * q_len * 2, 512)。
    pub fn default_capture_sizes(max_num_seqs: u32, uniform_decode_query_len: u32) -> Vec<u32> {
        let max = (max_num_seqs * uniform_decode_query_len * 2).min(512);
        let mut v: Vec<u32> = vec![1, 2, 4];
        v.extend((8..256).step_by(8));
        v.extend((256..=max).step_by(16));
        v.retain(|&s| s <= max);
        v.dedup();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_sizes_follow_vllm_rule() {
        let s = CudaGraphConfig::default_capture_sizes(1024, 1);
        assert_eq!(&s[..3], &[1, 2, 4]);
        assert!(s.contains(&8) && s.contains(&248) && s.contains(&256) && s.contains(&512));
        assert_eq!(*s.last().unwrap(), 512);
        let small = CudaGraphConfig::default_capture_sizes(4, 1);
        assert_eq!(small, vec![1, 2, 4, 8]);
    }
}
