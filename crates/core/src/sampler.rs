//! 每 step 的采样计划（host 侧，与 `BatchPlan.req_ids` 平行）与 logprobs 输出类型。
//! 顺序与 vLLM `Sampler.forward` 一致：penalties → temperature → min_p → top_k/top_p → 采样；logprobs 用 **原始** logits。
//! Philox (seed, offset)：seed 每请求固定（用户给或入队时随机），offset = 该请求已产出 token 数 → 同 seed 两次运行
//! 逐 token 相同，且与 batch 位置无关。

use crate::sample::SamplingParams;

/// 一个请求、一个采样行的参数。executor 直接按列上传。
#[derive(Debug, Clone, Default)]
pub struct SampleBatch {
    /// 随机行的温度（greedy 行为 0）。
    pub temperature: Vec<f32>,
    pub top_k: Vec<i32>,
    pub top_p: Vec<f32>,
    pub min_p: Vec<f32>,
    pub repetition_penalty: Vec<f32>,
    pub frequency_penalty: Vec<f32>,
    pub presence_penalty: Vec<f32>,
    pub seed: Vec<u64>,
    pub offset: Vec<u64>,
    /// 惩罚用的 token 历史，CSR：`hist_ids[hist_offsets[b]..hist_offsets[b+1]]`，前 `hist_prompt_len[b]` 个是 prompt。
    pub hist_ids: Vec<i32>,
    pub hist_offsets: Vec<i32>,
    pub hist_prompt_len: Vec<i32>,
    /// 每行要的 top-logprobs 个数；None = 不要 logprobs。
    pub logprobs: Vec<Option<u32>>,
    pub any_random: bool,
    pub any_penalty: bool,
    pub any_min_p: bool,
    pub any_topk_topp: bool,
    pub any_temperature: bool,
    /// 本步 logprobs 的最大 k（0 也可能需要：只要 chosen）。
    pub max_logprobs: Option<u32>,
}

impl SampleBatch {
    pub fn clear(&mut self) {
        *self = Self::default();
        self.hist_offsets.push(0);
    }

    pub fn num_rows(&self) -> usize {
        self.temperature.len()
    }

    /// 追加一行。`token_ids` 为该请求 prompt + 已产出 token；`num_prompt` 为 prompt 长度；`seed` 为请求固定种子。
    pub fn push(&mut self, sp: &SamplingParams, seed: u64, token_ids: &[u32], num_prompt: usize) {
        if self.hist_offsets.is_empty() {
            self.hist_offsets.push(0);
        }
        let greedy = sp.is_greedy();
        // greedy 行：温度 0，其余过滤关掉（vLLM greedy 直接 argmax，不受 top_k/p/min_p 影响）
        self.temperature.push(if greedy { 0.0 } else { sp.temperature });
        self.top_k.push(if greedy { -1 } else { sp.top_k });
        self.top_p.push(if greedy { 1.0 } else { sp.top_p });
        self.min_p.push(if greedy { 0.0 } else { sp.min_p });
        self.repetition_penalty.push(sp.repetition_penalty);
        self.frequency_penalty.push(sp.frequency_penalty);
        self.presence_penalty.push(sp.presence_penalty);
        self.seed.push(seed);
        self.offset.push(token_ids.len().saturating_sub(num_prompt) as u64);
        if sp.has_penalties() {
            self.hist_ids.extend(token_ids.iter().map(|&t| t as i32));
            self.any_penalty = true;
        }
        self.hist_offsets.push(self.hist_ids.len() as i32);
        self.hist_prompt_len.push(if sp.has_penalties() { num_prompt as i32 } else { 0 });
        self.logprobs.push(sp.logprobs);
        if let Some(k) = sp.logprobs {
            self.max_logprobs = Some(self.max_logprobs.map_or(k, |m| m.max(k)));
        }
        if !greedy {
            self.any_random = true;
            if sp.temperature != 1.0 {
                self.any_temperature = true;
            }
            if sp.min_p > 0.0 {
                self.any_min_p = true;
            }
            if sp.top_k > 0 || sp.top_p < 1.0 {
                self.any_topk_topp = true;
            }
        }
    }
}

/// 一个产出 token 的 logprob 信息（对应 OpenAI `logprobs.content[i]`）。
#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprobs {
    pub token_id: u32,
    /// log_softmax(原始 logits)[token_id]
    pub logprob: f32,
    /// 严格大于 chosen 的候选个数（0 = 它就是 argmax）；vLLM rank 从 1 起 = rank + 1。
    pub rank: u32,
    /// top-k 候选（降序），长度 = 请求的 top_logprobs。
    pub top: Vec<(u32, f32)>,
}

/// 入队时给没有 seed 的请求生成一个固定随机种子（不引 rand：用 RandomState 的随机 key 哈希请求 id）。
pub fn random_seed(req_id: u64) -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(req_id);
    h.write_u128(
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0),
    );
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offset_is_output_count_and_batch_independent() {
        let sp = SamplingParams { temperature: 0.8, ..Default::default() };
        let mut a = SampleBatch::default();
        a.clear();
        a.push(&sp, 7, &[1, 2, 3, 10, 11], 3);
        let mut b = SampleBatch::default();
        b.clear();
        b.push(&SamplingParams { temperature: 0.0, ..Default::default() }, 1, &[5], 1);
        b.push(&sp, 7, &[1, 2, 3, 10, 11], 3);
        assert_eq!(a.offset[0], 2);
        assert_eq!((a.seed[0], a.offset[0]), (b.seed[1], b.offset[1]));
        assert!(a.any_random && b.any_random && !b.any_penalty);
        assert_eq!(b.temperature, vec![0.0, 0.8]);
        assert_eq!(b.top_k, vec![-1, -1]);
    }

    #[test]
    fn penalty_history_is_csr() {
        let pen = SamplingParams { repetition_penalty: 1.3, ..Default::default() };
        let plain = SamplingParams::default();
        let mut sb = SampleBatch::default();
        sb.clear();
        sb.push(&plain, 1, &[9, 9], 2);
        sb.push(&pen, 2, &[1, 2, 3, 4], 3);
        assert_eq!(sb.hist_offsets, vec![0, 0, 4]);
        assert_eq!(sb.hist_ids, vec![1, 2, 3, 4]);
        assert_eq!(sb.hist_prompt_len, vec![0, 3]);
        assert!(sb.any_penalty && !sb.any_topk_topp && !sb.any_temperature);
    }
}
