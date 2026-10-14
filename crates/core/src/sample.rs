//! 采样参数。对应 vLLM `SamplingParams` 的子集。

#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub seed: Option<u64>,
    pub stop_token_ids: Vec<u32>,
    pub logprobs: Option<u32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 1.0,
            top_p: 1.0,
            top_k: -1,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
            stop_token_ids: Vec::new(),
            logprobs: None,
        }
    }
}

impl SamplingParams {
    #[inline]
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 1e-5
    }
}
