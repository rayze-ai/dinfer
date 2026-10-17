//! 采样参数。对应 vLLM `SamplingParams` 的子集（OpenAI chat 能映射到的部分）。

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SamplingParams {
    pub max_tokens: u32,
    /// 生成至少这么多 token 后才允许 eos / stop 生效（vLLM `min_tokens`）。
    pub min_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub seed: Option<u64>,
    pub stop_token_ids: Vec<u32>,
    /// stop 字符串（在 detokenize 后的原始文本上匹配；vLLM `check_stop_strings` 语义）。
    pub stop: Vec<String>,
    pub include_stop_str_in_output: bool,
    pub ignore_eos: bool,
    pub skip_special_tokens: bool,
    /// Some(k)：返回每个 token 的 logprob 与 top-k 候选（k 可为 0 = 只要 chosen）。
    pub logprobs: Option<u32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            min_tokens: 0,
            temperature: 1.0,
            top_p: 1.0,
            top_k: -1,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
            stop_token_ids: Vec::new(),
            stop: Vec::new(),
            include_stop_str_in_output: false,
            ignore_eos: false,
            skip_special_tokens: true,
            logprobs: None,
        }
    }
}

impl SamplingParams {
    #[inline]
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 1e-5
    }
    #[inline]
    pub fn has_penalties(&self) -> bool {
        self.repetition_penalty != 1.0 || self.frequency_penalty != 0.0 || self.presence_penalty != 0.0
    }

    /// 与 vLLM `SamplingParams._verify_args` 同口径的范围检查；错误信息直接给 400。
    pub fn validate(&self) -> Result<(), String> {
        if self.temperature < 0.0 {
            return Err(format!("temperature must be non-negative, got {}", self.temperature));
        }
        if !(0.0 < self.top_p && self.top_p <= 1.0) {
            return Err(format!("top_p must be in (0, 1], got {}", self.top_p));
        }
        if self.top_k < -1 || self.top_k == 0 {
            return Err(format!("top_k must be -1 (disable) or at least 1, got {}", self.top_k));
        }
        if !(0.0..=1.0).contains(&self.min_p) {
            return Err(format!("min_p must be in [0, 1], got {}", self.min_p));
        }
        if !(0.0..=2.0).contains(&self.repetition_penalty) || self.repetition_penalty == 0.0 {
            return Err(format!("repetition_penalty must be in (0, 2], got {}", self.repetition_penalty));
        }
        if !(-2.0..=2.0).contains(&self.frequency_penalty) {
            return Err(format!("frequency_penalty must be in [-2, 2], got {}", self.frequency_penalty));
        }
        if !(-2.0..=2.0).contains(&self.presence_penalty) {
            return Err(format!("presence_penalty must be in [-2, 2], got {}", self.presence_penalty));
        }
        if self.max_tokens == 0 {
            return Err("max_tokens must be at least 1".into());
        }
        if self.min_tokens > self.max_tokens {
            return Err(format!("min_tokens {} > max_tokens {}", self.min_tokens, self.max_tokens));
        }
        if let Some(k) = self.logprobs {
            if k > 20 {
                return Err(format!("top_logprobs must be <= 20, got {k}"));
            }
        }
        if self.stop.len() > 16 {
            return Err("at most 16 stop strings".into());
        }
        Ok(())
    }
}
