//! 采样参数。对应 vLLM `SamplingParams` 的子集（OpenAI chat 能映射到的部分）。

use std::sync::Arc;

/// 受限解码的每请求文法状态（vLLM `StructuredOutputGrammar`）。core 只认这个 trait，实现在 `dinfer-guided`
///（llguidance），由 server / service 在 submit 时构造并随 `SamplingParams.grammar` 传进来。
///
/// 协议：runner 每步在 `SampleBatch::push` 里按请求的输出序列**追赶** `consume_token`，再 `fill_mask` 得本步 bitmask；
/// 实现方不必记住已消费了多少（`GrammarHandle` 记）。
pub trait GrammarState: Send {
    /// bitmask 的 u32 字数（= ceil(vocab/32)，vocab 为实现方的词表大小；模型词表更大时多出的列按不允许处理）。
    fn mask_words(&self) -> usize;
    /// 把当前状态允许的 token 写进 `out[..mask_words]`（bit=1 允许）。文法已终止（接受态且无后续）时只允许 eos。
    fn fill_mask(&mut self, out: &mut [u32]) -> anyhow::Result<()>;
    /// 喂一个已采样的 token。
    fn consume_token(&mut self, tok: u32) -> anyhow::Result<()>;
    /// 回到初始状态（请求被回滚 / 重算时 `GrammarHandle` 会调用它再重放）。
    fn reset(&mut self) -> anyhow::Result<()>;
}

/// `GrammarState` + 已消费的输出 token 数 + mask 复用缓冲。放在 `Arc<Mutex>` 里是因为 `SamplingParams` 会被 clone 到
/// scheduler 的 `Request` 与 runner 的 `ReqState` 两处，状态必须共享；锁只在 runner 线程里逐请求短暂持有。
pub struct GrammarSlot {
    pub state: Box<dyn GrammarState>,
    pub consumed: usize,
    pub mask: Vec<u32>,
    /// 文法出错（非法 token / 内部错误）后置位：此后不再 mask（退化为无约束，不让请求卡死），错误记一次日志。
    pub failed: bool,
}

#[derive(Clone)]
pub struct GrammarHandle(pub Arc<parking_lot::Mutex<GrammarSlot>>);

impl GrammarHandle {
    pub fn new(state: Box<dyn GrammarState>) -> Self {
        let words = state.mask_words();
        Self(Arc::new(parking_lot::Mutex::new(GrammarSlot { state, consumed: 0, mask: vec![0; words], failed: false })))
    }
    /// 把文法状态追赶到 `output` 的末尾（已消费的前缀不重放；输出比已消费的短 = 回滚 → reset 重放），
    /// 再计算 mask。返回 `Some(&mask)`；文法已失效返回 None。
    pub fn advance_and_mask<'a>(slot: &'a mut GrammarSlot, output: &[u32]) -> Option<&'a [u32]> {
        if slot.failed {
            return None;
        }
        let r = (|| -> anyhow::Result<()> {
            if output.len() < slot.consumed {
                slot.state.reset()?;
                slot.consumed = 0;
            }
            for &t in &output[slot.consumed..] {
                slot.state.consume_token(t)?;
                slot.consumed += 1;
            }
            let words = slot.state.mask_words();
            slot.mask.resize(words, 0);
            slot.state.fill_mask(&mut slot.mask)
        })();
        match r {
            Ok(()) => Some(&slot.mask),
            Err(e) => {
                tracing::warn!(err = %e, "grammar: state error, request continues unconstrained");
                slot.failed = true;
                None
            }
        }
    }
}

impl std::fmt::Debug for GrammarHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("GrammarHandle(..)")
    }
}
impl PartialEq for GrammarHandle {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

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
    /// detokenize 时特殊 token 之间是否加空格（HF `spaces_between_special_tokens`；只影响 skip=false 的文本）。
    pub spaces_between_special_tokens: bool,
    /// Some(k)：返回每个 token 的 logprob 与 top-k 候选（k 可为 0 = 只要 chosen）。
    pub logprobs: Option<u32>,
    /// Some(k)：prefill 时对 prompt 每个位置（第 0 个为 None）返回 logprob + top-k（vLLM `prompt_logprobs`）。
    #[serde(default)]
    pub prompt_logprobs: Option<u32>,
    /// OpenAI `logit_bias`：(token_id, bias) 加到 logits（mask 之前、penalties 之前）。≤ 300 项。
    #[serde(default)]
    pub logit_bias: Vec<(u32, f32)>,
    /// 受限解码文法状态（response_format / tool_choice）；None = 不约束。不序列化（PD 导出的请求不带文法）。
    #[serde(skip)]
    pub grammar: Option<GrammarHandle>,
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
            spaces_between_special_tokens: true,
            logprobs: None,
            prompt_logprobs: None,
            logit_bias: Vec::new(),
            grammar: None,
        }
    }
}

impl SamplingParams {
    #[inline]
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 1e-5
    }
    /// 需要 logits mask / bias kernel（受限解码或 logit_bias）。
    #[inline]
    pub fn has_logits_mask(&self) -> bool {
        self.grammar.is_some() || !self.logit_bias.is_empty()
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
        if let Some(k) = self.prompt_logprobs {
            if k > 20 {
                return Err(format!("prompt_logprobs must be <= 20, got {k}"));
            }
        }
        if self.logit_bias.len() > 300 {
            return Err(format!("logit_bias has {} entries, at most 300", self.logit_bias.len()));
        }
        if let Some((t, b)) = self.logit_bias.iter().find(|(_, b)| !(-100.0..=100.0).contains(b)) {
            return Err(format!("logit_bias[{t}] = {b} out of [-100, 100]"));
        }
        Ok(())
    }
}
