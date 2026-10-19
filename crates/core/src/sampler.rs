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

    // ---- 受限解码 / logit_bias（docs/loops/guided-decoding.md）：无约束请求这些列全空，executor 不发任何额外 H2D ----
    /// 每行的 mask 行号：-1 无 mask，否则 `mask_bits[row * mask_words ..]`。
    pub mask_row: Vec<i32>,
    /// 紧凑的 bitmask 行（只含有约束的行），每行 `mask_words` 个 u32（bit=1 允许）。
    pub mask_bits: Vec<u32>,
    pub mask_words: usize,
    /// logit_bias CSR：`bias_ids[bias_offsets[b]..bias_offsets[b+1]]` / `bias_vals`。
    pub bias_offsets: Vec<i32>,
    pub bias_ids: Vec<i32>,
    pub bias_vals: Vec<f32>,
    /// 本步有任何行要 mask / bias（false 时 executor 不上传、eager 路径不发 kernel）。
    pub any_mask: bool,
    /// 每行要的 prompt_logprobs top-k（None = 不要）；只在该行是 prefill 行（本步含 prompt 位置）时有意义。
    pub prompt_logprobs: Vec<Option<u32>>,
    /// 每行的 prompt 长度（prompt_logprobs 用：位置 < num_prompt 的行才算 prompt 位置）。
    pub num_prompt: Vec<u32>,
    /// 本步 prompt_logprobs 的最大 k。
    pub max_prompt_logprobs: Option<u32>,
}

impl SampleBatch {
    pub fn clear(&mut self) {
        *self = Self::default();
        self.hist_offsets.push(0);
        self.bias_offsets.push(0);
    }

    pub fn num_rows(&self) -> usize {
        self.temperature.len()
    }

    /// 追加一行。`token_ids` 为该请求 prompt + 已产出 token；`num_prompt` 为 prompt 长度；`seed` 为请求固定种子。
    pub fn push(&mut self, sp: &SamplingParams, seed: u64, token_ids: &[u32], num_prompt: usize) {
        if self.hist_offsets.is_empty() {
            self.hist_offsets.push(0);
        }
        if self.bias_offsets.is_empty() {
            self.bias_offsets.push(0);
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
        // prompt_logprobs：只有 prefill 行（本步末尾还没走完 prompt，或刚好走完）要算
        let is_prefill_row = token_ids.len() <= num_prompt;
        let plp = sp.prompt_logprobs.filter(|_| is_prefill_row);
        self.prompt_logprobs.push(plp);
        self.num_prompt.push(num_prompt as u32);
        if let Some(k) = plp {
            self.max_prompt_logprobs = Some(self.max_prompt_logprobs.map_or(k, |m| m.max(k)));
            // prompt_logprobs 走 eager 的 logprobs 路径（raw logits 副本），让 executor / runner 按 "有 logprobs" 处理
            self.max_logprobs = Some(self.max_logprobs.map_or(k, |m| m.max(k)));
        }
        // logit_bias
        if !sp.logit_bias.is_empty() {
            self.bias_ids.extend(sp.logit_bias.iter().map(|&(t, _)| t as i32));
            self.bias_vals.extend(sp.logit_bias.iter().map(|&(_, b)| b));
            self.any_mask = true;
        }
        self.bias_offsets.push(self.bias_ids.len() as i32);
        // 受限解码：只有已进入生成阶段的行（token_ids 覆盖整个 prompt）才算 mask；chunked prefill 的中间 chunk 行不采样。
        let mut mask_row = -1;
        if let (Some(g), true) = (sp.grammar.as_ref(), token_ids.len() >= num_prompt) {
            let mut slot = g.0.lock();
            if let Some(m) = crate::sample::GrammarHandle::advance_and_mask(&mut slot, &token_ids[num_prompt..]) {
                let words = m.len();
                if self.mask_words == 0 {
                    self.mask_words = words;
                } else if self.mask_words != words {
                    // 同一 batch 里不同词表大小的文法（理论上不会：同一 tokenizer）：统一到较大者，短的补 0（不允许）
                    let w = self.mask_words.max(words);
                    if w != self.mask_words {
                        let rows = self.mask_bits.len() / self.mask_words;
                        let mut nb = vec![0u32; rows * w];
                        for r in 0..rows {
                            nb[r * w..r * w + self.mask_words]
                                .copy_from_slice(&self.mask_bits[r * self.mask_words..(r + 1) * self.mask_words]);
                        }
                        self.mask_bits = nb;
                        self.mask_words = w;
                    }
                }
                mask_row = (self.mask_bits.len() / self.mask_words) as i32;
                self.mask_bits.extend_from_slice(m);
                self.mask_bits.resize((mask_row as usize + 1) * self.mask_words, 0);
                self.any_mask = true;
            }
        }
        self.mask_row.push(mask_row);
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
    /// Some(p)：这是 prompt 第 p 个位置的 logprob（`prompt_logprobs`，p ≥ 1；p=0 无前文不算）；None = 生成 token。
    pub prompt_pos: Option<u32>,
    /// `prompt_logprobs`：挂在该步生成 token 的条目下（scheduler 的 logprobs 与 token 1:1 对齐，多出的条目会被丢），
    /// 每项 `prompt_pos = Some(p)`。server 展平。
    pub prompt: Vec<TokenLogprobs>,
}

impl TokenLogprobs {
    /// 只为承载 prompt_logprobs 的占位条目（该行没要生成 token 的 logprobs）：`token_id = u32::MAX`，server 丢弃。
    pub const PLACEHOLDER: u32 = u32::MAX;
    #[inline]
    pub fn is_placeholder(&self) -> bool {
        self.token_id == Self::PLACEHOLDER && self.prompt_pos.is_none()
    }
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
        assert!(!sb.any_mask && sb.mask_row == vec![-1, -1] && sb.bias_offsets == vec![0, 0, 0]);
    }

    /// 两个允许 token 的玩具文法：先只允许 a，消费 a 后只允许 b，再之后只允许 eos(=0)。
    struct Toy {
        step: usize,
    }
    impl crate::sample::GrammarState for Toy {
        fn mask_words(&self) -> usize {
            2
        }
        fn fill_mask(&mut self, out: &mut [u32]) -> anyhow::Result<()> {
            out.fill(0);
            let allow = [40u32, 41, 0][self.step.min(2)];
            out[(allow / 32) as usize] |= 1 << (allow % 32);
            Ok(())
        }
        fn consume_token(&mut self, tok: u32) -> anyhow::Result<()> {
            anyhow::ensure!(tok == [40u32, 41][self.step.min(1)], "bad token {tok}");
            self.step += 1;
            Ok(())
        }
        fn reset(&mut self) -> anyhow::Result<()> {
            self.step = 0;
            Ok(())
        }
    }

    #[test]
    fn grammar_mask_and_bias_columns() {
        let g = crate::sample::GrammarHandle::new(Box::new(Toy { step: 0 }));
        let sp = SamplingParams { grammar: Some(g.clone()), logit_bias: vec![(7, -100.0)], ..Default::default() };
        let plain = SamplingParams::default();
        let mut sb = SampleBatch::default();
        sb.clear();
        sb.push(&plain, 1, &[9, 9], 2);
        // 中间 chunk（token_ids 没覆盖 prompt）：不 mask，但 bias 仍加
        sb.push(&sp, 2, &[1, 2], 3);
        // 刚走完 prompt：输出空 → mask 允许 40
        sb.push(&sp, 2, &[1, 2, 3], 3);
        assert_eq!(sb.mask_row, vec![-1, -1, 0]);
        assert_eq!(sb.mask_words, 2);
        assert_eq!(sb.mask_bits, vec![0, 1 << 8]);
        assert_eq!(sb.bias_offsets, vec![0, 0, 1, 2]);
        assert_eq!(sb.bias_ids, vec![7, 7]);
        assert!(sb.any_mask);
        // 下一步：输出 [40] → 追赶 consume(40) → 允许 41
        let mut sb2 = SampleBatch::default();
        sb2.clear();
        sb2.push(&sp, 2, &[1, 2, 3, 40], 3);
        assert_eq!(sb2.mask_bits, vec![0, 1 << 9]);
        assert_eq!(g.0.lock().consumed, 1);
        // 回滚（输出变短）→ reset 重放
        let mut sb3 = SampleBatch::default();
        sb3.clear();
        sb3.push(&sp, 2, &[1, 2, 3], 3);
        assert_eq!(sb3.mask_bits, vec![0, 1 << 8]);
        // 非法 token → failed，不再 mask（退化为无约束）
        let mut sb4 = SampleBatch::default();
        sb4.clear();
        sb4.push(&sp, 2, &[1, 2, 3, 99], 3);
        assert_eq!(sb4.mask_row, vec![-1]);
        assert!(sb4.any_mask, "bias 仍在");
        assert!(g.0.lock().failed);
    }

    #[test]
    fn prompt_logprobs_only_on_prefill_rows() {
        let sp = SamplingParams { prompt_logprobs: Some(2), ..Default::default() };
        let mut sb = SampleBatch::default();
        sb.clear();
        sb.push(&sp, 1, &[1, 2, 3], 3); // prefill 末尾
        sb.push(&sp, 1, &[1, 2, 3, 4], 3); // decode
        assert_eq!(sb.prompt_logprobs, vec![Some(2), None]);
        assert_eq!(sb.num_prompt, vec![3, 3]);
        assert_eq!(sb.max_prompt_logprobs, Some(2));
        assert_eq!(sb.max_logprobs, Some(2));
    }
}
