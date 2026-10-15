//! 调度输出 / 一步结果。对应 vLLM `v1/core/sched/output.py` 与 `EngineCoreOutput` 的核心字段。

use crate::kv::{BlockId, KvBlocks};
use crate::request::{FinishReason, RequestId};
use crate::sample::SamplingParams;

/// 首次调度的请求：worker 需要全量数据。
#[derive(Debug, Clone)]
pub struct NewRequestData {
    pub req_id: RequestId,
    pub prompt_ids: Vec<u32>,
    pub sampling: SamplingParams,
    pub block_ids: KvBlocks,
    pub num_computed_tokens: u32,
}

/// 老请求：只传增量。
#[derive(Debug, Clone, Default)]
pub struct CachedRequestData {
    pub req_ids: Vec<RequestId>,
    /// 抢占后恢复的请求：`new_block_ids` 是全量而非增量。
    pub resumed: Vec<bool>,
    pub new_block_ids: Vec<Option<KvBlocks>>,
    pub num_computed_tokens: Vec<u32>,
}

#[derive(Debug, Clone, Default)]
pub struct SchedulerOutput {
    pub new_reqs: Vec<NewRequestData>,
    pub cached_reqs: CachedRequestData,
    /// (request, 本 step 调度的 token 数)，按 batch 顺序。
    pub num_scheduled_tokens: Vec<(RequestId, u32)>,
    pub total_num_scheduled_tokens: u32,
    /// req → 本 step 要校验的 draft token。
    pub scheduled_spec_tokens: Vec<(RequestId, Vec<u32>)>,
    /// 本 step 之前完成的请求（worker 释放状态）。
    pub finished_req_ids: Vec<RequestId>,
    pub preempted_req_ids: Vec<RequestId>,
    /// 新分配、需要清零的块（Mamba 状态）。
    pub blocks_to_zero: Vec<BlockId>,
}

impl SchedulerOutput {
    pub fn num_reqs(&self) -> usize {
        self.num_scheduled_tokens.len()
    }
    pub fn scheduled_tokens_of(&self, id: RequestId) -> Option<u32> {
        self.num_scheduled_tokens.iter().find(|(r, _)| *r == id).map(|(_, n)| *n)
    }
}

/// worker 一步的采样结果（按 `num_scheduled_tokens` 的顺序）。
#[derive(Debug, Clone, Default)]
pub struct ModelRunnerOutput {
    pub req_ids: Vec<RequestId>,
    /// 每请求本 step 产出的 token（prefill 中间 chunk 为空；投机解码为 1 + 接受数）。
    pub sampled: Vec<Vec<u32>>,
    /// 每请求下一步的 draft token（MTP 产出；无投机时为空）。与 req_ids 平行。
    pub drafts: Vec<Vec<u32>>,
}

/// `update_from_output` 的结果：扁平 token 缓冲 + 每请求一条记录（零小对象分配）。
#[derive(Debug, Clone, Default)]
pub struct StepResults {
    pub items: Vec<StepItem>,
    pub tokens: Vec<u32>,
}

#[derive(Debug, Clone, Copy)]
pub struct StepItem {
    pub req_id: RequestId,
    tok_start: u32,
    tok_len: u32,
    pub finish: Option<FinishReason>,
}

impl StepResults {
    pub fn with_capacity(n: usize) -> Self {
        Self { items: Vec::with_capacity(n), tokens: Vec::with_capacity(n * 2) }
    }
    #[inline]
    pub fn push(&mut self, req_id: RequestId, toks: &[u32], finish: Option<FinishReason>) {
        let tok_start = self.tokens.len() as u32;
        self.tokens.extend_from_slice(toks);
        self.items.push(StepItem { req_id, tok_start, tok_len: toks.len() as u32, finish });
    }
    #[inline]
    pub fn tokens_of(&self, i: usize) -> &[u32] {
        let it = &self.items[i];
        &self.tokens[it.tok_start as usize..(it.tok_start + it.tok_len) as usize]
    }
    pub fn iter(&self) -> impl Iterator<Item = (RequestId, &[u32], Option<FinishReason>)> + '_ {
        self.items.iter().map(move |it| {
            (it.req_id, &self.tokens[it.tok_start as usize..(it.tok_start + it.tok_len) as usize], it.finish)
        })
    }
    pub fn len(&self) -> usize {
        self.items.len()
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}
