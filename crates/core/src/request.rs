//! 请求生命周期。对应 vLLM `v1/request.py`。

use crate::sample::SamplingParams;

pub type RequestId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    Abort,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestStatus {
    Waiting,
    Running,
    Preempted,
    Finished(FinishReason),
}

#[derive(Debug)]
pub struct Request {
    pub id: RequestId,
    pub prompt_ids: Vec<u32>,
    pub sampling: SamplingParams,
    pub status: RequestStatus,
    /// 已经算过 KV 的 token 数（含 prefix cache 命中）。
    pub num_computed_tokens: u32,
    pub output_ids: Vec<u32>,
    /// 异步调度下已调度但尚未拿到采样结果的 token 数。
    pub num_output_placeholders: u32,
    /// 当前 step 的 draft token（投机解码）。
    pub spec_ids: Vec<u32>,
    pub arrival_ns: u64,
}

impl Request {
    pub fn new(id: RequestId, prompt_ids: Vec<u32>, sampling: SamplingParams, arrival_ns: u64) -> Self {
        Self {
            id,
            prompt_ids,
            sampling,
            status: RequestStatus::Waiting,
            num_computed_tokens: 0,
            output_ids: Vec::new(),
            num_output_placeholders: 0,
            spec_ids: Vec::new(),
            arrival_ns,
        }
    }

    #[inline]
    pub fn num_prompt_tokens(&self) -> u32 {
        self.prompt_ids.len() as u32
    }

    /// 总 token 数 = prompt + 已产出 + 占位。
    #[inline]
    pub fn num_tokens(&self) -> u32 {
        self.num_prompt_tokens() + self.output_ids.len() as u32 + self.num_output_placeholders
    }

    /// 本 step 还需要调度的 token 数（含 spec）。
    #[inline]
    pub fn num_tokens_with_spec(&self) -> u32 {
        self.num_tokens() + self.spec_ids.len() as u32
    }

    #[inline]
    pub fn is_finished(&self) -> bool {
        matches!(self.status, RequestStatus::Finished(_))
    }

    /// 还在 prefill 阶段（prompt 没算完）。
    #[inline]
    pub fn is_prefilling(&self) -> bool {
        self.num_computed_tokens < self.num_prompt_tokens()
    }
}
