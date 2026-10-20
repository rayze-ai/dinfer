//! 请求生命周期。对应 vLLM `v1/request.py`。

use crate::kv::BlockHasher;
use crate::sample::SamplingParams;

pub type RequestId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// 命中 eos 或 stop token。
    Stop,
    /// 达到 max_tokens 或 max_model_len。
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
    pub sampling: SamplingParams,
    pub status: RequestStatus,
    pub arrival_ns: u64,
    pub priority: i32,

    /// prompt + output 的完整 token 序列（prefix hash 基于它）。
    all_token_ids: Vec<u32>,
    num_prompt_tokens: u32,

    /// 已经算过 KV 的 token 数（含 prefix cache 命中、含异步调度的乐观推进）。
    pub num_computed_tokens: u32,
    /// 首次调度时由 prefix cache（GPU 块 + host tier）命中的 prompt token 数；对外报 `usage.prompt_tokens_details.cached_tokens`
    /// （vLLM `RequestOutput.num_cached_tokens` 同义）。
    pub num_cached_tokens: u32,
    /// 异步调度：已调度但采样结果未回的 token 数。
    pub num_output_placeholders: u32,
    /// 当前 step 的 draft token（投机解码）。
    pub spec_ids: Vec<u32>,
    /// prefix cache 用的链式块哈希。
    pub block_hashes: BlockHasher,
    pub num_preemptions: u32,
    /// PD 分离角色（design §10.3）。`None` = 普通单机请求。
    pub pd: Option<PdRole>,
}

/// PD 分离里请求的角色。
#[derive(Debug, Clone, PartialEq)]
pub enum PdRole {
    /// prefill 实例：首 token 采样后把该请求全部 KV 块导出（finish 时推 `kv_export`，先导出再 free）。
    Export,
    /// decode 实例：KV 已由 P 算好；首次调度只分块 + `kv_import`，不进 forward，
    /// `num_computed = prompt_len`，`first_token` 直接作为第一个输出。
    Import { first_token: u32 },
}

impl Request {
    pub fn new(id: RequestId, prompt_ids: Vec<u32>, sampling: SamplingParams, arrival_ns: u64) -> Self {
        let num_prompt_tokens = prompt_ids.len() as u32;
        Self {
            id,
            sampling,
            status: RequestStatus::Waiting,
            arrival_ns,
            priority: 0,
            all_token_ids: prompt_ids,
            num_prompt_tokens,
            num_computed_tokens: 0,
            num_cached_tokens: 0,
            num_output_placeholders: 0,
            spec_ids: Vec::new(),
            block_hashes: BlockHasher::default(),
            num_preemptions: 0,
            pd: None,
        }
    }

    /// 启用 prefix cache 哈希（由 Scheduler 在 add_request 时调用）。
    pub fn init_block_hasher(&mut self, hash_block_size: u32, salt: Vec<u8>) {
        self.block_hashes = BlockHasher::new(hash_block_size, salt);
        self.block_hashes.update(&self.all_token_ids);
    }

    #[inline]
    pub fn block_hashes_block_size(&self) -> u32 {
        self.block_hashes.hash_block_size().max(1)
    }

    #[inline]
    pub fn all_token_ids(&self) -> &[u32] {
        &self.all_token_ids
    }
    #[inline]
    pub fn prompt_ids(&self) -> &[u32] {
        &self.all_token_ids[..self.num_prompt_tokens as usize]
    }
    #[inline]
    pub fn output_ids(&self) -> &[u32] {
        &self.all_token_ids[self.num_prompt_tokens as usize..]
    }
    #[inline]
    pub fn num_prompt_tokens(&self) -> u32 {
        self.num_prompt_tokens
    }
    #[inline]
    pub fn num_output_tokens(&self) -> u32 {
        self.all_token_ids.len() as u32 - self.num_prompt_tokens
    }
    /// prompt + 已产出 output（不含占位、不含 spec）。
    #[inline]
    pub fn num_tokens(&self) -> u32 {
        self.all_token_ids.len() as u32
    }
    /// num_tokens + spec。
    #[inline]
    pub fn num_tokens_with_spec(&self) -> u32 {
        self.num_tokens() + self.spec_ids.len() as u32
    }
    #[inline]
    pub fn is_finished(&self) -> bool {
        matches!(self.status, RequestStatus::Finished(_))
    }
    /// 仍在 prefill（包含 chunked prefill 的中间 step）。
    #[inline]
    pub fn is_prefill_chunk(&self) -> bool {
        self.num_computed_tokens < self.num_tokens() + self.num_output_placeholders
    }

    pub fn append_output(&mut self, tok: u32) {
        self.all_token_ids.push(tok);
        let hbs = self.block_hashes.hash_block_size();
        // 只在恰好凑满一个 hash 块时才算（热路径：decode 每 step 一次）。
        if hbs > 0 && self.all_token_ids.len() as u32 % hbs == 0 {
            self.block_hashes.update(&self.all_token_ids);
        }
    }

    /// 本 step 是否跨过了一个 hash 块边界（用于决定要不要走 cache_blocks）。
    #[inline]
    pub fn crossed_hash_block_boundary(&self, before_len: u32) -> bool {
        let hbs = self.block_hashes.hash_block_size();
        hbs > 0 && before_len / hbs != self.num_tokens() / hbs
    }

    /// 抢占：清空计算进度，回到 waiting。
    pub fn reset_for_preemption(&mut self) {
        self.status = RequestStatus::Preempted;
        self.num_computed_tokens = 0;
        self.num_output_placeholders = 0;
        self.spec_ids.clear();
        self.num_preemptions += 1;
    }
}
