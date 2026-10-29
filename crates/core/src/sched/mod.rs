//! 调度器。对应 vLLM `v1/core/sched/scheduler.py` 的核心算法（`schedule:591`、
//! `_preempt_request:1571`、`_update_after_schedule:1616`、`update_from_output:2026`）。
//!
//! 与 vLLM 一致的要点：
//! - 没有 prefill/decode 阶段之分，只有 `num_computed_tokens` 追赶 `num_tokens_with_spec`。
//! - 先 running 再 waiting；running 分配失败则抢占 **末尾** running（FCFS）。
//! - chunked prefill：按 token 预算截断；`long_prefill_token_threshold` 防饿死。
//! - 异步调度：调度后立即把 `num_computed_tokens` 推进，用 `num_output_placeholders`
//!   占位，采样结果回来后再修正（拒绝的 draft 回退）。
//! - prefix cache：waiting 请求首次调度时查 `get_computed_blocks`。
//!
//! 性能：请求放在 slab（`Vec<Option<Slot>>`）里，running/waiting 存 slot 下标，
//! 热路径不查 HashMap；`RequestId → slot` 只在 add / update_from_output 入口用（恒等哈希）。
//!
//! 不做：encoder 预算、LoRA、KV connector、structured output、PP。

pub mod output;

pub use output::{
    CachedRequestData, ModelRunnerOutput, NewRequestData, SchedulerOutput, StepItem, StepResults,
};

use crate::config::{CacheConfig, SchedulerConfig};
use crate::kv::{GroupInfo, KvBlocks, KvCacheManager};
use crate::request::{FinishReason, PdRole, Request, RequestId, RequestStatus};
use crate::util::{id_hash_map, IdHashMap};
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct SchedulerParams {
    pub sched: SchedulerConfig,
    pub cache: CacheConfig,
    pub max_model_len: u32,
    /// 投机解码每 step 的 draft 数（0 = 关闭）。
    pub num_spec_tokens: u32,
    /// 是否启用异步调度（占位推进）。
    pub async_scheduling: bool,
    /// eos / stop token 集合（模型给）。
    pub eos_token_ids: Vec<u32>,
    /// 递归状态（GDN / KDA / Mamba）快照粒度（token 数，design §15.7）；0 = 关（非递归模型）。
    /// > 0 时：prefix cache 命中长度向下对齐到该粒度；prefill chunk 边界落在对齐点；跑过对齐点的 prefill 步登记快照。
    /// 必须是 `cache.block_size` 的整数倍。
    pub recurrent_snapshot_tokens: u32,
    /// 快照槽数（显存由模型侧分配；0 = 无快照池：命中仍对齐，但没有快照可续 → 整段重算）。
    pub recurrent_snapshot_slots: u32,
    /// 递归 prefill kernel 的 chunk 大小（token；GDN chunk prefill = 64）。对齐点相对本步段首的偏移必须是它的倍数
    /// kernel 才能在 chunk 边界顺带写快照；不是倍数（预算截断把段首切到非对齐处）时该步在对齐点结束（退回切步）。
    /// 0 = 不约束。
    pub recurrent_snapshot_chunk: u32,
}

/// 对齐位置 `pos`（快照粒度的整数倍、> 0）处的块 hash：`block_hashes[pos / hash_block_size - 1]`。
#[inline]
fn snapshot_hash_at(r: &Request, pos: u32) -> Option<crate::kv::BlockHash> {
    if pos == 0 {
        return None;
    }
    let hbs = r.block_hashes_block_size();
    r.block_hashes.hashes.get((pos / hbs) as usize - 1).copied()
}

#[derive(Debug)]
struct Slot {
    req: Request,
    /// worker 是否已经见过（决定走 new_reqs 还是 cached_reqs）。
    seen_by_worker: bool,
}

#[derive(Debug)]
pub struct Scheduler {
    p: SchedulerParams,
    kv: KvCacheManager,
    slab: Vec<Option<Slot>>,
    free_slots: Vec<usize>,
    id_to_slot: IdHashMap<usize>,
    waiting: VecDeque<usize>,
    running: Vec<usize>,
    /// 与 `SchedulerOutput::num_scheduled_tokens` 平行的 slot 列表（内部用，避免再查表）。
    sched_slots: Vec<usize>,
    finished_req_ids: Vec<RequestId>,
    preempted_req_ids: Vec<RequestId>,
    /// prefix cache 统计（token 数，vLLM 口径）：首次调度时查询 `num_tokens-1`，命中 `num_new_computed`。
    prefix_cache_queries: u64,
    prefix_cache_hits: u64,
    /// host tier 拉回的块数（指标）。
    host_tier_fetched: u64,
    /// scheduler 自行结束的请求（分不到块且无人可抢占）：engine 每步取走并推 Finished 事件。
    force_finished: Vec<(RequestId, FinishReason)>,
    /// PD：Export 请求 finish 时先推这里，下一次 `schedule()` 搬进 `out.kv_export`（runner 在 begin 最前面导出）。
    pd_pending_export: Vec<(RequestId, KvBlocks)>,
    /// 递归状态快照池（design §15.7；`recurrent_snapshot_tokens > 0 && slots > 0` 时存在）。
    snapshots: Option<crate::kv::StateSnapshotPool>,
    /// 统计：命中但快照已淘汰 → 整段重算的请求数；池满无槽可写的快照数。
    snapshot_resume_misses: u64,
    snapshot_write_skipped: u64,
}

impl Scheduler {
    pub fn new(p: SchedulerParams, groups: Vec<GroupInfo>, num_blocks: u32) -> Self {
        let kv = KvCacheManager::new(groups, num_blocks, p.cache.enable_prefix_caching, p.max_model_len);
        let cap = p.sched.max_num_seqs as usize * 2;
        if p.recurrent_snapshot_tokens > 0 {
            assert!(
                p.recurrent_snapshot_tokens % p.cache.block_size.max(1) == 0
                    && p.recurrent_snapshot_tokens % p.cache.hash_block_size.max(1) == 0,
                "recurrent_snapshot_tokens {} must be a multiple of block_size {} / hash_block_size {}",
                p.recurrent_snapshot_tokens,
                p.cache.block_size,
                p.cache.hash_block_size
            );
        }
        let snapshots = (p.recurrent_snapshot_tokens > 0 && p.recurrent_snapshot_slots > 0)
            .then(|| crate::kv::StateSnapshotPool::new(p.recurrent_snapshot_slots as usize));
        Self {
            p,
            kv,
            slab: Vec::with_capacity(cap),
            free_slots: Vec::new(),
            id_to_slot: id_hash_map(cap),
            waiting: VecDeque::new(),
            running: Vec::with_capacity(cap),
            sched_slots: Vec::with_capacity(cap),
            finished_req_ids: Vec::new(),
            preempted_req_ids: Vec::new(),
            prefix_cache_queries: 0,
            prefix_cache_hits: 0,
            host_tier_fetched: 0,
            force_finished: Vec::new(),
            pd_pending_export: Vec::new(),
            snapshots,
            snapshot_resume_misses: 0,
            snapshot_write_skipped: 0,
        }
    }

    /// 递归状态快照池统计：(lookup 命中, lookup 未命中, 淘汰, 命中后无快照整段重算, 池满放弃写)。
    pub fn snapshot_stats(&self) -> (u64, u64, u64, u64, u64) {
        let (h, m, e) = self.snapshots.as_ref().map_or((0, 0, 0), |p| (p.hits, p.misses, p.evictions));
        (h, m, e, self.snapshot_resume_misses, self.snapshot_write_skipped)
    }

    /// r8b：一步可跨任意多个对齐点，快照由 prefill kernel 在 chunk 边界顺带写出（design §15.7 "r8b 实现"），
    /// 调度**不再**按粒度切步。唯一例外：某个对齐点相对本步段首 `computed` 的偏移不是 kernel chunk 的倍数
    /// （粒度本身不是 chunk 的倍数，或上一步被预算截断在非对齐处）→ kernel 的 chunk 边界对不上该点，本步在该对齐点
    /// 结束（它成为段末，kernel 末态即快照）。粒度是 chunk 的倍数且段首对齐时永不切。
    #[inline]
    fn cap_to_snapshot_grain(&self, computed: u32, num_new: u32, num_prompt: u32) -> u32 {
        let g = self.p.recurrent_snapshot_tokens;
        let c = self.p.recurrent_snapshot_chunk;
        if g == 0 || c == 0 || self.snapshots.is_none() || num_new <= 1 || computed >= num_prompt {
            return num_new;
        }
        let end = (computed + num_new).min(num_prompt);
        let mut b = (computed / g + 1) * g;
        while b < end {
            if (b - computed) % c != 0 {
                return b - computed;
            }
            b += g;
        }
        num_new
    }

    /// 本步 `(computed, computed_after]` 内落在 prompt 内的每个对齐点各登记一个快照槽（已有同 hash 的快照则不重写）：
    /// `out.recurrent_snapshot (req, 对齐位置, 槽)`。位置 < computed_after 的由 prefill kernel 在 chunk 边界写出，
    /// == computed_after 的也可由 executor 步末 dtod（回退路径）。
    fn register_snapshots(&mut self, slot: usize, computed: u32, computed_after: u32, out: &mut SchedulerOutput) {
        let g = self.p.recurrent_snapshot_tokens;
        if g == 0 || computed_after == 0 {
            return;
        }
        let r = &self.slab[slot].as_ref().unwrap().req;
        let rid = r.id;
        let last = computed_after.min(r.num_prompt_tokens());
        let mut pos = (computed / g + 1) * g;
        while pos <= last {
            let Some(hash) = snapshot_hash_at(r, pos) else { break };
            let Some(pool) = self.snapshots.as_mut() else { return };
            if !pool.contains(hash) {
                match pool.insert(hash) {
                    Some(s) => {
                        pool.pin(s); // 本步内不许被后来的 insert 淘汰
                        out.recurrent_snapshot.push((rid, pos, s as u32));
                    }
                    None => self.snapshot_write_skipped += 1,
                }
            }
            pos += g;
        }
    }

    /// 取走本步 scheduler 自行结束的请求（见 running 循环的不自抢占分支）。
    pub fn take_force_finished(&mut self) -> Vec<(RequestId, FinishReason)> {
        std::mem::take(&mut self.force_finished)
    }

    /// 启用 KV host tier（槽位数由调用方按 host_gib / bytes_per_block 算）。
    pub fn enable_host_tier(&mut self, num_slots: usize, policy: Box<dyn crate::kv::host_tier::HostTierPolicy>) {
        self.kv.enable_host_tier(num_slots, policy);
    }
    /// (host 命中块数累计, 拉回块数, 推出块数, 淘汰数)
    pub fn host_tier_stats(&self) -> (u64, u64, u64, u64) {
        let s = self.kv.host_tier_stats();
        (s.hits_blocks, self.host_tier_fetched, s.offload_blocks, s.evictions)
    }

    pub fn add_request(&mut self, mut req: Request) {
        if self.p.cache.enable_prefix_caching {
            req.init_block_hasher(self.p.cache.hash_block_size, Vec::new());
        }
        let id = req.id;
        let slot = Slot { req, seen_by_worker: false };
        let idx = match self.free_slots.pop() {
            Some(i) => {
                self.slab[i] = Some(slot);
                i
            }
            None => {
                self.slab.push(Some(slot));
                self.slab.len() - 1
            }
        };
        self.id_to_slot.insert(id, idx);
        self.waiting.push_back(idx);
    }

    #[inline]
    fn req(&self, slot: usize) -> &Request {
        &self.slab[slot].as_ref().unwrap().req
    }
    #[inline]
    fn req_mut(&mut self, slot: usize) -> &mut Request {
        &mut self.slab[slot].as_mut().unwrap().req
    }

    #[inline]
    pub fn has_requests(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }
    /// PD：有 Export 请求刚 finish、块还没导出。engine 据此不能进空闲等待（导出发生在下一次 schedule）。
    pub fn has_pending_export(&self) -> bool {
        !self.pd_pending_export.is_empty()
    }
    pub fn num_running(&self) -> usize {
        self.running.len()
    }
    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }
    pub fn num_free_blocks(&self) -> u32 {
        self.kv.num_free_blocks()
    }
    pub fn kv_usage(&self) -> f32 {
        self.kv.usage()
    }
    /// (queries, hits) 累计 token 数。
    pub fn prefix_cache_stats(&self) -> (u64, u64) {
        (self.prefix_cache_queries, self.prefix_cache_hits)
    }
    pub fn request(&self, id: RequestId) -> Option<&Request> {
        self.id_to_slot.get(&id).map(|&s| self.req(s))
    }

    /// 测试用：改每步 draft 数（lookahead）。
    #[doc(hidden)]
    pub fn set_num_spec_tokens_for_test(&mut self, n: u32) {
        self.p.num_spec_tokens = n;
    }

    /// 设置下一步的 draft token（由 worker 的 drafter 产出）。
    pub fn update_draft_tokens(&mut self, drafts: &[(RequestId, Vec<u32>)]) {
        for (id, d) in drafts {
            if let Some(&s) = self.id_to_slot.get(id) {
                let r = self.req_mut(s);
                if r.status == RequestStatus::Running {
                    r.spec_ids.clear();
                    r.spec_ids.extend_from_slice(d);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // schedule
    // ------------------------------------------------------------------

    pub fn schedule(&mut self) -> SchedulerOutput {
        self.schedule_impl(false)
    }

    /// 只调度 waiting 里的新请求（跳过 running），预算全给它们；running 行一个都不动。
    /// 用于插队 prefill（Q3-o）：在飞 decode 步未结束时为新请求单独跑一步。
    pub fn schedule_waiting_only(&mut self) -> SchedulerOutput {
        self.schedule_impl(true)
    }

    fn schedule_impl(&mut self, waiting_only: bool) -> SchedulerOutput {
        let mut out = SchedulerOutput {
            finished_req_ids: std::mem::take(&mut self.finished_req_ids),
            preempted_req_ids: std::mem::take(&mut self.preempted_req_ids),
            ..Default::default()
        };
        let cap = self.running.len() + 8;
        out.num_scheduled_tokens.reserve(cap);
        out.cached_reqs.req_ids.reserve(cap);
        out.cached_reqs.resumed.reserve(cap);
        out.cached_reqs.new_block_ids.reserve(cap);
        out.cached_reqs.num_computed_tokens.reserve(cap);
        self.sched_slots.clear();
        if let Some(pool) = self.snapshots.as_mut() {
            pool.unpin_all(); // 上一步的 resume / snapshot 已入流
        }

        let mut token_budget = self.p.sched.max_num_batched_tokens;
        // B47：整步 prefill 新 token 配额（0 = 不限）。只约束 prefill 行（含 running 里的 chunk 续算与 waiting 新请求），decode 行不计。
        let mut prefill_quota =
            if self.p.sched.prefill_tokens_per_step > 0 { self.p.sched.prefill_tokens_per_step } else { u32::MAX };
        let num_lookahead = self.p.num_spec_tokens;
        let max_num_seqs = self.p.sched.max_num_seqs as usize;
        let max_model_len = self.p.max_model_len;

        // long_prefill_token_threshold：只有多于 1 个候选请求时才生效（防饿死）。
        let num_eligible = self.running.len() + self.waiting.len();
        let long_prefill_threshold =
            if num_eligible > 1 { self.p.sched.long_prefill_token_threshold } else { 0 };

        let mut any_preempted = false;

        // ---- 1. running ----（waiting_only 时整段跳过）
        let exempt_decode = self.p.sched.decode_exempt_from_budget;
        let mut i = 0;
        while !waiting_only && i < self.running.len() {
            let slot = self.running[i];
            let r = &self.slab[slot].as_ref().unwrap().req;
            // decode 行（算 1 个 token 且已有历史）在 exempt 模式下不占预算；其余行预算用完就停
            let is_decode = r.num_tokens_with_spec() + r.num_output_placeholders - r.num_computed_tokens == 1
                && r.num_computed_tokens > 0;
            let charge_budget = !(exempt_decode && is_decode);
            if charge_budget && token_budget == 0 {
                break;
            }

            // 异步调度：上一步已到 max_tokens，不再多调度一步。
            if r.num_output_placeholders > 0
                && r.num_computed_tokens + 2 - r.num_output_placeholders
                    >= r.num_prompt_tokens() + r.sampling.max_tokens
            {
                i += 1;
                continue;
            }

            let mut num_new = r.num_tokens_with_spec() + r.num_output_placeholders - r.num_computed_tokens;
            if long_prefill_threshold > 0 && num_new > long_prefill_threshold {
                num_new = long_prefill_threshold;
            }
            if charge_budget {
                num_new = num_new.min(token_budget);
            }
            if !is_decode {
                num_new = num_new.min(prefill_quota);
                if num_new == 0 {
                    i += 1;
                    continue; // 本步 prefill 配额用完：这条留到下一步（不抢占、不出队）
                }
            }
            // 不超过 max_model_len（留出本 step 要采样的 1 个 token）。
            num_new = num_new.min(max_model_len.saturating_sub(r.num_computed_tokens + 1));
            num_new = self.cap_to_snapshot_grain(r.num_computed_tokens, num_new, r.num_prompt_tokens());
            if num_new == 0 {
                i += 1;
                continue;
            }

            // 分配；失败则抢占末尾 running。**不抢占自己**：若末尾就是本请求（它是唯一 / 最后一个 running）
            // 且仍分不到块，说明池里没有别人能让 → 以 Length 结束它（否则它会被抢占→重排→再分不到→空步，
            // 2026-10-17 host tier A/B 撞到的 livelock；vLLM 靠启动期 max_model_len ≤ 池容量校验避免，我们两个都做）。
            let new_blocks = loop {
                let r = &self.slab[slot].as_ref().unwrap().req;
                if let Some(b) = self.kv.allocate_slots(slot, r, num_new, 0, None, num_lookahead) {
                    break Some(b);
                }
                let victim = *self.running.last().unwrap();
                if victim == slot {
                    // 末尾就是自己。若还有别的 running（排在前面、本步已拿到块），本步跳过我、不抢占：
                    // 下一步别人释放 / 完成后再试（vLLM 同样 "不自抢占"）。
                    if self.running.len() > 1 {
                        break None;
                    }
                    // 真正只剩我一个且仍分不到块：池装不下这个请求，只能以 Length 结束（不然就是 livelock）
                    let rid = r.id;
                    tracing::warn!(
                        req = rid,
                        tokens = r.num_tokens(),
                        free_blocks = self.kv.num_free_blocks(),
                        "sole running request cannot get a KV block: finishing with Length"
                    );
                    self.req_mut(slot).status = RequestStatus::Finished(FinishReason::Length);
                    self.finish_slot(slot);
                    self.force_finished.push((rid, FinishReason::Length));
                    break None;
                }
                self.running.pop();
                self.preempt(victim);
                any_preempted = true;
            };
            let Some(new_blocks) = new_blocks else { break };

            let r = &mut self.slab[slot].as_mut().unwrap().req;
            let rid = r.id;
            out.num_scheduled_tokens.push((rid, num_new));
            self.sched_slots.push(slot);
            if !is_decode {
                prefill_quota -= num_new;
            }
            if charge_budget {
                token_budget -= num_new;
            }
            if !r.spec_ids.is_empty() {
                // 这一步覆盖到的 spec 数 = num_new + computed - num_tokens - placeholders
                let n_spec = (num_new + r.num_computed_tokens)
                    .saturating_sub(r.num_tokens() + r.num_output_placeholders)
                    as usize;
                if n_spec > 0 {
                    let mut s = std::mem::take(&mut r.spec_ids);
                    s.truncate(n_spec);
                    out.scheduled_spec_tokens.push((rid, s));
                } else {
                    r.spec_ids.clear();
                }
            }
            let computed_before = r.num_computed_tokens;
            let computed_after = computed_before + num_new;
            out.cached_reqs.req_ids.push(rid);
            out.cached_reqs.resumed.push(false);
            out.cached_reqs.new_block_ids.push(if new_blocks.is_empty() { None } else { Some(new_blocks) });
            out.cached_reqs.num_computed_tokens.push(computed_after);
            self.register_snapshots(slot, computed_before, computed_after, &mut out);
            i += 1;
        }

        // ---- 2. waiting（本 step 有抢占则不接新请求）----
        if !any_preempted {
            while token_budget > 0 {
                let Some(&slot) = self.waiting.front() else { break };
                if self.running.len() >= max_num_seqs {
                    break;
                }
                let r = &self.slab[slot].as_ref().unwrap().req;
                // PD Import（design §10.3）：KV 由 P 算好，首次调度只分块 + 导入，不进 forward、不占 token 预算。
                if let (Some(PdRole::Import { first_token }), 0) = (&r.pd, r.num_computed_tokens) {
                    let first_token = *first_token;
                    let prompt_len = r.num_tokens();
                    let Some(new_blocks) = self.kv.allocate_slots(slot, r, prompt_len, 0, None, 0) else {
                        break; // 块不够：等下一 step
                    };
                    self.waiting.pop_front();
                    let s = self.slab[slot].as_mut().unwrap();
                    let r = &mut s.req;
                    let rid = r.id;
                    r.status = RequestStatus::Running;
                    r.num_computed_tokens = prompt_len;
                    r.append_output(first_token);
                    self.running.push(slot);
                    out.kv_import.push((rid, new_blocks.clone()));
                    // 满块登记哈希：后续同前缀请求可 GPU 命中（与单机 prefill 后的 cache_blocks 等价）
                    let r_ro: &Request = r;
                    self.kv.cache_blocks(slot, r_ro, prompt_len);
                    debug_assert!(!s.seen_by_worker);
                    s.seen_by_worker = true;
                    out.new_reqs.push(NewRequestData {
                        req_id: rid,
                        prompt_ids: r_ro.prompt_ids().to_vec(),
                        sampling: r_ro.sampling.clone(),
                        block_ids: new_blocks,
                        num_computed_tokens: prompt_len,
                        known_output: vec![first_token],
                        resume_from: None,
                    });
                    continue;
                }

                // prefix cache（只在首次调度，即 num_computed_tokens == 0 时查）。
                let (mut computed, mut num_new_computed) = self.kv.get_computed_blocks(r);
                // 递归状态按块缓存（design §15.7）：命中长度向下对齐到快照粒度，且该对齐位置必须有状态快照，
                // 否则递归层无法续算 → 整段重算（KV 块命中作废）。
                let mut resume: Option<(u32, crate::kv::BlockHash, usize)> = None;
                if self.p.recurrent_snapshot_tokens > 0 && num_new_computed > 0 {
                    let g = self.p.recurrent_snapshot_tokens;
                    let aligned = num_new_computed / g * g;
                    let hit = snapshot_hash_at(r, aligned).and_then(|h| {
                        let s = self.snapshots.as_mut()?.lookup(h)?;
                        Some((h, s))
                    });
                    match hit {
                        Some((h, s)) if aligned > 0 => {
                            computed.truncate_to(aligned, &self.kv.group_block_sizes());
                            num_new_computed = aligned;
                            resume = Some((aligned, h, s));
                        }
                        _ => {
                            if aligned > 0 {
                                self.snapshot_resume_misses += 1;
                            }
                            computed = KvBlocks::none();
                            num_new_computed = 0;
                        }
                    }
                }
                if self.p.cache.enable_prefix_caching && r.num_computed_tokens == 0 {
                    self.prefix_cache_queries += r.num_tokens().saturating_sub(1) as u64;
                    self.prefix_cache_hits += num_new_computed as u64;
                }
                // host tier（KV 分层）：GPU 未命中的前缀再到主机索引里找；命中的块当作本步要算的 token 来分配
                // GPU 块（走正常 allocate_slots），但不进 token 预算、不进 forward——executor 在 forward 前
                // 从 host 槽拉进这些块，num_computed 直接推进到命中末尾。
                let gpu_hit_blocks: Vec<usize> = computed.groups.iter().map(|g| g.len()).collect();
                let (host_hits, host_tokens) = if r.num_computed_tokens == 0 && self.kv.host_tier_enabled() {
                    self.kv.host_tier_lookup(r, &gpu_hit_blocks)
                } else {
                    (Vec::new(), 0)
                };
                let num_computed = r.num_computed_tokens + num_new_computed;
                let mut num_new = r.num_tokens() - num_computed - host_tokens;
                if long_prefill_threshold > 0 && num_new > long_prefill_threshold {
                    num_new = long_prefill_threshold;
                }
                num_new = num_new.min(token_budget).min(prefill_quota);
                num_new = self.cap_to_snapshot_grain(num_computed, num_new, r.num_prompt_tokens());
                if num_new == 0 {
                    break;
                }
                let computed_opt = if computed.is_empty() { None } else { Some(&computed) };
                let Some(new_blocks) = self.kv.allocate_slots(
                    slot,
                    r,
                    num_new + host_tokens,
                    num_new_computed,
                    computed_opt,
                    num_lookahead,
                ) else {
                    break; // 块不够：等下一 step
                };
                // host 命中的块 = 新分配块里的前 host_hits.len() 个（紧接 GPU 命中块之后）
                if !host_hits.is_empty() {
                    let orders = self.kv.register_fetched(slot, r, &gpu_hit_blocks, &new_blocks, &host_hits);
                    self.host_tier_fetched += orders.len() as u64;
                    out.kv_fetch.extend(orders);
                }
                let num_computed = num_computed + host_tokens;

                self.waiting.pop_front();
                let s = self.slab[slot].as_mut().unwrap();
                let r = &mut s.req;
                let rid = r.id;
                let was_preempted = r.status == RequestStatus::Preempted;
                if r.num_computed_tokens == 0 && !was_preempted {
                    // 首次调度：prefix cache 命中数（GPU 块 + host tier）= 本步不用算的 prompt token 数
                    r.num_cached_tokens = num_computed;
                }
                r.status = RequestStatus::Running;
                r.num_computed_tokens = num_computed;
                self.running.push(slot);
                out.num_scheduled_tokens.push((rid, num_new));
                self.sched_slots.push(slot);
                prefill_quota -= num_new;
                token_budget -= num_new;

                // 合并命中块 + 新块作为 worker 看到的全量 block 表。
                let mut all = if computed.is_empty() {
                    KvBlocks::empty(new_blocks.groups.len().max(1))
                } else {
                    computed
                };
                for (g, nb) in all.groups.iter_mut().zip(new_blocks.groups) {
                    g.extend(nb);
                }
                let computed_after = num_computed + num_new;
                if let Some((pos, _, snap)) = resume {
                    self.snapshots.as_mut().unwrap().pin(snap);
                    out.recurrent_resume.push((rid, snap as u32));
                    debug_assert_eq!(pos, num_computed);
                }
                if s.seen_by_worker {
                    // 抢占后恢复：走 cached_reqs，block 表全量替换。
                    out.cached_reqs.req_ids.push(rid);
                    out.cached_reqs.resumed.push(was_preempted);
                    out.cached_reqs.new_block_ids.push(Some(all));
                    out.cached_reqs.num_computed_tokens.push(computed_after);
                } else {
                    s.seen_by_worker = true;
                    out.new_reqs.push(NewRequestData {
                        req_id: rid,
                        prompt_ids: r.prompt_ids().to_vec(),
                        sampling: r.sampling.clone(),
                        block_ids: all,
                        num_computed_tokens: num_computed,
                        known_output: Vec::new(),
                        resume_from: resume.map(|(p, h, _)| (p, h)),
                    });
                }
                self.register_snapshots(slot, num_computed, computed_after, &mut out);
            }
        }

        out.total_num_scheduled_tokens = out.num_scheduled_tokens.iter().map(|(_, n)| n).sum();
        out.blocks_to_zero = self.kv.take_blocks_to_zero();
        out.kv_offload = self.kv.take_pending_offload();
        out.kv_export = std::mem::take(&mut self.pd_pending_export);

        // ---- 3. 调度后推进（异步调度）----
        self.update_after_schedule(&out);
        out
    }

    fn update_after_schedule(&mut self, out: &SchedulerOutput) {
        let async_sched = self.p.async_scheduling;
        let num_spec = self.p.num_spec_tokens as usize;
        let has_spec = !out.scheduled_spec_tokens.is_empty();
        for (k, &(rid, n)) in out.num_scheduled_tokens.iter().enumerate() {
            let slot = self.sched_slots[k];
            let r = &mut self.slab[slot].as_mut().unwrap().req;
            r.num_computed_tokens += n;
            if async_sched && !r.is_prefill_chunk() {
                // 这一步会产出 1 个 token + 被校验的 spec 数（占位）。
                let n_spec = if has_spec {
                    out.scheduled_spec_tokens
                        .iter()
                        .find(|(id, _)| *id == rid)
                        .map_or(0, |(_, s)| s.len() as u32)
                } else {
                    0
                };
                r.num_output_placeholders += 1 + n_spec;
                // 下一步的 spec 用占位（真实 draft 由 worker 回填）。
                r.spec_ids.clear();
                r.spec_ids.resize(num_spec, 0);
            }
        }
    }

    fn preempt(&mut self, slot: usize) {
        self.kv.free(slot);
        let r = &mut self.slab[slot].as_mut().unwrap().req;
        debug_assert_eq!(r.status, RequestStatus::Running);
        r.reset_for_preemption();
        let rid = r.id;
        self.waiting.push_front(slot);
        self.preempted_req_ids.push(rid);
    }

    // ------------------------------------------------------------------
    // update_from_output
    // ------------------------------------------------------------------

    /// 用 worker 的采样结果更新状态；返回每请求的新 token 与完成原因。
    pub fn update_from_output(&mut self, sched: &SchedulerOutput, mro: &ModelRunnerOutput) -> StepResults {
        let mut results = StepResults::with_capacity(mro.req_ids.len());
        let mut stopped: Vec<usize> = Vec::new();
        let async_sched = self.p.async_scheduling;
        let prefix_cache = self.p.cache.enable_prefix_caching;
        let has_spec = !sched.scheduled_spec_tokens.is_empty();
        let max_model_len = self.p.max_model_len;
        let eos = std::mem::take(&mut self.p.eos_token_ids);

        for (idx, &rid) in mro.req_ids.iter().enumerate() {
            let Some(&slot) = self.id_to_slot.get(&rid) else { continue };
            let r = &mut self.slab[slot].as_mut().unwrap().req;
            if r.status == RequestStatus::Preempted {
                // 异步调度：步 n 在飞时 schedule(n+1) 抢占了它（placeholders 已清零、块已 free），步 n 的结果现在才到。
                // runner 的 token_ids 已经追加了这个 token（恢复后从它之后续算），scheduler 也必须追加，否则两边序列
                // 差 1 → 恢复后位置错位、输出错（2026-10-20 golden mixed+preempt case 抓到：ref 第 8 个 token 丢失）。
                // vLLM `_update_from_output` 同样对非 running 请求照常 append。
                let generated = &mro.sampled[idx];
                let mut finish = None;
                for &tok in generated {
                    r.append_output(tok);
                    if let Some(f) = check_stop(r, &eos, max_model_len) {
                        finish = Some(f);
                        break;
                    }
                }
                if !generated.is_empty() {
                    results.push(rid, generated, finish);
                }
                if let Some(f) = finish {
                    r.status = RequestStatus::Finished(f);
                    stopped.push(slot);
                }
                continue;
            }
            if r.status != RequestStatus::Running {
                continue; // 已完成
            }
            let generated = &mro.sampled[idx];

            // 投机解码：拒绝的 draft 回退 computed / placeholders。
            if has_spec {
                if let Some((_, spec)) = sched.scheduled_spec_tokens.iter().find(|(id, _)| *id == rid) {
                    let num_draft = spec.len() as u32;
                    let num_accepted = (generated.len() as u32).saturating_sub(1);
                    let num_rejected = num_draft.saturating_sub(num_accepted);
                    if num_rejected > 0 {
                        r.num_computed_tokens = r.num_computed_tokens.saturating_sub(num_rejected);
                        r.num_output_placeholders = r.num_output_placeholders.saturating_sub(num_rejected);
                    }
                }
            }

            let before_len = r.num_tokens();
            let mut finish = None;
            let mut n_used = 0usize;
            for &tok in generated {
                r.append_output(tok);
                if async_sched {
                    r.num_output_placeholders = r.num_output_placeholders.saturating_sub(1);
                }
                n_used += 1;
                if let Some(f) = check_stop(r, &eos, max_model_len) {
                    finish = Some(f);
                    break;
                }
            }
            if let Some(f) = finish {
                r.status = RequestStatus::Finished(f);
                stopped.push(slot);
            } else if prefix_cache && r.crossed_hash_block_boundary(before_len) {
                // 新产出的 token 凑满了新块 → 登记。
                let n = r.num_computed_tokens - r.num_output_placeholders;
                let r: &Request = r;
                self.kv.cache_blocks(slot, r, n);
            }
            match mro.logprobs.get(idx) {
                Some(lp) if !lp.is_empty() => {
                    results.push_with_logprobs(rid, &generated[..n_used], lp, finish)
                }
                _ => results.push(rid, &generated[..n_used], finish),
            }
            // 首个输出 token 所在的步：把 prefix cache 命中数带给 server（usage.prompt_tokens_details.cached_tokens）
            if n_used > 0 && before_len == r.num_prompt_tokens() {
                results.items.last_mut().unwrap().num_cached_tokens = Some(r.num_cached_tokens);
            }
        }
        self.p.eos_token_ids = eos;

        for slot in stopped {
            self.finish_slot(slot);
        }
        results
    }

    fn finish_slot(&mut self, slot: usize) {
        let rid = self.req(slot).id;
        // PD Export：先把块按位置记下（含未满尾块），再 free。块在下一步 begin 的导出之前不会被复用
        // （allocate 只发生在 schedule 里，而 schedule 把 pending 搬进 out.kv_export 后 runner 才 begin）。
        if matches!(self.req(slot).pd, Some(PdRole::Export)) {
            if let Some(b) = self.kv.get_blocks(slot) {
                self.pd_pending_export.push((rid, b));
            }
        }
        self.kv.free(slot);
        if let Some(pos) = self.running.iter().position(|&s| s == slot) {
            self.running.remove(pos); // 保持 FCFS 顺序
        }
        if let Some(pos) = self.waiting.iter().position(|&s| s == slot) {
            self.waiting.remove(pos);
        }
        self.finished_req_ids.push(rid);
        self.id_to_slot.remove(&rid);
        self.slab[slot] = None;
        self.free_slots.push(slot);
    }

    /// 外部中止。
    pub fn abort(&mut self, rid: RequestId) {
        if let Some(&slot) = self.id_to_slot.get(&rid) {
            self.req_mut(slot).status = RequestStatus::Finished(FinishReason::Abort);
            self.finish_slot(slot);
        }
    }
}

/// 对应 vLLM `sched/utils.py::check_stop`：先 eos / stop_token_ids（受 `min_tokens` / `ignore_eos` 约束），再 length。
/// stop **字符串** 不在这里（需要 detokenize，由 server 侧 OutputStream 判定后 abort）。
fn check_stop(r: &Request, eos: &[u32], max_model_len: u32) -> Option<FinishReason> {
    let last = *r.output_ids().last()?;
    let sp = &r.sampling;
    if r.num_output_tokens() > sp.min_tokens
        && ((!sp.ignore_eos && eos.contains(&last)) || sp.stop_token_ids.contains(&last))
    {
        return Some(FinishReason::Stop);
    }
    if r.num_tokens() >= max_model_len || r.num_output_tokens() >= sp.max_tokens {
        return Some(FinishReason::Length);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::KvCacheSpec;
    use crate::sample::SamplingParams;

    fn params(max_seqs: u32, budget: u32, blocks: u32) -> (SchedulerParams, Vec<GroupInfo>, u32) {
        let p = SchedulerParams {
            sched: SchedulerConfig {
                max_num_seqs: max_seqs,
                max_num_batched_tokens: budget,
                enable_chunked_prefill: true,
                long_prefill_token_threshold: 0,
                decode_exempt_from_budget: false,
                prefill_tokens_per_step: 0,
            },
            cache: CacheConfig { block_size: 16, hash_block_size: 16, ..Default::default() },
            max_model_len: 4096,
            num_spec_tokens: 0,
            async_scheduling: false,
            eos_token_ids: vec![2],
            recurrent_snapshot_tokens: 0,
            recurrent_snapshot_slots: 0,
            recurrent_snapshot_chunk: 0,
        };
        let g = vec![GroupInfo {
            spec: KvCacheSpec::Mla { block_size: 16, head_size: 576, dtype_bytes: 2 },
            block_size: 16,
        }];
        (p, g, blocks)
    }
    fn mk(max_seqs: u32, budget: u32, blocks: u32) -> Scheduler {
        let (p, g, b) = params(max_seqs, budget, blocks);
        Scheduler::new(p, g, b)
    }
    /// 每个 id 的 prompt 不同，避免意外 prefix 命中。
    fn req(id: u64, prompt: u32, max_tokens: u32) -> Request {
        let base = 100 + id as u32 * 10_000;
        let sp = SamplingParams { max_tokens, ..Default::default() };
        Request::new(id, (base..base + prompt).collect(), sp, 0)
    }
    /// 相同 prompt（测 prefix cache）。
    fn req_same(id: u64, prompt: u32, max_tokens: u32) -> Request {
        let sp = SamplingParams { max_tokens, ..Default::default() };
        Request::new(id, (100..100 + prompt).collect(), sp, 0)
    }
    fn step(s: &mut Scheduler, toks: &[(u64, Vec<u32>)]) -> (SchedulerOutput, StepResults) {
        let o = s.schedule();
        let mro = ModelRunnerOutput {
            req_ids: toks.iter().map(|(i, _)| *i).collect(),
            sampled: toks.iter().map(|(_, t)| t.clone()).collect(),
            drafts: vec![Vec::new(); toks.len()],
            logprobs: vec![],
            mode: None,
            phases: None,
        };
        let r = s.update_from_output(&o, &mro);
        (o, r)
    }

    #[test]
    fn prefill_then_decode_then_finish_by_eos() {
        let mut s = mk(8, 64, 32);
        s.add_request(req(1, 10, 100));
        let (o, _) = step(&mut s, &[(1, vec![7])]);
        assert_eq!(o.new_reqs.len(), 1);
        assert_eq!(o.num_scheduled_tokens, vec![(1, 10)]);
        let (o, _) = step(&mut s, &[(1, vec![8])]);
        assert_eq!(o.num_scheduled_tokens, vec![(1, 1)], "decode step");
        assert!(o.new_reqs.is_empty() && o.cached_reqs.req_ids == vec![1]);
        assert!(o.cached_reqs.new_block_ids[0].is_none(), "no new block needed");
        let (_, r) = step(&mut s, &[(1, vec![2])]);
        assert_eq!(r.items[0].finish, Some(FinishReason::Stop));
        assert_eq!(r.tokens_of(0), &[2]);
        assert!(!s.has_requests());
        let o = s.schedule();
        assert_eq!(o.finished_req_ids, vec![1]);
    }

    #[test]
    fn chunked_prefill_splits_by_budget() {
        let mut s = mk(8, 32, 64);
        s.add_request(req(1, 70, 4));
        let (o, r) = step(&mut s, &[(1, vec![])]);
        assert_eq!(o.num_scheduled_tokens, vec![(1, 32)]);
        assert!(r.tokens_of(0).is_empty());
        let (o, _) = step(&mut s, &[(1, vec![])]);
        assert_eq!(o.num_scheduled_tokens, vec![(1, 32)]);
        let (o, _) = step(&mut s, &[(1, vec![9])]);
        assert_eq!(o.num_scheduled_tokens, vec![(1, 6)], "last chunk");
        let (o, _) = step(&mut s, &[(1, vec![9])]);
        assert_eq!(o.num_scheduled_tokens, vec![(1, 1)]);
    }

    #[test]
    fn fcfs_second_request_waits_for_budget_then_runs_alongside() {
        let mut s = mk(8, 40, 64);
        s.add_request(req(1, 30, 10));
        s.add_request(req(2, 30, 10));
        // step 1：r1 全部 30，r2 只拿到剩余预算 10（prefill 未完 → 无输出）
        let (o, _) = step(&mut s, &[(1, vec![1]), (2, vec![])]);
        assert_eq!(o.num_scheduled_tokens, vec![(1, 30), (2, 10)]);
        // step 2：r1 decode 1，r2 补完剩余 20（prefill 完成 → 产出第 1 个 token）
        let (o, _) = step(&mut s, &[(1, vec![1]), (2, vec![1])]);
        assert_eq!(o.num_scheduled_tokens, vec![(1, 1), (2, 20)]);
        // step 3：两者都 decode
        let (o, _) = step(&mut s, &[(1, vec![1]), (2, vec![1])]);
        assert_eq!(o.num_scheduled_tokens, vec![(1, 1), (2, 1)]);
    }

    #[test]
    fn preempts_last_running_when_out_of_blocks() {
        // 4 blocks × 16 = 64 token 容量；两个 32-token 请求正好占满。
        let mut s = mk(8, 1000, 4);
        s.add_request(req(1, 32, 100));
        s.add_request(req(2, 32, 100));
        let (o, _) = step(&mut s, &[(1, vec![1]), (2, vec![1])]);
        assert_eq!(o.num_scheduled_tokens.len(), 2);
        // 各 33 token，decode 1 个需要第 3 块：r1 分配失败 → 抢占末尾 r2 → r1 拿到。
        let (o, _) = step(&mut s, &[(1, vec![1])]);
        assert_eq!(o.num_scheduled_tokens, vec![(1, 1)]);
        assert_eq!(s.num_waiting(), 1);
        assert_eq!(s.request(2).unwrap().status, RequestStatus::Preempted);
        assert_eq!(s.request(2).unwrap().num_computed_tokens, 0);
        let o = s.schedule();
        assert_eq!(o.preempted_req_ids, vec![2]);
    }

    #[test]
    fn preempted_request_resumes_reusing_surviving_cached_blocks() {
        // 池 4 块。step1 后 r1=[0,1] r2=[2,3]，四块都带 hash。
        // step2：r1 要第 3 块 → 抢占 r2（释放 3,2 到 LRU 尾）→ r1 拿到 3（hash 被淘汰）→ r1 完成释放 0,1,3。
        // r2 恢复：块 2 的 hash 仍在 → 命中前 16 token，重算 17。与 vLLM 行为一致。
        let mut s = mk(8, 1000, 4);
        s.add_request(req(1, 32, 2));
        s.add_request(req(2, 32, 100));
        step(&mut s, &[(1, vec![1]), (2, vec![1])]);
        step(&mut s, &[(1, vec![1])]);
        assert!(s.request(1).is_none());
        assert_eq!(s.request(2).unwrap().status, RequestStatus::Preempted);
        let o = s.schedule();
        assert_eq!(o.num_scheduled_tokens, vec![(2, 17)], "16 cached + 17 recomputed = 33");
        assert!(o.new_reqs.is_empty(), "already seen by worker");
        assert_eq!(o.cached_reqs.req_ids, vec![2]);
        assert_eq!(o.cached_reqs.resumed, vec![true]);
        assert_eq!(o.cached_reqs.num_computed_tokens, vec![33]);
        assert_eq!(o.cached_reqs.new_block_ids[0].as_ref().unwrap().groups[0].len(), 3, "full table");
    }

    #[test]
    fn preempted_request_without_prefix_cache_replays_everything() {
        let (mut p, g, b) = params(8, 1000, 4);
        p.cache.enable_prefix_caching = false;
        let mut s = Scheduler::new(p, g, b);
        s.add_request(req(1, 32, 2));
        s.add_request(req(2, 32, 100));
        step(&mut s, &[(1, vec![1]), (2, vec![1])]);
        step(&mut s, &[(1, vec![1])]);
        let o = s.schedule();
        assert_eq!(o.num_scheduled_tokens, vec![(2, 33)], "replays prompt + 1 output");
        assert_eq!(o.cached_reqs.new_block_ids[0].as_ref().unwrap().groups[0].len(), 3);
    }

    #[test]
    fn prefix_cache_hit_skips_computed_prefix() {
        let mut s = mk(8, 1000, 32);
        s.add_request(req_same(1, 48, 1));
        step(&mut s, &[(1, vec![2])]); // 完成，块进入 LRU 但仍缓存
        s.add_request(req_same(2, 48, 1));
        let o = s.schedule();
        // 命中 32（48-1 规则下最多 2 块），只算 16
        assert_eq!(o.num_scheduled_tokens, vec![(2, 16)]);
        assert_eq!(o.new_reqs[0].num_computed_tokens, 32);
        assert_eq!(o.new_reqs[0].block_ids.groups[0].len(), 3);
    }

    /// 递归快照（design §15.7）：粒度 = 4 块（64 token）、池 4 槽。
    fn mk_snap(blocks: u32, grain: u32, slots: u32) -> Scheduler {
        let (mut p, g, b) = params(8, 1000, blocks);
        p.recurrent_snapshot_tokens = grain;
        p.recurrent_snapshot_slots = slots;
        p.recurrent_snapshot_chunk = 16;
        p.cache.enable_prefix_caching = true;
        Scheduler::new(p, g, b)
    }

    #[test]
    fn snapshot_one_step_spans_grains_and_registers_each() {
        // r8b：150 token 的 prompt、预算 1000 → 一步算完 0..150，跨 64 / 128 两个对齐点 → 登记 2 槽 (pos 64, 128)，不切步
        let mut s = mk_snap(64, 64, 4);
        s.add_request(req_same(1, 150, 4));
        let o = s.schedule();
        assert_eq!(o.num_scheduled_tokens, vec![(1, 150)], "不再按粒度切步");
        let mut snaps: Vec<(u64, u32)> = o.recurrent_snapshot.iter().map(|&(r, p, _)| (r, p)).collect();
        snaps.sort();
        assert_eq!(snaps, vec![(1, 64), (1, 128)]);
        let slots: std::collections::BTreeSet<u32> = o.recurrent_snapshot.iter().map(|x| x.2).collect();
        assert_eq!(slots.len(), 2, "两个不同的槽");
        assert!(o.recurrent_resume.is_empty());
        assert!(o.new_reqs[0].resume_from.is_none());
        let mro = ModelRunnerOutput {
            req_ids: vec![1], sampled: vec![vec![]], drafts: vec![vec![]], logprobs: vec![], mode: None, phases: None,
        };
        s.update_from_output(&o, &mro);
        let (_, m, _, _, skipped) = s.snapshot_stats();
        assert_eq!((m, skipped), (0, 0));
    }

    #[test]
    fn snapshot_one_step_spans_four_grains() {
        // 300 token prompt、预算 1000：一步跨 64/128/192/256 四个对齐点 → 登记 4 槽
        let mut s = mk_snap(64, 64, 8);
        s.add_request(req_same(1, 300, 4));
        let o = s.schedule();
        assert_eq!(o.num_scheduled_tokens, vec![(1, 300)]);
        let mut pos: Vec<u32> = o.recurrent_snapshot.iter().map(|x| x.1).collect();
        pos.sort();
        assert_eq!(pos, vec![64, 128, 192, 256], "一步跨 4 个对齐点登记 4 槽");
    }

    #[test]
    fn snapshot_grain_not_chunk_multiple_splits_at_each_grain() {
        // 粒度 32 < chunk 64（golden tiny 的口径）：对齐点 32 相对段首 0 的偏移不是 64 的倍数 → 在 32 结束；
        // 之后段首 32、下一对齐点 64 偏移 32 仍非倍数 → 再切。等价于 r8 的逐粒度切步（kernel 末态即快照）。
        let (mut p, g, b) = params(8, 1000, 64);
        p.recurrent_snapshot_tokens = 32;
        p.recurrent_snapshot_slots = 4;
        p.recurrent_snapshot_chunk = 64;
        p.cache.enable_prefix_caching = true;
        let mut s = Scheduler::new(p, g, b);
        s.add_request(req_same(1, 100, 4));
        let mro = ModelRunnerOutput {
            req_ids: vec![1], sampled: vec![vec![]], drafts: vec![vec![]], logprobs: vec![], mode: None, phases: None,
        };
        for want in [32u32, 64, 96] {
            let o = s.schedule();
            assert_eq!(o.num_scheduled_tokens, vec![(1, 32)]);
            assert_eq!(o.recurrent_snapshot.iter().map(|x| x.1).collect::<Vec<_>>(), vec![want]);
            s.update_from_output(&o, &mro);
        }
        let o = s.schedule();
        assert_eq!(o.num_scheduled_tokens, vec![(1, 4)]);
        assert!(o.recurrent_snapshot.is_empty());
    }

    #[test]
    fn snapshot_unaligned_segment_start_falls_back_to_split() {
        // 上一步被预算截断在 40（非 chunk 16 的倍数）→ 本步跨 64 时在 64 结束（kernel 的 chunk 边界对不上 64-40=24）
        let (mut p, g, b) = params(8, 1000, 64);
        p.recurrent_snapshot_tokens = 64;
        p.recurrent_snapshot_slots = 4;
        p.recurrent_snapshot_chunk = 16;
        p.cache.enable_prefix_caching = true;
        p.sched.max_num_batched_tokens = 40;
        let mut s = Scheduler::new(p, g, b);
        s.add_request(req_same(1, 150, 4));
        let o = s.schedule();
        assert_eq!(o.num_scheduled_tokens, vec![(1, 40)]);
        assert!(o.recurrent_snapshot.is_empty());
        let mro = ModelRunnerOutput {
            req_ids: vec![1], sampled: vec![vec![]], drafts: vec![vec![]], logprobs: vec![], mode: None, phases: None,
        };
        s.update_from_output(&o, &mro);
        // 预算 40 本来只能到 80；对齐点 64 的相对偏移 24 不是 16 的倍数 → 截到 64
        let o = s.schedule();
        assert_eq!(o.num_scheduled_tokens, vec![(1, 24)], "段首非对齐 → 在对齐点结束");
        assert_eq!(o.recurrent_snapshot.iter().map(|x| x.1).collect::<Vec<_>>(), vec![64]);
        s.update_from_output(&o, &mro);
        // 之后段首 64 对齐，预算 40 → 64..104，不跨 128 → 不登记
        let o = s.schedule();
        assert_eq!(o.num_scheduled_tokens, vec![(1, 40)]);
        assert!(o.recurrent_snapshot.is_empty());
        s.update_from_output(&o, &mro);
        // 104..144 跨 128：104 % 16 = 8 非对齐 → 截到 128
        let o = s.schedule();
        assert_eq!(o.num_scheduled_tokens, vec![(1, 24)]);
        assert_eq!(o.recurrent_snapshot.iter().map(|x| x.1).collect::<Vec<_>>(), vec![128]);
    }

    #[test]
    fn snapshot_hit_aligns_down_and_resumes_from_snapshot() {
        // 任务口径：命中 1300、粒度 1024 → resume 1024。这里缩比：块 16、粒度 64；prompt 150 → KV 可命中 144（149 向下取块），
        // 对齐到 64 → 128，且 @128 有快照 → resume 128。
        let mut s = mk_snap(64, 64, 4);
        s.add_request(req_same(1, 150, 1));
        let mro0 = ModelRunnerOutput {
            req_ids: vec![1], sampled: vec![vec![]], drafts: vec![vec![]], logprobs: vec![], mode: None, phases: None,
        };
        for _ in 0..2 {
            let o = s.schedule();
            s.update_from_output(&o, &mro0);
        }
        step(&mut s, &[(1, vec![2])]); // 最后一块 + 采样 → 完成
        assert!(!s.has_requests());
        s.add_request(req_same(2, 150, 1));
        let o = s.schedule();
        assert_eq!(o.new_reqs[0].num_computed_tokens, 128, "144 向下对齐到 128");
        assert_eq!(o.num_scheduled_tokens, vec![(2, 22)]);
        let (pos, _h) = o.new_reqs[0].resume_from.expect("resume_from");
        assert_eq!(pos, 128);
        assert_eq!(o.recurrent_resume.len(), 1);
        assert_eq!(o.recurrent_resume[0].0, 2);
        assert_eq!(o.new_reqs[0].block_ids.groups[0].len(), 128 / 16 + 2, "命中 8 块 + 新 2 块");
        let (h, m, _, miss_resume, _) = s.snapshot_stats();
        assert_eq!((h, m, miss_resume), (1, 0, 0));
        let (q, hits) = s.prefix_cache_stats();
        assert_eq!((q, hits), (149 * 2, 128), "两条请求都计 query；命中按对齐后的数");
    }

    #[test]
    fn snapshot_hit_without_snapshot_recomputes_from_zero() {
        // 池 0 槽：命中仍按粒度对齐，但没有快照 → 整段重算（KV 命中作废）；且**不切步**（engine-gap 8l：
        // 无池时切步只剩代价，6 K 单发 1 → 6 步 +32 ms TTFT）
        let mut s = mk_snap(64, 64, 0);
        s.add_request(req_same(1, 150, 1));
        let mro0 = ModelRunnerOutput {
            req_ids: vec![1], sampled: vec![vec![]], drafts: vec![vec![]], logprobs: vec![], mode: None, phases: None,
        };
        for _ in 0..2 {
            let o = s.schedule();
            assert!(o.recurrent_snapshot.is_empty(), "无池不登记");
            s.update_from_output(&o, &mro0);
        }
        step(&mut s, &[(1, vec![2])]);
        s.add_request(req_same(2, 150, 1));
        let o = s.schedule();
        assert_eq!(o.new_reqs[0].num_computed_tokens, 0);
        assert!(o.new_reqs[0].resume_from.is_none());
        assert_eq!(o.num_scheduled_tokens, vec![(2, 150)], "从 0 起、无池不切步，一步算完");
        assert_eq!(s.snapshot_stats().3, 1, "记一次命中无快照");
    }

    #[test]
    fn snapshot_grain_zero_keeps_old_behaviour() {
        let mut s = mk(8, 1000, 32);
        s.add_request(req_same(1, 48, 1));
        step(&mut s, &[(1, vec![2])]);
        s.add_request(req_same(2, 48, 1));
        let o = s.schedule();
        assert_eq!(o.new_reqs[0].num_computed_tokens, 32);
        assert!(o.new_reqs[0].resume_from.is_none());
        assert!(o.recurrent_resume.is_empty() && o.recurrent_snapshot.is_empty());
    }

    #[test]
    fn num_cached_tokens_recorded_and_surfaced_on_first_output() {
        let mut s = mk(8, 1000, 32);
        s.add_request(req_same(1, 48, 1));
        let (_, r1) = step(&mut s, &[(1, vec![2])]);
        assert_eq!(r1.items[0].num_cached_tokens, Some(0), "冷请求首 token 的步报 cached=0");
        s.add_request(req_same(2, 48, 2));
        let (_, r2) = step(&mut s, &[(2, vec![3])]);
        assert_eq!(s.request(2).unwrap().num_cached_tokens, 32);
        assert_eq!(r2.items[0].num_cached_tokens, Some(32), "首个输出 token 的步带出命中数");
        let (_, r3) = step(&mut s, &[(2, vec![4])]);
        assert_eq!(r3.items[0].num_cached_tokens, None, "后续 decode 步不再带");
    }

    #[test]
    fn max_tokens_finishes_with_length() {
        let mut s = mk(8, 1000, 32);
        s.add_request(req(1, 4, 2));
        step(&mut s, &[(1, vec![5])]);
        let (_, r) = step(&mut s, &[(1, vec![6])]);
        assert_eq!(r.items[0].finish, Some(FinishReason::Length));
    }

    #[test]
    fn long_prefill_threshold_only_with_multiple_requests() {
        let (mut p, g, b) = params(8, 1000, 64);
        p.sched.long_prefill_token_threshold = 8;
        let mut s = Scheduler::new(p, g, b);
        s.add_request(req(1, 40, 1));
        let o = s.schedule();
        assert_eq!(o.num_scheduled_tokens, vec![(1, 40)], "single request ignores threshold");
        // r1 prefill 完成产出 1 个 token（max_tokens=1 → 完成）。此时加入 r2 与 r3：两个候选 → 阈值生效。
        s.update_from_output(
            &o,
            &ModelRunnerOutput {
                req_ids: vec![1],
                sampled: vec![vec![1]],
                drafts: vec![Vec::new(); 1],
                logprobs: vec![],
                mode: None,
                phases: None,
            },
        );
        s.add_request(req(2, 40, 1));
        s.add_request(req(3, 40, 1));
        let o = s.schedule();
        assert_eq!(o.num_scheduled_tokens, vec![(2, 8), (3, 8)], "both capped at threshold");
    }

    /// 异步调度 + 抢占：步 n 在飞时 schedule(n+1) 抢占了请求 2，步 n 的采样结果到达时仍要追加进请求 2（runner 已追加）。
    #[test]
    fn async_preempted_request_keeps_in_flight_token() {
        // 4 块 × 16 = 64 token；两个 30-token 请求各 2 块；decode 各长到 32 要第 3 块 → 只能给一个，另一个被抢占
        let (mut p, g, b) = params(8, 1000, 4);
        p.async_scheduling = true;
        let mut s = Scheduler::new(p, g, b);
        s.add_request(req(1, 30, 100));
        s.add_request(req(2, 30, 100));
        let o1 = s.schedule(); // prefill 30 + 30（各 2 块；第 2 块剩 2 格）
        assert_eq!(o1.num_scheduled_tokens.len(), 2);
        let mro = |toks: Vec<Vec<u32>>| ModelRunnerOutput {
            req_ids: vec![1, 2],
            sampled: toks,
            drafts: vec![Vec::new(); 2],
            logprobs: vec![],
            mode: None,
            phases: None,
        };
        s.update_from_output(&o1, &mro(vec![vec![10], vec![20]]));
        let o2 = s.schedule(); // decode 31（在飞）
        s.update_from_output(&o2, &mro(vec![vec![11], vec![21]]));
        let o3 = s.schedule(); // decode 32（在飞）：各自第 2 块的最后一格
        // o4：位置 33 要第 3 块 → 池空 → 抢占末尾请求 2（此时 o3 的结果还没回）
        let o4 = s.schedule();
        assert_eq!(o4.num_scheduled_tokens, vec![(1, 1)], "req 2 preempted");
        assert_eq!(s.request(2).unwrap().status, RequestStatus::Preempted);
        // o3 的结果到：请求 2 的 token 22 必须进它的序列
        let r3 = s.update_from_output(&o3, &mro(vec![vec![12], vec![22]]));
        assert_eq!(r3.items.len(), 2, "both requests produce a token for step 3");
        let r = s.request(2).unwrap();
        assert_eq!(r.output_ids(), &[20, 21, 22], "in-flight token of the preempted request is kept");
        assert_eq!(r.num_computed_tokens, 0);
        assert_eq!(r.num_output_placeholders, 0);
    }

    #[test]
    fn async_scheduling_uses_placeholders_and_spec_rollback() {
        let (mut p, g, b) = params(8, 1000, 64);
        p.async_scheduling = true;
        p.num_spec_tokens = 2;
        let mut s = Scheduler::new(p, g, b);
        s.add_request(req(1, 10, 100));
        let o = s.schedule();
        assert_eq!(o.num_scheduled_tokens, vec![(1, 10)]);
        {
            let r = s.request(1).unwrap();
            assert_eq!(r.num_computed_tokens, 10);
            assert_eq!(r.num_output_placeholders, 1, "1 token in flight");
            assert_eq!(r.spec_ids.len(), 2, "placeholder drafts");
        }
        // step 2 可以在 step 1 结果回来前调度：1 bonus + 2 spec = 3 tokens
        let o2 = s.schedule();
        assert_eq!(o2.num_scheduled_tokens, vec![(1, 3)]);
        assert_eq!(o2.scheduled_spec_tokens[0].1.len(), 2);
        assert_eq!(s.request(1).unwrap().num_output_placeholders, 4);
        s.update_from_output(
            &o,
            &ModelRunnerOutput {
                req_ids: vec![1],
                sampled: vec![vec![50]],
                drafts: vec![Vec::new(); 1],
                logprobs: vec![],
                mode: None,
                phases: None,
            },
        );
        assert_eq!(s.request(1).unwrap().num_output_placeholders, 3);
        // step 2 结果：bonus + 1 accepted（拒绝 1）
        s.update_from_output(
            &o2,
            &ModelRunnerOutput {
                req_ids: vec![1],
                sampled: vec![vec![51, 52]],
                drafts: vec![Vec::new(); 1],
                logprobs: vec![],
                mode: None,
                phases: None,
            },
        );
        let r = s.request(1).unwrap();
        assert_eq!(r.num_output_tokens(), 3);
        assert_eq!(r.num_computed_tokens, 10 + 3 - 1, "one rejected draft rolled back");
        assert_eq!(r.num_output_placeholders, 0);
    }

    #[test]
    fn slots_are_reused_after_finish() {
        let mut s = mk(8, 1000, 64);
        for id in 0..20 {
            s.add_request(req(id, 4, 1));
            step(&mut s, &[(id, vec![9])]);
        }
        assert!(s.slab.len() <= 2, "slab should not grow: {}", s.slab.len());
        assert!(!s.has_requests());
    }

    #[test]
    fn eos_in_middle_of_multi_token_output_truncates() {
        let mut s = mk(8, 1000, 64);
        s.add_request(req(1, 4, 100));
        let (_, r) = step(&mut s, &[(1, vec![5, 2, 9])]);
        assert_eq!(r.tokens_of(0), &[5, 2], "tokens after eos dropped");
        assert_eq!(r.items[0].finish, Some(FinishReason::Stop));
    }

    /// decode 免预算：预算 8。r1 已在 decode，r2 新来 20 token 的 prompt → 同一步 r1 1 token + r2 拿满 8
    /// （不免预算时 r1 先占 1，r2 只剩 7，见上面 fcfs 用例的分配方式）。
    #[test]
    fn decode_exempt_from_budget_coschedules_decode_and_prefill() {
        let (mut p, g, b) = params(4, 8, 64);
        p.sched.decode_exempt_from_budget = true;
        let mut s = Scheduler::new(p, g, b);
        s.add_request(req(1, 4, 16));
        let (o, _) = step(&mut s, &[(1, vec![1])]); // r1 prefill 4 → 第 1 个 token
        assert_eq!(o.num_scheduled_tokens, vec![(1, 4)]);
        s.add_request(req(2, 20, 16));
        let (o, _) = step(&mut s, &[(1, vec![1]), (2, vec![])]);
        assert_eq!(o.num_scheduled_tokens, vec![(1, 1), (2, 8)], "decode 行免预算，prefill 拿到完整 8");
    }

    /// 显存压力下最后一个 running 请求分不到块时会抢占自己 → 本步空；下一步它从 waiting 重新 prefill。
    /// 回归 2026-10-17 host tier A/B 撞到的 livelock：空步必须能推进（下一步 num_scheduled 非空），
    /// 且被抢占请求的块真的回到了池里。
    #[test]
    fn self_preempt_under_pressure_makes_progress() {
        // 池 4 块 × 16 = 64 token；一个 48-token prompt + 长输出：decode 到 65 token 时需要第 5 块
        let mut s = mk(8, 1000, 4);
        s.add_request(req(1, 48, 100));
        let (o, _) = step(&mut s, &[(1, vec![1])]);
        assert_eq!(o.num_scheduled_tokens, vec![(1, 48)]);
        let mut empty_steps = 0;
        let mut forced = Vec::new();
        for _ in 0..40 {
            let (o, _) = step(&mut s, &[(1, vec![1])]);
            forced.extend(s.take_force_finished());
            if o.num_scheduled_tokens.is_empty() {
                empty_steps += 1;
                assert!(empty_steps <= 1, "scheduler emits empty steps repeatedly (livelock)");
            } else {
                empty_steps = 0;
            }
            if s.request(1).is_none() {
                break;
            }
        }
        assert!(s.request(1).is_none(), "sole request that cannot grow must be finished, not spun");
        assert_eq!(forced, vec![(1, FinishReason::Length)]);
        assert_eq!(s.kv_usage(), 0.0, "its blocks must be freed");
    }

    /// 多个 running 时末尾请求分不到块：不得被 Length 结束、不得抢占自己；本步跳过，下一步前面的请求释放后再分到。
    #[test]
    fn tail_request_with_others_present_is_skipped_not_finished() {
        // 池 6 块 = 96 token。r1 48 token（3 块）、r2 48 token（3 块）→ 池满。
        let mut s = mk(8, 1000, 6);
        s.add_request(req(1, 48, 2));
        s.add_request(req(2, 48, 100));
        let (o, _) = step(&mut s, &[(1, vec![1]), (2, vec![1])]);
        assert_eq!(o.num_scheduled_tokens.len(), 2);
        // 两者都要第 4 块：r1 先分（抢占末尾 r2 → r2 被抢占）。r1 还剩 1 个 token 到 max_tokens。
        let (o, _) = step(&mut s, &[(1, vec![1])]);
        assert_eq!(o.num_scheduled_tokens, vec![(1, 1)]);
        assert_eq!(s.request(2).unwrap().status, RequestStatus::Preempted);
        assert!(s.take_force_finished().is_empty(), "r2 was preempted, not force-finished");
        // r1 完成 → 释放；r2 从 waiting 回来重算
        let o = s.schedule();
        assert!(s.request(1).is_none());
        assert_eq!(o.num_scheduled_tokens.first().map(|x| x.0), Some(2));
        assert!(s.take_force_finished().is_empty());
    }

    /// PD Import（design §10.3）：首次调度只分块 + kv_import，不进 forward；第二步正常 decode 1 token。
    #[test]
    fn pd_import_first_step_skips_forward_then_decodes() {
        let mut s = mk(8, 64, 32);
        let mut r = req(1, 40, 100); // 40 token = 2 满块 + 8 尾
        r.pd = Some(PdRole::Import { first_token: 777 });
        s.add_request(r);
        let o = s.schedule();
        assert!(o.num_scheduled_tokens.is_empty(), "import step must not forward");
        assert_eq!(o.kv_import.len(), 1);
        assert_eq!(o.kv_import[0].0, 1);
        assert_eq!(o.kv_import[0].1.groups[0].len(), 3, "40 tokens → 3 blocks incl. partial tail");
        assert_eq!(o.new_reqs.len(), 1);
        assert_eq!(o.new_reqs[0].num_computed_tokens, 40);
        assert_eq!(o.new_reqs[0].known_output, vec![777]);
        assert_eq!(s.num_running(), 1);
        // 没有 forward 行，update 不需要调；第二步应为 1 token 的 decode
        let (o2, _) = step(&mut s, &[(1, vec![9])]);
        assert_eq!(o2.num_scheduled_tokens, vec![(1, 1)]);
        assert!(o2.kv_import.is_empty());
        // 导入后前缀登记：同 prompt 的新请求应 GPU 命中 2 个满块
        let mut r2 = req(1, 40, 100);
        r2.id = 2;
        s.add_request(r2);
        let (o3, _) = step(&mut s, &[(1, vec![10]), (2, vec![11])]);
        let nr = o3.new_reqs.iter().find(|n| n.req_id == 2).unwrap();
        assert_eq!(nr.num_computed_tokens, 32, "2 full blocks hit from imported KV");
    }

    /// PD Export：max_tokens=1 的请求 finish 时，kv_export 带全部块（含尾块），且在下一次 schedule 的输出里。
    #[test]
    fn pd_export_emits_all_blocks_on_finish() {
        let mut s = mk(8, 64, 32);
        let mut r = req(1, 40, 1);
        r.pd = Some(PdRole::Export);
        s.add_request(r);
        let (o, _) = step(&mut s, &[(1, vec![5])]); // prefill 40 → 首 token → Length finish
        assert_eq!(o.num_scheduled_tokens, vec![(1, 40)]);
        assert!(o.kv_export.is_empty(), "export is emitted on the NEXT schedule, before any begin");
        let free_before = s.num_free_blocks();
        let o2 = s.schedule();
        assert_eq!(o2.kv_export.len(), 1);
        assert_eq!(o2.kv_export[0].0, 1);
        assert_eq!(o2.kv_export[0].1.groups[0].len(), 3);
        assert_eq!(s.num_free_blocks(), free_before, "blocks freed, none reallocated yet");
        assert_eq!(s.num_running(), 0);
    }
}
