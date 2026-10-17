//! 多 group KV cache 管理器。对应 vLLM `kv_cache_manager.py` + `kv_cache_coordinator.py`
//! + `single_type_kv_cache_manager.py` 的子集：
//! - 每个 group 一张 `slot → blocks` 表；所有 group 共享一个 `BlockPool`
//!   （vLLM 把不同 page 大小的 group 放进同一池，靠 page 对齐；P1 先要求所有 group 的
//!   `page_size_bytes` 相等，Mamba 组通过 `block_size` 折算，不等则 panic）。
//! - prefix cache：只对 attention 类 group 做链式 hash 查找（前缀封闭）；
//!   `Mamba` 组只在 hit 边界取"最后一个"状态块（align 模式简化版：仅在 block 对齐的
//!   token 数处登记/命中；调度器不强制对齐 chunk，所以 Mamba 命中率低于 vLLM align 模式。P7 再补）。
//! - `get_computed_blocks` → `allocate_slots` → `cache_blocks` → `free` 四个入口与 vLLM 同名。
//!
//! 性能：请求用调度器分配的 `slot: usize` 索引（dense Vec），不走 HashMap；
//! decode 稳态（无新块、无命中、未跨 hash 块边界）走零分配快路径。

use super::block_pool::{BlockId, BlockPool};
use super::spec::KvCacheSpec;
use crate::request::Request;

pub const MAX_GROUPS: usize = 8;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KvBlocks {
    /// 每个 group 一份 block id 列表。`groups.is_empty()` 表示"没有新块"（零分配）。
    pub groups: Vec<Vec<BlockId>>,
}

impl KvBlocks {
    pub fn empty(num_groups: usize) -> Self {
        Self { groups: vec![Vec::new(); num_groups] }
    }
    /// 零分配的"无新块"。
    #[inline]
    pub const fn none() -> Self {
        Self { groups: Vec::new() }
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.groups.iter().all(|g| g.is_empty())
    }
    #[inline]
    pub fn group(&self, gid: usize) -> &[BlockId] {
        self.groups.get(gid).map_or(&[], |g| g.as_slice())
    }
}

#[derive(Debug, Clone)]
pub struct GroupInfo {
    pub spec: KvCacheSpec,
    /// 该 group 的 block 覆盖多少 token（Mamba 为"每请求固定 slot"，用 block_size 表示折算）。
    pub block_size: u32,
}

#[derive(Debug, Default)]
struct ReqState {
    blocks: Vec<Vec<BlockId>>, // per group
    /// 每 group 已登记进 prefix cache 的块数。
    num_cached: Vec<u32>,
    /// 已经处理过登记的 hash 块数（`num_tokens / hash_block_size`），用于跳过未跨边界的 step。
    cached_hash_blocks: u32,
    active: bool,
}

#[derive(Debug)]
pub struct KvCacheManager {
    pub groups: Vec<GroupInfo>,
    is_mamba: [bool; MAX_GROUPS],
    pool: BlockPool,
    reqs: Vec<ReqState>,
    enable_caching: bool,
    max_model_len: u32,
    /// 新分配、需要清零的块（Mamba 状态块必须清零）。每 step 由 runner 取走。
    blocks_to_zero: Vec<BlockId>,
    /// KV 分层：host 索引（None = 关）；本步产生的推出指令（scheduler 每步取走）。
    host_tier: Option<super::host_tier::HostTierIndex>,
    pending_offload: Vec<(BlockId, u32)>,
}

/// 自由函数，避免对整个 `KvCacheManager` 可变借用。
fn state_mut_or_init(reqs: &mut Vec<ReqState>, slot: usize, num_groups: usize) -> &mut ReqState {
    if slot >= reqs.len() {
        reqs.resize_with(slot + 1, ReqState::default);
    }
    let s = &mut reqs[slot];
    if !s.active {
        s.active = true;
        s.blocks.clear();
        s.blocks.resize_with(num_groups, Vec::new);
        s.num_cached.clear();
        s.num_cached.resize(num_groups, 0);
        s.cached_hash_blocks = 0;
    }
    s
}

impl KvCacheManager {
    pub fn new(groups: Vec<GroupInfo>, num_blocks: u32, enable_caching: bool, max_model_len: u32) -> Self {
        assert!(!groups.is_empty() && groups.len() <= MAX_GROUPS, "1..={MAX_GROUPS} KV groups");
        let page = groups[0].spec.page_size_bytes();
        let mut is_mamba = [false; MAX_GROUPS];
        for (i, g) in groups.iter().enumerate() {
            assert_eq!(
                g.spec.page_size_bytes(),
                page,
                "P1 requires uniform page size across groups (pad Mamba spec to match)"
            );
            is_mamba[i] = matches!(g.spec, KvCacheSpec::Mamba { .. });
        }
        Self {
            groups,
            is_mamba,
            pool: BlockPool::new(num_blocks, enable_caching),
            reqs: Vec::new(),
            enable_caching,
            max_model_len,
            blocks_to_zero: Vec::new(),
            host_tier: None,
            pending_offload: Vec::new(),
        }
    }

    #[inline]
    pub fn num_free_blocks(&self) -> u32 {
        self.pool.num_free()
    }
    #[inline]
    pub fn usage(&self) -> f32 {
        self.pool.usage()
    }
    #[inline]
    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }

    fn state(&self, slot: usize) -> Option<&ReqState> {
        self.reqs.get(slot).filter(|s| s.active)
    }

    /// prefix cache 查找。返回 (命中的块, 命中 token 数)。
    /// 命中长度 ≤ `num_tokens - 1`（最后一个 token 必须重算以产 logits）。
    /// 所有 attention 类 group 取共同前缀；Mamba 组在该长度处取一个状态块（若有）。
    pub fn get_computed_blocks(&self, req: &Request) -> (KvBlocks, u32) {
        let n = self.groups.len();
        if !self.enable_caching || req.num_computed_tokens > 0 {
            return (KvBlocks::none(), 0);
        }
        let max_hit = req.num_tokens().saturating_sub(1);
        let hashes = &req.block_hashes.hashes;
        let hbs = req.block_hashes_block_size();
        if hashes.is_empty() {
            return (KvBlocks::none(), 0);
        }

        // 1) attention 组：共同最长前缀。
        let mut hit_len = max_hit;
        let mut per_group: Vec<Vec<BlockId>> = vec![Vec::new(); n];
        for (gid, blocks) in per_group.iter_mut().enumerate() {
            if self.is_mamba[gid] {
                continue;
            }
            let bs = self.groups[gid].block_size;
            debug_assert_eq!(bs % hbs, 0, "block_size must be a multiple of hash_block_size");
            let step = (bs / hbs) as usize;
            let max_blocks = (hit_len / bs) as usize;
            for b in 0..max_blocks {
                let hi = (b + 1) * step - 1; // 该 block 末尾对应的 hash 下标
                let Some(h) = hashes.get(hi) else { break };
                match self.pool.get_cached(gid as u8, *h) {
                    Some(id) => blocks.push(id),
                    None => break,
                }
            }
            hit_len = hit_len.min(blocks.len() as u32 * bs);
        }
        if hit_len == 0 {
            return (KvBlocks::none(), 0);
        }
        // 所有 attention 组截到共同 hit_len；Mamba 组在 hit_len 边界找状态块。
        for (gid, blocks) in per_group.iter_mut().enumerate() {
            if self.is_mamba[gid] {
                let hi = (hit_len / hbs) as usize - 1;
                match hashes.get(hi).and_then(|h| self.pool.get_cached(gid as u8, *h)) {
                    Some(id) => *blocks = vec![id],
                    None => return (KvBlocks::none(), 0), // P1 简化：不做逐级回退
                }
            } else {
                blocks.truncate((hit_len / self.groups[gid].block_size) as usize);
            }
        }
        (KvBlocks { groups: per_group }, hit_len)
    }

    /// 为请求分配本 step 所需的新块。失败返回 None（不改状态）。
    /// `num_new_computed_tokens` / `new_computed` 来自 `get_computed_blocks`（仅首次调度时非空）。
    pub fn allocate_slots(
        &mut self,
        slot: usize,
        req: &Request,
        num_new_tokens: u32,
        num_new_computed_tokens: u32,
        new_computed: Option<&KvBlocks>,
        num_lookahead_tokens: u32,
    ) -> Option<KvBlocks> {
        debug_assert!(num_new_tokens > 0 || num_new_computed_tokens > 0);
        let n = self.groups.len();
        let total_computed = req.num_computed_tokens + num_new_computed_tokens;
        let num_tokens_need_slot =
            (total_computed + num_new_tokens + num_lookahead_tokens).min(self.max_model_len);

        // 需要多少新块（按 group 累加）。
        let mut need = [0u32; MAX_GROUPS];
        let mut need_total = 0u32;
        let existing = self.state(slot);
        for (gid, nd) in need.iter_mut().enumerate().take(n) {
            let have = existing.map_or(0, |s| s.blocks[gid].len() as u32);
            let computed = new_computed.map_or(0, |c| c.group(gid).len() as u32);
            *nd = if self.is_mamba[gid] {
                // 每请求固定 1 个状态块（P1：不做 checkpoint 多块）。
                u32::from(have + computed == 0)
            } else {
                num_tokens_need_slot.div_ceil(self.groups[gid].block_size).saturating_sub(have + computed)
            };
            need_total += *nd;
        }

        let to_cache = (total_computed + num_new_tokens).min(req.num_tokens());

        // 快路径：running 请求、无新块、无命中 → 只需（可能）登记缓存。
        if need_total == 0 && new_computed.is_none() {
            self.cache_blocks_inner(slot, req, to_cache);
            return Some(KvBlocks::none());
        }

        // 命中块里 ref==0 的也要算进容量（touch 会把它们从空闲队列摘走）。
        let evictable = new_computed
            .map_or(0, |c| c.groups.iter().flatten().filter(|&&b| self.pool.ref_cnt(b) == 0).count() as u32);
        if need_total + evictable > self.pool.num_free() {
            return None;
        }

        // 字段级拆借：state 来自 self.reqs，pool / blocks_to_zero / is_mamba 各自独立。
        let Self { reqs, pool, blocks_to_zero, is_mamba, .. } = self;
        let state = state_mut_or_init(reqs, slot, n);

        // 先接纳命中块。
        if let Some(c) = new_computed {
            for gid in 0..n {
                let cg = c.group(gid);
                if !cg.is_empty() {
                    pool.touch(cg);
                    state.blocks[gid].extend_from_slice(cg);
                    state.num_cached[gid] = state.blocks[gid].len() as u32;
                }
            }
        }
        // 再分配新块。
        let mut out = KvBlocks::empty(n);
        for (gid, og) in out.groups.iter_mut().enumerate() {
            if need[gid] == 0 {
                continue;
            }
            let ids = pool.allocate(need[gid]).expect("checked above");
            if is_mamba[gid] {
                blocks_to_zero.extend_from_slice(&ids);
            }
            state.blocks[gid].extend_from_slice(&ids);
            *og = ids;
        }

        // 登记可缓存的满块（只到 request.num_tokens，不含 draft / lookahead）。
        self.cache_blocks_inner(slot, req, to_cache);
        Some(out)
    }

    /// 把 `[0, num_tokens)` 内已满的块登记进 prefix cache（幂等）。
    pub fn cache_blocks(&mut self, slot: usize, req: &Request, num_tokens: u32) {
        self.cache_blocks_inner(slot, req, num_tokens);
    }

    fn cache_blocks_inner(&mut self, slot: usize, req: &Request, num_tokens: u32) {
        if !self.enable_caching {
            return;
        }
        let Some(state) = self.reqs.get_mut(slot).filter(|s| s.active) else { return };
        let hbs = req.block_hashes_block_size();
        if hbs == 0 {
            return;
        }
        let num_full_hash = num_tokens / hbs;
        if num_full_hash <= state.cached_hash_blocks {
            return; // 没跨过新的 hash 块边界：无事可做（decode 稳态 15/16 的 step 走这里）
        }
        let hashes = &req.block_hashes.hashes;
        for (gid, g) in self.groups.iter().enumerate() {
            let bs = g.block_size;
            if self.is_mamba[gid] {
                // 状态块在 block 对齐的 token 边界登记；边界变化时重新登记（旧键由 LRU 淘汰）。
                if num_tokens % bs != 0 {
                    continue;
                }
                let Some(&id) = state.blocks[gid].first() else { continue };
                let hi = (num_tokens / hbs) as usize - 1;
                if let Some(&h) = hashes.get(hi) {
                    self.pool.cache_block(id, gid as u8, h);
                }
            } else {
                let step = (bs / hbs) as usize;
                let num_full = (num_tokens / bs) as usize;
                let start = state.num_cached[gid] as usize;
                let end = num_full.min(state.blocks[gid].len());
                for b in start..end {
                    let hi = (b + 1) * step - 1;
                    let Some(&h) = hashes.get(hi) else { break };
                    self.pool.cache_block(state.blocks[gid][b], gid as u8, h);
                    state.num_cached[gid] = (b + 1) as u32;
                    // host tier：新凑满的块推到主机（executor 在本步结束后异步做）
                    if let Some(t) = self.host_tier.as_mut() {
                        if let Some(slot_h) = t.insert(gid as u8, h) {
                            t.stats.offload_blocks += 1;
                            self.pending_offload.push((state.blocks[gid][b], slot_h));
                        }
                    }
                }
            }
        }
        state.cached_hash_blocks = num_full_hash;
    }

    /// 释放请求的全部块（倒序，让末尾块先被淘汰）。
    pub fn free(&mut self, slot: usize) {
        let Some(state) = self.reqs.get_mut(slot).filter(|s| s.active) else { return };
        state.active = false;
        let blocks = std::mem::take(&mut state.blocks);
        for g in blocks {
            self.pool.free(g.into_iter().rev());
        }
    }

    pub fn get_blocks(&self, slot: usize) -> Option<KvBlocks> {
        self.state(slot).map(|s| KvBlocks { groups: s.blocks.clone() })
    }

    pub fn take_blocks_to_zero(&mut self) -> Vec<BlockId> {
        std::mem::take(&mut self.blocks_to_zero)
    }
    /// 启用 host tier（槽位数 = host_gib / bytes_per_block_all_layers）。
    pub fn enable_host_tier(&mut self, num_slots: usize) {
        if num_slots > 0 {
            self.host_tier = Some(super::host_tier::HostTierIndex::new(num_slots));
        }
    }
    pub fn host_tier_enabled(&self) -> bool {
        self.host_tier.is_some()
    }
    pub fn host_tier_stats(&self) -> super::host_tier::HostTierStats {
        self.host_tier.as_ref().map_or_else(Default::default, |t| t.stats)
    }
    pub fn take_pending_offload(&mut self) -> Vec<(BlockId, u32)> {
        std::mem::take(&mut self.pending_offload)
    }
    /// host tier 命中：GPU 命中 `gpu_hit_blocks` 块之后，hashes 还能连续命中多少块（只看 attention 组 0）。
    /// 返回命中的 host 槽位列表（顺序 = 块顺序）。本轮要求单组、block_size == hash_block_size。
    pub fn host_tier_lookup(&mut self, req: &Request, gpu_hit_blocks: usize) -> Vec<u32> {
        let Some(t) = self.host_tier.as_mut() else { return Vec::new() };
        if self.groups.len() != 1 || self.is_mamba[0] {
            return Vec::new();
        }
        let hbs = req.block_hashes_block_size();
        let bs = self.groups[0].block_size;
        if hbs == 0 || bs != hbs {
            return Vec::new();
        }
        let max_blocks = (req.num_tokens().saturating_sub(1) / bs) as usize;
        let n = req.block_hashes.hashes.len().min(max_blocks);
        t.lookup_prefix(0, &req.block_hashes.hashes[..n], gpu_hit_blocks)
    }
    /// host tier 拉回的块：登记哈希（forward 前会被填好，之后与普通缓存块无差别）。
    /// `first_hash_idx` = 这些块在请求 hash 列表里的起始下标（= GPU 命中块数）。
    pub fn register_fetched_blocks(&mut self, slot: usize, req: &Request, blocks: &[BlockId], first_hash_idx: usize) {
        let hashes = &req.block_hashes.hashes;
        for (i, &b) in blocks.iter().enumerate() {
            if let Some(&h) = hashes.get(first_hash_idx + i) {
                self.pool.cache_block(b, 0, h);
            }
        }
        if let Some(state) = self.reqs.get_mut(slot).filter(|s| s.active) {
            let n = (first_hash_idx + blocks.len()) as u32;
            state.num_cached[0] = state.num_cached[0].max(n);
            state.cached_hash_blocks = state.cached_hash_blocks.max(n);
        }
    }

    pub fn reset_prefix_cache(&mut self) -> bool {
        self.pool.reset_prefix_cache()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::SamplingParams;

    fn mla(bs: u32) -> GroupInfo {
        GroupInfo {
            spec: KvCacheSpec::Mla { block_size: bs, head_size: 576, dtype_bytes: 2 },
            block_size: bs,
        }
    }
    fn mgr(num_blocks: u32, bs: u32) -> KvCacheManager {
        KvCacheManager::new(vec![mla(bs)], num_blocks, true, 4096)
    }
    fn req(id: u64, toks: Vec<u32>) -> Request {
        let mut r = Request::new(id, toks, SamplingParams::default(), 0);
        r.init_block_hasher(16, vec![]);
        r
    }

    #[test]
    fn allocate_then_full_prefix_hit() {
        let mut m = mgr(10, 16);
        let r1 = req(1, (0..48).collect()); // 3 full blocks
        let (c, n) = m.get_computed_blocks(&r1);
        assert!(c.is_empty() && n == 0);
        let got = m.allocate_slots(1, &r1, 48, 0, None, 0).unwrap();
        assert_eq!(got.groups[0].len(), 3);
        assert_eq!(m.num_free_blocks(), 7);

        // 同前缀第二个请求：命中前 2 块（第 3 块因 num_tokens-1 规则不算）。
        let r2 = req(2, (0..48).collect());
        let (c, n) = m.get_computed_blocks(&r2);
        assert_eq!(n, 32);
        assert_eq!(c.groups[0], got.groups[0][..2].to_vec());
        let got2 = m.allocate_slots(2, &r2, 16, 32, Some(&c), 0).unwrap();
        assert_eq!(got2.groups[0].len(), 1, "only the tail block is new");
        assert_eq!(m.num_free_blocks(), 6);

        m.free(1);
        assert_eq!(m.num_free_blocks(), 7);
        m.free(2);
        assert_eq!(m.num_free_blocks(), 10);
    }

    #[test]
    fn chunked_prefill_allocates_incrementally() {
        let mut m = mgr(8, 16);
        let mut r = req(1, (0..40).collect());
        let a = m.allocate_slots(1, &r, 16, 0, None, 0).unwrap();
        assert_eq!(a.groups[0].len(), 1);
        r.num_computed_tokens = 16;
        let b = m.allocate_slots(1, &r, 24, 0, None, 0).unwrap();
        assert_eq!(b.groups[0].len(), 2);
        assert_eq!(m.get_blocks(1).unwrap().groups[0].len(), 3);
        // decode 稳态：无新块 → 零分配 none()
        r.num_computed_tokens = 40;
        let c = m.allocate_slots(1, &r, 1, 0, None, 0).unwrap();
        assert!(c.groups.is_empty());
    }

    #[test]
    fn out_of_blocks_returns_none_and_keeps_state() {
        let mut m = mgr(2, 16);
        let r = req(1, (0..48).collect());
        assert!(m.allocate_slots(1, &r, 48, 0, None, 0).is_none());
        assert_eq!(m.num_free_blocks(), 2);
        assert!(m.get_blocks(1).is_none());
    }

    #[test]
    fn lookahead_tokens_reserve_blocks() {
        let mut m = mgr(8, 16);
        let r = req(1, (0..16).collect());
        let a = m.allocate_slots(1, &r, 16, 0, None, 4).unwrap();
        assert_eq!(a.groups[0].len(), 2);
    }

    #[test]
    fn decode_caches_block_exactly_when_boundary_crossed() {
        let mut m = mgr(8, 16);
        let mut r = req(1, (0..16).collect());
        m.allocate_slots(1, &r, 16, 0, None, 0).unwrap();
        r.num_computed_tokens = 16;
        // 产出 15 个 token：第 2 块未满，不应登记
        for t in 0..15 {
            r.append_output(1000 + t);
            m.allocate_slots(1, &r, 1, 0, None, 0).unwrap();
            r.num_computed_tokens += 1;
        }
        let probe = req(2, r.all_token_ids().to_vec());
        assert_eq!(m.get_computed_blocks(&probe).1, 16, "only first block cached");
        // 第 16 个产出 token → 第 2 块满 → 登记
        r.append_output(2000);
        m.cache_blocks(1, &r, 32);
        let probe = req(2, [r.all_token_ids(), &[9]].concat());
        assert_eq!(m.get_computed_blocks(&probe).1, 32);
    }

    #[test]
    fn hybrid_mamba_group_gets_one_state_block() {
        let mamba = GroupInfo {
            spec: KvCacheSpec::Mamba { shapes: vec![vec![9216]], dtype_bytes: vec![2], block_size: 16 },
            block_size: 16,
        };
        let mut m = KvCacheManager::new(vec![mla(16), mamba], 8, true, 4096);
        let r = req(1, (0..32).collect());
        let a = m.allocate_slots(1, &r, 32, 0, None, 0).unwrap();
        assert_eq!(a.groups[0].len(), 2);
        assert_eq!(a.groups[1].len(), 1);
        assert_eq!(m.take_blocks_to_zero(), a.groups[1]);
        // 第二个相同请求：Mla 命中 16（32-1 规则），Mamba 状态只在 32 边界登记 → 整体不命中。
        let r2 = req(2, (0..32).collect());
        let (_, n) = m.get_computed_blocks(&r2);
        assert_eq!(n, 0);
    }
}
