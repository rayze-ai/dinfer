//! KV 分层第一层：主机 pinned 内存索引（design §10.2，loop q3-hosttier）。
//! 这里只管 **索引与槽位**（纯 host 数据结构）；字节搬运由 executor 的 copy stream 做，
//! scheduler 通过 `SchedulerOutput::{kv_fetch, kv_offload}` 下发搬运指令。
//! 语义：`(group, BlockHash) → HostSlot`，LRU 淘汰；命中只认连续前缀（与 GPU prefix cache 一致）。
use super::hash::BlockHash;
use super::BlockId;
use super::manager::GroupInfo;
use crate::request::Request;
use std::collections::HashMap;

/// 命中单元策略（**模型相关**，由模型 crate 注入；core 只放容器）：
/// 什么算 host 层的一个单元、多组如何合成前缀命中、一个单元覆盖多少 token、拉回的块如何登记。
/// Qwen3：一个单元 = 一个 16-token 块 × 全部层（`FullAttentionPolicy`）。
/// GLM（MLA 潜变量 + 索引器 K 两组）/ K3（KDA 状态块只在边界有效 + 全注意力）要各自实现——
/// 规则应与 GPU 侧 `KvCacheManager::get_computed_blocks` 的多组合成一致（attention 组取最短前缀、Mamba 组边界找状态块）。
pub trait HostTierPolicy: Send + std::fmt::Debug {
    /// 请求首次调度：GPU 已命中 `gpu_hit_blocks[gid]` 块后，在 host 索引里还能连续命中哪些单元。
    /// 返回 (命中的 host 槽位序列, 这些单元覆盖的 token 数)。
    fn lookup(&self, idx: &mut HostTierIndex, groups: &[GroupInfo], req: &Request, gpu_hit_blocks: &[usize]) -> (Vec<HostSlot>, u32);
    /// 为命中的单元挑出要填的 GPU 块：`new_blocks[gid]` 是本步新分配的块（含将被 fetch 填充的部分）。
    /// 返回 (块, 槽) 对，顺序 = 槽序列顺序；以及要登记哈希的 (gid, block, hash) 列表。
    fn fetch_targets(&self, groups: &[GroupInfo], req: &Request, gpu_hit_blocks: &[usize], new_blocks: &[Vec<BlockId>], slots: &[HostSlot]) -> (Vec<(BlockId, HostSlot)>, Vec<(u8, BlockId, BlockHash)>);
}

/// 默认策略：单个全注意力组，block_size == hash_block_size，一个单元 = 一块 × 全层。
#[derive(Debug, Default, Clone, Copy)]
pub struct FullAttentionPolicy;

impl HostTierPolicy for FullAttentionPolicy {
    fn lookup(&self, idx: &mut HostTierIndex, groups: &[GroupInfo], req: &Request, gpu_hit_blocks: &[usize]) -> (Vec<HostSlot>, u32) {
        if groups.len() != 1 || !matches!(groups[0].spec, super::spec::KvCacheSpec::FullAttention { .. }) {
            return (Vec::new(), 0);
        }
        let hbs = req.block_hashes_block_size();
        let bs = groups[0].block_size;
        if hbs == 0 || bs != hbs {
            return (Vec::new(), 0);
        }
        let max_blocks = (req.num_tokens().saturating_sub(1) / bs) as usize;
        let n = req.block_hashes.hashes.len().min(max_blocks);
        let hits = idx.lookup_prefix(0, &req.block_hashes.hashes[..n], gpu_hit_blocks.first().copied().unwrap_or(0));
        let tokens = hits.len() as u32 * bs;
        (hits, tokens)
    }
    fn fetch_targets(&self, _groups: &[GroupInfo], req: &Request, gpu_hit_blocks: &[usize], new_blocks: &[Vec<BlockId>], slots: &[HostSlot]) -> (Vec<(BlockId, HostSlot)>, Vec<(u8, BlockId, BlockHash)>) {
        let g0 = &new_blocks[0];
        let first = gpu_hit_blocks.first().copied().unwrap_or(0);
        let mut orders = Vec::with_capacity(slots.len());
        let mut regs = Vec::with_capacity(slots.len());
        for (i, &s) in slots.iter().enumerate() {
            orders.push((g0[i], s));
            if let Some(&h) = req.block_hashes.hashes.get(first + i) {
                regs.push((0u8, g0[i], h));
            }
        }
        (orders, regs)
    }
}

pub type HostSlot = u32;

#[derive(Debug, Default, Clone, Copy)]
pub struct HostTierStats {
    pub hits_blocks: u64,
    pub fetch_blocks: u64,
    pub offload_blocks: u64,
    pub evictions: u64,
}

#[derive(Debug)]
pub struct HostTierIndex {
    map: HashMap<(u8, BlockHash), HostSlot>,
    key_of: Vec<Option<(u8, BlockHash)>>,
    /// LRU：最近使用的在尾部；用 "最后使用轮次" 近似，淘汰时线性找最小（容量万级，每步最多几十次）。
    last_use: Vec<u64>,
    free: Vec<HostSlot>,
    tick: u64,
    pub stats: HostTierStats,
}

impl HostTierIndex {
    pub fn new(num_slots: usize) -> Self {
        Self {
            map: HashMap::with_capacity(num_slots),
            key_of: vec![None; num_slots],
            last_use: vec![0; num_slots],
            free: (0..num_slots as u32).rev().collect(), // pop 得到 0,1,2,…
            tick: 0,
            stats: HostTierStats::default(),
        }
    }
    pub fn capacity(&self) -> usize {
        self.key_of.len()
    }
    /// 从 `hashes[start..]` 起连续命中的 (slot) 列表；命中的槽位 touch（LRU）。
    pub fn lookup_prefix(&mut self, group: u8, hashes: &[BlockHash], start: usize) -> Vec<HostSlot> {
        self.tick += 1;
        let mut out = Vec::new();
        for h in &hashes[start.min(hashes.len())..] {
            match self.map.get(&(group, *h)) {
                Some(&s) => {
                    self.last_use[s as usize] = self.tick;
                    out.push(s);
                }
                None => break,
            }
        }
        self.stats.hits_blocks += out.len() as u64;
        out
    }
    pub fn contains(&self, group: u8, h: BlockHash) -> bool {
        self.map.contains_key(&(group, h))
    }
    /// 为一个新 hash 取槽位（必要时淘汰 LRU）；已存在则返回 None（不需要 offload）。
    pub fn insert(&mut self, group: u8, h: BlockHash) -> Option<HostSlot> {
        if self.map.contains_key(&(group, h)) {
            return None;
        }
        self.tick += 1;
        let slot = match self.free.pop() {
            Some(s) => s,
            None => {
                if self.key_of.is_empty() {
                    return None;
                }
                let (victim, _) =
                    self.last_use.iter().enumerate().min_by_key(|(_, &t)| t).map(|(i, &t)| (i as u32, t)).unwrap();
                if let Some(k) = self.key_of[victim as usize].take() {
                    self.map.remove(&k);
                }
                self.stats.evictions += 1;
                victim
            }
        };
        self.map.insert((group, h), slot);
        self.key_of[slot as usize] = Some((group, h));
        self.last_use[slot as usize] = self.tick;
        Some(slot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn h(i: u128) -> BlockHash {
        BlockHash(i)
    }
    #[test]
    fn prefix_hit_and_lru_eviction() {
        let mut t = HostTierIndex::new(3);
        assert_eq!(t.insert(0, h(1)), Some(0));
        assert_eq!(t.insert(0, h(2)), Some(1));
        assert_eq!(t.insert(0, h(3)), Some(2));
        assert_eq!(t.insert(0, h(2)), None, "already present");
        // 连续前缀命中：1,2 命中（touch），4 不在 → 停
        assert_eq!(t.lookup_prefix(0, &[h(1), h(2), h(4)], 0), vec![0, 1]);
        // 容量满：插 h(5) 淘汰最久未用的 h(3)（slot 2）
        assert_eq!(t.insert(0, h(5)), Some(2));
        assert!(!t.contains(0, h(3)) && t.contains(0, h(5)));
        assert_eq!(t.stats.evictions, 1);
    }
}
