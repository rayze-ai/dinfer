//! KV 分层第一层：主机 pinned 内存索引（design §10.2，loop q3-hosttier）。
//! 这里只管 **索引与槽位**（纯 host 数据结构）；字节搬运由 executor 的 copy stream 做，
//! scheduler 通过 `SchedulerOutput::{kv_fetch, kv_offload}` 下发搬运指令。
//! 语义：`(group, BlockHash) → HostSlot`，LRU 淘汰；命中只认连续前缀（与 GPU prefix cache 一致）。
use super::hash::BlockHash;
use std::collections::HashMap;

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
