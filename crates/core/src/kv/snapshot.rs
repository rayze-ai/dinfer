//! 递归状态快照池（design §15.7）：GDN / KDA / Mamba 这类"状态不可按块复用"的层，只在 prefix cache 的
//! 对齐位置（`recurrent_snapshot_tokens` 的整数倍）保存一份完整状态，键 = 该位置的 KV 块 hash（链式，天然带前缀语义）。
//!
//! 池只管 **键 → 槽、LRU、pin**；显存由模型侧按槽数分配（qwen35：每槽 48 层 rec+conv ≈ 154 MB）。
//! 对应 vLLM `MambaManager`（`mamba_cache_mode=align`）的"状态块只在对齐位置缓存"；我们的改进：粒度可配、
//! 池独立于 KV 块池（不占 KV 块、不参与 KV LRU）。
//!
//! 时序契约（scheduler 持有）：
//! - `insert(hash)` 在 prefill 步**调度时**登记，executor 在该步 forward 之后把活槽 dtod 进快照槽；
//! - `lookup(hash)` 命中 → 本步 `pin`，executor 在该步 forward 之前把快照槽 dtod 进活槽；下一次 `schedule()` 开头 `unpin_all`；
//! - 同一流上"写快照（步 n）→ 读快照（步 n+1）"天然有序；同一步内先 insert 再 lookup 的槽不会被当作可淘汰的受害者
//!   （insert 把它标成最新，且本步 lookup 的槽都 pin 住，淘汰只挑未 pin 且最久未用的）。

use super::hash::BlockHash;
use std::collections::HashMap;

#[derive(Debug)]
pub struct StateSnapshotPool {
    cap: usize,
    by_hash: HashMap<BlockHash, usize>,
    /// 每槽的键（None = 空槽）。
    keys: Vec<Option<BlockHash>>,
    /// LRU 时间戳（越大越新）。
    last_used: Vec<u64>,
    pinned: Vec<bool>,
    free: Vec<usize>,
    tick: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl StateSnapshotPool {
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            by_hash: HashMap::with_capacity(cap),
            keys: vec![None; cap],
            last_used: vec![0; cap],
            pinned: vec![false; cap],
            free: (0..cap).rev().collect(),
            tick: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    pub fn cap(&self) -> usize {
        self.cap
    }
    pub fn len(&self) -> usize {
        self.by_hash.len()
    }
    pub fn is_empty(&self) -> bool {
        self.by_hash.is_empty()
    }

    /// 查命中：刷新 LRU，不 pin（调用方决定）。
    pub fn lookup(&mut self, hash: BlockHash) -> Option<usize> {
        self.tick += 1;
        match self.by_hash.get(&hash).copied() {
            Some(s) => {
                self.last_used[s] = self.tick;
                self.hits += 1;
                Some(s)
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// 只看不改（不计统计、不刷新 LRU）。
    pub fn contains(&self, hash: BlockHash) -> bool {
        self.by_hash.contains_key(&hash)
    }

    /// 登记一个新键并返回槽：已有则返回现有槽（刷新 LRU）；池满则淘汰"未 pin 且最久未用"的槽；
    /// 全被 pin 住（或 cap 0）→ None（调用方放弃本次快照）。
    pub fn insert(&mut self, hash: BlockHash) -> Option<usize> {
        self.tick += 1;
        if let Some(&s) = self.by_hash.get(&hash) {
            self.last_used[s] = self.tick;
            return Some(s);
        }
        let s = match self.free.pop() {
            Some(s) => s,
            None => {
                let victim = (0..self.cap)
                    .filter(|&s| !self.pinned[s])
                    .min_by_key(|&s| self.last_used[s])?;
                if let Some(k) = self.keys[victim].take() {
                    self.by_hash.remove(&k);
                }
                self.evictions += 1;
                victim
            }
        };
        self.keys[s] = Some(hash);
        self.by_hash.insert(hash, s);
        self.last_used[s] = self.tick;
        Some(s)
    }

    pub fn pin(&mut self, slot: usize) {
        self.pinned[slot] = true;
    }
    pub fn unpin(&mut self, slot: usize) {
        self.pinned[slot] = false;
    }
    pub fn unpin_all(&mut self) {
        self.pinned.iter_mut().for_each(|p| *p = false);
    }
    pub fn is_pinned(&self, slot: usize) -> bool {
        self.pinned[slot]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(i: u128) -> BlockHash {
        BlockHash(i)
    }

    #[test]
    fn insert_lookup_and_lru_eviction() {
        let mut p = StateSnapshotPool::new(2);
        let a = p.insert(h(1)).unwrap();
        let b = p.insert(h(2)).unwrap();
        assert_ne!(a, b);
        assert_eq!(p.lookup(h(1)), Some(a)); // 刷新 1 → 2 成为最久未用
        let c = p.insert(h(3)).unwrap();
        assert_eq!(c, b, "evicts the least recently used slot");
        assert_eq!(p.lookup(h(2)), None);
        assert_eq!(p.lookup(h(1)), Some(a));
        assert_eq!(p.evictions, 1);
        assert_eq!(p.len(), 2);
    }

    #[test]
    fn insert_existing_is_idempotent() {
        let mut p = StateSnapshotPool::new(1);
        let a = p.insert(h(7)).unwrap();
        assert_eq!(p.insert(h(7)), Some(a));
        assert_eq!(p.len(), 1);
        assert_eq!(p.evictions, 0);
    }

    #[test]
    fn pinned_slots_are_never_evicted() {
        let mut p = StateSnapshotPool::new(2);
        let a = p.insert(h(1)).unwrap();
        let b = p.insert(h(2)).unwrap();
        p.pin(a);
        p.pin(b);
        assert_eq!(p.insert(h(3)), None, "all pinned → refuse");
        p.unpin(a);
        assert_eq!(p.insert(h(3)), Some(a), "only the unpinned one is a victim");
        assert!(p.contains(h(2)) && p.contains(h(3)) && !p.contains(h(1)));
        p.unpin_all();
        assert!(!p.is_pinned(b));
    }

    #[test]
    fn zero_capacity_refuses() {
        let mut p = StateSnapshotPool::new(0);
        assert_eq!(p.insert(h(1)), None);
        assert_eq!(p.lookup(h(1)), None);
    }
}
