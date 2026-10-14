//! 物理 block 池：引用计数 + LRU 空闲双链表 + hash → block 索引。
//! 对应 vLLM `block_pool.py` + `kv_cache_utils.py::FreeKVCacheBlockQueue`。
//!
//! 空闲队列语义（与 vLLM 一致）：
//! - 带 hash 的块释放时 **追加到尾部**（FIFO → LRU 淘汰）；
//! - 不带 hash 的块释放时 **插到头部**（LIFO → GPU 局部性）；
//! - `allocate` 从头部取；取到带 hash 的块时把它从 hash 索引里摘掉（淘汰）。
//! - prefix 命中时 `touch`：ref 0→1 的块要从空闲队列里 O(1) 摘除。

use super::hash::BlockHash;
use std::collections::HashMap;

pub type BlockId = u32;
const NIL: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, Default)]
struct Node {
    prev: u32,
    next: u32,
}

#[derive(Debug)]
pub struct BlockPool {
    ref_cnt: Vec<u32>,
    /// 块当前登记的 (group, hash)；None 表示未缓存。
    hash_of: Vec<Option<(u8, BlockHash)>>,
    /// 空闲双链表。`in_free[i]` 表示块 i 是否在链表里。
    link: Vec<Node>,
    in_free: Vec<bool>,
    head: u32,
    tail: u32,
    num_free: u32,
    /// (group, hash) → block。同一 hash 在同一 group 只保留一个块。
    cached: HashMap<(u8, BlockHash), BlockId>,
    enable_caching: bool,
}

impl BlockPool {
    pub fn new(num_blocks: u32, enable_caching: bool) -> Self {
        let n = num_blocks as usize;
        let mut link = vec![Node::default(); n];
        for (i, nd) in link.iter_mut().enumerate() {
            nd.prev = if i == 0 { NIL } else { (i - 1) as u32 };
            nd.next = if i + 1 == n { NIL } else { (i + 1) as u32 };
        }
        Self {
            ref_cnt: vec![0; n],
            hash_of: vec![None; n],
            link,
            in_free: vec![true; n],
            head: if n == 0 { NIL } else { 0 },
            tail: if n == 0 { NIL } else { (n - 1) as u32 },
            num_free: num_blocks,
            cached: HashMap::new(),
            enable_caching,
        }
    }

    #[inline]
    pub fn num_free(&self) -> u32 {
        self.num_free
    }
    #[inline]
    pub fn num_total(&self) -> u32 {
        self.ref_cnt.len() as u32
    }
    #[inline]
    pub fn usage(&self) -> f32 {
        1.0 - self.num_free as f32 / self.num_total().max(1) as f32
    }
    #[inline]
    pub fn ref_cnt(&self, id: BlockId) -> u32 {
        self.ref_cnt[id as usize]
    }
    #[inline]
    pub fn block_hash(&self, id: BlockId) -> Option<&(u8, BlockHash)> {
        self.hash_of[id as usize].as_ref()
    }

    // ---- 空闲链表原语 ----

    fn unlink(&mut self, id: BlockId) {
        debug_assert!(self.in_free[id as usize]);
        let Node { prev, next } = self.link[id as usize];
        if prev == NIL {
            self.head = next;
        } else {
            self.link[prev as usize].next = next;
        }
        if next == NIL {
            self.tail = prev;
        } else {
            self.link[next as usize].prev = prev;
        }
        self.in_free[id as usize] = false;
        self.num_free -= 1;
    }

    fn push_back(&mut self, id: BlockId) {
        debug_assert!(!self.in_free[id as usize]);
        self.link[id as usize] = Node { prev: self.tail, next: NIL };
        if self.tail == NIL {
            self.head = id;
        } else {
            self.link[self.tail as usize].next = id;
        }
        self.tail = id;
        self.in_free[id as usize] = true;
        self.num_free += 1;
    }

    fn push_front(&mut self, id: BlockId) {
        debug_assert!(!self.in_free[id as usize]);
        self.link[id as usize] = Node { prev: NIL, next: self.head };
        if self.head == NIL {
            self.tail = id;
        } else {
            self.link[self.head as usize].prev = id;
        }
        self.head = id;
        self.in_free[id as usize] = true;
        self.num_free += 1;
    }

    // ---- 分配 / 释放 ----

    /// 从空闲队首取 n 个块（可能淘汰缓存块）。不足返回 None，状态不变。
    pub fn allocate(&mut self, n: u32) -> Option<Vec<BlockId>> {
        if n > self.num_free {
            return None;
        }
        let mut out = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let id = self.head;
            self.unlink(id);
            if let Some(key) = self.hash_of[id as usize].take() {
                // 被重用即淘汰出 prefix cache。
                if self.cached.get(&key) == Some(&id) {
                    self.cached.remove(&key);
                }
            }
            debug_assert_eq!(self.ref_cnt[id as usize], 0);
            self.ref_cnt[id as usize] = 1;
            out.push(id);
        }
        Some(out)
    }

    /// prefix 命中共享：ref +1；若原本在空闲队列则摘除。
    pub fn touch(&mut self, ids: &[BlockId]) {
        for &id in ids {
            if self.ref_cnt[id as usize] == 0 {
                self.unlink(id);
            }
            self.ref_cnt[id as usize] += 1;
        }
    }

    /// 释放一组块（按淘汰优先级排序：先传的先被淘汰）。
    pub fn free(&mut self, ordered: impl IntoIterator<Item = BlockId>) {
        let mut evict_last = Vec::new();
        let mut evict_first = Vec::new();
        for id in ordered {
            let c = &mut self.ref_cnt[id as usize];
            debug_assert!(*c > 0, "double free of block {id}");
            *c -= 1;
            if *c == 0 {
                if self.hash_of[id as usize].is_none() || !self.enable_caching {
                    evict_first.push(id);
                } else {
                    evict_last.push(id);
                }
            }
        }
        // 无 hash 的块：LIFO，插到头部（保持传入顺序：先传的更靠前 → 更先被取走）。
        for &id in evict_first.iter().rev() {
            self.push_front(id);
        }
        for id in evict_last {
            self.push_back(id);
        }
    }

    // ---- prefix cache 索引 ----

    pub fn get_cached(&self, group: u8, hash: BlockHash) -> Option<BlockId> {
        self.cached.get(&(group, hash)).copied()
    }

    /// 把一个已满的块登记到 prefix cache。若同 key 已有别的块，保留旧的（先到先得）。
    pub fn cache_block(&mut self, id: BlockId, group: u8, hash: BlockHash) {
        if !self.enable_caching {
            return;
        }
        debug_assert!(self.ref_cnt[id as usize] > 0);
        if self.hash_of[id as usize].is_some() {
            return; // 已登记
        }
        let key = (group, hash);
        if self.cached.contains_key(&key) {
            return;
        }
        self.cached.insert(key, id);
        self.hash_of[id as usize] = Some(key);
    }

    /// 清空 prefix cache（所有块都空闲时才允许）。
    pub fn reset_prefix_cache(&mut self) -> bool {
        if self.num_free != self.num_total() {
            return false;
        }
        self.cached.clear();
        for h in &mut self.hash_of {
            *h = None;
        }
        true
    }

    #[cfg(test)]
    fn free_list(&self) -> Vec<BlockId> {
        let mut v = Vec::new();
        let mut cur = self.head;
        while cur != NIL {
            v.push(cur);
            cur = self.link[cur as usize].next;
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(x: u64) -> BlockHash {
        BlockHash(x as u128)
    }

    #[test]
    fn alloc_free_roundtrip() {
        let mut p = BlockPool::new(4, true);
        let a = p.allocate(3).unwrap();
        assert_eq!(a, vec![0, 1, 2]);
        assert_eq!(p.num_free(), 1);
        assert!(p.allocate(2).is_none());
        p.free(a);
        assert_eq!(p.num_free(), 4);
        // 无 hash 块 LIFO：0,1,2 插回头部，顺序保持。
        assert_eq!(p.free_list(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn shared_block_refcount() {
        let mut p = BlockPool::new(2, true);
        let a = p.allocate(1).unwrap();
        p.touch(&a);
        p.free(a.clone());
        assert_eq!(p.num_free(), 1, "still referenced once");
        p.free(a);
        assert_eq!(p.num_free(), 2);
    }

    #[test]
    fn cached_blocks_are_lru_and_evicted_on_reuse() {
        let mut p = BlockPool::new(3, true);
        let a = p.allocate(2).unwrap(); // [0,1]
        p.cache_block(a[0], 0, h(10));
        p.cache_block(a[1], 0, h(11));
        p.free(a.clone()); // 带 hash → 追加到尾部：free = [2, 0, 1]
        assert_eq!(p.free_list(), vec![2, 0, 1]);
        assert_eq!(p.get_cached(0, h(10)), Some(0));

        // 命中 hash 11 → touch 把块 1 从空闲队列摘出
        let hit = p.get_cached(0, h(11)).unwrap();
        p.touch(&[hit]);
        assert_eq!(p.free_list(), vec![2, 0]);

        // 分配 2 个：先拿 2（无 hash），再拿 0（淘汰 hash 10）
        let b = p.allocate(2).unwrap();
        assert_eq!(b, vec![2, 0]);
        assert_eq!(p.get_cached(0, h(10)), None, "evicted");
        assert_eq!(p.get_cached(0, h(11)), Some(1), "still cached (ref>0)");
    }

    #[test]
    fn same_hash_different_group_are_distinct() {
        let mut p = BlockPool::new(2, true);
        let a = p.allocate(2).unwrap();
        p.cache_block(a[0], 0, h(7));
        p.cache_block(a[1], 1, h(7));
        assert_eq!(p.get_cached(0, h(7)), Some(0));
        assert_eq!(p.get_cached(1, h(7)), Some(1));
    }
}
