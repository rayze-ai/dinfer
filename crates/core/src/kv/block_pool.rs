//! 物理 block 池：空闲链表 + 引用计数。对应 vLLM `block_pool.py` 的最小子集。
//! LRU 淘汰与 hash → block 映射在 P1 加。

pub type BlockId = u32;

#[derive(Debug)]
pub struct BlockPool {
    ref_cnt: Vec<u32>,
    free: Vec<BlockId>,
}

impl BlockPool {
    pub fn new(num_blocks: u32) -> Self {
        // 倒序压栈，这样 pop 出来的 id 从 0 递增，便于测试与 debug。
        let free = (0..num_blocks).rev().collect();
        Self { ref_cnt: vec![0; num_blocks as usize], free }
    }

    #[inline]
    pub fn num_free(&self) -> u32 {
        self.free.len() as u32
    }

    #[inline]
    pub fn num_total(&self) -> u32 {
        self.ref_cnt.len() as u32
    }

    /// 分配 n 个 block；不足则返回 None 且不改变状态。
    pub fn allocate(&mut self, n: u32) -> Option<Vec<BlockId>> {
        if (self.free.len() as u32) < n {
            return None;
        }
        let at = self.free.len() - n as usize;
        let mut ids = self.free.split_off(at);
        // 栈顶在末尾；翻转后与 free 列表的压栈顺序一致（首次分配得到 0,1,2,…）。
        ids.reverse();
        for &id in &ids {
            self.ref_cnt[id as usize] = 1;
        }
        Some(ids)
    }

    /// 引用 +1（prefix cache 命中共享）。
    pub fn touch(&mut self, id: BlockId) {
        self.ref_cnt[id as usize] += 1;
    }

    /// 引用 -1，归零则回收。
    pub fn free(&mut self, id: BlockId) {
        let c = &mut self.ref_cnt[id as usize];
        debug_assert!(*c > 0, "double free of block {id}");
        *c -= 1;
        if *c == 0 {
            self.free.push(id);
        }
    }

    pub fn free_all(&mut self, ids: impl IntoIterator<Item = BlockId>) {
        for id in ids {
            self.free(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_free_roundtrip() {
        let mut p = BlockPool::new(4);
        let a = p.allocate(3).unwrap();
        assert_eq!(a, vec![0, 1, 2]);
        assert_eq!(p.num_free(), 1);
        assert!(p.allocate(2).is_none());
        p.free_all(a);
        assert_eq!(p.num_free(), 4);
    }

    #[test]
    fn shared_block_refcount() {
        let mut p = BlockPool::new(2);
        let a = p.allocate(1).unwrap();
        p.touch(a[0]);
        p.free(a[0]);
        assert_eq!(p.num_free(), 1, "still referenced once");
        p.free(a[0]);
        assert_eq!(p.num_free(), 2);
    }
}
