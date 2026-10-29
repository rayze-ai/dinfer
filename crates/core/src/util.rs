//! 小工具：面向 u64 键的恒等哈希（RequestId 是单调计数器，无需打散）。

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

#[derive(Default, Clone, Copy)]
pub struct IdHasher(u64);

impl Hasher for IdHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // 只会被 u64/usize 键调用；保守处理任意长度。
        for chunk in bytes.chunks(8) {
            let mut b = [0u8; 8];
            b[..chunk.len()].copy_from_slice(chunk);
            self.0 = self.0.rotate_left(5) ^ u64::from_le_bytes(b);
        }
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.0 = i as u64;
    }
}

pub type IdHashMap<V> = HashMap<u64, V, BuildHasherDefault<IdHasher>>;

pub type IdHashSet<K = u64> = std::collections::HashSet<K, BuildHasherDefault<IdHasher>>;
pub fn id_hash_set<K>() -> IdHashSet<K> {
    std::collections::HashSet::with_hasher(BuildHasherDefault::default())
}
pub fn id_hash_map<V>(capacity: usize) -> IdHashMap<V> {
    HashMap::with_capacity_and_hasher(capacity, BuildHasherDefault::default())
}
