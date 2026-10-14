//! prefix cache 的块哈希：链式 xxh3-128。对应 vLLM `kv_cache_utils.py::hash_block_tokens`。
//!
//! `hash(block_i) = xxh3_128(parent_hash ‖ token_ids[block_i] ‖ extra)`。
//! 第一块的 parent 是一个固定种子（vLLM 用 `NONE_HASH`，由随机/环境变量决定；我们固定为常量，
//! 多租户隔离靠 `cache_salt` 混入 extra）。

use xxhash_rust::xxh3::xxh3_128_with_seed;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockHash(pub u128);

const ROOT: u128 = 0x6d69_6e66_6572_2d72_6f6f_742d_6861_7368; // "dinfer-root-hash"
const SEED: u64 = 0x5f44_494e_4645_525f; // "_DINFER_"

#[inline]
pub fn hash_block(parent: Option<BlockHash>, tokens: &[u32], extra: &[u8]) -> BlockHash {
    let parent = parent.map_or(ROOT, |h| h.0);
    let mut buf = Vec::with_capacity(16 + tokens.len() * 4 + extra.len());
    buf.extend_from_slice(&parent.to_le_bytes());
    for &t in tokens {
        buf.extend_from_slice(&t.to_le_bytes());
    }
    buf.extend_from_slice(extra);
    BlockHash(xxh3_128_with_seed(&buf, SEED))
}

/// 增量维护一个请求的块哈希列表：只对"新凑满"的块算 hash。
#[derive(Debug, Default, Clone)]
pub struct BlockHasher {
    pub hashes: Vec<BlockHash>,
    hash_block_size: u32,
    salt: Vec<u8>,
}

impl BlockHasher {
    pub fn new(hash_block_size: u32, salt: Vec<u8>) -> Self {
        Self { hashes: Vec::new(), hash_block_size, salt }
    }

    /// 根据当前完整 token 序列补齐 hash（幂等）。
    pub fn update(&mut self, all_tokens: &[u32]) {
        let bs = self.hash_block_size as usize;
        let num_full = all_tokens.len() / bs;
        while self.hashes.len() < num_full {
            let i = self.hashes.len();
            let parent = if i == 0 { None } else { Some(self.hashes[i - 1]) };
            // salt 只混进第一块，后续块通过链式 parent 继承。
            let extra: &[u8] = if i == 0 { &self.salt } else { &[] };
            self.hashes.push(hash_block(parent, &all_tokens[i * bs..(i + 1) * bs], extra));
        }
    }

    /// 0 表示未启用。
    #[inline]
    pub fn hash_block_size(&self) -> u32 {
        self.hash_block_size
    }

    pub fn truncate_to_tokens(&mut self, num_tokens: usize) {
        self.hashes.truncate(num_tokens / self.hash_block_size as usize);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chained_and_salted() {
        let a = hash_block(None, &[1, 2, 3], b"");
        let b = hash_block(None, &[1, 2, 3], b"salt");
        let c = hash_block(Some(a), &[4, 5, 6], b"");
        let d = hash_block(Some(b), &[4, 5, 6], b"");
        assert_ne!(a, b);
        assert_ne!(c, d, "salt propagates through the chain");
        assert_eq!(a, hash_block(None, &[1, 2, 3], b""));
    }

    #[test]
    fn incremental_update_is_idempotent() {
        let mut h = BlockHasher::new(4, vec![]);
        let toks: Vec<u32> = (0..10).collect();
        h.update(&toks[..5]);
        assert_eq!(h.hashes.len(), 1);
        h.update(&toks);
        assert_eq!(h.hashes.len(), 2);
        h.update(&toks);
        assert_eq!(h.hashes.len(), 2);
        let mut fresh = BlockHasher::new(4, vec![]);
        fresh.update(&toks);
        assert_eq!(h.hashes, fresh.hashes);
    }
}
