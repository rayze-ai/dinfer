//! KV cache 管理。对应 vLLM `v1/core/{kv_cache_manager,block_pool,kv_cache_utils,kv_cache_coordinator}.py`。

pub mod block_pool;
pub mod hash;
pub mod host_tier;
pub mod manager;
pub mod snapshot;
pub mod spec;

pub use block_pool::{BlockId, BlockPool};
pub use hash::{hash_block, BlockHash, BlockHasher};
pub use manager::{GroupInfo, KvBlocks, KvCacheManager};
pub use snapshot::StateSnapshotPool;
pub use spec::KvCacheSpec;
