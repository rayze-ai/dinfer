//! KV cache 管理。对应 vLLM `v1/core/{kv_cache_manager,block_pool,kv_cache_utils}.py`。
//!
//! P0 只放类型与 BlockPool 的最小实现；prefix cache、多 group、Mamba 状态在 P1 补齐。

pub mod block_pool;
pub mod spec;

pub use block_pool::{BlockId, BlockPool};
pub use spec::KvCacheSpec;
