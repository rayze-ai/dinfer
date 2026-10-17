# Q3-r PinnedHostTier（KV 分层第一层，loop 规格）

> design §10.2 的第一步：把 prefix cache 的容量从显存（4090 17 GiB ≈ 158k token）扩到主机 pinned 内存。
> 范围（本轮）：单机、单层 host tier、**fetch 同步**（主流上排在本步 forward 前，PCIe 拉 1024 token 的块 ~5–10 ms，仍远快于重算 ~20 ms）、**offload 异步**（copy stream，等主流事件后 D2H）。远端 / PD 复用同一 order 形态，下轮。

## 设计（落地版，比 §10.2 的 trait 简化：先不抽 trait，直接实现；接口形态保留给 RemoteTier）
- **scheduler 侧**（`core::kv::host_tier::HostTierIndex`，纯 host 索引：`HashMap<(group,BlockHash) → HostSlot>` + LRU 空闲链；容量 = `kv_host_gib / bytes_per_block`）：
  - 首次调度：`get_computed_blocks` 得 GPU 命中 h_gpu 后，继续用请求的 block hashes 在 HostTierIndex 查连续命中 h_host；`allocate_slots(num_new = h_host_tok + 本步 prefill)`；对 h_host 个新块各发一条 `KvFetch { host_slot, block }`，并立刻 `pool.cache_block`（它们马上就是有效缓存）。`num_computed_tokens` 推进 h_gpu + h_host，本步只算其后的 prompt。
  - `cache_blocks_inner` 登记新凑满的块时：若 host 索引没有该 hash → 分配 HostSlot，发 `KvOffload { block, host_slot }`（LRU 淘汰旧 hash）。
  - `SchedulerOutput` 新增 `kv_fetch: Vec<(BlockId, HostSlot)>`、`kv_offload: Vec<(BlockId, HostSlot)>`。
- **executor 侧**（`Qwen3Executor` 持 `copy_stream` + pinned 竞技场 `[num_host_slots × bytes_per_block_all_layers]`）：
  - `ModelExecutor::kv_fetch(orders)`：主流上对每 (slot, block) 每层一次 `cuMemcpyHtoDAsync`（28 次 × 64 KB），**同步排在本步 forward 前**（同流，无需事件）。
  - `ModelExecutor::kv_offload(orders)`：copy stream `wait(main_done)` 后每层一次 `cuMemcpyDtoHAsync`；主流下次 fetch 前 `wait(copy_done)`（host slot 若刚被 offload 写、又被 fetch 读的情况）。
  - host slot 布局 `[layer][block_size × Hkv × 2 × D]` 连续，与设备 `cache[layer][block]` 一一对应。
- **runner**：`begin/begin_cut_in` 里 `update_states` 之后、enqueue 之前 `exec.kv_fetch(&so.kv_fetch)`；`finish` 之后 `exec.kv_offload(&so.kv_offload)`（异步，不等）。
- 开关：`--kv-host-gib N`（默认 0 = 关）。指标：`host_tier_hits_total / host_tier_fetch_blocks_total / host_tier_offload_blocks_total`。

## 验收
1. core 单测：HostTierIndex 命中 / LRU；scheduler 用例：GPU 容量只够 1 个请求、第 2 个同前缀请求先 offload 再 fetch（断言 kv_fetch 非空、num_computed 推进）。
2. 3090 e2e：新用例 `e2e_qwen3_host_tier`：`--num-blocks 64 --kv-host-gib 1`，长前缀 A 跑完 → 跑 B（不同前缀，把 A 挤出 GPU）→ 再跑 A，`/metrics` host_tier_hits > 0 且输出与第一次逐字相同。
3. 4090 GPU 3：`bench --prefix-repeat 0.5`（50% 请求共享一个 1024-token 前缀）在 `--num-blocks 2048`（32k token，不够装）下，开 tier vs 关 tier 的 TTFT。
迭代上限 6 轮。

## 记录
| 轮 | 块 | 现象 | 根因 | 处理 |
|---|---|---|---|---|
| 1 | 实现 | 2026-10-17 | — | 索引 + scheduler 两处接入 + executor 拷贝 + runner 接线 + CLI + 指标 |
| 2 | core 侧 | `kv/host_tier.rs` 索引（HashMap + 线性 LRU，单测）；manager：`host_tier_lookup / register_fetched_blocks / pending_offload`，`cache_blocks_inner` 登记时顺带 insert → offload；scheduler 首次调度：GPU 命中之后查 host 命中 → 多分配 host_tokens 的块、发 `kv_fetch`、num_computed 推进；`SchedulerOutput.kv_fetch/kv_offload`；runner 在 enqueue 前 `kv_fetch`、后 `kv_offload`；metrics 三个计数 | — | executor 侧（copy stream + pinned 竞技场）→ CLI → e2e |
| 3 | executor 侧 | `HostTierStore`（pinned 竞技场 `[slot][layer][block]` + copy stream + offload_done 事件）；`kv_fetch` 主流同步 HtoDAsync 逐层逐块；`kv_offload` copy stream 等主流事件后 DtoHAsync；`enable_host_tier(host_gib)` 算槽位数回给 scheduler；CLI `--kv-host-gib`。host_tier 单测 LRU 期望修正（free 槽 pop 顺序 0,1,2）| — | 编译 → core 单测 → 3090 e2e 用例 |
| 4 | 全链路编译 | qwen3 crate 禁 unsafe → pinned 竞技场与裸拷贝下沉到 `dinfer_kernels::{HostArena, Stream::{htod_from_arena, dtoh_to_arena}}`；executor `HostTierStore{arena, copy, offload_done}`、`enable_host_tier(host_gib) -> num_slots`；service 用槽位数 `sched.enable_host_tier`；CLI `--kv-host-gib`。core 35/35（含 host_tier LRU 单测）、real_parity 3/3 | — | e2e 用例 `e2e_qwen3_host_tier`（--num-blocks 64 --kv-host-gib 1：A → B 挤出 → A 再来 host 命中、逐字相同）|
| 5 | 3090 e2e | `e2e_qwen3_host_tier` PASS（A 700 token 前缀 → offload；B 挤出 A；A 再来 host 命中、fetch > 0、greedy 6 token 逐字相同）；`prefix_cache` 用例不受影响 | — | 4090 baseline（含新 e2e）+ GPU 3 前缀复用负载 tier on/off TTFT 对比 |
