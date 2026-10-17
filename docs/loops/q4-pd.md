# Q4 PD 分离（loop 规格）

设计与四个决策见 `docs/design.md` §10.3。本文件只放环境、观察方法、停止条件、迭代记录。

## 环境
- 4090 测试机：P 在 GPU 3 `:8101`（`--role prefill --kv-port 9101`），D 在 GPU 3 `:8102`（`--role decode --kv-port 9102`，同卡两进程先验流程），proxy `:8100`（`dinfer proxy --prefill :8101 --decode :8102`）。GPU 0 `:8012` 常驻单机对照不动。
- 本机 3090：e2e `e2e_qwen3_pd`（三进程同卡，tiny 块数）。
- 代码落点：`crates/core/src/pd/`（迁移消息、`KvTransport` trait、TCP 实现、`IncomingKv` 表）；scheduler 新增请求来源 `WaitingForKv`；`models/qwen3/src/executor.rs` 复用 `HostTierStore` 的 pack/unpack + arena；`crates/server/src/proxy.rs`；CLI `serve --role {single,prefill,decode}`、`proxy`。

## 观察方法
- 正确性：`e2e_qwen3_pd` greedy 输出 vs 单机逐字比对；D 的 `/metrics` `pd_kv_recv_units_total > 0`、`pd_fallback_total == 0`。
- 故障：`kill -9` P / D、`iptables`（或直接关 kv-port 监听）各一次，请求 200 + `pd_fallback_total{reason}` 计数正确。
- 性能：`bench --concurrency 8 --prompt-len 1024 --max-tokens 128` 打 proxy `:8100` vs 单机 `:8012`，`bench_compare.sh test/qwen3/<日期>/pd1 "PD v1 TCP"`。

## 停止条件（机器可判定）
`scripts/verify/q4.sh` exit 0：e2e PASS + 三种故障回退 PASS + bench TTFT ≤ 1.3× 单机、TPOT ≤ 单机。迭代上限 12 轮。

## 记录
| 轮 | 块 | 现象 | 根因 | 处理 |
|---|---|---|---|---|
| 1 | core::pd 骨架（2026-10-18） | `crates/core/src/pd.rs`：线协议 `magic|hlen|header(JSON)|payload`（`MigrationHeader{request_id, model_tag, prompt_ids, sampling, first_token, units:(gid,BlockHash), unit_bytes}`）、`KvTransport` trait + `TcpTransport`、D 侧 `IncomingKv`（put/take/gc/stats）+ `spawn_receiver` 线程（model_tag 不符拒收）。`SamplingParams` / `BlockHash` 加 serde derive | — | 3 单测过（roundtrip / TCP 收发+拒收 / gc）；workspace 编译过。下一轮：scheduler 接 `IncomingKv`（带 request_id 的新请求 → `register_fetched` 路径 + `WaitingForKv`），executor `kv_import(payload) / kv_export(blocks) -> payload` 复用 pack/unpack + arena |
| 2 | core 接线方案定稿（未动代码，2026-10-18） | 接入点已查清：`Request{all_token_ids, num_prompt_tokens}` 私有、`append_output(tok)`；scheduler 首次调度块在 `sched/mod.rs:322–372`（`get_computed_blocks` → host_tier_lookup → `allocate_slots(slot, r, num_new+host_tokens, num_new_computed, computed, lookahead)` → `register_fetched`）；`seen_by_worker` 在 :387–395 决定 `new_reqs`；`finish_slot` :524 先 `kv.free(slot)`；runner `kv_fetch` 在 forward 前 :261、`kv_offload` 在 forward 后 :291；engine `Cmd::Submit` :360 建 `Request::new`，`publish_step` :124 推 `Finished`；executor `enable_host_tier` :643 建 `HostTierStore{arena, copy, staging, blocks_buf, layer_ptrs, ...}`，`kv_fetch/kv_offload` :822/:844 是 pack/unpack 范本 | — | **方案**：① `Request` 加 `pd: Option<PdRole>`，`PdRole::Export`（P，首 token 后导出）/ `PdRole::Import{first_token}`（D）；`Submission` 同步加字段，`ChatBackend::submit` 透传。② **按位置不按哈希**：P 导出该请求全部块（含最后一个未满块，prompt_len 不是 16 倍数时哈希路径会丢尾块），`SchedulerOutput.kv_export: Vec<(RequestId, Vec<BlockId>)>`；D 侧 `kv_import: Vec<(BlockId, usize /*payload 块下标*/)>`。③ D 首次调度 `Import` 请求：跳过 prefix/host 查找，`allocate_slots(slot, r, prompt_len, 0, None, 0)`，推 `kv_import`，`num_computed_tokens = prompt_len`，`append_output(first_token)`，然后走正常 `cache_blocks`（满块登记哈希，供后续 prefix 命中），进 running。该请求本步**不进 forward**（没有新 token 要算：首 token 已由 P 采样；下一步 decode 正常算）。④ P 侧 `Export` 请求 finish 时 `finish_slot` **先推 `kv_export` 再 free**；runner 在 step 末同步执行 `exec.kv_export(blocks) -> Vec<u8>`（pack → staging → pinned，64 块一批，~6 ms/112 MB，首 token 已返回不影响 TTFT），engine 拿到 payload 后在独立线程 `KvTransport::send`；`kv_import(blocks, payload)` 反向。⑤ `ModelExecutor` 加 `kv_export / kv_import` 默认空实现；qwen3 executor 复用 `HostTierStore`（`--role prefill/decode` 时即使 `kv_host_gib=0` 也建一个 64 槽小 arena）。⑥ `IncomingKv` 挂在 engine：`Cmd::Submit` 带 `PdRole::Import` 时先 `take(request_id)`，没到就把 Submission 放 `pd_pending` 队列，每步开头重查，超 2 s 转普通请求（`pd_fallback_total{reason="timeout"}`）。⑦ 单测（CPU，mock executor）：Import 请求首步不进 forward、`num_computed == prompt_len`、第二步 decode 1 token；Export 请求 finish 产出 `kv_export` 且块仍可读；超时回退。下一轮按 ①→⑦ 写 |
