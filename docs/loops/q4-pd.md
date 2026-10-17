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
