# loop: P1 调度器性能门禁

## 目标

`schedule() + update_from_output()`，1024 个 decode 请求一步，中位数 < 200 µs（prefix cache 开/关都要过）。

## 环境

- 本机 3090 主机 CPU，`cargo bench -p dinfer-core --bench scheduler`（criterion，release）。
- 更细粒度：`cargo run --release -p dinfer-core --example probe_sched`（手工计时，分 schedule / update 两段）。
- 热点：`perf record -g` 对 `target/release/examples/probe_sched`，`perf report --no-children`。

## 观察方法

1. `scripts/verify/p1.sh` 末段解析 criterion 输出，两条 bench 的中位数都 < 阈值才 exit 0。
2. `probe_sched` 看 schedule 与 update 各占多少；perf 看前 10 个符号。

## 停止条件（机器可判定）

`scripts/verify/p1.sh` exit 0。

## 迭代上限

6 轮。超过则记录现状进 progress.md，门禁暂放宽到实测值 ×1.2，P3 再回头。

## 记录

| 轮 | 改动 | pc=false | pc=true | 备注 |
|---|---|---|---|---|
| 0 | 初版（HashMap<RequestId, Request>，每步 cache_blocks 全量） | 298 | 449 | |
| 1 | slab + slot 索引、StepResults 扁平缓冲、cache_blocks 跨边界早退、恒等哈希 | 197 | 310 | pc=true 差 110 µs 不合理 |
| 2 | 发现 bench 本身的问题：`iter_batched` 把每次重建 1024 请求的 setup 计入；改为持久状态 `iter` | **95** | **146** | `probe_sched` 手工计时 62 / 92 µs 与之吻合。pc=true 多出的 ~50 µs 是每 token 一次 hash 块边界检查 + 每 16 步一次 xxh3 + cache_block 登记，合理。**门禁通过，loop 结束** |

## 结论

- 真实每步开销：schedule ≈ 35 µs（pc=off）/ 56 µs（pc=on），update ≈ 27 / 36 µs。
- 本机没装 perf（`which perf` 命中的是无效路径），热点分析靠 `probe_sched` 分段计时。P3 起若需要再装 `linux-tools`。
- 后续若要再压：`get_cached` 的 HashMap<(u8,u128)> 可换 open-addressing；`append_output` 可改为只在 `num_tokens % hbs == 0` 时才走 hasher（已做）。
