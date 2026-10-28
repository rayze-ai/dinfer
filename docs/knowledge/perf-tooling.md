# 性能工具链：怎样拿到可信的"谁大"（perf-method 的工具页）

> 与 [perf-method.md](perf-method.md) 分工：那页讲方法论与口径，这页讲**把测量做对**的具体手法与坑（编译验证、nsys、A/B 脚本）。来源：loop `kaon3-decode-opt` 2026-10-28。

## 1. 判瓶颈先做"整段 skip"，但 skip 结果只在**确认改动已编进二进制**后才算数（2026-10-28 kaon3-decode-opt）
- **判断**：改 kernel / 加 skip 开关后，必须 `strings 二进制 | grep <新字面量>` 验证进了，再看结果；否则"skip 全 0 影响"可能只是老二进制（D5–D16 整天的"瓶颈在 host"全是假象，D45 新二进制 skip 立刻看到 MoE 5.3 / paged 3.2 / dense 2.8 ms）。
- **为什么**：三条编译链路都会静默跳过：`cargo -p dinfer` 不重编依赖 crate；rsync 保留旧 mtime 让 cargo 认为源没变；改 `.cu` 后 nvcc 增量不重跑。`remote_*.sh put` 已加 `--no-times`。
- **怎么用**：Step B 验证脚本（`scripts/dev/kaon3_stepB_verify.sh`）固定做"备份旧二进制 → 编 → strings 验 marker → A/B/A2/B2 同轮 + 固定 prompt 输出"。
- **证据**：loop kaon3-decode-opt D42 / D45 / D49。

## 2. 图内 kernel 清单要用 nsys `--cuda-graph-trace=node`，且必须 `--cuda-flush-interval` + 不 INT nsys 本体（2026-10-28）
- **判断**：定 decode 图内谁大，nsys 节点级清单（kernel 名 / 次数 / 时长 / grid）比任何 host 计时或 NO_GRAPH 分段都准；一步 decode 1298 kernel 的分布半小时就把"按 max_tokens 行空跑的 norm/rope"（3 ms，18%）揪出来。
- **为什么**：CUPTI 缓冲默认只在 `cudaProfilerStop()` 落盘，被 INT 杀的服务不调它 → 报告里只有加载期 memcpy、无 kernel 表（两个版本 nsys 各失败一次）；INT 发给 nsys 本体会打断报告生成（无 .nsys-rep）。
- **怎么用**：`scripts/dev/kaon3_nsys_decode.sh`（flush 1 s，只 INT dinfer，等 nsys 自然退出）+ `scripts/dev/nsys_kern.py`（按 graphExecId 聚类取 decode 图实例，逐 kernel 聚合）。看 grid 是否等于 max_tokens / max_rows 这类上限值——那是"按缓冲满长跑"的标志。
- **证据**：loop kaon3-decode-opt D52 / D54；B1 一刀 TPOT −19.4%。

## 3. 卡型旋钮（norm layout 等）先用现成 env 三态 A/B，再落进 ArchProfile（2026-10-28）
- **判断**：能用环境变量切的实现选择，先同轮三态 A/B（含输出一致），赢家写进 `ArchProfile` 按 `Arch` 分支，不留 env 依赖。B2：Sm120 h=2816 单行 norm Block256 2.4 us < Block1024 6.8 us < warp，−3.4%。
- **怎么用**：A/B 脚本每个变体都要抓固定 prompt 的输出；输出坏（"is is is…"）的变体数字作废（D57 `KAON3_MEGA=0` 11.33 ms 就是坏的）。
