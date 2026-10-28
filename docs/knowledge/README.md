# 知识库（docs/knowledge/）

> 与 `pitfalls.md`（按时间堆的事故记录）、`loops/`（逐轮实验日志）、`progress.md`（交接）的分工：这里只放**提炼后可复用的判断与方法**，每条带"为什么"和"证据"链接，不复制事实。每次收功（`/project-skill wrap`）把本次 loop 里可复用的教训提炼进来；重启会话先读本索引再读 `progress.md`。

| 页 | 内容 | 适用时机 |
|---|---|---|
| [perf-method.md](perf-method.md) | 性能迭代方法论：先 profile 再优化、口径、门禁、常见误读 | 任何性能 loop 开工前 |
| [perf-tooling.md](perf-tooling.md) | 把测量做对：改动必验二进制、nsys 图内 kernel 清单抓法、卡型旋钮 A/B 脚本 | 任何 Step A 测量 / Step B 验证前 |
| [kernel-rules.md](kernel-rules.md) | 写 / 改 CUDA kernel 的硬规则与已验证的结构选择（按卡型） | 动 `crates/kernels` 前 |
| [blackwell-mma.md](blackwell-mma.md) | sm_100/103/120 低精度 MMA：gencode、NVFP4/FP8 指令与布局、实测吞吐 | 写 Blackwell MMA 前 |
| [megakernel.md](megakernel.md) | persistent kernel 的结构事实（组数 / 流水 / CTAs per SM） | 做单层 persistent 前 |
| [model-onboarding.md](model-onboarding.md) | 接新模型的方法与盲区清单（golden 怎么造、对拍口径、与 vLLM 的差异点） | 开新 `models/<m>` 前 |
| [executor-common.md](executor-common.md) | executor 公共件（async / graph / 槽表 / plan 上传）用法，模型 crate 该剩什么 | 给模型接 executor 前 |
| [parity-localization.md](parity-localization.md) | 数值对不上时怎么定位：口径/1-ulp/放大器三分、整跑 vs 隔离、A/B 开关、调参照自己的算子、前置条件 | 端到端读数不齐时 |
| [segment-isolation.md](segment-isolation.md) | 逐段隔离定位（`--layer-iso` + `SUB_LAYER`）、参照数据自证三项、口径红线、数据布局 | 数值对不上、要抓参照中间量前 |
| [bitexact-replication.md](bitexact-replication.md) | 定位到算子后怎样写到逐位一致：源码→PTX→SASS、Triton cache/nvdisasm、允许区间反推、探针干扰 | 复刻参照 kernel 数值前 |
| [subagent-ops.md](subagent-ops.md) | 多子代理并行作业：brief 模板、文件分工、远端共用目录、后台任务超时 | 派子代理前 |
| [infra-b300.md](infra-b300.md) | B300-50 / 4090 / 3090 三机口径：连接、口径一致性、GPU 编号、容器、bench 流程 | 上远端机器前 |

**维护规则**：一条知识 = 一个判断 + 为什么 + 怎么用 + 证据（commit / loop 行 / bench 目录）。过时的删，不要注释掉。单页 ≤ 120 行，超了拆页。
