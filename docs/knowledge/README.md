# 知识库（docs/knowledge/）

> 与 `pitfalls.md`（按时间堆的事故记录）、`loops/`（逐轮实验日志）、`progress.md`（交接）的分工：这里只放**提炼后可复用的判断与方法**，每条带"为什么"和"证据"链接，不复制事实。每次收功（`/project-skill wrap`）把本次 loop 里可复用的教训提炼进来；重启会话先读本索引再读 `progress.md`。

| 页 | 内容 | 适用时机 |
|---|---|---|
| [perf-method.md](perf-method.md) | 性能迭代方法论：先 profile 再优化、口径、门禁、常见误读 | 任何性能 loop 开工前 |
| [kernel-rules.md](kernel-rules.md) | 写 / 改 CUDA kernel 的硬规则与已验证的结构选择（按卡型） | 动 `crates/kernels` 前 |
| [model-onboarding.md](model-onboarding.md) | 接新模型的方法与盲区清单（golden 怎么造、对拍口径、与 vLLM 的差异点） | 开新 `models/<m>` 前 |
| [executor-common.md](executor-common.md) | executor 公共件（async / graph / 槽表 / plan 上传）用法，模型 crate 该剩什么 | 给模型接 executor 前 |
| [subagent-ops.md](subagent-ops.md) | 多子代理并行作业：brief 模板、文件分工、远端共用目录、后台任务超时 | 派子代理前 |
| [infra-b300.md](infra-b300.md) | B300-50 / 4090 / 3090 三机口径：连接、口径一致性、GPU 编号、容器、bench 流程 | 上远端机器前 |

**维护规则**：一条知识 = 一个判断 + 为什么 + 怎么用 + 证据（commit / loop 行 / bench 目录）。过时的删，不要注释掉。单页 ≤ 120 行，超了拆页。
