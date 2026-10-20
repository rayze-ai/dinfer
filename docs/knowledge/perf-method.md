# 性能迭代方法论（提炼自 Qwen3-0.6B Q3 44 轮、Qwen3.5 里程碑 6 共 9 轮）

## 1. 先 profile 再优化，kernel 级，不要按直觉开工
- **判断**：改动前必须有 nsys **node 级**（`--cuda-graph-trace=node`）的"每 decode 步分 kernel 耗时表"，而且要**只取纯 decode 步**（排除含 prefill kernel 的步、warmup、收尾小批）。
- **为什么**：Qwen3.5 第 2、3 轮（CUDA graph、GEMM 自调优）各花一轮却几乎零收益，因为真因是 `Linear::forward` 对预分配缓冲推断出 M=2112，两轮都在给一个假形状优化。node 级 trace 一眼看到 `128x192 2cta` 大 tile 名字才暴露。
- **怎么用**：`scripts/dev/b300_nsys_decode.sh` + `bootstrap/nsys_steps.py`（按 `sample_fused` 切步、按是否含 prefill kernel 分类）。看到 decode 步里出现大 M 的 tile 名、自调优日志 µs 与 nsys 实测对不上 → 先查调用形状。
- 证据：`docs/loops/qwen35-perf.md` 3c–3e、`pitfalls.md`"Linear::forward 推断行数"。

## 2. nsys 的次数 ÷ 步数前，先用与 M 无关的 kernel 反推步数
- **判断**：`Instances / 步数` 得到"每步次数"时，步数要用 `sample_fused` 之类每步恰一次的 kernel 数，且排除 warmup / 自调优 / graph 捕获。
- **为什么**：第 8d 行把 warmup（M=1，256 步）的 `small_m<1>` 算进 c32 步数，得出"96 次/步、省 1 ms"的假 bug，第 9 轮一半精力花在证伪。
- 证据：loop 8d → 9（疑点 1）。

## 3. 分清"GPU 空闲"与"kernel 慢"：span vs kernel 之和
- **判断**：每步 span − kernel 之和 > 0.3 ms → 是 host 往返（同步 D2H、scheduler、上传），上 async scheduling；≈ 0 或 < 0（相邻步重叠）→ 只能减 kernel 时间 / 数量。
- **为什么**：Qwen3.5 第 8 轮 async 前 span 15.57 > kernel 14.99；接 async 后 13.39 < 14.10，之后所有收益都来自 kernel 本身。
- 证据：loop 6d / 8c。

## 4. 物理下限先算出来，再决定做局部还是换结构
- **判断**：decode TPOT 下限 = 权重字节 / HBM 带宽（27B bf16 单卡 B300 = 6.5 ms）。实测 / 下限 > 60% 后，同结构再抠上限约 1.1×；要质变只能减 kernel 数（megakernel）或减字节（低精度）。
- 证据：`design.md` §14.1。

## 5. 口径与门禁
- bench 两边**预算一致**（`--max-num-batched-tokens` 与 vLLM 容器同值；预算是 TTFT↔TPOT 的 tradeoff 不是代码问题）；每次换 seed；每组前 2 条预热不计；流式 TTFT = 首个内容 chunk。
- 每轮改动 → `bench_compare.sh <dir> "一句话"` → 进 `perf-trend.*` → commit；没进 CSV 视为未完成。
- 性能改动过**两卡门禁**（4090 sm89 + B300 sm_103，任一卡退化 > 3% 不合入）；kernel 内不写卡型 / 模型调优数值（`ArchProfile` + 启动自调优，环境变量只覆盖）。
- 证据：`.claude/CLAUDE.md` 硬性规则；`scripts/verify/bench_compare.sh`。

## 6. 常见误读清单
| 现象 | 错误结论 | 真因 | 证据 |
|---|---|---|---|
| 启动日志自调优 16 µs，TPOT 不变 | 自调优没命中 | 运行时 M 是缓冲行数 2112 | loop 3e |
| "每步 ~900 kernel 的边界效应 = 3.8 ms"（§14.1） | megakernel 能拿回 1.4× | graph 下单层 11 kernel 基线离权重流下限只差 17%；边界效应被 graph 吃掉大半，megakernel 的占用代价反而更大 | loops/mega-sm100.md 结论 |
| CUDA graph 零收益 | launch 不是瓶颈 | 对，但原因是 kernel 在算假 M | loop 2b/3e |
| 新 kernel 微基准 4×，bench 持平 | B300 与 3090 不同 | 分发条件 `tokens == num_seqs` 因缓冲推断永不成立 | loop 4b/4c |
| 卡空却报 OOM | GPU 编号错 | 每层激活缓冲 × 预算 16384 = 170 GB | pitfalls "nsys 起服务 OOM" |
| 某 kernel 次数/步异常高 | 分发 bug | nsys 把 warmup 段算进来 | loop 9 疑点 1 |

## 7. 引擎层面 vs 模型层面，以及测试程序本身（用户 2026-10-20）
- **模型层面**（四组闭环、固定长度、ignore_eos）量 kernel + graph；**引擎层面**（泊松 + 长度分布 + 多轮前缀 + 不 ignore_eos）量调度 / prefix cache / 尾部 / 容量，口径 `requirements.md` 验收 v2 / `design.md` §15。Qwen3.5 四组 1.10× 时引擎层面 goodput 只有 vLLM 的 0.07–0.44×——两者可以完全脱节，**只报引擎层面的 goodput / p99 给用户**。
- **先怀疑工具再怀疑引擎**：首个引擎数据点里 TTFT 5 ms 是 server 提前发空 role chunk 的伪像；goodput 0.00 是编错 feature 的二进制。每轮必带：压测端 CPU 集合与 `getrusage`、`--self-check` 的计时误差、与第三方工具的交叉差、客户端 vs 服务端自报的对账。压测端与服务共核 / CPU > 1 核 → 数据作废。
