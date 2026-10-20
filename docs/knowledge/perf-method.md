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

## 7. 引擎层面 vs 模型层面：两套口径，各管一件事（用户 2026-10-20）
- **判断**：四组闭环（固定并发 / 固定长度 / ignore_eos，`bench_compare.sh`）是**模型层面**——量 kernel + graph + 自调优，每步 batch 恒定、prefill/decode 天然错开、prefix cache 只命中模板、无尾部。它只做每轮性能改动的**回归门禁**。"超越 vLLM" 的判定用**引擎层面**：泊松开环到达率扫描 + prompt / 输出长度分布 + 不 ignore_eos + 共享前缀 / 多轮，指标 = SLO 下 goodput（首要：**SLO 下最大 req/s**）、TTFT / TPOT 的 p99 / max、prefix cache 命中率、饱和后行为、长跑漂移（`bench_engine.sh` → `engine-trend.*`）。
- **为什么**：Qwen3.5 四组 1.10× 的数字下，调度器 / 抢占 / prefix cache / 尾部 / 容量全被抹平；B300 首个 `vllm bench serve` 点（rate 8）TPOT p50/p99 29.5/45 vs vLLM 16.6/19.4——差距在尾部而非均值，闭环看不到。
- **怎么用**：改 kernel → 跑四组门禁；改调度 / cache / 抢占 → 跑 `bench_engine.sh`，两边各自最佳配置写进 summary 头；两边同 seed（同到达序列）；先 `bench_engine_vllm_tool.sh` 对账一次（同 rate 同长度 p50/p99 差 < 10%）再信自己的数字。对账 TTFT 用 `ttft_first_chunk_ms`（vLLM 工具把 dinfer 的空 role delta 当首 token → ~5 ms 伪像）。
- **误读**：goodput ≥ 0.95×名义 rate 判"扛住"会把短跑（每率 96 条）全好的点判掉——收尾长尾压低实际 req/s；用 good/completed ≥ 0.95。
- 证据：`docs/design.md` §15；`docs/loops/engine-bench.md`；`test/qwen35/2026-10-20/engine_smoke`。

## 8. 引擎层面差距的定位顺序（engine-gap 第 1–4 轮沉淀，2026-10-20）
1. **先把工具做可信**（§7）：钉核、Rust bench、交叉、对账。前两轮数字因此作废。
2. **步模式差值**：bench 前后抓 `/metrics` 的 `engine_step_execute_{prefill,decode,mixed}` count/sum 差 → 三种步的次数与均值；再用 nsys 按"步内 kernel 含 prefill / decode 标志"分类（`bootstrap/nsys_mixed.py`）拿 GPU 真实 span。两者对照：metrics 口径含 async 等待，nsys 是 GPU。
3. **TPOT − 纯 decode 步 span = 步间开销**：engine2 里 23.8 − 13.2 = 10 ms/token，全在混批步（52 × 125 ms 占 decode 墙钟 32%）。
4. **先看混批步是否比"纯 prefill + 纯 decode"之和还贵**：我们 mixed 125 > 70 + 13，说明混批路径本身有额外成本（eager 不走 graph、两种 attention 都 launch、自调优桶按总 M 命中），这是引擎实现问题，不是负载不可避免。
5. 读 vLLM 源码核机制时要看**默认路径**：`throttle_prefills` 只在 DP 子类生效，单卡默认 False——差点把一个不存在的机制当成差距来源。
