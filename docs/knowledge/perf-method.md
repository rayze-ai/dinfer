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
| decode 步 31% 是 H2D memcpy | 上传路径有问题 | 切步脚本把**加载期权重 H2D** 摊进了 decode 步；稳态每步 H2D < 0.5% | engine-gap 6c |

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

## 9. 引擎口径基线与归因链（Qwen3.5-27B B300，2026-10-20，`engine_1`）
- 基线：goodput dinfer 1.73 / 1.92 / 1.40 / 0.57 vs vLLM 1.72 / 2.55 / 3.06 / 3.14（rate 2/4/8/12，SLO 500/20）；rate 2 持平、≥ 4 落后、SLO 下最大 req/s ≈ 2 vs ≥ 12。同一时刻四组闭环是 1.10×——两口径可以完全脱节。
- 归因链（每步一个工具）：① 客户端 TPOT 23 ≈ `/metrics` decode 步均值 23（含 async 等待）→ TPOT 由 decode 步节拍决定；② nsys 纯 decode 步 span 13–19 ms → 差在 GPU 之外或 GPU 上的非 kernel 项；③ nsys 按步分类 `nsys_mixed.py` **必须先切稳态窗口**（最后一次大 H2D 之后），否则加载期拷贝摊进步里（6c 误读）；稳态下 decode 步 ≈ kernel 之和 → 差在 kernel；prefill 步 GPU 只忙 27% → host 空转（alloc / 懒加载 / 首捕获 / 同步 D2H）。④ 切片（`long_prefill_token_threshold`）把 mixed 步变小但总量不变、TPOT 不动 → 排除"mixed 步是主因"。
- 教训：闭环 bench 的短 prompt / 固定并发把"随长度 / 在飞数增长的 host 成本"全藏住了；引擎口径下先看 memcpy / memset / host 空转，再看 kernel。
- **第 7 轮收口（2026-10-20）**：按 correlationId 关联 RUNTIME↔MEMCPY 后，"H2D 31%"（加载期拷贝摊入）、"步间 8 ms host 空转"（= 等别人的 prefill/mixed 步）两条都被证伪；四段计时（`engine_step_phase_{upload,launch,wait}_{mode}`）显示 decode upload 0.06 / launch 0.9 ms，**host 侧已无可观空转**；服务期一次性开销（kernel 懒加载 71 ms + 首捕获 35 ms + alloc）用 warmup + 全桶预捕获 + 一次分配清零。**引擎口径的剩余差距回到 kernel**：decode 步 11.6 ms（与闭环同）、1024-token prefill ≈ 100 ms kernel（vLLM 更快 + prefix cache 命中），decode 请求在等这些步。教训：引擎口径下"GPU 忙 / 不忙"要用稳态窗口 + 按 correlationId 的 API↔GPU 关联判，切步脚本的粗分类只能当线索。

## 8. 三类趋势表，测量必落其一（用户 2026-10-20）

- **判断**：任何性能测量都要进 CSV 并由脚本生成页面，页面刷新可见；只写 loop 表行视为未完成。三类：`perf-trend`（四组闭环，`bench_compare.sh`）、`engine-trend`（开环泊松 / SLO goodput，`engine_trend.py`）、`probe-trend`（TTFT-长度曲线、kernel 分解、微基准伸缩、第三方 kernel 对照，`probe_trend.py`，输入 `test/<model>/<日期>/probe_*/probe.csv` 长表）。`test/index.html` 每模型一行链到三页。
- **为什么**：2026-10-20 engine-gap 8c–8g 四组探针测量只写了 markdown 表行，用户看页面发现没有；形状不同不是不进 CSV 的理由，长表（group/metric/side/x/value）能装任何探针。
- **怎么用**：探针开始前先建 `probe_<名>/probe.csv` 写表头，**一个目录只放一个对比项**（kernel 微基准 / 单发 TTFT-长度 / kernel 分解各自成目录，group 名分别用 `gdn_scale`-类 / `ttft_len` / `kernel_1k`，页面按 group 名归类），每拿到一组数就追加一行，结束跑 `python3 scripts/dev/probe_trend.py --root test/<model> && python3 scripts/dev/bench_index.py`，commit 生成物。
- **证据**：`docs/loops/engine-gap.md` 8h；`test/qwen35/2026-10-20/probe_prefill_1/`。

## 9. 调度策略改动必测"单发长 prompt 的 prefill 步数"（2026-10-20 r8 切步）

- **判断**：任何改 prefill chunk 边界 / 对齐 / 预算的调度改动，合入前除了 engine 口径（多请求、prefix-repeat），还要测**单发 6 K prompt** 的 TTFT 与 `/metrics` 的 engine_steps 增量；步数变了就有每步 ~6 ms 的固定开销乘进去。
- **为什么**：r8 的 1024 对齐切步让 6 K 单发从 1 步变 6 步，TTFT +32 ms（+10%），把 r9 kernel 的 −80 ms 实效遮掉一半；engine 口径（64 请求混批）和四组闭环（p128 / p1024）都看不出来，因为那些 prompt 要么短于粒度、要么多请求本来就多步。A/B 时只关池（`--gdn-snapshot-gib 0`）不关粒度，第一轮误判为"不是快照"的同时也差点把切步放过。
- **怎么用**：`ttft1.py <port> 5 860` + 前后 `engine_steps_total` 差值；步数 > 1 要说清为什么；无登记需要时不切。
- **证据**：`docs/loops/engine-gap.md` 8k–8m；`test/qwen35/2026-10-20/probe_prefill_2/`。

## 10. 目标要换算成每个口径的阈值再动手（2026-10-20 用户："全面超过 vLLM 1.2×"）

- **判断**：单一总目标（如"1.2×"）要拆成四组闭环 + 单发 TTFT（1K/3K/6K）+ engine（goodput / TTFT p99 / TPOT p99）各自要到的数，否则做完一轮不知道算不算达标。
- **为什么**：2026-10-20 收口时四组 1.04–1.11×、6 K TTFT 落后 9%、engine goodput 1.12×，同一个"超过 vLLM"在这三个口径上差得完全不同；只看一组会误判。
- **怎么用**：开工先写基线表（dinfer / vLLM / 倍数），每轮只更新这张表并存 `test/<model>/<日期>/`；缺口按"每 token / 每步"折算成 µs 再排序。

## 11. 带宽型 / 算力型 kernel 的上限先用同形状探针量，不用标称规格算（2026-10-21 r11b + r11a）

- **判断**：给 kernel 定"还能再快多少"之前，先写一个**同形状的裸探针**量硬上限：流式读（`probe.cu`：只 load 不做计算，任意 grid / 行块 / 读模式）量 HBM 实际可达带宽；`mma.sync m16n8k16` 空转（`mma_peak.cu`）量 tensor-core 实际可达算力。用探针数字当分母，不用 HBM / 峰值规格当分母。
- **为什么**：B300 标称 HBM 8 TB/s，实测**小 kernel（≤400 MB、单次 launch）流式读只有 5.9–6.3 TB/s**（ramp + tail 摊不开），cuBLASLt 的 6.0–6.6 已经贴顶——按 8 TB/s 算出来的"decode GEMM 还能从 6.6 提到 7.4、省 0.8 ms"这条路根本不存在，r11b 为此烧了 5 轮。同理 B300 标称 bf16 2.25 PF/s 是 **tcgen05 / FP8** 的数，**legacy `mma.sync` 峰值只有 537–575 TF**（实测，4 warps/SM 就到 537）；按 2.25 PF 定的"6 K 单层 ≤ 700 µs"需 636 TF，物理不可达，r11a 为此烧了 7 轮里的 2 轮才确认。
- **怎么用**：① 动手前先跑探针，把上限写进 loop 页的停止条件旁边；② 探针要覆盖真实规模（≤400 MB 的 kernel 用 400 MB 量，别用 2.5 GB 的 lm_head 结果外推——它能到 7.1 是靠 ramp 摊薄）；③ 差距里"探针上限 × 85%"就是现实天花板，达不到就换指令集（mma.sync → tcgen05）或换策略（融合减 kernel 数），不要再调 tile。

## 12. nsys 按 grid 归类 GEMM 时必须同时核 N/tile 与 count/步（2026-10-21 r11b）

- **判断**：`nsys stats` 里同一个 tile 名（如 `64x8`）可能被**多个不同形状**共用；按名字把耗时归到某个算子会得出"某算子只有 2.6 TB/s"的错误结论。判据：`grid = N / tile_n` 且 **count/步** 与该算子的调用次数对得上，两者都对才算同一条目。
- **为什么**：engine-gap 9f 把 `64x8`（grid 544 = 34816/64）归给了 in_proj，实际是 gate_up 的另一个 M 桶；in_proj 真实是 `128x8 4x1` grid 132、28.5 µs = 5.9 TB/s。误读导致 r11b 的 brief 直接错指了优化目标。
- **怎么用**：nsys 分解脚本里对每个 GEMM tile 名同时打印 `grid`、`N`、`calls/step`，三元组对不上就标"未归属"，不要猜。

## 13. "等访存"占大头时，先查访存是否合并，再想 TMA / 加深流水（2026-10-21 r12a）

**判断**：kernel 里"等装载"占比很高、但算出来的总流量又远低于 HBM 带宽时，**不是带宽问题、也不是 outstanding 不够**，
八成是**相邻线程访存不合并**——每个 16 B 请求独占一条 128 B cache line，L2 流量放大 8×，带宽全浪费在无用字节上。

**证据**（attn tcgen05，B300 GPU 5）：

| | 等 K/V 占比 | QK 发射 | softmax | T=6019 |
|---|---|---|---|---|
| 合并前（相邻线程读相邻 key，地址相隔 `kv_stride=16 KB`） | **91%** | — | 22% | 5158 µs |
| 合并后（相邻线程读同一 key 的相邻 16 B 片） | **3%** | 67% | 34% | 3311 µs |

**怎么改**：让相邻线程覆盖**同一行内的连续字节**，而不是跨行。
例：一个 key 的 256 维 = 32 个 16 B 片；64 key × 32 片 = 2048 片 / 128 线程 = 每线程 16 片连续，
这样一条 128 B cache line 被 8 个线程一次吃满。

**为什么比 TMA 优先**：TMA 解决的是"outstanding 请求数不够"，那是**第二个**嫌疑；
合并问题用普通 `cp.async` 就够了，改一行索引逻辑，不需要 mbarrier 改造。先量再选，别直奔复杂方案。

**顺带的一条**：判断"带宽受限 vs 延迟受限"要用**有效吞吐**比：算 kernel 实际搬的字节 / 实际耗时，
得到一个 GB/s，和实测流式上限（B300 约 6 TB/s）比。远低于它 → 延迟受限。
r12a 一开始算成 3.5 GB / 3.5 TB/s 就 1 ms，是把 L2 流量当成了 HBM 流量（全部 K/V 其实只有 24.7 MB）。

## 14. 1 CTA/SM 的 UMMA 结构对"短 K、频繁 commit"的 GEMM 不利（2026-10-21 r12a）

`umma_peak` 能到 2236 TF/s（标称 99.4%），是因为它 K 方向长、连续 4096 轮把发射开销摊掉。
attn 的 QK 是"16 条 K=16 全部累加到同一个 TMEM 累加器 → 必须串行 → commit + 等"，每 tile 都付一次发射 + 同步，
**1 个 CTA 只有一条 MMA 发射流，没有第二条填发射间隙** → 每 tile 8.9 µs vs 理想 0.56（16×）。

**用法**：给 GEMM 选结构时先看"发射/依赖比"——K 越长、commit 越少，越适合 1 CTA/SM 的 UMMA。
GDN 的 chunk 内 GEMM（K=128 连续、state 现成在 smem）属于好的那一类；attention 的 QK/PV 属于坏的那一类。

