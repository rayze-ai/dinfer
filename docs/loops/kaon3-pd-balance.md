# loop: kaon3-pd-balance —— prefill / decode 资源分配做成动态控制（用户 2026-10-29："应该是动态的"）

> 背景：E6 后 decode 流最高优先级 = 二元开关（decode 全抢；prefill 被拖 50%，TTFT 回吐）；硬件优先级只分两档。
> vLLM 靠单前向混批 + token 预算在调度层分配，没有 kernel 争抢。我们两流并行，需要一个**连续可调的量**和一个**闭环**。

## 可调量（执行器 / 调度器各一个，模型无关，放 core）
- **Q1 调度层**：每步 prefill 新 token 配额 `prefill_tokens_per_step`（`SchedulerParams`，B47 已有，现为静态 CLI）。改成运行时可写（`Scheduler::set_prefill_quota(u32)`）。配额小 → 副流 prefill 子批短 → decode 图等得短 / 两个子批之间 decode 多跑几步；代价 TTFT 变长、每子批固定开销。
- **Q2 kernel 层**：prefill GEMM 的 SM 份额 `prefill_sm_fraction`（0.25..1.0）。persistent kernel 的 grid = ⌈SM × fraction⌉；G9 接 CUTLASS 时带入（它本身 persistent，grid (1,170,1)）；我们 p64 也可改 persistent（CTA 循环取 tile）。这是真正的比例旋钮。模型 crate 只负责把 fraction 传给自己的 kernel（`ModelExecutor::set_prefill_sm_fraction(f32)` 默认空实现）。

## 信号（都已有）
- TTFT 压力：`Scheduler::num_waiting()`、waiting 队首的 `arrival_ns` 距今（最老等待时长）、每步 `late-begin` 的 prefill tok。
- TPOT 压力：`num_running()`（在飞 decode 行数）、`[host-time]` 纯 decode 步 exec 的滑动中位数（decode 步被拖 = exec 涨）、`metrics.time_per_output_token` 滑动分位。
- 目标：`--slo-ttft-ms` / `--slo-tpot-ms`（bench 已有同名参数；serve 加 CLI，默认 0 = 不启用控制器，保持 E6 静态行为）。

## 控制律（第一版，简单、可解释）
每步末（`publish_step` 后）跑一次，周期 ≥ 50 步或 200 ms：
```
e_ttft = (oldest_wait_ms - slo_ttft) / slo_ttft        # >0 = prefill 饿
e_tpot = (decode_exec_p50_ms - slo_tpot) / slo_tpot    # >0 = decode 饿
u = clamp(u + k * (e_ttft - e_tpot), u_min, u_max)     # u ∈ [0,1] = 给 prefill 的份额
prefill_tokens_per_step = lerp(512, 4096, u)
prefill_sm_fraction     = lerp(0.25, 1.0, u)
```
k 取 0.1，死区 ±5%。两个误差都 ≤ 0 时 u 向默认（0.5）回归。日志每次变化打一行 `[pd-balance] u=… quota=… frac=… e_ttft=… e_tpot=…`。

## 环境 / 口径
5090 kaon3-full；bench 用 **非闭环**负载（`--request-rate` 泊松到达，`scripts/verify/bench_poisson.sh`）——闭环 c16 一波同到的形状太规整，看不出控制器的意义；rate 取使 vLLM TTFT≈SLO 的值（先扫 8/12/16 req/s）。对比：E6 静态（控制器关）vs 控制器开，同 SLO 下 goodput（同时满足 TTFT/TPOT SLO 的请求比例，bench 已算）与 p99。

## 停止条件
① 同 rate 下 goodput ≥ 静态 E6 + 5 个百分点，且 TPOT p99 不升；② 控制器输出 u 收敛（50 步内波动 < 0.1），不震荡；③ 关掉控制器（SLO=0）行为与 E6 逐字同（默认路径零影响）；④ 单测：假信号序列下 u 的轨迹。
不达 → 记负结果，退回静态 E6 + 旋钮 Q1/Q2 手动。

## 顺序
1. Q1 运行时可写 + 控制器骨架（只动 Q1）+ 泊松 bench 口径（半天）→ 先看调度层单旋钮够不够。
2. Q2 随 G9（CUTLASS persistent）带入；p64 persistent 化另算。
3. 自适应 k / 死区按数据调。

## 记录
H 编号。

## H1 2026-10-29 实现（core，模型无关）
- `Scheduler::{set_prefill_tokens_per_step, prefill_tokens_per_step, oldest_wait_ns(now_ns)}`。
- `engine::PdBalance`：`PD_BALANCE_SLO_TTFT_MS` / `PD_BALANCE_SLO_TPOT_MS` 都 > 0 启用（A/B 期 env）；`PD_BALANCE_K`（0.1）、`PD_BALANCE_PERIOD`（20 步）；每步末喂纯 decode 步 exec（滑动 64）与 waiting 队首等待；u 初值 0.5 → quota 2304；输出 `[pd-balance] u quota e_ttft e_tpot …`（HOST_TIME）。
- h1 链：开环泊松 rate 12/16/20、160 请求、混合 prompt、SLO 200/12，A（关）B（开）两轮交错，看 goodput / p99。
