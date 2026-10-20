# 引擎层面差距定位（Qwen3.5-27B B300 单卡；口径 design §15 / requirements 验收 v2）

> 起点 §15.4：`vllm bench serve` random 1024/256（range 0.5）+ 512 共享前缀，rate 4 / 8：goodput dinfer **1.25 / 0.35** vs vLLM **2.84 / 5.07**；TPOT mean/p99 dinfer 20.8/25.7、29.5/45.2 vs vLLM 17.0/20.6、16.6/19.4。四组闭环 c8 是 10.2 vs 11.2 → 差距全在负载形态。

## 环境
B300 GPU 0 dinfer `:8012`（预算 16384、graph eager、async 开、prefix cache **关**）、GPU 1 vLLM `:8013`（预算 16384、prefix cache 开）；第三方工具在 vLLM 容器内 `vllm bench serve`；我们的 `bench_engine.sh`（子代理实现中）之后作主口径。

## 观察方法
- 每次 bench 前后抓 dinfer `/metrics` 的 `engine_step_execute_{prefill,decode,mixed}_seconds_{count,sum}` 差值 → 三种步的次数与均值。
- 同 rate 下 prefix 512 vs prefix 0 两组，隔离 prefix cache 的贡献。
- TTFT 两口径：首 chunk / 首个带内容 chunk（dinfer 首 chunk 是空 role delta，和 vLLM 一样，但工具把它当首 token → 5 ms 伪像；vLLM 同样发空 role delta却报 161 ms？→ 待核：可能 vLLM 的空 delta 是在 prefill 完成后才发，而我们在请求接受时就发）。

## 停止条件
rate 8 下 dinfer goodput ≥ vLLM（≥ 5.0），TPOT p99 ≤ 20 ms；再扫 2–24。

## 假设（按先后验证）
1. **预算 16384 一步吞掉多条 1024 prefill，把同批 decode 的 TPOT 顶到 40+ ms**（vLLM 有 `long_prefill_token_threshold` 默认 = 4% × max_model_len 把单条 prefill 切片，我们默认 0 = 关）→ 看 mixed 步均值；试 `long_prefill_token_threshold=328`（8192×4%）或预算 2048–4096。
2. **prefix cache 缺失**：vLLM 512 前缀命中 → prefill 少一半；对比 prefix 0 两边差距是否收窄。
3. 空 role delta 发送时机 → TTFT 口径修正（不影响 goodput 的 TPOT 项，但影响 TTFT 项）。

## 记录
| 轮 | 改动 / 观察 | 结果 |
|---|---|---|
| 0 | §15.4 数据点（prefix 512，rate 4/8） | goodput 1.25/0.35 vs 2.84/5.07 |
| 1 | **engine1**（rate 8，64 req）：dinfer p512 goodput 0.35 / TPOT 29.4，p0 **0.80 / 24.5**；vLLM p0 **2.42** / TPOT 19.9（p512 时 5.07 / 16.6）。dinfer 步模式差值（p512）：prefill 8 步 avg 113 ms、**decode 524 步 avg 32.1 ms**、mixed 53 步 avg 131 ms；p0：decode 598 步 **avg 29.4 ms**、mixed 69 步 93.7 ms。**关键**：纯 decode 步均值 29–32 ms，而四组闭环 c8 的 decode 步是 ~10 ms → 不是混批把 TPOT 顶高，是**这种负载下 decode 步本身慢 3×**。候选：① `engine_step_execute` 在 async 下量的是 `finish` 等上一步的时间（含排队），口径需核（c32 时也是 25.8 vs TPOT 15.7）；② 序列长（1024–1536 + 输出）→ gqa decode v3 / GDN 没变但 KV 读变多（闭环 c8 是 128 prompt）；③ 每步 batch 小且在变（泊松到达 ~3 在飞）→ graph 桶 4/8 命中但 M 小、权重读占满 → 单步固定 ~10 ms 下限 × 并发少 = TPOT 不应变大……需要 nsys 这类负载下的纯 decode 步 span。prefix cache：vLLM p512→p0 goodput 5.07→2.42（命中值一半的 goodput），dinfer p512→p0 反而 0.35→0.80（p512 下 prompt 更长 1024+512 没命中更慢）→ **prefix cache 是第二大项，但 decode 步慢是第一** | 定位 decode 步 |
| 1a | **TTFT 伪像根因**：`crates/server/src/chat.rs` 在**请求接受时**就发空 role delta（`for i in 0..n { send(role chunk) }` 在 `consume` 之前），而 vLLM 的 role chunk 随首个输出一起发；工具把首个 SSE chunk 当首 token → dinfer TTFT 5 ms。修：role chunk 改为随首个 content delta 懒发（`role_sent[i]`）。之后 dinfer 的 TTFT 会变成真实值（≥ prefill 时间），goodput 的 TTFT 项才有意义 | 修复中 |
| 2 | **测试程序本身**（用户 11:4x 提醒）：B300 主机 128 核 2 NUMA；dinfer 进程 CPU 集合 0-127 未钉；**`vllm bench serve`（Python，容器内）rate 8 时 CPU 229–410%、线程 192–322**；我们的 Rust `bench` 同负载 CPU 0.1–0.9%、129 线程。两工具对 vLLM :8013 同负载：vllm-tool TTFT p99 233 / TPOT 18.1；Rust bench TTFT p99 269（TTFT1st 同值，说明 vLLM 的 role chunk 随首 token 发）/ 吞吐 4.81 req/s——p99 差 15%，超 10% 阈值，工具自身开销可能是差源。**nsys（GPU 5，vllm-tool rate 8 p0 ×32）：dinfer 纯 decode 步 span 14.9 ms = kernel 之和 15.2（GPU 满），TPOT 18.3 / p99 21.9**——同一负载在 GPU 0 的 engine1 是 decode 步 29.4、TPOT 24.5。差 2×：GPU 5 这次是单独跑、GPU 0 那次压测工具（4 核 Python）与 dinfer / vLLM 共 CPU 且 vLLM 容器在同机。**假设：engine1 的 dinfer 数字被压测工具 + 同机 vLLM 的 CPU 竞争拖慢了（async 调度线程被抢）**；下一轮：dinfer / vLLM / 压测端三方 `taskset` 隔离到不同核组，Rust bench 作主口径，重跑 rate 8 p0 两边 | 工具开销是变量 |
| 3 | **engine2，首个可信口径**（dinfer `taskset 0-31`，Rust bench `taskset 96-127` client CPU 0.3 s / 10 MB；vLLM 容器 cpuset 未设 = 0-127，记录在案）：rate 8、prompt 512/1024/1536 = 0.25/0.5/0.25、out 256、×64、无前缀、ignore_eos。**dinfer 5.08 req/s、TTFT 164 / p99 297、TPOT 23.3 / p90 29.0 / p99 29.6；vLLM 5.37 req/s、TTFT 123 / p99 250、TPOT 16.9 / p90 18.7 / p99 19.0**。dinfer 步差值：prefill 11 步 avg 70 ms、decode 1060 步 avg **23.8 ms**、mixed 52 步 avg 121 ms（`engine_step_execute` 口径含 async finish 等待）。结论：隔离后差距收窄（engine1 的 29.4/45 → 23.3/29.6）但 **TPOT 仍落后 40%**，且 p90≈p99 说明是**每步都慢**不是尾部抖动；四组闭环 c8（prompt 128）是 10.2 vs 11.2 → 差在**长序列（1000–1800 token）下的 decode 步**。候选：① 长 KV 下 `gqa_decode_v3` D=256 的 splits / 带宽（c8-long 闭环 10.7 vs 11.2 已接近，但那是 p1024 固定）；② 批小（在飞 ~5）+ 预算 16384 让每个 1536 prefill 一步吞下 → mixed 步 121 ms 把同批 decode 顶高（vLLM 同预算也如此却 p99 19 → 它的 prefill 快？或 `max_num_partial_prefills` 限流）；③ async 下 `engine_step_execute` 不是 GPU 时间——要 nsys 这组负载的纯 decode 步 span。下一轮：GPU 5 nsys（同负载、同钉核、Rust bench）+ 试 `long_prefill_token_threshold` 2048 | 首个可信差距：TPOT 1.4× |
| 4 | **nsys（GPU 5，dinfer 钉 32-63，Rust bench 钉 96-127，同 engine2 负载）**：bench 侧 TPOT 23.8 / p99 30.1（与 GPU 0 一致 → 隔离后数字稳定、可复现）；**纯 decode 步 661 步 span 13.2 ms = kernel 之和 13.4（GPU 满）**；步模式：decode 1050 步、mixed 52 步 avg 125 ms、prefill 11 步 70 ms。**关键推断：GPU 上一个 decode 步 13.2 ms，客户端却每 token 等 23.8 ms → 每 token 平均多等 ~10 ms，不在 decode kernel 里**。来源只能是 ① mixed 步：52 步 × 125 ms = 6.5 s，占总 decode 墙钟（1050 × 13.2 + 6.5 s ≈ 20.4 s）的 **32%**——每条 1024–1536 的新请求到达都被整条吞进一步（预算 16384），那一步里所有在飞 decode 请求各停 125 ms；平均每 20 个 decode 步插一个 mixed → 每 token 摊 6 ms；② async `finish` 等待 / scheduler 间隙 ~4 ms。vLLM 同预算 TPOT p99 19 → 它把长 prefill 切片或限并发 prefill（`max_num_partial_prefills` / `long_prefill_token_threshold`），decode 不被整条 prefill 卡。**第 5 轮**：`--long-prefill-token-threshold`（已加 CLI）取 512 / 1024 / 2048 扫，看 mixed 步 avg 与 TPOT p99；对照 vLLM 日志里的实际切片参数 | mixed 步是 TPOT 差源 |
