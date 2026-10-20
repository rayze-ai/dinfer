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
| 1 | 跑中：dinfer rate 8 prefix 512（抓 metrics 差值）、dinfer rate 8 prefix 0、vLLM rate 8 prefix 0 | — |
| 1a | **TTFT 伪像根因**：`crates/server/src/chat.rs` 在**请求接受时**就发空 role delta（`for i in 0..n { send(role chunk) }` 在 `consume` 之前），而 vLLM 的 role chunk 随首个输出一起发；工具把首个 SSE chunk 当首 token → dinfer TTFT 5 ms。修：role chunk 改为随首个 content delta 懒发（`role_sent[i]`）。之后 dinfer 的 TTFT 会变成真实值（≥ prefill 时间），goodput 的 TTFT 项才有意义 | 修复中 |
