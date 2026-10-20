# 引擎层面趋势 — qwen35

口径：泊松开环到达率扫描；goodput = 满足 SLO（TTFT 且 TPOT）的完成数 / 墙钟；**SLO 下最大 req/s** = goodput ≥ 0.95×rate 的最大到达率；cache hit = Σcached_tokens / Σprompt_tokens；TTFT = 首个带内容 chunk。详见 docs/design.md §15、docs/loops/engine-bench.md。

| 轮 | 卡 | 改动 | SLO 下最大 req/s dinfer | vLLM | 比 |
|---|---|---|---|---|---|
| 2026-10-20/engine_smoke | 3090 | 引擎口径工具冒烟（tiny 全词表 qwen35，3090，仅 dinfer） | 8 | - | - |

## 2026-10-20/engine_smoke（3090）— 引擎口径工具冒烟（tiny 全词表 qwen35，3090，仅 dinfer）

SLO：TTFT ≤ 500 ms 且 TPOT ≤ 20 ms。负载：prompt mix 64:0.6,256:0.4；out mix 16:0.5,48:0.5；不 ignore_eos；prefix_repeat 0.5；multi_turn 1；每率 24 条

| rate | 端 | 完成/失败 | goodput | req/s | out tok/s | TTFT p50/p90/p99/max | TPOT p50/p90/p99/max | cache hit |
|---|---|---|---|---|---|---|---|---|
| 4 | dinfer | 24/0 | 3.82 (24/24) | 3.82 | 122 | 1.9/2.0/2.2/2.2 | 0.3/0.3/0.3/0.3 | 0.000 |
| 8 | dinfer | 24/0 | 7.61 (24/24) | 7.61 | 244 | 1.9/2.3/2.5/2.5 | 0.3/0.3/0.3/0.3 | 0.000 |
