# 引擎层面：dinfer vs vLLM，泊松到达扫描（2026-10-20 11:47:31，NVIDIA GeForce RTX 3090）

dinfer: http://127.0.0.1:8012（qwen35）；vLLM: 无（unknown）
seed=1792468049  每率 24 条（+2 预热）  rates=4 8
负载：泊松开环；prompt mix=64:0.6,256:0.4；输出 max_tokens mix=16:0.5,48:0.5，不 ignore_eos（实际输出 = min(eos, 抽样)）；prefix_repeat=0.5（共享前缀 256 词 + 16 随机词）；multi_turn=1；temperature 0；enable_thinking=false
SLO：TTFT ≤ 500 ms 且 TPOT ≤ 20 ms → good；goodput = good/墙钟（vLLM --goodput 同义）。TTFT = 首个带内容 chunk（TTFT1st = 任意首 chunk，对账用）。cache hit = Σcached_tokens/Σprompt_tokens（usage.prompt_tokens_details）。
CPU：dinfer 服务进程 0-15；vLLM 服务进程 ?；压测进程 taskset 8-15（nproc=16）。⚠ 压测核与服务核重叠：10 11 12 13 14 15 8 9 
压测工具自校验：measurement error vs mock truth: TTFT p50 +1551 µs (max +1661), TPOT p50 +4 µs, ITL p50 +146 µs (p99 +257); parse 90.0 µs/req
服务端口径（每率 bench 前后 /metrics 差值，scripts/dev/metrics_delta.py）：srv TTFT/TPOT = 服务自报直方图增量的 mean/p50/p99 ms（桶内插值，只对量级），与客户端列的差 = 网络 + 工具开销；
step = dinfer engine_step_execute_{prefill,decode,mixed} count 增量（均值 ms）——**async 调度下量的是含 finish 等待的 host 侧步时，不等于 GPU 步时间**；srv cache = 服务自报 prefix_cache hits/queries。
client = 压测进程开销（bench JSON client 段）：平均占用核数 / 峰值 RSS MiB / SSE+JSON 解析 ms 每请求。

| rate | 端 | 完成/失败 | goodput | req/s | out tok/s | TTFT p50/p90/p99/max | TTFT1st p50 | TPOT p50/p90/p99/max | cache hit | len-stop | srv TTFT mean/p50/p99 | srv TPOT mean/p50/p99 | srv cache | step P/D/M (mean ms) | client cores/RSS/parse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | dinfer | 24/0 | 3.82 (24/24) | 3.82 | 122.1 | 1.9/2.0/2.2/2.2 | 1.0 | 0.3/0.3/0.3/0.3 | 0.000 | 24/24 | 1.1/0.5/8.7 (n=26) | 0.3/0.5/1.0 (n=998) | 0q | 26 (1.0) / 998 (0.6) / 0 | 0.00/5/0.11 |
| 8 | dinfer | 24/0 | 7.61 (24/24) | 7.61 | 243.6 | 1.9/2.3/2.5/2.5 | 1.0 | 0.3/0.3/0.3/0.3 | 0.000 | 24/24 | 0.9/0.6/4.7 (n=26) | 0.3/0.5/1.0 (n=998) | 0q | 26 (0.8) / 998 (0.6) / 0 | 0.01/4/0.14 |
