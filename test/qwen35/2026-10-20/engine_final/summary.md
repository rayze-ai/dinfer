# 引擎层面：dinfer vs vLLM，泊松到达扫描（2026-10-20 18:36:56，NVIDIA B300 SXM6 AC）

dinfer: http://127.0.0.1:8012（Qwen3.8-27B）；vLLM: 无（unknown）
seed=1792492614  每率 64 条（+2 预热）  rates=4 8
负载：泊松开环；prompt mix=128:0.5,1024:0.35,4096:0.15；输出 max_tokens mix=128:0.5,512:0.35,1024:0.15，不 ignore_eos（实际输出 = min(eos, 抽样)）；prefix_repeat=0.5（共享前缀 1024 词 + 16 随机词）；multi_turn=1；temperature 0；enable_thinking=false
SLO：TTFT ≤ 500 ms 且 TPOT ≤ 20 ms → good；goodput = good/墙钟（vLLM --goodput 同义）。TTFT = 首个带内容 chunk（TTFT1st = 任意首 chunk，对账用）。cache hit = Σcached_tokens/Σprompt_tokens（usage.prompt_tokens_details）。
CPU：dinfer 服务进程 0-31；vLLM 服务进程 ?；压测进程 taskset 120-127（nproc=128）。
压测工具自校验：measurement error vs mock truth: TTFT p50 +1351 µs (max +1440), TPOT p50 -0 µs, ITL p50 +87 µs (p99 +162); parse 46.3 µs/req
服务端口径（每率 bench 前后 /metrics 差值，scripts/dev/metrics_delta.py）：srv TTFT/TPOT = 服务自报直方图增量的 mean/p50/p99 ms（桶内插值，只对量级），与客户端列的差 = 网络 + 工具开销；
step = dinfer engine_step_execute_{prefill,decode,mixed} count 增量（均值 ms）——**async 调度下量的是含 finish 等待的 host 侧步时，不等于 GPU 步时间**；混批步占比 = M/(P+D+M)。两边单卡默认都是混批（vLLM 单卡无 prefill throttle）。srv cache = 服务自报 prefix_cache hits/queries（dinfer *_total / vllm:prefix_cache_*）；preempt = num_preemptions[_total] 增量。
client = 压测进程开销（bench JSON client 段）：平均占用核数 / 峰值 RSS MiB / SSE+JSON 解析 ms 每请求。

| rate | 端 | 完成/失败 | goodput | req/s | out tok/s | TTFT p50/p90/p99/max | TTFT1st p50 | TPOT p50/p90/p99/max | cache hit | len-stop | srv TTFT mean/p50/p99 | srv TPOT mean/p50/p99 | srv cache | preempt | step P/D/M (mean ms) | client cores/RSS/parse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | dinfer | 64/0 | 2.37 (61/64) | 2.48 | 921.3 | 47.5/230.6/271.0/274.9 | 47.5 | 13.8/18.2/20.1/20.8 | 0.493 | 63/64 | 62.2/38.3/458.8 (n=66) | 13.9/12.5/47.7 (n=23933) | 0.479 | 0 | 8 (65.8) / 2161 (23.0) / 95 (62.6) | 0.02/7/0.98 |
| 8 | dinfer | 64/0 | 3.21 (64/64) | 3.21 | 1194.5 | 53.7/73.7/85.8/95.3 | 53.7 | 16.1/19.3/19.6/19.9 | 0.809 | 64/64 | 39.6/36.8/77.4 (n=66) | 15.1/13.3/53.7 (n=23998) | 0.811 | 0 | 2 (25.9) / 1664 (24.0) / 91 (46.6) | 0.02/8/1.00 |
