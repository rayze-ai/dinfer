# 引擎层面：dinfer vs vLLM，泊松到达扫描（2026-10-20 14:56:37，NVIDIA B300 SXM6 AC）

dinfer: http://127.0.0.1:8412（Qwen3.8-27B）；vLLM: 无（unknown）
seed=1792479395  每率 64 条（+2 预热）  rates=4 8
负载：泊松开环；prompt mix=128:0.5,1024:0.35,4096:0.15；输出 max_tokens mix=128:0.5,512:0.35,1024:0.15，不 ignore_eos（实际输出 = min(eos, 抽样)）；prefix_repeat=0.5（共享前缀 1024 词 + 16 随机词）；multi_turn=1；temperature 0；enable_thinking=false
SLO：TTFT ≤ 500 ms 且 TPOT ≤ 20 ms → good；goodput = good/墙钟（vLLM --goodput 同义）。TTFT = 首个带内容 chunk（TTFT1st = 任意首 chunk，对账用）。cache hit = Σcached_tokens/Σprompt_tokens（usage.prompt_tokens_details）。
CPU：dinfer 服务进程 0-31；vLLM 服务进程 ?；压测进程 taskset 96-127（nproc=128）。
压测工具自校验：measurement error vs mock truth: TTFT p50 +1318 µs (max +1432), TPOT p50 +0 µs, ITL p50 +88 µs (p99 +166); parse 46.8 µs/req
服务端口径（每率 bench 前后 /metrics 差值，scripts/dev/metrics_delta.py）：srv TTFT/TPOT = 服务自报直方图增量的 mean/p50/p99 ms（桶内插值，只对量级），与客户端列的差 = 网络 + 工具开销；
step = dinfer engine_step_execute_{prefill,decode,mixed} count 增量（均值 ms）——**async 调度下量的是含 finish 等待的 host 侧步时，不等于 GPU 步时间**；混批步占比 = M/(P+D+M)。两边单卡默认都是混批（vLLM 单卡无 prefill throttle）。srv cache = 服务自报 prefix_cache hits/queries（dinfer *_total / vllm:prefix_cache_*）；preempt = num_preemptions[_total] 增量。
client = 压测进程开销（bench JSON client 段）：平均占用核数 / 峰值 RSS MiB / SSE+JSON 解析 ms 每请求。

| rate | 端 | 完成/失败 | goodput | req/s | out tok/s | TTFT p50/p90/p99/max | TTFT1st p50 | TPOT p50/p90/p99/max | cache hit | len-stop | srv TTFT mean/p50/p99 | srv TPOT mean/p50/p99 | srv cache | preempt | step P/D/M (mean ms) | client cores/RSS/parse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | dinfer | 64/0 | 2.15 (61/64) | 2.26 | 819.4 | 59.4/171.9/401.0/416.6 | 59.4 | 14.8/17.1/20.4/21.5 | 0.348 | 63/64 | 86.3/54.5/472.5 (n=66) | 14.3/12.5/74.6 (n=23445) | 0.338 | 0 | 9 (48.0) / 2305 (22.5) / 119 (58.3) | 0.02/7/0.92 |
| 8 | dinfer | 64/0 | 3.04 (64/64) | 3.04 | 1109.9 | 60.0/88.1/103.7/104.3 | 60.0 | 15.7/18.4/18.8/19.0 | 0.807 | 63/64 | 47.6/47.1/93.4 (n=66) | 14.8/13.1/65.6 (n=23532) | 0.809 | 0 | 2 (28.4) / 1767 (23.6) / 88 (54.7) | 0.02/7/0.89 |
