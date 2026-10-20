# 引擎层面：dinfer vs vLLM，泊松到达扫描（2026-10-20 18:11:10，NVIDIA B300 SXM6 AC）

dinfer: http://127.0.0.1:8012（Qwen3.8-27B）；vLLM: 无（unknown）
seed=1792491068  每率 64 条（+2 预热）  rates=4 8
负载：泊松开环；prompt mix=128:0.5,1024:0.35,4096:0.15；输出 max_tokens mix=128:0.5,512:0.35,1024:0.15，不 ignore_eos（实际输出 = min(eos, 抽样)）；prefix_repeat=0.5（共享前缀 1024 词 + 16 随机词）；multi_turn=1；temperature 0；enable_thinking=false
SLO：TTFT ≤ 500 ms 且 TPOT ≤ 20 ms → good；goodput = good/墙钟（vLLM --goodput 同义）。TTFT = 首个带内容 chunk（TTFT1st = 任意首 chunk，对账用）。cache hit = Σcached_tokens/Σprompt_tokens（usage.prompt_tokens_details）。
CPU：dinfer 服务进程 0-31；vLLM 服务进程 ?；压测进程 taskset 120-127（nproc=128）。
压测工具自校验：measurement error vs mock truth: TTFT p50 +1309 µs (max +1458), TPOT p50 -5 µs, ITL p50 +85 µs (p99 +151); parse 43.5 µs/req
服务端口径（每率 bench 前后 /metrics 差值，scripts/dev/metrics_delta.py）：srv TTFT/TPOT = 服务自报直方图增量的 mean/p50/p99 ms（桶内插值，只对量级），与客户端列的差 = 网络 + 工具开销；
step = dinfer engine_step_execute_{prefill,decode,mixed} count 增量（均值 ms）——**async 调度下量的是含 finish 等待的 host 侧步时，不等于 GPU 步时间**；混批步占比 = M/(P+D+M)。两边单卡默认都是混批（vLLM 单卡无 prefill throttle）。srv cache = 服务自报 prefix_cache hits/queries（dinfer *_total / vllm:prefix_cache_*）；preempt = num_preemptions[_total] 增量。
client = 压测进程开销（bench JSON client 段）：平均占用核数 / 峰值 RSS MiB / SSE+JSON 解析 ms 每请求。

| rate | 端 | 完成/失败 | goodput | req/s | out tok/s | TTFT p50/p90/p99/max | TTFT1st p50 | TPOT p50/p90/p99/max | cache hit | len-stop | srv TTFT mean/p50/p99 | srv TPOT mean/p50/p99 | srv cache | preempt | step P/D/M (mean ms) | client cores/RSS/parse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | dinfer | 64/0 | 2.37 (64/64) | 2.37 | 956.2 | 51.4/89.5/244.1/248.4 | 51.4 | 13.9/15.1/16.6/17.0 | 0.470 | 64/64 | 51.1/39.4/200.5 (n=66) | 13.4/12.5/48.6 (n=26046) | 0.455 | 0 | 11 (54.5) / 2275 (23.1) / 92 (59.0) | 0.02/7/1.04 |
| 8 | dinfer | 64/0 | 3.12 (64/64) | 3.12 | 1245.0 | 49.9/68.3/77.1/83.5 | 49.9 | 15.5/17.7/19.0/19.2 | 0.815 | 63/64 | 38.2/37.1/73.4 (n=66) | 14.7/13.2/49.8 (n=25740) | 0.817 | 0 | 0 / 1699 (24.2) / 99 (45.4) | 0.02/7/1.05 |
