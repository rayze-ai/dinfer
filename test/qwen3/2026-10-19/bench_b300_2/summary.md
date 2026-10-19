# dinfer vs vLLM（2026-10-19 13:36:24，NVIDIA B300 SXM6 AC）

dinfer: http://127.0.0.1:8012（qwen3-0.6b）；vLLM: http://127.0.0.1:8013（unknown）
seed：1792388182（每次运行不同，prompt 不与历史轮次重复；两边 prefix cache 只命中模板前缀）

口径：流式；TTFT = 首个带内容 chunk；TPOT = (latency − TTFT)/(out−1)；ignore_eos=true；temperature 0；enable_thinking=false；每组前 2 个预热请求不计。

| 并发 | prompt | out | 端 | 请求 | req/s | out tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | ITL p50 (ms) | latency p50 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 128 | 128 | dinfer | 16/16 | 6.63 | 848.0 | 2.1 / 3.6 | 1.2 / 1.2 | 1.2 | 151.0 |
