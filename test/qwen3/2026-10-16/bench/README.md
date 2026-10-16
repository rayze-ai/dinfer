# 首轮 dinfer vs vLLM（2026-10-16，4090 / CUDA 13.0，dinfer `--release`，vLLM v0.31.0 Docker）

`scripts/verify/bench_compare.sh test/2026-10-16/bench`，bench 自检 PASS。两边：Qwen3-0.6B bf16，max_model_len 8192，gpu_memory_utilization 0.9，prefix cache 开，greedy，ignore_eos，思考关，每请求 128 输出 token。表见 `summary.md`，各组 JSON 同目录。

## 结论（dinfer / vLLM）

| 组 | out tok/s | TTFT p50 ms | TPOT p50 ms |
|---|---|---|---|
| c1 p128 | 221 / 447（0.49×） | 83.7 / 17.5 | 3.9 / 2.1 |
| c8 p128 | 1465 / 2911（0.50×） | 85.6 / 28.6 | 4.6 / 2.5 |
| c32 p128 | 2707 / 8065（0.34×） | 376 / 69 | 10.0 / 3.4 |
| c8 p1024 | 56 / 1916（0.03×） | 9369 / 55 | 60.7 / 3.7 |

## 诊断（已确认的事实；首版写成"冷形状停顿"是误判，已改）

1. **prefill attention kernel 慢得离谱，且与 T² 成正比**：同一 1514-token prompt，改一个前缀词让 prefix cache 不命中 → 6.36 s；原样重发（整段前缀命中，只算最后一块）→ 0.12 s。此前以为"同形状第二次就快"是 kernel 热身，实际是 prefix cache 命中。vLLM 同 prompt 0.04 s（150 倍）。GPU 全程 100% 利用率、SM 满频，是算力浪费不是停顿：`gqa_prefill_kernel` 的 grid 只有 (num_reqs × heads) = 16 个 block，每个 block 串行遍历该 head 的全部 T 个 query，每个 query 对全部 key 扫三遍（max、sum、PV），纯标量 fp32。T=1514 时每 block 做 3 × T²/2 × D ≈ 4.4 × 10⁸ 次 MAC × 16 block，在 16 个 SM 上跑 → 秒级。c8 p1024 组每个请求 1.2k token 且互不相同（无前缀命中）→ TTFT 9 s，decode 步被别人的 prefill 卡住 → TPOT 60 ms。这是该组 34 倍差距的全部来源，也拉高了 c32 组的 TTFT 尾部（171 token prompt 也要 ~60 ms）。
2. **decode TPOT 热态 3.9 ms vs 2.1 ms（c1）**：参考 kernel（GQA 逐 token 单 block 循环、无 CUDA graph、每 step 多次小 H2D/D2H、采样管线多次 kernel）。vLLM 有 CUDA graph + FlashAttention。
3. 并发 32 时 dinfer 吞吐只到 c8 的 1.85 倍（vLLM 2.8 倍）：decode 步时间 4.6 → 10 ms，decode attention kernel 每请求一个 block 串行扫 key，不随 batch 并行。
4. GEMM 走 cuBLASLt，与 vLLM 同级；`cold_shape` 微基准见 `test/2026-10-16/cold_shape.md`。

## 下一步（Q3 规格的输入）
按收益排序：① **prefill attention 重写**（flash 风格：query 分块 × heads × reqs 的 grid，K/V tile 进 shared memory，online softmax 一遍，warp 级并行；预期 1514 token 从 6.4 s 到 ≤ 50 ms，直接消掉 c8 p1024 的 34× 与 c32 的 TTFT 尾部）；② decode attention 改 split-K / 多 warp + CUDA graph；③ 采样管线融合、H2D 批量化。
