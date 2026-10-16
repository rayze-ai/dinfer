# 首轮 dinfer vs vLLM（2026-10-16，4090 / CUDA 13.0，dinfer `--release`，vLLM v0.31.0 Docker）

`scripts/verify/bench_compare.sh test/2026-10-16/bench`，bench 自检 PASS。两边：Qwen3-0.6B bf16，max_model_len 8192，gpu_memory_utilization 0.9，prefix cache 开，greedy，ignore_eos，思考关，每请求 128 输出 token。表见 `summary.md`，各组 JSON 同目录。

## 结论（dinfer / vLLM）

| 组 | out tok/s | TTFT p50 ms | TPOT p50 ms |
|---|---|---|---|
| c1 p128 | 221 / 447（0.49×） | 83.7 / 17.5 | 3.9 / 2.1 |
| c8 p128 | 1465 / 2911（0.50×） | 85.6 / 28.6 | 4.6 / 2.5 |
| c32 p128 | 2707 / 8065（0.34×） | 376 / 69 | 10.0 / 3.4 |
| c8 p1024 | 56 / 1916（0.03×） | 9369 / 55 | 60.7 / 3.7 |

## 诊断（已确认的事实）

1. **每个新 prompt 长度（新的 GEMM M 维）首次出现时有一次性大停顿，且随 M 线性增长**：389 token 0.47 s、639 token 0.87 s、889 token 1.26 s、1139 token 1.64 s；同长度第二次 0.02–0.07 s。vLLM 同 prompt 0.02–0.04 s。c8 p1024 组每个请求的 prompt 长度都不同（1221 ± 几十），于是几乎每个请求都吃一次 1.5–3 s 停顿 → TTFT 9 s、TPOT 60 ms（decode 步被别的请求的冷停顿卡住）。这是 c8 p1024 组 34 倍差距的全部来源，也拉高了 c32 组的 TTFT 尾部。嫌疑在 cuBLASLt 每次调用重新做启发式搜索 / 建描述符（`Gemm::forward_bf16` 无按形状缓存），待下一步定位；修法：按 (M,N,K) 缓存算法或启动时按 M 分桶预热，并把 M 向上对齐到 2 的幂 / 64 的倍数以减少形状数。
2. **decode TPOT 热态 3.9 ms vs 2.1 ms（c1）**：这是 Q3 真正要啃的部分：参考 kernel（GQA 逐 token 单 block 循环、无 CUDA graph、每 step 多次小 H2D/D2H、采样管线多次 kernel）。vLLM 有 CUDA graph + FlashAttention。
3. **TTFT 热态 84 ms vs 18 ms（c1 p171）**：prefill 本身也慢 4–5 倍：GQA prefill kernel 是逐 query token 一个 block、对全部 key 两遍扫（softmax 两 pass + PV 累加），无 tiling；GEMM 走 cuBLASLt 应与 vLLM 同级，所以差在 attention 与 kernel 数。
4. 并发 32 时 dinfer 吞吐只到 c8 的 1.85 倍（vLLM 2.8 倍）：batch 变大 decode 步时间从 4.6 → 10 ms，attention kernel 不随 batch 并行是主因（每请求一个 block 串行扫 key）。

## 下一步（Q3 规格的输入）
按收益排序：① 冷形状停顿（确定根因后修，预期直接消掉 c8 p1024 的 34×）；② GQA decode kernel 改 split-K / 多 warp 并行 + CUDA graph；③ prefill attention tiling（flash 风格）；④ 采样管线融合、H2D 批量化。
