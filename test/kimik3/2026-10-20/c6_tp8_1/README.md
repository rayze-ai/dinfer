# K3 C-6 全 93 层 tp8 首轮（B300-50 8 卡独占，2026-10-20 14:3x–14:5x）

修复 `cudaFuncSetAttribute` static 缓存（`08fd4ec`，pitfalls 2026-10-20）后首次跑通。预算 **512**（每层各持激活缓冲，16384 放不下；ActScratch `f3962d7` 对拍后回 16384），**decode CUDA graph 关**（LatentMoe / AttnRes 体内 D2H），NCCL all-reduce 每层 3 次。c8-long 没跑完（r9 子代理按用户指示停 serve 释放 8 卡）。vLLM 基线 = `bench_b300_vllm1`（tp8，预算 16384）。

| 组 | dinfer TPOT p50 | vLLM | 倍数 | dinfer TTFT p50 | vLLM | 完成 |
|---|---|---|---|---|---|---|
| c1 p128 | 168.8 ms | 8.70 | 19.4× 慢 | 1822 | 193 | 16/16 |
| c8 p128 | 324.5 | 13.09 | 24.8× | 7314 | 368 | 32/32 |
| c32 p128 | 817.3 | 19.15 | 42.7× | 32958 | 421 | 32/32（脚本 n = max(c×4,16)；vLLM 基线那轮是 128 个） |
| c8-long p1024 | — | 13.11 | — | — | 508 | 未跑 |

输出吞吐 c8 / c32 都是 ~21 tok/s，**不随并发涨**：每步墙钟 ≈ 170 ms × (1 + 并发比例)，是 host / 同步瓶颈，不是 GPU 算力（5 层 tp8 冒烟 c8 TPOT 13.5 ms → 93 层 324 ms，每层 3.5 ms = 3 次 all-reduce + 每层 MoE `dtoh_sync_copy(offsets)` + AttnRes host softmax 的同步点）。性能工作在 C-7 起（见 k3-serve 3i）。
