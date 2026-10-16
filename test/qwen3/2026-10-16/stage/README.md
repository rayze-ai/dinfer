# Q3-d 逐段 GPU 耗时：dinfer vs vLLM（2026-10-16，4090 GPU 3，各自独立实例，nsys --cuda-graph-trace=node）

负载同为 `bench --concurrency 8 --num-requests 24 --prompt-len 256 --max-tokens 64`（8 次 prefill + 约 190 个 decode step）。表见 `stage-compare.md`；原始 CSV 同目录（nsys-rep / sqlite 太大不进 git，在 4090 `/data/yaochen/infer/dinfer/test/qwen3/2026-10-16/stage/`）。

## 读数（ms/step = 阶段总 GPU 时间 / decode step 数）

| 阶段 | dinfer ms/step | vLLM ms/step | 说明 |
|---|---|---|---|
| attention | 0.888 | 0.770 | dinfer：split-K decode 85 ms + paged flash 52 ms + prefill flash 21 ms + cache_write 7 ms；vLLM：flash_fwd_splitkv 108 + combine 29 |
| gemm | **1.510** | 2.030 | dinfer 的 1.51 里 gemv 占 187 ms（65%）：batch ≤ 8 时 cuBLASLt 选 gemvx。**微基准（cold_shape，3090）显示 M=1..8 的 gemv 与 M=16 的 tensorop 耗时相同**（qkv 0.016 / gate_up 0.020 / down 0.014 ms），即小 M 已是带宽上限，gemv 不是问题；vLLM 同样 M 走 cutlass 用时相近。两边 GEMM 差距主要是 vLLM 多了 torch.compile 的融合 kernel 计数 |
| norm | 0.163 | 0.110 | dinfer 每层 3 次 rmsnorm（输入 / q_norm / k_norm）+ fused_add；vLLM 两次 fused |
| act/mlp | 0.109 | 0.097 | 同级 |
| rope/embed | 0.057 | 0 | vLLM 把 rope 融进 triton kernel（归到 other） |
| sampling | 0.013 | 0.051 | dinfer 图内 4 个 kernel；vLLM 的 sampler 有 sort |
| copy/misc | 0.146 | 4.288* | dinfer：copy_cols（qkv 拆分）28 ms；vLLM 的 823 ms 是 **`at::native::Fill` 808 ms**，是 profile 期间的 KV cache 清零 / 预热填充（启动期），不是每 step 开销 → 剔除后 vLLM 约 3.7 ms/step 总量 |
| other | 0.025 | 0.686 | vLLM 的 triton 融合 kernel（rmsnorm+rope+…）归在这里 |
| **total** | **2.91** | 8.03* / **≈3.7（剔 Fill）** | 同负载 bench：dinfer TPOT 3.9 ms、vLLM 2.7 ms |

\* nsys 抓的是整个进程生命周期，vLLM 的启动期 Fill 污染了总量；按 kernel 看 vLLM 真正的 per-step GPU 时间 ≈ 3.7 ms（含 prefill 份额）。

## 结论

1. **GPU 时间 dinfer 已不比 vLLM 多**（2.9 vs ≈3.7 ms/step），但端到端 TPOT 反而慢（3.9 vs 2.7 ms）→ 差距在 **GPU 空闲**：每 step 的 CPU 侧（调度 + 7 次 H2D + D2H 同步等待 + 回填）与 GPU 没有重叠。vLLM 是异步调度（CPU 准备下一步时 GPU 跑当前步）。这是 Q3 下一个最大项：**异步 step 流水 / 双缓冲**（我们现在 `runner.execute` 是同步的：launch → `dtoh_sync_copy` 等结果 → 再调度）。
2. GEMM：小 M 的 gemv 路径经微基准确认**不慢于** tensorop（权重 1.5 GB/step 读一遍是 decode 的带宽下限，M ≤ 16 都在这个底上）；GEMM 这块没有算法层面的余量，只剩减少 kernel 数（qkv 拆分并入、q/k norm 融合）。
3. attention 已持平；norm 多一倍（3 次 vs 2 次）→ e 阶段融合（q_norm/k_norm 并入 qkv GEMM 后处理，rmsnorm+残差已有 fused 版）。
4. copy_cols（qkv 拆分）0.15 ms/step 纯浪费 → e 阶段让 attention kernel 直接按 stride 读 qkv_buf。
5. TTFT：HTTP 侧 CPU（模板 0.02 ms + 编码 0.3 ms + 解码 µs 级）可忽略；engine 内 174-token prefill 稳态 5.0 ms（首次 14.4 ms 含 cuBLASLt 启发式），非流式 wall 6 ms。bench 的 43 ms（p50 = p99，极稳定）减 5 ms ≈ 38 ms，与 **Nagle + 延迟 ACK 40 ms** 一致：SSE 首个 role chunk 和首个内容 chunk 都是小写入，复用连接上第二段被攒住；uvicorn 默认 TCP_NODELAY，axum 默认不设。修法 `axum::serve(...).tcp_nodelay(true)`（本轮加），待 bench6 验证。
