# Q3 perf 迭代记录（2026-10-16）

| 轮 | 块 | 结果 | 时间 |
|---|---|---|---|
| 0 | 基线 | bench_compare 四组 + 诊断见 bench/README.md | 16:1x |
| 0b | 加载 | 首测：dinfer 进程起 15:30:46 → listening 15:30:50（≈ 4 s，含 CUDA 初始化 + 1.5 GB 权重 mmap→bf16→H2D + KV 分配）；vLLM：Loading weights 0.50 s、Model loading 5.6 s、torch.compile 33.9 s、CUDA graph 2×2 s，init engine 46.1 s，到 Starting server ≈ 53 s。dinfer 加载已比 vLLM 快一个量级（无编译）；细分见 `startup timing` 日志（本轮加） | 16:2x |
| 1 | a | 新 prefill kernel 3090 ref 3/3；4090 验收中 | 16:4x |

## 加载时间（用户要求记录，2026-10-16，4090）

| 阶段 | dinfer | vLLM v0.31.0 |
|---|---|---|
| 权重读入 + H2D | 含在下面 ≈ 4 s 里（mmap → bf16 Vec → 逐张量 H2D，198 个张量，1.5 GB） | Loading weights 0.50 s；Model loading 5.6 s |
| 编译 / 图捕获 | 无 | torch.compile 33.9 s + CUDA graph 2 × 2 s |
| 进程起 → 可服务 | **≈ 4 s**（15:30:46 → 15:30:50 listening，含 CUDA 初始化、KV 14.5 GiB 分配） | **≈ 53 s**（07:14:07 → 07:15:00） |

dinfer 已快一个量级，主因是没有编译期；但 dinfer 的"权重 0.5 s"级别还没做到：当前每个张量 mmap→`Vec<bf16>` 复制一次再 H2D（同步），1.5 GB 走两遍内存。可做：mmap 页直接 pinned 注册 + 异步 H2D，或多线程按张量并行。细分数字看服务日志 `startup timing`（本轮加，重启后生效）。Q3 期间 CUDA graph 捕获会加几秒，届时再看。
| 3b | a | GPU 2 OOM 根因 = 孤儿 VLLM::EngineCore 占卡（pitfalls）；清掉后 real_parity 3/3 PASS（新 prefill kernel 下 greedy 仍 = HF）；sampling_parity 5 个在模型加载 OOM：测试按 max_tokens=8192 分 logits 行 → 改 `load_with_budget(max_logits_rows=max_seqs)` | 16:5x |
| 4 | a | paged 路径也换 flash tile kernel（读 paged cache，history+causal）；D=128 paged torch 对拍 `gqa_paged_d128_flash_ref`；3090 gqa 4/4 PASS；4090 验收 + bench3 进行中 | 17:0x |
| 5 | a | bench3：c8 p1024 TTFT 9.4 s → 0.50 s、吞吐 6.2×；c32 吞吐 +29%；仍是 vLLM 的 0.17–0.54×。见 bench3/summary.md | 17:2x |
| 6 | b | nsys：decode attention 占 85% GPU 时间 → split-K decode kernel；GPU 2 是用户 k8s pod（qwen3-06b-vllm），测试改 GPU 3 | 17:4x |
| 7 | b | split-K decode：bench4 vs vLLM c1_p128_o128: tok/s 327/456 = 0.72×, TTFT 43/11 ms, TPOT 2.7/2.1 ms；c8_p128_o128: tok/s 2283/2944 = 0.78×, TTFT 44/25 ms, TPOT 3.2/2.5 ms；c32_p128_o128: tok/s 5048/8536 = 0.59×, TTFT 99/65 ms, TPOT 5.8/3.3 ms；c8_p1024_o128: tok/s 615/2103 = 0.29×, TTFT 497/36 ms, TPOT 9.2/3.5 ms | 18:0x |
| 8 | c | CUDA graph 可行性探针 PASS（需 disable_event_tracking）；见 crates/kernels/tests/graph_probe.rs | 18:2x |
| 9 | c | CUDA graph + 固定缓冲实现完成，编译中 | 18:5x |
| 10 | c | 3090 real_parity 3/3 走 CUDA graph，decode ≈3.1 ms/步（原 7.5） | 19:0x |
| 11 | c | 4090 parity 8/8；bench5 vs vLLM：c1_p128_o128: tok/s 423/455 = 0.93×, TTFT 43/12 ms, TPOT 2.0/2.1 ms；c8_p128_o128: tok/s 2548/2943 = 0.87×, TTFT 43/26 ms, TPOT 2.9/2.5 ms；c32_p128_o128: tok/s 5166/8577 = 0.60×, TTFT 92/61 ms, TPOT 5.7/3.3 ms；c8_p1024_o128: tok/s 630/2104 = 0.30×, TTFT 497/36 ms, TPOT 8.8/3.5 ms | 19:1x |
| 12 | d | stage-compare 完成，见 stage/README.md；结论：GPU 时间已追平，差在 CPU/GPU 串行 | 19:3x |
| 13 | d | TTFT 43 ms = 5 ms prefill + ~38 ms Nagle 停顿（怀疑，待验证） | 19:4x |
| 16 | d | bench6：TTFT 43→6.5 ms（Nagle 修复），已优于 vLLM；差距全在 execute | 19:5x |
| 17 | d | 分段实测 GPU-bound（wait 2.29 ms / 2.36 ms）；异步流水放弃，转 kernel 融合 | 20:0x |
| 18 | e | qkv_post 融合 + 跨度读 + 残差并入，kernel/层 17→8，编译中 | 20:2x |
| 19 | e | 3090 ref 20/20 + parity 8/8 通过融合路径 | 20:3x |
