# Q3 perf 迭代记录（2026-10-16）

| 轮 | 块 | 结果 | 时间 |
|---|---|---|---|
| 0 | 基线 | bench_compare 四组 + 诊断见 bench/README.md | 16:1x |
| 0b | 加载 | 首测：dinfer 进程起 15:30:46 → listening 15:30:50（≈ 4 s，含 CUDA 初始化 + 1.5 GB 权重 mmap→bf16→H2D + KV 分配）；vLLM：Loading weights 0.50 s、Model loading 5.6 s、torch.compile 33.9 s、CUDA graph 2×2 s，init engine 46.1 s，到 Starting server ≈ 53 s。dinfer 加载已比 vLLM 快一个量级（无编译）；细分见 `startup timing` 日志（本轮加） | 16:2x |
