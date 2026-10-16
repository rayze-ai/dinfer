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
