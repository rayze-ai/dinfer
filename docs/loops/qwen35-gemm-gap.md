# qwen35 prefill GEMM 缺口（2026-10-21 起）

## 为什么开这一条（用户批评）

> "分析肯定要先全局，再到局部啊，哪有你这样先分析占用小的局部"

成立。r11d 我从 ho kernel **内部**的百分比（load 60%、vp 28%）出发优化了两天，
从未量过 ho 段在整步里占多少。全局 profile 一出来，结论是：

## 全局分布（B300-50，6K prefill，nsys `--cuda-graph-trace=node`）

命令：`bash scripts/dev/q35_nsys_prefill.sh gap1 5 8021 "860"`
报告：`/tmp/q35_gap1.nsys-rep`；汇总：`nsys stats --report cuda_gpu_kern_sum`

| 类别 | 时间 | 占比 | 次数 |
|---|---|---|---|
| **cuBLAS (nvjet)** | **245.1 ms** | **67.4%** | 4008 |
| **small_m_kernel（我们的 GEMM）** | **88.1 ms** | **24.2%** | 280 |
| cublasLt splitK reduce | 10.5 ms | 2.9% | 2144 |
| 其他杂项（norm/embed/sample…） | 9.5 ms | 2.6% | 2272 |
| recurrent_decode | 5.3 ms | 1.5% | 336 |
| conv1d_update | 3.4 ms | 0.9% | 336 |
| gated_rmsnorm | 1.0 ms | 0.3% | 384 |
| conv1d_prefill | 0.8 ms | 0.2% | 48 |
| **`ho_mma_kernel`（r11d 优化的那段）** | **0.81 ms** | **0.22%** | 48 |
| **`wy_mma_kernel`** | **0.11 ms** | **0.03%** | 48 |

**结论：GDN 全链 < 3%；ho+wy = 0.25%。** r11d 把 ho 从 7843 优化到 1068 µs（7.3×），
在端到端的上限是 0.2% —— 这解释了 A/B 压测里 UMMA 开关「无差异」的现象，
不是噪声，是分母太小。

**战场是 GEMM（91.6%）**，其中 cuBLAS 67.4% / 自己的 small_m 24.2%。

## vLLM 侧结构（`/data/ref/vllm`，main de8291e / 0.31.0）

- 默认路径 `default_unquantized_gemm` = `torch.nn.functional.linear(x, w, bias)`
  → 底层同样是 cuBLASLt → 同样出 `nvjet` kernel。**库是同一个，差别只能在调用方式。**
- 可选 `--linear-backend=flashinfer_cutedsl` → FlashInfer `mm_bf16`（CuTe DSL）；
  不满足 `can_implement` 时回退 `F.linear`。默认不开。
- 所以要对齐/超越，要看的是：**调用次数、M 的粒度（是否把大 GEMM 拆碎）、algo 选择**。

## 待查（进行中）

1. 我们的 4008 次 cuBLAS 调用的 **shape 分布**（`GEMM_SHAPE_STATS=1` 打印）。
   重点怀疑：prefill 时 M 是否被按 chunk（64 token）切碎，导致 cuBLAS 效率塌陷。
2. 自己的 `small_m_kernel` 88 ms / 280 次（平均 315 µs）分在 m=1/2/4/8/16/32 六个模板上，
   其中 `<32>` 41.5 ms、`<16>` 20.4 ms —— 这些本该是 cuBLAS 的活还是自研更优，要按 M 重新划线。
3. 和 vLLM 同样 workload 的 kernel 数 / 总时间对比（nsys 两边各跑一次）。

## 坑

- **serve 的 main 走 `std::process::exit`，`Drop` 不执行** → 挂在 Drop 里的统计 dump 永远是空的
  （实测 `0 distinct shapes, total 0 calls`）。要打印就只能在首次出现时立即打，或显式注册 atexit。
- `nsys` 的 `--sqlite` 参数在部分版本不可用（本次 `--sqlite` 未生效，只拿到 .nsys-rep）。
- `nsys profile` 用 `DINFER_BIN` 时**环境变量不会传给被 profile 的子进程**（UMMA=1 没生效，
  profile 里出现的是 `ho_mma_kernel` 而非 UMMA kernel）。
