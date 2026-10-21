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

## ⚠️ 第一次 profile 作废（2026-10-21）

`q35_nsys_prefill.sh gap1` 抓的 `q35_gap1.nsys-rep` **不能用来做单次 prefill 的分解**：

- 时间跨度 **7327 ms**，而一次 6K prefill 的 TTFT 只有 280 ms；
- 100 ms 分桶的 GPU busy 稳定在 80–105 ms —— 这是**稳态 decode**，不是 prefill 尖峰；
- 里面混了 11 次 prefill 请求 + 全部 decode 步 + 启动/抓图。

所以基于它算出的「cuBLAS 245 ms / 67.4%、small_m 88 ms / 24.2%」是**7.3 秒累计量的相对比例**，
不等于「单次 prefill 内部的比例」。**结论方向仍可能对（GEMM 占大头），但数字不可引用。**

要引用必须重抓：干净窗口 + 单次 6K 请求 + `--cuda-graph-trace=node` 关掉（它会干扰）。

## ⚠️ GEMM_SHAPE_STATS 的正确读法

`M ∈ {1,2,4,8,16,32,64}` 那 6 个形状是 **decode 的 CUDA graph 桶**（启动时抓图触发），
不是 prefill。真正 prefill 的只有 `m=6019`（和 `m=130` 这个待查值）。

prefill 每层 5 个 GEMM（Qwen3.5-27B，hidden=5120）：
| shape | 层 |
|---|---|
| m=6019 n=34816 k=5120 | QKV（MLA） |
| m=6019 n=16480 k=5120 | gate_up（FFN） |
| m=6019 n=5120 k=6144 | down（FFN） |
| m=6019 n=5120 k=17408 | o_proj |
| m=6019 n=14336 k=5120 | 待确认 |

**M 没有被拆碎** —— 每层都是完整 6019 行，「大 GEMM 被切小」的猜测被否定。

## 6K prefill TTFT 同条件实测（2026-10-21）

| 端 | TTFT 中位 | 备注 |
|---|---|---|
| dinfer（GPU 5，:8412） | **280 ms** | 276–286，8 次取样 |
| vLLM（GPU 1，:8013） | **262 ms** | 260–269，6 次取样 |

**dinfer 慢 6.9%（+18 ms）。** 不是 2×、不是 10×。

这解释了为什么在 kernel 层面找不到"大瓶颈"：差距是 7%，摊在很多小地方；
而 c8-long 吞吐差 4%（579 vs 557 tok/s）与 TTFT 差 7% 量级一致，互相印证。

**要达到用户目标（全面 1.2×），6K prefill 需要从 280 → 217 ms，即再快 22%。**

## 前面几份 profile 的问题（已作废，勿引用）

1. `q35_gap1`：跨 7.3 秒，混 11 次 prefill + 全部 decode + 抓图。
2. `pf2`（10 请求）/ `pf3`（1 请求）：窗口 6.5–7.7 s 内仍混有 decode 步。
   `small_m_mma` 的 M 模板上限是 32（`gemm_tune.small_m_max`），prefill 的 M=6019
   **不可能**进那里 —— 那些调用全是 decode 的 CUDA graph 桶。
3. 因此「small_m_mma 占 64.3%」不能算作 prefill 的构成。

**唯一仍然成立的结论：ho_mma + wy_mma = 0.1–0.2%（三份 profile 一致）。**

## 正在做

vLLM 同条件 nsys profile（GPU 6、:8023、无 prefix cache），与 dinfer 的单次 6K
请求逐 kernel 对比，定位那 7% 差在哪。

## vLLM prefill 结构（同条件 profile，2026-10-21）

容器 `vllm/vllm-openai:nightly`（0.32.1rc1），GPU 6，`:8023`，`--tensor-parallel-size 1`，
nsys 以 `--entrypoint` 挂进容器。3 个 6K 随机 prompt。

**窗口定位方法（不再靠猜）**：用 prefill 独有的 `flash_fwd_sm100` kernel 的起止时间框定 ——
27 个 flash kernel，跨度 47842.7–49203.5 ms，窗口内 313 个 kernel、564.2 ms GPU busy。

| 类别 | 时间 | 占比 | 次数 |
|---|---|---|---|
| **GEMM** | **526.1 ms** | **93.2%** | **131** |
| elementwise | 21.7 ms | 3.8% | 104 |
| norm | 9.8 ms | 1.7% | 52 |
| rotary | 6.6 ms | 1.2% | 26 |

**3 个请求只有 131 个 GEMM kernel，平均每个请求 44 个。** 这就是 vLLM 的结构：
GEMM 极少、极大，几乎全部时间在 GEMM 上，其他都是零头。

## dinfer 侧对应数据（待补）

同窗口方法在 dinfer 上暂时失效：`conv1d_prefill_blocked` 只框出 23.7 ms（conv 阶段），
`gqa_fa` 系列在 prefill 窗口里不出现（走的是别的 attention 实现）。
定位 dinfer 的 prefill 窗口仍在做——改用 `GEMM_SHAPE_STATS` 直接按 shape 测，
绕开窗口划分（窗口我已经错了很多次，不再依赖它）。

## TTFT 三方（同一台机、随机 prompt、无缓存命中）

| 端 | TTFT 中位 |
|---|---|
| dinfer | 280 ms |
| vLLM（宿主机容器，0.26.1） | 262 ms |
| vLLM（本容器，0.32.1） | **257.5 ms** |

dinfer 慢 8.7%（对 0.32.1）/ 6.9%（对 0.26.1）。

## prefill GEMM 直接实测（GEMM_TUNE_M=6019）

| shape | 单次 | vs cuBLAS默认 | 层数 | 小计 |
|---|---|---|---|---|
| gdn.in_proj | 482 us | -0% | 48 | 23.1 ms |
| gdn.out_proj | 180 us | -4% | 48 | 8.6 ms |
| mlp.gate_up | 980 us | +0% | 48 | 47.0 ms |
| mlp.down | 488 us | -13% | 48 | 23.4 ms |
| attn.qkv | 410 us | -2% | 16 | 6.6 ms |
| lm_head | 6826 us | +0% | 1 | 6.8 ms |
| 合计 | | | | 115.5 ms |

结论：
1. prefill 大 GEMM 全部走 cuBLASLt，没走 small_m/small_mma。
2. cuBLAS 已接近峰值：lm_head 15.3 TFLOP / 6.826 ms = 2242 TF/s（峰值约 2250）。

缺口：GEMM 合计 115.5 ms，TTFT 277 ms，160 ms 不在这六个 GEMM 里。
tr3.csv 抓不到 prefill，在其上做的窗口分析全部作废。
