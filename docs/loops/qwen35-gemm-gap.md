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

## prefill 分块扫描（2026-10-21）

6019 token，改 --max-num-batched-tokens 强制切块，测 TTFT：

| max-num-batched-tokens | chunk 数 | TTFT | 相对单块 |
|---|---|---|---|
| 512 | 12 | 366.4 ms | +89.4 |
| 1024 | 6 | 333.0 ms | +56.0 |
| 2048 | 3 | 303.5 ms | +26.5 |
| 4096 | 2 | 292.8 ms | +15.8 |
| 16384 | 1 | 277.0 ms | — |

边际成本递减：1→2 +15.8、2→3 +10.7、3→6 +9.8/chunk、6→12 +5.6/chunk。
说明每次 scheduler step 有固定代价（launch / 层循环 / 逐步 norm），不是按 token 分摊。

**这个实验解释了 vLLM 在 1K 时慢（81.8 vs 49.5 ms，它的每步固定代价更大），
但不解释 6K 单块时 dinfer 慢 8.7%** —— 那边只有 1 个 chunk，切块开销不参与。

## 仍未解释

6K 单块 prefill：TTFT 277 ms，六个大 GEMM 实测合计 115.5 ms，GPU 利用率 97.4%，
CPU 入队仅 11.89 ms。**约 160 ms 找不到落点。**
已排除：GEMM 效率（近峰值）、GPU 空闲、CPU 瓶颈、切块开销、chunked prefill 配置差异（两边都是 16384）。

## GPU 时间轴分层分段（LAYER_PROFILE=2，CUDA 事件，2026-10-21）

这是第一次真正拿到 **GPU 侧**的分段：
 里的  不行（跨 kernel 无意义），改用 （device.rs 已有，
走裸驱动 API，disable_event_tracking 后 cudarc safe event 会报 INVALID_HANDLE）。
每层 5 个边界事件，64 层共 325 个，只同步一次。

| t | total | attn | mlp | postnorm |
|---|---|---|---|---|
| 1021 | 48.7 ms | 23.5 | 23.2 | 0.96 |
| 6019 | 278.3 ms | **142.6** | **127.9** | 3.9 |

**6K：attn 51.2% / mlp 46.0%。** 两个加起来 97%，没有第三个大头。

对照 autotune 实测的 GEMM：
- mlp 的 gate_up 47.0 + down 23.4 = 70.4 ms，而 mlp 段实测 127.9 ms
- attn 的 qkv 6.6（full attn）+ gdn in_proj 23.1 + out_proj 8.6 = 38.3 ms，而 attn 段实测 142.6 ms

**attn 段里只有约 38 ms 是 GEMM，其余约 105 ms 在 GDN 的 chunk 路径（conv / prep / wy / ho / cumsum / gate）。**

注意：norm 列显示负数，是事件下标错位（ 与  之间量的是层首 norm 之后的间隙），
不影响 attn/mlp/postnorm 三项的归属；total 与三项之和吻合，说明分段边界本身是对的。

## GQA 内部三段（GQA_PROFILE=1，kernel 内 clock64，2026-10-21）

| 阶段 | 占比 |
|---|---|
| QK mma | 45.4% |
| PV mma | 40.8% |
| softmax | 13.8% |

q_len=6019 时单次约 246 us。**两个 mma 合计 86%，softmax 只占 14%** —— 
「softmax 慢」这个常见猜测在这里不成立。

## 前置校验：scripts/verify/remote_sync.sh（新增）

远端树缺文件（gqa_fa_sm100.cu、umma_peak_sm100.cu 等）会让所有 `CUDA_ARCHS=100,103`
构建在**链接期**才失败，而默认 86,89,90 不编这些文件、一切正常 —— 「带 sm100 的构建
从来没成功过」被隐藏了很久。首次运行抓到 4 个缺失文件，补齐后 169/169 齐备。

**构建带 sm100 前必须先跑这个脚本**（1 秒，exit 1 = 有缺失并打印清单）。

## sm100 干净基线（2026-10-21 定稿）

关掉所有 profile 开关（GDN_PROFILE/ATTN_PROFILE/LAYER_PROFILE 全不设）：

| 长度 | TTFT 中位 | 开 profile 时 |
|---|---|---|
| 1021 | 49.5 ms | 54.2 ms |
| 3009 | 137.3 ms | 140.5 ms |
| 6019 | 282.0 ms | 284.0 ms |

与 sm100 之前的 49.5/137.1/283.1 **一致（±0.5 ms）** —— 证明历史 profile 未被
「远端缺 gqa_fa_sm100.cu / umma_peak_sm100.cu」污染（gdn_ho_umma 默认关，走的一直是 mma.sync）。

**profile 开关自身成本**：短 prompt +4.7 ms（325 个事件 record+sync），长 prompt +2 ms。

分段（6019）：attn_gdn 101 / mlp 124 / attn_full 42 ms。

## vLLM vs dinfer：GDN 段直接对照（2026-10-21）

重抓 vLLM trace（关 prefix cache、容器内 nsys、TRTLLM prefill attention），
三档长度各一次请求：

| 长度 | vLLM GDN | dinfer GDN | 结论 |
|---|---|---|---|
| 1021 | 2.2 ms | 10.7 ms (48×0.222) | vLLM 快 4.9× |
| 6019 | **143.3 ms** | **48–58 ms** | **dinfer 快 2.5×** |

**6K 时 dinfer 的 GDN 反而更快。** `GatedDeltaNetChunked` 在 6K 窗口里 768 次
（≈48 层 × 16 次），dinfer 一层只有 3 次 launch，重复更少。

## vLLM 6K prefill 完整构成（窗口 21015 kernel / 453.5 ms busy）

| kernel | 时间 | 次数 |
|---|---|---|
| GDN chunked | 143.3 ms | 768 |
| **index_elementwise** | **100.6 ms** | 768 |
| **elementwise (128,2)** | **55.8 ms** | 783 |
| vectorized_gather | 21.2 ms | 767 |
| full attention (fmha) | 8.1 ms | 256 |
| GEMM | ~36 ms | 512 |

**vLLM 有 156 ms 花在 inductor 生成的 elementwise 上，dinfer 无对应项。**
dinfer 更重的是 MLP 段（124 ms）。

两者大致抵消 —— 所以 dinfer 的思路没错，差距不在 GDN 也不在 attention。
**唯一 dinfer 明显更重的是 MLP（124 ms）。**

## MLP 段实测与 autotune 偏差（2026-10-21）

MLP_PROFILE=1 拆三层（GPU 事件，t=6019，64 层）：

| 段 | 生产实测 | autotune 测 | 差 |
|---|---|---|---|
| gate_up | 1.123 ms | 0.980 ms | +15% |
| silu_and_mul | 0.115 ms | — | |
| down | 0.660 ms | 0.488 ms | +35% |
| 合计 | 1.899 ms | 1.468 ms | +29% |

64 × 1.899 = 121.5 ms，对上 MLP 段实测 124 ms。
按 autotune 数字应为 94 ms，**多花 23.7 ms ≈ 追赶 vLLM 所需的 24 ms**。

## 撤回：改 autotune scratch 尺寸

原假设「autotune 的 A 缓冲太小（786 KB，驻留 L2）」——**错**。
`GEMM_TUNE_M=6019` 时 `m_max=6019`，`a_scratch = 6019 × 6144 × 2 = 74 MB`，
早已是真实尺寸。改 scratch 不影响任何东西。

## 真实原因（最可能）：权重超出 L2

gate_up 权重 16480 × 5120 × 2 = **168 MB**，B300 L2 约 126 MB —— **单层权重就超 L2**。
autotune 对同一份 w 连续跑多次，首轮之后全命中 L2；生产路径 48 层权重轮转，
每层都从 HBM 重读。

**这是测量方法的固有偏差，无法通过调 scratch 修正**：要修只能让 autotune 每轮换权重，
那就失去了「选 algo」的意义。**结论：autotune 的绝对耗时数字对生产是系统性偏乐观的，
只能用来看相对排序（选哪个 algo），不能当性能依据。**
