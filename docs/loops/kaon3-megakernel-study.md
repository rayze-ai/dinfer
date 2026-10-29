# kaon3 megakernel 可行性研究（只分析，不写 kernel；2026-10-29）

> 问题：如果只剩"一步 decode 一个 persistent kernel"这条路，收益上限多少、改动多大、怎么分阶段。
> 数据来源：远端 `tmp_nsys_c8b.sqlite`（b39+诊断二进制：graph id=2 = M=1 65 步，id=11 = M=8 122 步）、`tmp_nsys_c1b.sqlite`（b29 时代，1058 kernel/步，只作历史对照）；
> 解析脚本 `/w/nsys_hist.py`（只读 sqlite，本机副本 `/tmp/mkstudy/nsys_hist.py`，按 graphId 分组、时长 / 间隙直方图）。
> 端到端现状（b41，bench 口径）：c1 TPOT **4.78**（vLLM 4.69）、c8 **7.19**（vLLM 6.24）。
> **所有 nsys 数字都是 nsys 态：< 10 us 的 kernel 时长膨胀 ~2×（D153：fp8 gemv nsys 10.7 us，微基准 6.2 / 下限 4.8；perf-method §6）。** 下面凡标 "nsys" 的，小 kernel 部分按 ÷2 折算成"实际"。

## A. 每步 kernel 清单（nsys 态）

| 口径 | kernel/步 | kernel 和 | 步内 gaps（≤200 us） | 步内 span | 备注 |
|---|---|---|---|---|---|
| c1b（b29，历史） | 1058 | 4.89 ms | 0.63 ms（另 0.69 ms 步间） | 5.5 ms | D148 口径；B23/B30 后已减 390 launch |
| **c8b id=2（M=1，当前结构）** | **668** | **4.52 ms** | **0.55 ms**（1022 个 <1 us 间隙 = 0.32 + 30 个 2–5 us = 0.11） | 5.07 ms | bench c1 4.78 ⇒ nsys 膨胀 ~0.3 ms |
| **c8b id=11（M=8）** | **670** | **7.55 ms** | **0.44 ms** | 8.0 ms | bench c8 7.19 |

每层 ≈ 22 kernel：quant_fp8、rmsnorm_vec ×4、rmsnorm_scale2、rmsnorm_quant、rmsnorm_grouped_quant、gelu_grouped_quant、norm_quant、attn_pre_fused、attn(v3/d512)、merge、qkv/o_proj/down fp8 gemv ×3、gateup_act、router_gemv、route_bucket_small、MoE pipe4 gate‖up、MoE pipe4 down、combine。步级另有 embedding / lm_head / softcap / argmax 等 6 个。

## B. kernel 时长分布（nsys 态）

| 档 | M=1 n/步 | M=1 时长/步 | 占比 | M=8 n/步 | M=8 时长/步 | 占比 |
|---|---|---|---|---|---|---|
| < 3 us | 363 | 0.62 ms | 13.7% | 329 | 0.60 ms | 7.9% |
| 3–8 us | 117 | 0.52 ms | 11.6% | 98 | 0.35 ms | 4.7% |
| 8–20 us | 164 | 2.03 ms | 44.8% | 142 | 1.68 ms | 22.3% |
| > 20 us | 24 | 1.36 ms | 29.9% | 101 | 4.92 ms | 65.1% |

**"边界效应"粗上限 = < 8 us 的 kernel 时长 + 全部步内 gaps**：
- M=1：nsys 1.15 + 0.55 = **1.69 ms**（33% of 5.07）；按小 kernel ÷2 折算 ≈ 0.57 + 0.55 ≈ **1.1 ms 实际（23% of 4.78）**。
- M=8：nsys 0.95 + 0.44 = **1.39 ms**（17% of 8.0）；折算 ≈ **0.9 ms 实际（12% of 7.19）**。
- 这是把 480 个小 kernel 与 670 个边界**全部归零**的数；megakernel 里它们变成阶段屏障，不会归零（见 C）。
- 对照 D154 实测：-240 launch = -0.41 ms（1.7 us/launch 含 kernel 体）⇒ 剩余 668 launch 全融的线性外推上限 ≈ 1.1 ms，与上行一致。
- M=8 的 65% 在 > 20 us 的 kernel 里（MoE gate‖up 64 us ×30 = 1.94 ms、MoE down 38 us ×30 = 1.14、gemv<8> 15 us ×85 = 1.30、lm_head 0.88）—— **c8 落后 vLLM 的 0.95 ms 全在这些大 kernel 的带宽效率**（MoE 只到 ~1.0 TB/s，D180；gemv<8> 占用，D156/D186），不是边界。

## C. DRAM 物理下限（5090 1.79 TB/s）

权重字节/步（形状取 D60/D153 微基准口径；MoE 专家 3.35 MB 含 scale）：

| 段 | 字节 | 下限 |
|---|---|---|
| 滑窗层 ×25：qkv fp8 3072×2816 8.65 MB + o_proj 2816×4096 11.5 MB + dense 17.8 MB + router 0.7 MB = 38.7 MB | 967 MB | 0.54 ms |
| 全注意力层 ×5：qkv ≈ 17 MB + o_proj K=8192 ≈ 23 MB + dense 17.8 + router 0.7 ≈ 59 MB | 294 MB | 0.16 ms |
| MoE c1：8 专家 × 3.35 MB × 30 | 804 MB | 0.45 ms |
| lm_head bf16 262144×2816 | 1480 MB | 0.83 ms |
| **c1 合计** | **3.55 GB** | **1.98 ms** |
| MoE c8：29 专家 × 3.35 MB × 30（替换上行 MoE） | 2.91 GB | 1.63 ms |
| **c8 合计** | **5.65 GB** | **3.16 ms** |

（KV 读 400 token 级 < 20 MB，忽略；激活 / 中间量 < 50 MB。）

- 现状 / 下限：**c1 4.78 / 1.98 = 2.4×；c8 7.19 / 3.16 = 2.3×**。vLLM 同卡 c1 4.69 / 1.98 = 2.4×（E）。
- 这 2.4× 里：lm_head 已 1.05×；MoE 两段实际 ~0.9+0.3 ms vs 下限 0.45（2.7×，是**每 CTA 串行 K 链 / 在飞字节不足**，D181–D183，不是边界）；fp8 gemv M=1 已 0.8–1.0×（D153）；边界 + 小 kernel ≈ 1.1 ms（B）；attention + 其它 ≈ 0.5 ms。
- **megakernel 理论区间** = 下限 + 阶段屏障。一层 decode 的真依赖级数 = 14（input_norm → qkv → attn_pre → attn → merge → o_proj → res/norm → gate‖up → down → norm+router → route → MoE gate‖up → MoE down → combine），30 层 = **420 级**。B300 实测每级 3–10 us（megakernel.md；flag 屏障跨 170 CTA 最少 ~1.5 us）：
  - 乐观 1.5 us/级：c1 = 1.98 + 0.63 = **2.6 ms**；中 3 us：**3.2 ms**；B300 实测中位 6 us：**4.5 ms**；悲观 10 us：**6.2 ms**。
  - 且这个区间**还假设每个阶段都达到独立 kernel 的带宽效率**。本卡直接证据相反：kaon3 dense persistent mega（`KAON3_MEGA`，mma tile）86.9 us/层 vs 3 个带宽型 kernel 15 us/层（D60/D61，**5.8× 慢**，R26 wait = exec 2.5×、59 任务铺 170 SM）；B300 单层 0.91/0.82/0.59×。
  - 换言之：megakernel 要赢现状 4.78，需要**同时**做到屏障 ≤ 6 us/级 **和** 每阶段 ≥ 现有独立 kernel 效率；两条在两张卡、两个模型上都没做到过。

## D. 改动规模

| 阶段 | 现有 mega_* 可复用 | 需新写 | 难点 | LOC 估 |
|---|---|---|---|---|
| input_norm / 各 norm / quant / res | `mega_layer_sm120.cu` S_IN/S_RES/S_PRE_FF/S_POST、`mega_mlp_core.cuh` rmsnorm_row_warp | 改成 M≤16 行驻 smem、与现有 fused kernel 逐位对齐 | 全行归约只能"最后到达 CTA 做"（信号量） | 200 |
| qkv / o_proj / dense down gemv | 骨架（S_QKV/S_OPROJ/S_DOWN）但现实现是 **mma tile（输了 5.8×）** | 把 `mlp_fp8_decode.cu` 的 warp-行 GEMV 改成 persistent task 体 | 常驻 1 CTA/SM 时占用 / 在飞字节输给独立 kernel（B300 4.5 vs 6.5 TB/s；D156 gemv<8> 已是占用瓶颈） | 400 |
| dense gate‖up + act | S_GATE/S_UP/S_ACT | 同上，用 gateup_act 口径 | 同上 | 150 |
| attention sliding | `mega_attn_sm120.cu`（v3 core，256 线程壳只 128 活）+ merge | split 任务表（现 grid 256 = splits）、把 merge 融进 o_proj 前级 | 半数 lane 空转；splits 动态 | 250 |
| attention full d512 | **无**（core 只 D∈{128,256}） | 新 core 或 d512 kernel 改 task 体 | 5 层单独一类 task | 300 |
| router gemv + route/bucket | **无** | 128 块 gemv + 末到块做 softmax/top-8/bucket | route 现在就是 1 block 11.5 us 的延迟链（B43 在做） | 250 |
| MoE grouped GEMM ×2 | **无**（pipe4 是独立 kernel，BM16/BK256/STAGES3） | pipe4 体改 persistent task + **动态任务表**（任务数 = Σ_e ⌈rows_e/16⌉ × N 块，由 route 阶段在 device 写；`StagePlan.count` 现为 host 静态） | mega_common 调度器要支持 device 侧 count；pipe4 smem 38 KB + gemv 的 smem/寄存器同一上限 → 占用再掉 | 700 |
| combine + norm + res | **无** | 末到 CTA 归约 8 专家 | 小 | 150 |
| lm_head bf16 1.48 GB | **无** | 65536 task 的 gemvx 等价 | 已 1.05× 下限，persistent 版只会变慢（B300 GEMV 经验） —— 应**留在 kernel 外** | 0（不进） |
| 跨层循环 / ready 自复位 / graph 内 | mega_common 单层 15 stage | 30 层 × 14 级 = 420 个 ready 计数器 + 自复位 | 一处计数错 = 静默垃圾（cp.async wait_group 教训） | 200 |
| host 侧（layer.rs 接入、MegaBufs、回退开关、FFI） | 部分（R29 曾发现 FFI 失配） | | | 500 |
| 对拍 / 门禁（Rust GPU 测试） | `mega_layer_parity.rs` 骨架 | 每阶段 vs 独立 kernel 逐位 / ulp | | 500 |
| **合计** | 2.2k 行骨架可借形不可借体 | **≈ 3.5–4k 新 LOC** | | |

A/B 轮数：每类阶段至少一轮独立占用调优（B300 一层 GDN 用了 2 轮仍输）→ 7 类 ≥ 7 轮 + 2 轮整层 + 2 轮多层 ≈ **≥ 11 轮**，每轮含对拍门禁。期间 c1/c8 现有路线停摆。

## E. 对照 vLLM

vLLM 同卡 c1 4.69 = 下限 2.4×，也是几百个 kernel + cudagraph；我们 4.78 = 1.02×。边界效应是双方共有的 ~1 ms；**赢 vLLM 不需要 megakernel**（c1 差 0.09 ms，c8 差 0.95 ms 全在 MoE / gemv<8> 带宽效率，B 节）。megakernel 要赢的是下限，而本卡 / B300 的四次实测（dense mega 5.8× 慢、B300 三档 < 1×、R26 wait 2.5×、gemv<8> 占用）都说明 persistent grid 在这类小 M 带宽型负载上**先输占用再谈边界**。

## 结论：**no-go**（整步 megakernel）

1. 上限：边界 + 小 kernel 实际 ≈ **1.1 ms（c1 23%）/ 0.9 ms（c8 12%）**，而 420 级阶段屏障按已实测的 3–10 us/级要吃回 **1.3–4.2 ms** —— 期望值为负，与 B300 结论（边界效应 < 20% 不值架构重写）一致，本卡 dense mega 5.8× 慢是最直接的证据。
2. c8 的真缺口（MoE 1.0 TB/s、gemv<8> 占用）与边界无关，persistent grid 还会让占用更差。
3. 代价 3.5–4k LOC、≥ 11 轮、现路线停摆。

**如果一定要做（最小可验证切片，一轮定生死）**：只融"attention 段"一层一段 —— input_norm → qkv gemv → attn_pre → attn(sliding, splits=1) → merge+quant → o_proj → post_attn_norm+res+pre_ff_norm+quant，8 个 kernel（nsys 每层 ≈ 2+16+2.3+4.7+2.2+0.6+6.2+1.3+3.2 ≈ 38 us，launch 上限 8 × 1.7 = 14 us/层 = 0.4 ms/步）→ 1 个 persistent 段，用 `mega_layer_sm120.cu` S_IN…S_PRE_FF 的 StagePlan，**GEMM task 体必须换成 `mlp_fp8_decode.cu` 的 warp-行 GEMV（不用 mma tile）**，只做 25 个滑窗层、M=1。判据：`mega_layer_parity` 逐位 + bN ABAB **c1 TPOT 降 ≥ 0.3 ms（4.78 → ≤ 4.48）且 c8 不退化**才进第二段；不达标即封存，不再开第二轮。预估 600–800 LOC、2 轮。

**替代路线（推荐，基于 B 的分布）**：M=1 每层 12 个 < 3 us kernel 还能再融 6–8 个（-180…-240 launch，按 D154 1.7 us/launch ≈ **-0.3…-0.4 ms c1**，4.78 → ~4.4 < vLLM 4.69）：
① merge + quant_fp8 → 1（或 merge 并进 o_proj gemv 的 A 装载）；② router_gemv + route_bucket_small → 1（末到块做 route；叠加 B43 并行 softmax/扫描，route 11.5 → ~5 us 另 -0.2 ms）；③ rmsnorm_grouped_quant 并进 route 块（M=1 只 8 行）；④ gelu_grouped_quant 并进 MoE gate‖up dual 的末到 CTA；⑤ combine + 后 norm + res → 1；⑥ down gemv 后的 post_ff_norm / rmsnorm_scale2 / rmsnorm_quant 三连 → gemv 末到 CTA 尾声或 1 个 layer_tail。
c8 另走带宽效率：MoE split-K×2（D184 方向，DRAM 冷场景在飞字节翻倍）+ gemv<8> 8-warp / A 驻留减半（D156 10.3 vs 15.2 us，-0.4 ms）—— 这两刀是 c8 7.19 → ~6.3 的主体，与 megakernel 无关。
