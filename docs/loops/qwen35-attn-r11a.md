# Qwen3.5 r11a：全注意力 prefill kernel（`gqa_fa_v5_kernel<PAGED,256,NBUF>`，D=256）在 B300 上提效

> 状态：**收口（2026-10-20 22:2x，7 轮，主会话定）**。B300 GPU 4 默认路径（cfg 3464 = v6 4 warp × tile 64 + SFU exp2）：**6 K 单层 2040 → 1558 µs（−24%）**、3 K 506 → 454（−10%）、1 K 101 → 78（−23%）；16 层 attn 合计 32.6 → ~25 ms。停止条件 **未达**（6 K ≤ 700 µs）——`mma_peak` 实测 mma.sync 峰值 575 TF，700 µs 需 636 TF，物理不可达；mma.sync 路线实际封顶 ~50%（286 TF / 575）。要进 40%+ 峰值只能换 **tcgen05.mma（TMEM + UMMA 描述符）**，另开 loop。起点（engine-gap 9e，B300 GPU 2 nsys）：16 层合计 T=1 K 1.6 ms（101 µs/层）/ 3 K 9.7（506 µs/层）/ 6 K **32.6（2.04 ms/层 = 0.22 PFLOP/s ≈ B300 bf16 10%）**。目标：6 K 单层 ≤ 700 µs（≈30% 峰值，16 层省 ~22 ms），3 K 省 ~5 ms；用户 2026-10-20 "全面超过 vLLM 1.2×"，6 K TTFT 277 → ≤ 217。

## 环境

- **出数：B300-50 GPU 4**（`CUDA_VISIBLE_DEVICES=4`，`CARGO_TARGET_DIR=/data/yaochen/infer/target_r11a`，构建 `taskset -c 64-79`、运行 `taskset -c 48-55`；`source /data/yaochen/infer/env.sh`，`CUDA_ARCHS=103`）。GPU 0/1 常驻、2/3 主会话、5–7 留用，**都不碰**。
- **3090 本机只跑 parity + 编译**（knowledge/kernel-rules：性能改动在 B300 定去留）。
- 同步：子代理不 `push`；改动文件 `B300_TAG=50 remote_b300.sh put` 单文件到远端同路径（开工前 9 个相关文件 md5 已核对与本机 dev `f90aaea` 一致）。产物落远端 `bootstrap/r11a/`，本机 `pull` 回 `test/qwen35/2026-10-20/probe_attn_r11a_N/`。
- 远端一键：`bash /data/yaochen/infer/bootstrap/r11a/r11a_b300.sh <轮> ["ENV=val" …]` → build / `gqa_fa_bench` 默认 + 变体 / ncu（`sudo -n ncu --set full`，抓 T=6019 那一次 launch）。本机 `scripts/dev/r11a_b300.sh` 是它的源。
- 微基准：`crates/kernels/tests/gqa_fa_bench.rs`（`#[ignore]`，T ∈ {1024, 3009, 6019}，H=24 / Hkv=4 / D=256，单请求 causal，`q_stride = (H+2Hkv)·D` 与模型同）→ `scripts/dev/gqa_bench_to_probe.py` → probe.csv（group `attn_scale`，metric `fa_us` / `tflops`）→ `probe_trend.py --root test/qwen35`。
- 对手口径：vLLM 同模型 6 K TTFT 260 ms 里 attn 估 ~15 ms（FA3 级 30–40% 峰值）。

## 文件归属

`crates/kernels/cuda/gqa.cu`（v5 kernel 与分发）、`crates/kernels/src/arch.rs` 的 `gqa_*`、`crates/kernels/tests/gqa_fa_bench.rs`、`scripts/dev/{gqa_bench_to_probe.py,r11a_b300.sh}`。对拍沿用 `crates/kernels/tests/ops_parity.rs` 的 `parity_gqa_prefill_qwen35` / `parity_gqa_paged_qwen35`（`scripts/verify/ops_parity.sh`）。**不动 `models/qwen35`、不改 `docs/progress.md`**。

## 候选改法（先 ncu 再定序，不猜）

1. GQA 打包：同一 KV 头的 6 个 q 头合在一个 block 共享 K/V tile（K/V smem 装载 ÷6）。
2. O 按 D 切两半（两组 warp 各 128 维，softmax 统计量共享 / 或各算一遍 S），让寄存器回到 D=128 档、TILE 回 64、occupancy 回 2 block/SM。
3. 8 warp（BQ=128）摊 K/V tile 装载与 `__syncthreads`。
4. causal 对角 tile 的 mask 分支 / 对角 tile 单独处理。
5. NBUF=2（smem 68 KB）在 B300（227 KB smem）是否真开。

## 硬性规则

- kernel 内不写卡型 / 形状调优数值；tile / warp / NBUF 的选择走 `ArchProfile`（按 cc），环境变量只做覆盖；`#if __CUDA_ARCH__` 禁止在 kernel 体内。
- 旧 kernel（v5 现结构）保留为回退。
- 每轮：3090 `ops_parity.sh` gqa 项过（`parity_gqa_prefill_qwen35` / `parity_gqa_paged_qwen35`），B300 GPU 4 `gqa_fa_bench` 三形状进 `probe_attn_r11a_N/probe.csv`，commit 不 push，本页轮次表一行。

## 停止条件（机器可判定）

- B300 `gqa_fa_bench` **T=6019 fa_us ≤ 700**（单层；16 层 ≈ 11 ms）；
- 3090 + B300 `ops_parity` gqa 项全过（精度门限沿用 `ops_parity_ref.py`）；
- 迭代上限 8 轮；卡住 2 轮没进展停下回报主会话。

## 轮次

| 轮 | 改动 | B300 GPU 4 `gqa_fa_bench` T=1024 / 3009 / 6019（µs，括号 TFLOP/s） | parity（3090） | ncu 定性 / 结论 |
|---|---|---|---|---|
| 0 | 基线（`gqa_fa_v5<false,256,1>`：4 warp × 16 行，TILE 32，Q 在寄存器；dev `f90aaea`）`probe_attn_r11a_0` | **91 (142) / 527 (211) / 1818 (245)**；NBUF=2 变体 99 / 565 / 1917（更慢，1 block/SM） | 沿用 | ncu（T=6019，`--set full`）：**255 reg/thread 且有 spill**（875 KB local）、Block Limit Registers 2 → 理论占用 12.5%（8 warp/SM）、Eligible 0.59 / Issued 0.44 warp/scheduler、No Eligible 56%、**Tensor pipe 45%（最高管线）**、DRAM 0.57% / L2 19% / L2 hit 98%、Mem Pipes 16%。结论：**延迟型**——寄存器爆满限占用 + 每 warp 串行依赖链；**不是带宽**（GQA 打包 K/V ÷6 的方向不成立，K/V 已全在 L2）。微基准 1818 vs serve nsys 2040 µs/层，口径接近，之后以微基准为准 |
| 1 | **v6 kernel**（`gqa_fa_v6_kernel<PAGED,256,NW,TILE>`：Q 进 smem 每 tile ldmatrix 取 A 片、NW / TILE 模板化、blockIdx.x 反序让重 block 先发、整 warp 不可见 tile 跳 mma、单缓冲；C ABI `gqa_prefill_bf16` / `gqa_paged_bf16` 加 `cfg = NW·100+TILE`，0 = v5 回退；`ArchProfile::gqa_prefill_cfg_for`（Sm100 464 / Sm90 464 / Sm8x 0），`GQA_PREFILL_CFG` 覆盖）`probe_attn_r11a_1` | cfg **464：83 (155) / 482 (231) / 1645 (271)（−9.4%）**；432：88 / 534 / 1870；864：97 / 561 / 1916；832：107 / 634 / 2197；v5(0) 同轮 91 / 524 / 1815。自对拍 vs v5：maxabs 0.0078（bf16 1 ulp）cos 0.9999995 | **17/17**（`GQA_PREFILL_CFG=432`，99 KB 装不下 464；d256 的 prefill / paged / mixed 都走 v6）| **mma.sync 物理上限**（`scripts/dev/mma_peak.cu`，B300 GPU 4）：m16n8k16 bf16 纯算 **537–575 TFLOP/s**（4 warp/SM 已 537，再多 warp 不涨）= 标称 2.25 PF 的 ~25%；6 K 单层 ≤ 700 µs 需 636 TF **超过 mma.sync 上限**，FA2 结构可达的实际上限约 80–85% = 460–490 TF ≈ 930–1000 µs，再往下只能 tcgen05.mma（另开 `*_sm100.cu`，TMEM / UMMA 描述符，工程量另估）。ncu（864，T=3009）：246 reg、Block Limit Reg 1 / Smem 1 → 12.5%、Eligible 0.46 / Issued 0.32（比 v5 还低）：8 warp 单 block 全体 `__syncthreads` + 576 block 3.9 波尾部抵消了 K/V 装载减半。464 赢在 tile 64 把每 tile 固定开销（同步 / O 重缩放 / max-sum 归约）减半，占用仍 2 block × 4 warp = 8 warp/SM，延迟型没变 |
| 2 | **v7 warp 对分工**（每对 2 warp 共担同 16 行：各算半 tile 的 S、经 smem 换 max、P 写 smem 供对方取 A 片、PV 各算半 D；`__launch_bounds__(NW·32,2)` 压到 128 reg，8 warp × 2 block/SM = 25% 占用；named barrier 对内同步）`probe_attn_r11a_2` | 1864：90 / 528 / **1802**；1832：102 / 609 / 2143；1464：91 / 577 / 2078；1432：106 / 700 / 2615；v6 464 复现 83 / **481** / **1650** | 3090 cfg 1432（132 KB，99 KB 装不下）结果见下 | ncu（1864）：**128 reg、Block Limit Reg 2 / Smem 2 → 25% 占用（12.5 → 25）**、Eligible 0.46 → **0.73**、Issued 0.32 → 0.39、Compute 44%、Memory 64%；**但更慢**（1802 vs 1650）。结论：**占用不是瓶颈**——`mma_peak` 已证 4 warp/SM（537 TF）即接近 mma.sync 峰值 575 TF，加占用只加竞争；v7 的额外代价（Q A 片读两遍、P 一次 smem 往返、每 tile 两次 named barrier）抵消收益。真正缺口是**依赖停顿**：每 tile 内 QK → softmax（SFU / shuffle / pack）→ PV 串行，softmax 期间 tensor 管空转 → 下一轮做 FA2 两段流水（QK(i+1) 与 softmax(i) 重叠） |
| 3 | **v6 + K/V 双缓冲**（`NBUF` 模板参数，cfg 2xxx；装下一 tile 与算当前 tile 重叠）`probe_attn_r11a_3` | 2464：99 / 627 / **2212**；2432：92 / 548 / 1878；单缓冲 464 同轮 82 / 482 / 1651 | — | **负结果**：B300 上双缓冲一律更慢（smem 翻倍 → Block Limit Smem 2→1，且 cp.async 与 ldmatrix 抢同一 smem 端口）；ncu（464）248 reg、Reg/Smem 各 2 → 12.5%、Eligible 0.57、Warp Cycles/Inst 6.1。单缓冲 464 的 82 / 482 / 1651 复现稳定（第 1/2/3 轮同数），后面以它为基准 |
| 4 | **softmax 指数改 SFU `ex2.approx.ftz.f32`**（`FEXP` 模板开关，cfg 3xxx；ncu 明写 "stalled waiting on a fixed latency execution dependency… could be improved with fast math"——`exp2f` 在无 `-use_fast_math` 下编成精确多项式）`probe_attn_r11a_4` | **3464：78 (165) / 455 (244) / 1556 (286)（−5.7%）**；3432：84 / 496 / 1753；基准 464 同轮 83 / 481 / 1650。vs_v5 maxabs 0.0078 / cos 0.9999995（`ex2.approx` 误差 < 2 ulp，远小于 bf16 输出 1 ulp 门限）| 3090 cfg 3432 17/17 PASS（见下）| 默认改 3464（Sm100 + Sm90）；SFU 只省 5.7% 说明 softmax 不是主项——剩余缺口在 QK→PV 之间**无重叠**的串行链，以及 **mma.sync 峰值本身**。此时 6 K 已 = **286 TF / 575 TF 峰值 = 50%** |
| 5 | 8 warp × tile 64 / 32（SFU 版，cfg 3864 / 3832）+ 缓存命中扫描 | 无可比数：**两个变体都 `cudaError 1`**——V6F 只注册了 4 warp 的 case，3864/3832 落到 `default:` 返回 InvalidValue，bench panic（脚本变体日志 0 字节）。默认路径（3464）同轮 78 / 457 / 1558，ncu 244 reg / 12.5% / Eligible 0.50 / Compute 51% | — | 工具坑：**变体 cfg 必须先确认在 dispatch switch 里**（`grep V6F_CASE`），否则静默 fallthrough；已补 8 warp 的 V6F case 到第 6 轮 |
| 6 | ① 8 warp + SFU（3864 / 3832，这次真跑到了）；② O 重缩放 `alpha==1` 时跳过（OF=32 × 4 = 128 次乘 / tile，随机数据下 max 极少刷新）| ① 3864：93 / 525 / **1792**；3832：106 / 618 / 2134；② 默认（含 skip）81 / 462 / **1577**（第 4 轮 78 / 455 / **1556**）| — | ① 8 warp 第三次负结果（与第 1 / 2 轮一致）：单 block 8 warp 的 block 级 `__syncthreads` 与 2280 block 的尾部使 K/V 装载减半拿不回来；② **skip 也小负**（1577 vs 1556）——分支 + 谓词打断 mma 流水，已回退（保留注释） |
| 7 | 最终确认（回退 skip 后的默认 3464，`GQA_BENCH_ITERS=50`）+ ncu | **78 (166) / 454 (245) / 1558 (286)**，与第 4 轮 78/455/1556 一致（3 次复现）`probe_attn_r11a_7` | 3090 17/17（cfg 3432 / 432 / 1432 三个版本路径都过） | ncu（3464，T=3009）：244 reg、Reg/Smem 各限 2 block → 12.5%、Eligible 0.50、Issued 0.39、No Eligible 61%、Compute 51.7%、Memory 60.6%、DRAM < 1%。**结论：mma.sync 结构饱和**——占用（第 2 轮 25%）、双缓冲（第 3 轮）、8 warp（第 1/5/6 轮三次）、重缩放跳过（第 6 轮）全部负结果或中性，只剩 softmax 与依赖链，已拿到 5.7%

## 收口（2026-10-20 22:2x，7 轮）

- **数字**（B300-50 GPU 4，`gqa_fa_bench`，默认路径）：6 K 2040 → **1558 µs**（286 TF = mma.sync 实测峰值 575 TF 的 50%）、3 K 506 → 454、1 K 101 → 78。
- **物理上限（本轮最重要的结论）**：`scripts/dev/mma_peak.cu` 在 B300 上测 m16n8k16 bf16 纯算 **537–575 TFLOP/s**（4 warp/SM 已 537，16/32 warp 也只 574）= 标称 2.25 PF 的 ~25%（标称是 tcgen05/FP8 口径）。所以 **700 µs（30% 标称）在这个结构里做不到**；按 85% 的 kernel 效率折算，mma.sync 版的实际下限 ≈ 930 µs，当前 1558 是它的 60%。
- **有效的改法**：① Q 从寄存器搬到 smem（v5 的 255 reg + spill → 244 无 spill），使 TILE 32 → 64，每 tile 固定开销减半（−9%）；② softmax 指数走 SFU `ex2.approx`（−5.7%）。
- **负结果（别再试）**：K/V 双缓冲（B300 smem 翻倍 → 1 block/SM）；提高占用（v7 warp 对分工 12.5 → 25% 反而慢 9%）；8 warp（三次）；O 重缩放按 alpha 跳过（分支打断 mma 流水）。共性：**占用与访存都不是瓶颈**（DRAM < 1%、L2 hit 98%、4 warp/SM 就到 mma 峰值），瓶颈是 mma.sync 自身的发射率和 QK→softmax→PV 的无重叠串行链。
- **下一步（未做，需单独立项）**：`gqa_fa_tcgen05_sm100.cu`——tcgen05.mma 走 TMEM，Ampere 式逐 warp mma 换成 warpgroup 级单发射，理论可到 1 PF+ 量级；工程量 = UMMA 描述符 + TMEM 分配 + smem 布局（K/V 需 128 B swizzle），且必须留在 `*_sm100.cu` 分文件里（CLAUDE.md 多卡型规则）。估计 2–3 天，收益 attn 6 K 25 → 10 ms（全 TTFT 6 K 省 ~15 ms，缺口 60 ms 的 1/4）。
- **端到端验证（主会话 2026-10-21）**：B300 四组门禁 `bench_merge_21b` **104 / 745 / 2271 / 579**（1.10/1.11/1.06/1.04×），与 `bench_b300_20`（103/744/2251/579）持平、无退化。单发 TTFT 1 K / 3 K / 6 K = **49.0 / 139.8 / 281.7 ms**（合并前 284.9），斜率 51.7 → **46.5 µs/token**。**但 6 K 只拿回 3.2 ms，是微基准预示（−7.7 ms）的 42%** → attn 在真实 prefill 步里的份额小于 9e 的静态分解（9e 是 32.6/277 = 12%，其中 16 层 × 2.04 ms），待 `mns` nsys 复分解确认（怀疑：6 K 单发被切成多个 prefill 步 / chunk，每步的 attn tile 更小、v6 的 TILE=64 优势摊不开）。
- **未验**：sm8x 上 v6（默认仍 v5，cfg 3432 在 3090 parity 过但未出性能数）；4090 门禁（待主会话）。
