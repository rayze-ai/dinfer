# Qwen3.5 r9：GDN chunk prefill 大 T 伸缩（wy / ho / conv）

> 状态：**r0–r7 收口（2026-10-20，3090 + B300 都测过）**：B300 默认路径 1×1024 全链 311 → 208、8×1024 1662 → 1109、1×8192 → 1271（wy 2.5× / ho 1.3× / conv 1.0×）；停止条件 8×1024 ≤ 800 与 1×8192 wy+ho ≤ 1000 都**没到**（差 28% / 19%），两段版 / PF=1 两卡都不赢（默认关）。8 轮上限到，待主会话与用户定下一步（剩余在 wy 的 (I+A)⁻¹ 仍是 fp32 标量乘、ho 单序列串行链）。来源 `docs/loops/engine-gap.md` 8c–8h：长 prompt TTFT 斜率 dinfer 51.5 vs vLLM ~39 µs/token，差 12.5，其中 GDN 三段约 8（fla 同形状 1 K 持平、8 K 只涨 2.4× 而我们线性 5.3×）。用户 2026-10-20 决定：**按当前方案优化（修自己的 kernel），实在不行再谈替代**。

## 环境

- 开发 / parity / 微基准：3090 本机（`cargo test -p dinfer-kernels --release --features ref-tests --test gdn_parity`，`gdn_chunk_bench --ignored`）。
- 验收：B300-50 单卡（C-6 结束后给 GPU 6；`CUDA_ARCHS=103`），sm_103。4090 回归门禁（`scripts/verify/bench_gate.sh`，任一卡退化 > 3% 不合入）。
- 文件归属：`crates/kernels/cuda/gdn_chunk.cu`、`crates/kernels/src/ops.rs` 的 `gdn_chunk_*`、`crates/kernels/src/arch.rs` 的 `gdn_chunk_*`、`crates/kernels/tests/gdn_parity.rs`。**不动 `models/qwen35`**（r8 在改 executor / gdn.rs / service.rs），kernel ABI 若必须改，加新入口不删旧。
- 对手口径：vLLM `third_party/flash_linear_attention` fla `chunk_gated_delta_rule`（Triton）：`solve_tril` 分块三角逆（16×16 子块逆 + 矩阵乘拼 64×64，全 tensor core）；`chunk_fwd_h` 只递推 h 并把每 chunk 的 h 写回全局；`chunk_fwd_o` 对所有 (chunk, head) 全并行算 o = q·h + A·v。B300 实测：1×1024 359 µs、1×3009 349、1×8192 853（含 l2norm）。

## 观察方法

1. `gdn_chunk_bench` **加形状** 1×4096、1×8192（单长序列是 ho 并行度最差情形，现有三形状都没覆盖），每轮记 prep / wy / ho / conv 各段 µs 进 `test/qwen35/<日期>/probe_gdn_r9_N/probe.csv`（group `gdn_scale`，side dinfer；fla 数字作 side vllm 对照），`python3 scripts/dev/probe_trend.py --root test/qwen35`。
2. nsys `gdn_chunk_bench`（`cuda_gpu_kern_sum`）看 wy / ho 的 min / max 与 SM 占用；B300 上再对 6 K 单发 prefill 做一次 nsys 分解（engine-gap 待办）。
3. 正确性：`gdn_parity` 14/14（3090 + B300），`scripts/verify/qwen35_golden.sh`（主会话在 B300 跑，子代理不跑）。

## 改法（按序，每步独立可测）

1. **ho 装载 cp.async 双缓冲**：算 chunk c 时预取 c+1 的 q / w / kg / Aq（`gdn_decode.cuh` 已有 `cp_async16_smem` / `cp_async_commit_g`，共享头抽到 `arch.cuh` 或新 `.cuh`）。预期 ho 每 chunk 10 → 6–7 µs。
2. **ho 拆两段**：`h-pass`（每 (seq, head, V 分片) 一 block 串行递推 h，只做 h = h·g + v'ᵀ·kg，每 chunk 的 h bf16 写回全局 `[chunks, HV, V, K]`）+ `o-pass`（grid (chunk, head)，全并行 o = (q·hᵀ)·dec + Aq·v'）。串行段算量减半，并行段吃满 SM。中间 h 缓冲走 `GdnChunk` 工作区（启动一次分配，不在 kernel 内算尺寸）。
3. **wy 三角逆分块**：64 步串行前代 → 4 个 16×16 子块各自求逆（迭代 / 小前代）+ 矩阵乘拼 64×64，全部 mma；去掉逐步 `__syncthreads`。
4. **conv1d 行数**：每 block 行数由 `ArchProfile` 给（按 T 与 SM 数算），目标 8×1024 222 → < 80 µs（60 MB / 8 TB/s ≈ 8 µs 下限）。
5. 若 1–4 后仍差：wy 的 (chunk, head) 一 block 4 warp 太重 → 升 8 warp 或拆 V 分片。

## 硬性规则

- kernel 内不写卡型 / 形状调优数值（BV、行数、warp 数、双缓冲深度 → `ArchProfile` + 自调优），`#if __CUDA_ARCH__` 禁止在 kernel 体内；按 arch 分文件（`*_sm8x.cu` / `*_sm100.cu`）只在确有分叉时。
- 旧 kernel 保留为回退（环境变量 `GDN_CHUNK_HO=legacy` 之类只做覆盖）。
- 每轮 commit + 记本页轮次表 + probe.csv。

## 事故（2026-10-20 14:5x）

r9 子代理两次 `remote_b300.sh push`（rsync `--delete`）把 B300 远端 `test/qwen35/2026-10-20/engine_r8/`（r8 刚跑完、尚未 pull 的 engine 产物）删了，本机也没有副本，只能重跑。规则（主会话）：**子代理不 push**，同步代码发消息让主会话推；非得推用 `remote_b300.sh put <文件> <远端路径>` 单文件；产物落远端 `bootstrap/` 或 `/tmp`，由主会话 pull。pitfalls 已有同类条目（另一会话 push 删 k3 refs）。

## 为什么 B300 上还是没超过 vLLM（2026-10-20 收口后分析）

**先承认方法问题（用户问"3090 上验证的在 B300 真能最优吗"）**：r0–r5 六轮全在 3090 迭代，B300 只有 r6 / r7 两轮"复测 + 在 3090 设计的候选里按 B300 数选门限"，**没有一轮是以 B300 的瓶颈为目标改 kernel**。两卡瓶颈不同：3090（936 GB/s、82 SM）上 wy / ho 撞的是**带宽墙**（conv 84% 带宽；两段版因 h16 多搬 200 MB 输）；B300（8 TB/s、148 SM、HBM 延迟型）上同一批 kernel 撞的是**延迟 / 并行度墙**——两段版 / PF=1 在 B300 也输，但输的原因没在 B300 上查过。所以 r9 的准确结论是：**3090 设计空间收口，B300 设计空间没进。**

**B300 nsys 分解（GPU 6，`dinfer_gdn_r9.nsys-rep`，默认路径 kernel，min = 1×1024 / max = 1×8192 或 8×1024，µs）**
- `wy_mma<64,128,TRI=1>` 89 / 532；`ho_mma<64,32,128,PF=0>` 82 / 659；`conv1d_prefill_blocked<8,4>` 36 / 318（rows 扫描的 max）；prep 的 norm + cumsum ~18 / 82。
- 两段版：`ho_hpass<64,32,128,PF=0>` 93 / 721 + `ho_opass<64,128,128>` 36 / 289 —— **h-pass 单独就比融合版慢**（721 vs 659），说明 B300 上 ho 的瓶颈不是"每 chunk 四个乘法太多"，而是**每 block 串行走 128 个 chunk 的延迟链本身**：每 chunk 约 5 µs = 装载（全局 → smem，无预取）+ 4 次 `__syncthreads` + mma 发射延迟，148 SM 只有 192 个 block、每 SM 1–2 个 block、4 warp，**掩不住延迟**。fla 的 chunk_fwd_h 也是串行扫 chunk，但 num_warps 2–4、BV 32/64 由 autotune 选、num_stages 2–4 做软件流水，同一条串行链每 chunk 更短。
- wy：fla 把 wy 拆成 4 个小 kernel（kkt 点积 → solve_tril 分块逆 → recompute_w_u → 各自 autotune BK/BV/num_warps），中间量 A / Ai / w / u 全 **bf16**；我们一个 kernel 做完，中间量 aq / u / w / kg 全 **f32**（写 2× 字节，ho 再读 2×），1×8192 下 aq 100 MB + u/w/kg 各 200 MB。3090 上这是带宽，B300 上是 L2 / HBM 延迟。

**下一轮（r10）该怎么做（都在 B300 上迭代，3090 只跑 parity）**
1. 中间量 f32 → bf16（aq / u / w / kg；fla 同口径，parity 容差已经是 bf16 级）：wy 写量减半、ho 读量减半，预计 ho 1×8192 659 → ~450。
2. ho 串行链每 chunk 延迟：真正的 cp.async 双缓冲（smem 两份，不是寄存器预取；B300 smem 227 KB 装得下两份 75 KB），4 warp → 8 warp 分 V 列，减少 `__syncthreads` 次数（装载与 (2) 合并一次）。目标每 chunk 5 → 3 µs。
3. wy 的 (I+A)⁻¹ 非对角块 fp32 标量乘 → mma（16×16×16 刚好一条 m16n8k16 ×2）。
4. 门限自调优：`ArchProfile` 给候选，启动时按真实形状各跑一次取优（现在是按卡写死的分档）。

## 备忘（用户 2026-10-20）

- 本轮 fla 基线用的是 B300 上既有的 `vllm/vllm-openai:nightly`（0.26.1rc1 / torch 2.13 cu130，与 engine-gap 8d 同口径），**本次不改**；**后续 vLLM 对照一律用最新版本**（镜像要更新：机器无外网，走 `remote_b300.sh put` 中转，见 infra-b300）。

## 停止条件（机器可判定）

- `gdn_parity` 14/14（3090 + B300 sm_103）；
- `gdn_chunk_bench`（B300）：**8×1024 全链 ≤ 800 µs**（现 1662）、**1×8192 wy+ho ≤ 1000 µs**（fla 853 含 l2norm；现估 ~2500）；
- 单发 6 K TTFT 斜率（probe ttft_len 复测）**≤ 45 µs/token**（现 51.5；GDN 8 µs/token 拿回大半）；
- 4090 `bench_gate.sh` 四组不退化 > 3%；B300 `bench_compare.sh` 四组不退化。
- 迭代上限：8 轮或 2 天；到上限未达标 → 回主会话与用户讨论替代方案（cuDNN SDPA 只对注意力；GDN 没有现成 C 库）。

## 轮次

| 轮 | 改动 | 3090 1×1024 / 8×1024 / 1×8192（wy / ho / conv µs） | B300 | parity | 结论 |
|---|---|---|---|---|---|
| 0 | 基线（第 5A 轮 mma 版）；bench 加 1×4096 / 1×8192 | 3090：1×1024 **542 / 381 / 74**，8×1024 **4219 / 2460 / 427**，1×8192 **4205 / 2961 / 424**（ho 取 bv32、conv 取各档最优；全链 956 / 6787 / 7352）；B300 1×1024 134 / 160 / 41，8×1024 912 / 664 / 222 | 全链 311 / 1662 | 3090 15/15 | 起点。3090 上 wy / ho 都 ~8× 线性涨（1 K → 8 K），conv 5.7×；ho 1 block/SM（smem 75 KB）× 4 warp 纯延迟链；wy 的 64 步前代（每步 2 次 `__syncthreads`）占大头。`probe_gdn_r9_0` |
| 1 | ho 装载重写：每线程固定向量数（QV=4 × 16 B）进寄存器 → 转 bf16 写 smem；PF 模板参数可选预取下一 chunk；新 C 入口 `gdn_chunk_ho_v2`（旧 `gdn_chunk_ho` 保留）；`GdnChunkHo{bv,prefetch}` + `ArchProfile::gdn_chunk_ho_prefetch` + `GDN_CHUNK_HO_PREFETCH` 覆盖 | **ho（bv32，PF=0）381 → 278 / 2460 → 1857 / 2961 → 2043**；PF=1 反而 441 / 3148 / 3473（寄存器 255 顶满、occupancy 掉）；wy / conv 不变 | 待 | 3090 15/15（含 pf 两 case） | 装载路径重写本身赢 27–31%，『预取』不赢 → 默认 PF=0，PF=1 留给 B300 复测。`probe_gdn_r9_1` |
| 2 | ho 两段版：`ho_hpass_kernel`（grid (V/BV,HV,nseq)，只做 v'=u−w·hᵀ 与 h 递推，每 chunk 入口 h 写 `h16` bf16 [chunks,HV,V,K]、v' 写 `v16`）+ `ho_opass_kernel`（grid (chunks,HV)，8 warp，o=(q·hᵀ)·dec+Aq·v' 全并行）；`GdnChunkScratch::{h16,v16}`、`GdnChunkHo.split`、`ArchProfile::gdn_chunk_ho_split`、`GDN_CHUNK_HO_SPLIT` 覆盖 | 3090 split（bv32 / bv16）：1×1024 **318 / 340**（融合 278 / 347），8×1024 **2111 / 2489**（融合 1816 / 2678），1×8192 **2634 / 2696**（融合 2045 / 2792）；h-pass 本身 ≈ 融合版 70–80%，o-pass 再加 15–25% | 待 | 3090 15/15（含 sp 4 case） | 3090 上拆了不赢：h-pass 串行链少了两个乘法但每 chunk 多写 h16（32 KB/块）+ v16，3090 带宽 936 GB/s 吃不下；o-pass 1×8192 读 h16 201 MB ≈ 215 µs 下限。B300 带宽 8× 应改观 → 默认门限先按 3090 实测**关**（`gdn_chunk_ho_split` 返回 false 待 B300 数字再开）。`probe_gdn_r9_2` |
| 3 | wy：(I+A)⁻¹ 改分块三角逆（`TRI=1`：4 个 16×16 对角块各 warp 内 16 步前代只 `__syncwarp`，6 个非对角块 M_ij=−M_ii·Σ A_ik·M_kj fp32 乘，按列原地）；sM f32 复用 1) 之后死掉的 sK∪sQ（smem 72 → 56 KB，1 → 2 block/SM）；`GdnChunkWy{Legacy,Mma,MmaSerial}`，`GDN_CHUNK_WY=serial` 回 64 步 | **wy 542 → 274 / 4219 → 1906 / 4205 → 1910**（2.0–2.2×）；其中 smem 瘦身（mma64 列：342 / 2287 / 2284）占大半，分块逆再 −17–20%；ho / conv 不变 | 待 | 3090 15/15（wy_mma_vs_legacy 两版都对老 fp32） | 1×8192 wy+ho = 1910+2045 = 3955 vs 基线 7166 → 1.75×；全链 7352 → 4140。`probe_gdn_r9_3` |
| 4 | A）o-pass 按 V 分片 VT=64（smem 78 → 49 KB，2 block/SM）；B）conv 行数目标改 6 block/SM、档位加 16 / 8（bench 扫 rows 8/16） | A）split 1×8192 **2607 → 2954**（退 13%）、1×1024 322 → 335：q / Aq 装两次 > occupancy；B）conv rows8 / rows16 在所有形状都慢于 rows32–256（1×1024 73 → 105、8×1024 441 → 513），rows256 在 8×1024 = 336 MB / 426 µs = **84% 带宽**，3090 上 conv 已到头 | 待 | 3090 15/15 | 两条都**不采**：o-pass 加 VT=128（8 warp 两组 64 列）为首选档、VT=64/32/16 留给 V 不整除；conv 回 2 波规则（档位不加）。B300 conv 222 µs 只 7% 带宽是 kernel 结构（每线程串行 load→FMA→store）而非行数，留 B300 单独做。`probe_gdn_r9_4` |
| 5 | 第 4 轮两条回正：o-pass 加 VT=128 档（8 warp 两组 64 列，`gdn_chunk_opass_vt` 首选），conv 回 2 波规则（bench 保留 rows 8/16 扫描列） | 3090 默认路径（wy Mma + ho 融合 bv32）：**1×1024 274 / 273 / 75，8×1024 1906 / 1892 / 427，1×8192 1910 / 2053 / 424**；全链 654 / 3842 / 4137；split（VT=128）回到第 2 轮水平 316 / 2112 / 2587（仍慢于融合，默认关） | 待 | 3090 15/15；精度：默认路径 out max_err 2.4e-4 cos 0.99999、state rel 6.7e-3（与第 5A 轮相同）；split 与融合逐位相同；wy 分块逆 vs fp32 老版 u 7.7e-4 / w 3.8e-4 / kg 0 | **3090 侧收口**：1×8192 wy+ho 7166 → 3963（1.81×）、8×1024 全链 6787 → 3842（1.77×），停止条件的 2× 没到但 ho / conv 在 3090 已到带宽 / 寄存器墙，剩余杠杆（ho 两段、PF=1、conv 结构）都要 B300 带宽才可能赢。`probe_gdn_r9_5` |
| 6 | **B300-50 GPU 6 sm_103 首测**（r5 代码；用户要求停 K3 C-6 上 B300）| B300 wy / ho(bv32) / conv(最优档)：8×1024 **531 / 498 / 220**（r0 912 / 664 / 222），1×8192 **532 / 660 / 219**；全链 8×1024 1110、1×8192 1538；1×8192 wy+ho = 1192。split（bv32sp）648 / 1001 仍慢于融合；PF=1 649 / 874 也慢；bv16 694 / 946 慢于 bv32；conv rows128（老规则选的）310 vs rows16 220 | 全链 1110 / 1538 | chunk 14/15：chunk_qwen 全 11 变体过（out 2.4e-4 / state rel 6.7e-3，与 3090 逐位同）；**`gdn_recurrent_decode_b32_qwen_shape` 的 bv0_legacy 变体 max_err 0.017 > 0.01 失败**（decode 老 fp32 kernel，r9 没碰；**pre-r9 树（e71c579）在同卡复测同样失败** max_err 0.0117 cos 0.99794 → 既有问题、与 r9 无关，且两次 max_err 不同（0.017 / 0.012）说明老 fp32 decode kernel 在 sm_103 上有非确定性，另开 issue） | 停止条件：1×8192 wy+ho ≤ 1000 ✅（1192）；8×1024 全链 ≤ 800 ❌（1110，但默认门限还是 3090 的 bv16 / rows128，按 B300 数改 `gdn_chunk_ho_bv` 恒 32、`gdn_conv_rows_for(Sm100)` 8 块/SM 后预计 ≈ 1331）。两段版 / PF=1 B300 也不赢 → 默认保持关。`probe_gdn_r9_6` |
| 7 | B300 第 2 次：`ArchProfile` 按 B300 数改门限（`gdn_chunk_ho_bv` 恒最大可整除档 32；`gdn_conv_rows_for(Sm100)` 8 块/SM 可下探 16/8，Sm8x 不低于 32）| **B300 默认路径 wy / ho / conv**：1×1024 **102 / 87 / 38**（r0 134 / 160 / 41），8×1024 **532 / 498 / 219**（r0 912 / 664 / 222），1×8192 **533 / 659 / 219**；全链 1×1024 311 → **208**、8×1024 1662 → **1109**、1×8192 → **1271**（fla 含 l2norm 359 / — / 853）| 208 / 1109 / 1271 | chunk 14/15（decode_b32 bv0_legacy 既有失败，与 r9 无关）；chunk_qwen 全 11 变体 out 2.4e-4 / state rel 6.7e-3 | **停止条件**：1×8192 wy+ho ≤ 1000 ❌（533 + 659 = 1192，差 19%）；8×1024 全链 ≤ 800 ❌（1109；wy 532 + ho 498 已超）；两段版 / PF=1 在 B300 也慢于融合（bv32sp 642 / 1004、bv32pf 648 / 875），默认关是对的。**vLLM fla 同形状基线补齐（B300 GPU 6 nightly 容器，含 l2norm / 不含）**：1×1024 360 / 289、8×1024 745 / 684、32×128 410 / 380、1×4096 447 / 416、1×8192 853 / 792 µs → dinfer 全链（含 prep）vs fla 含 l2norm：**1×1024 208 vs 360（快 1.7×）、32×128 593 vs 410（慢 1.4×）、8×1024 1109 vs 745（慢 1.5×）、1×4096 660 vs 447（慢 1.5×）、1×8192 1271 vs 853（慢 1.5×）**；伸缩 1 K → 8 K：dinfer 6.1×、fla 2.4×。剩余差距全在 **wy（533 vs fla 整链的 ~60%）与 ho 单序列串行链**；8 轮上限到（r0–r7），回主会话。`probe_gdn_r9_7` |
