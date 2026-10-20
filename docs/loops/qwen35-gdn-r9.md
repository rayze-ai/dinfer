# Qwen3.5 r9：GDN chunk prefill 大 T 伸缩（wy / ho / conv）

> 状态：**3090 侧 5 轮收口（2026-10-20），待 B300 验收**：默认路径 wy 2.2× / ho 1.45×；两段版 ho、PF=1、conv 行数三条在 3090 不赢，默认关，B300 复测后再定门限。来源 `docs/loops/engine-gap.md` 8c–8h：长 prompt TTFT 斜率 dinfer 51.5 vs vLLM ~39 µs/token，差 12.5，其中 GDN 三段约 8（fla 同形状 1 K 持平、8 K 只涨 2.4× 而我们线性 5.3×）。用户 2026-10-20 决定：**按当前方案优化（修自己的 kernel），实在不行再谈替代**。

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
