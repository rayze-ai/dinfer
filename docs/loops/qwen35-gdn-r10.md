# Qwen3.5 r10：GDN chunk prefill 在 B300 上追平 fla（wy / ho 中间量 bf16、ho 串行链、门限自调优）

> 状态：**开工（2026-10-20 17:2x）**。前置：r9 收口 + fla per-kernel 分解（`qwen35-gdn-r9.md` "fla 分解"）。目标按分解重定：wy 532 → ≤ 350、ho 498 / 659 → ≤ 350 / 450（8×1024 / 1×8192），即每段追到 fla 的 1.05× 内。

## 环境

- **出数：B300-50 GPU 7**（`CUDA_VISIBLE_DEVICES=7`，`CARGO_TARGET_DIR=/data/yaochen/infer/target_r10`，构建 `taskset -c 64-95`、运行 `taskset -c 48-63`；`source /data/yaochen/infer/env.sh`，`CUDA_ARCHS=103`）。GPU 0/1 主会话 bench、GPU 2–5 留 K3、GPU 6 fla 探针，**都不碰**。
- **3090 本机只跑 parity + 编译**（knowledge/kernel-rules：性能改动在 B300 定去留，3090 看不出 B300 的延迟型瓶颈）。
- 同步：子代理不 `push`；改动文件 `remote_b300.sh put` 单文件到远端同路径（远端树无 .git，与本机 dev 同内容，开工前 md5 已核对一致）。产物落远端 `bootstrap/`，本机 `pull` 回 `test/qwen35/2026-10-20/probe_gdn_r10_N/`。
- 对手口径：fla 分解表（r9 页）。vLLM 侧 conv 走 `causal_conv1d_fn`，第 1 轮顺手测进 probe（`fla_conv_us`），全链才可比。

## 文件归属

`crates/kernels/cuda/gdn_chunk.cu`、`crates/kernels/src/ops.rs` 的 `gdn_chunk_*` / `GdnChunkScratch`、`crates/kernels/src/ffi.rs` 对应签名、`crates/kernels/src/arch.rs` 的 `gdn_chunk_*`、`crates/kernels/tests/gdn_parity.rs`。**不动 `models/qwen35`**（`GdnChunkScratch::new` 签名不变，字段类型可变——gdn.rs 只持有 / 传引用）。

## 改法（按序，每步独立可测、独立可回退）

1. **中间量 bf16**：`GdnChunkScratch` 的 aq / u / w / kg f32 → bf16；`wy_mma_kernel` 直接写 bf16（u / w 已在寄存器 f32 累加，只改 store；kg 的 `k0*dk` 直接 pack）；`ho_mma_kernel` / `ho_hpass_kernel` 的 `fetch()` 改 uint4 直载 bf16 进 smem（省掉 `pack_bf16x2` 转换与一半寄存器 → PF=1 预取在 BV=32 可能回正）；`ho_opass_kernel` 同。老 fp32 `wy_kernel` / `ho_kernel`（legacy 回退路径）改读写 bf16 保持可用。parity：`gdn_parity` 容差已是 bf16 级，aq 的 bf16 化若让 `wy_tc_vs_fp32` 对拍飘，收紧到比 u / w / kg（它们本来就被 ho 转 bf16 用）。
2. **ho 拆 o 全并行 + h-pass 双缓冲**：fla 模式——h-pass 只递推 h（smem 两份 cp.async 双缓冲，装载 c+1 与算 c 重叠，4 warp → 8 warp 分 V 列合并 `__syncthreads`），o-pass 用 r9 已有 `ho_opass_kernel`（h16 bf16）。r9 两段版慢在 h-pass 721 > 融合 659，原因就是无双缓冲 + 多写 h16；第 1 轮 bf16 后 h16 相对开销减半再看。
3. **wy 非对角块 mma**：(I+A)⁻¹ 的 6 个非对角块 `M_ij = −M_ii·ΣA_ik·M_kj` 现 fp32 标量乘 → m16n8k16 ×2。
4. **门限自调优**：`ArchProfile::gdn_chunk_*` 的 bv / rows / warps 从"按卡分档"改为"候选集 + 启动时按真实 (T, HV) 各跑一次取优"（kernel 体内仍无数值）。

## 硬性规则

- kernel 内不写卡型 / 形状调优数值，`#if __CUDA_ARCH__` 禁止在 kernel 体内；按 arch 分文件只在确有分叉时。
- 每轮：`gdn_parity` 14/14（3090），B300 `gdn_chunk_bench` 五形状进 `probe_gdn_r10_N/probe.csv`（`gdn_bench_to_probe.py`），commit 不 push，本页轮次表一行。
- 旧路径保留为回退（环境变量只做覆盖）。

## 停止条件（机器可判定）

- `gdn_parity` 14/14（3090 + B300 sm_103）；
- B300 `gdn_chunk_bench`：**8×1024 wy ≤ 350 且 ho ≤ 350**、**1×8192 ho ≤ 450**（fla 334 / 332 / 440 的 1.05×）；全链不含 conv 8×1024 ≤ 785（fla 746 ×1.05）；
- 4090 `bench_gate.sh` 四组不退化 > 3%；B300 `bench_compare.sh` 四组不退化（主会话跑）。
- 迭代上限：8 轮或 2 天；卡住 2 轮回主会话。

## 轮次

| 轮 | 改动 | B300 GPU 7：1×1024 / 8×1024 / 1×8192（prep / wy / ho / conv / chain µs） | parity（3090） | 结论 |
|---|---|---|---|---|
| 0 | 基线 = r9 第 7 轮（`probe_gdn_r9_7`，GPU 6） | 18/102/87/38/208 · 82/532/498/219/1109 · 82/533/659/221/1271 | 14/14 | fla：wy 334 / ho 332 / 440 |
| 1 | **aq / u / w / kg f32 → bf16**（`GdnChunkScratch` 四字段、wy 存 pack、ho / hpass fetch 改 uint4 直载省 pack，opass aq 直拷，legacy 两 kernel 跟改；C 入口按 kernels.h 约定 void*）`d2cc649` | 18/76/65/39/**185** · 82/505/405/221/**988** · 82/505/690(bv32pf **520**)/219/**1271** | 15/15（3090 + B300） | 8×1024 chain −11%（wy −5%、ho −19%）；1×8192 默认 bv32 反而 659→690，但 **bv32pf 520**（r9_7 为 685）：bf16 把 PF 的寄存器开销减半，预取从"全形状退"翻成"长形状赢 25%"。wy 只降 5% → wy 的 ~200 µs 差不在写量，在 u/w 两 GEMM 的 stage 装载 / M·stage 算力（下一轮看 ncu）。**第 2 轮：`ArchProfile::gdn_chunk_ho_prefetch` 在 sm_100 开 PF**（kernel 不变，只改选档）+ hpass 双缓冲 |
| 2 | **`ArchProfile::gdn_chunk_ho_prefetch_for(Sm100) = true`**（kernel 不变，默认路径 = bv32pf；Sm8x 仍关；`GDN_CHUNK_HO_PREFETCH=0/1` 覆盖）`55f3e0e` | 18/76/66/38/**178** · 82/505/378/221/**967** · 82/505/520/219/**1103** | 14/15（B300：`gdn_recurrent_decode_b32_qwen_shape` 的 `bv0_legacy/out` max_err 0.009–0.016 波动于 0.01 门限，**r9_b300.log / r9_b300_7.log 同一测试已经在 0.0078 擦线**，是老 fp32 decode kernel + 随机 100 MB 状态的既有飘动，与本轮改动无关；3090 15/15） | 1×8192 chain 1271 → **1103**（−13%），8×1024 988 → 967。累计 r9_7 → r10_2：8×1024 1109 → 967（−13%）、1×8192 1271 → 1103（−13%）。ho 1×8192 520 vs fla fwd_h+fwd_o 440 还差 18%；wy 505 vs 334 差 51% —— **wy 现在是最大差距**。第 3 轮 ho smem 双缓冲（cp.async，Sm100 smem 够两份）或先转 wy？按分解：wy 差 171 µs > ho 差 80 µs，**先 wy** |

### 第 3 轮前置：B300 ncu `wy_mma_kernel<64,128,TRI=1>`（8×1024，GPU 7，`probe_gdn_r10_3/ncu_wy_b300.txt`；**ncu 在 B300 要 `sudo -n`**，普通用户 ERR_NVGPUCTRPERM）

| 指标 | B300 | 3090（对照） | 读法 |
|---|---|---|---|
| Duration | 422 µs | 1140 µs | bench 505 含 launch 间隙 |
| SM 吞吐 / Mem 吞吐 / DRAM | 34% / 62% / **6%** | 27% / 32% | 不是 HBM 带宽；Mem 62% 全是 smem（ldmatrix / sts） |
| 占用：理论 / 实际 | 31% / 29%（5 block/SM，regs 96 与 smem 45.6 KB 都限 5） | 17% / 16%（smem 限 2） | 资源没满，**warp 数不够掩延迟** |
| 每 scheduler 可发射 warp | **0.53**，64% 周期无可发射 warp | 0.22，80% | 典型延迟型 |
| 每指令周期 | 12.8，其中 **4.6 短记分板（smem 依赖）= 36%** | 9.8 | 分块逆的 sM 标量链 + stage 的 2 B 标量装载 |
| grid | 3072 block × 128 线程 = 4 warp | — | 148 SM × 5 = 740 常驻 → 4.15 波 |

结论：wy 不缺带宽、不缺占用槽位，缺**每 block 的并行 warp**和**更少的 smem 依赖链**。fla 把 wy 拆成 3 个 kernel 各自 autotune num_warps（kkt / wu 常用 8 warp）。第 3 轮改法：`kWyWarps` 4 → 8（u / w 阶段 8 warp 各持 64×64 片的 8 列 tile；点积阶段 8 warp 各 16 行 × 一半列；分块逆 4 对角块给 4 warp、另 4 warp 并行做非对角块的 S 累加），smem 不变。目标 505 → ≤ 400。
| 3 | **wy 8 warp**（`wy_mma_kernel<BT,K,TRI,NW>`，点积 / u·w 阶段每 warp 16 行 × 一半列，对角块逆仍 4 warp；`GdnChunkWy::Mma8` = wy_mode 3；`ArchProfile::gdn_chunk_wy_warps_for(Sm100)=8`，Sm8x 仍 4）`7f3d6dc` | 18/65/65/39/**161** · 82/**435**/377/220/**898** · 82/435/521/219/**1034** | 14/15（decode_b32 同前擦线；mma8 与 4 warp 逐位一致） | wy 506 → 435（−14%，ncu 预判对：延迟型加 warp 有效）；8×1024 chain 967 → **898**、1×8192 1103 → **1034**。累计 r9_7 → r10_3：8×1024 **1109 → 898（−19%）**、1×8192 **1271 → 1034（−19%）**；不含 conv：678 vs fla 746（**已反超 9%**）、815 vs 853（反超 4%）。3090 上 mma8 也快 19%（1712 → 1387）但 Sm8x 默认暂不改（规则：B300 出数、3090 只跑 parity；若主会话想顺手开 4090 门禁再定）。wy 435 vs fla 334 仍差 30%：ncu 短记分板的另一半是 stage 装载 2 B 标量 → 第 4 轮 16 B 向量装载 |
| 4 | **wy stage 装载 2 B 标量 → 16 B uint4**（8 列/线程，v / kn 读、stage 写、kg 写全部 16 B；smem 不变）`0a6f93a`；本轮起在 r8b 合并树（`cb7ee9a`）上出数 | 18/57/65/38/**148** · 82/**360**/380/221/**822** · 82/360/524/219/**963** | 4/4 chunk（B300）；3090 mma8 逐位同 4 warp | wy 435 → 360（−17%；两轮合计 506 → 360，−29%）；8×1024 chain 898 → **822**、1×8192 1034 → **963**。累计 r9_7 → r10_4：**8×1024 1109 → 822（−26%）、1×8192 1271 → 963（−24%）**；不含 conv 601 vs fla 746（**反超 19%**）、744 vs 853（反超 13%）。wy 360 vs fla 334 还差 8%。**异常**：ho 未改，但 bv32（无预取）从 692 → 497、373，反而比 bv32pf（524 / 380）快了——wy 的 16 B 写改变了 ho 读 w/kg 时的 L2 命中态，预取收益被抹平。第 2 轮"Sm100 开预取"的依据失效，但一次数据不翻档（规则：≥ 2 卡 × 全形状）；这正是改法 4"门限自调优"的用例——预取与否按启动时真实形状实测。下一轮：先看 ho ncu，再定 ho 双缓冲 vs 自调优 |

### 第 5 轮前置：B300 ncu `ho_mma_kernel<64,32,128,PF=1>`（1×8192 默认路径，GPU 7，`probe_gdn_r10_4/ncu_ho_b300.txt`）

| 指标 | 值 | 读法 |
|---|---|---|
| Duration | 615 µs（bench 524 含图外；ncu 的 clock 锁频偏高） | — |
| DRAM / SM / **Mem Busy** | 11% / 31% / **77%** | 不是 HBM；Mem Busy 77% = **smem 管线**（ldmatrix / lds_u32 / sts），ho 是 smem 带宽型 |
| 寄存器 | **242 / 线程** | PF=1 的 rq/rw/rg/ra 预取寄存器 + hacc[H_TILES][4] + oacc/wacc；255 上限边缘 |
| 占用 | 理论 12.5% / 实际 12.1%（**2 block/SM**，regs 与 smem 75 KB 同时限 2） | 单 (seq, hv, V 片) 一个 block、每 SM 8 warp |
| 可发射 warp / scheduler | 0.37，69% 周期无可发射 | 延迟 + smem 管线双重 |
| grid | 1536 block（1×8192：1 seq × 48 hv × 4 V 片 × … = 192 block？实为 192×8 chunks 并行？）→ 5.19 波 | 1×8192 只有 192 个 (seq,hv,片) 串行链；ncu 这里采的是 32×128 形状的 launch（1536 = 32×48） |

结论：ho 的瓶颈 = **smem 管线 77%**（四个 mma 的 B 操作数 h16 用 32 位 `lds_u32` 逐 lane 读 `[v][k]`，每次 mma 两条 lds；fla 用 Triton 的 `tl.dot` 走 ldmatrix 全程）+ 占用 12.5%。两条可选：
- A）**h16 的 B 读改 ldmatrix.x4**（`sh16` 按 [v][k] 行主序，mma B 需 col-major [k][v]：用 `ldsm_x4_t` 从 [v][k] 转置取），每 ks 步 4 条 lds_u32 ×NT_V → 1 条 ldsm_x4 / 2 tile；smem 事务减 ~4×。改 kernel 体内 (1)(2) 两处 B 读，不碰 smem 布局 / snap / 入口。
- B）cp.async 双缓冲 + 8 warp（改法 2）：regs 已 242，再加 8 warp 的 oacc/wacc 只会溢出；双缓冲 smem 75 → 130 KB 把占用压到 1 block/SM。**B300 上 B 不划算，先做 A**。
- 第 4 轮 bv32（无预取）反超 bv32pf 的现象与此一致：PF 的寄存器预取把 regs 顶到 242 → 2 block/SM；无预取版 regs 更少可能 3 block/SM，掩延迟反而更好。等 ncu 的 hpass / bv32 无 PF 行再定预取默认。

ho_mma 原始计数（第 4 轮代码，8×1024 launch，`r10_ncu_ho3.txt`，第 5 轮对照基线）：inst 112.8 M，其中 **shared_ld 16.0 M**（lds_u32，B 操作数 h16 + sdec/sb 等）、ldsm 4.3 M、shared_st 6.1 M；每发射停顿（周期）：wait 1.40 / short_scoreboard 1.13 / long_scoreboard 1.02 / **mio_throttle 0.92** / not_selected 0.22 / barrier 0.20；smem bank 冲突：ld 0.29 M、**st 0.73 M**（svT 转置写 + sh16 写）。第 5 轮（B 读改 ldmatrix）预期：shared_ld 16 M → ~4 M、mio_throttle 与 short_scoreboard 下降；st 冲突不动（第 6 轮候选：svT 行跨 / swizzle）。

### 第 6 轮候选（从第 4 轮代码的 ncu 计数推，待第 5 轮数字确认方向）：svT 的 2 B 转置写是 st bank 冲突源

`svT[vv*TS + i0]`（TS=72 bf16 = 36 word）：同一 warp 内 lane(g, t4) 写 word = (8·t4·36 + g/2) mod 32 → g=2m 与 g=2m+1 落同一 word、t4 四档落 bank 0/0/0/0 偏移（8·36 = 288 ≡ 0 mod 32），**32 lane 只占 16 bank，每条 st 2 路冲突**；sh16 的 `sts_u32`（KS=136 → 68 word，68g mod 32 = 4g）32 lane 恰铺 32 bank，无冲突。所以 st 冲突 0.73 M 全在 svT（每 chunk 4 条 × NT_V × 4 warp）。改法：v' 不转置，按 [j][v] 行主序 32 位 `pack_bf16x2` 存（与 sh16 同款、无冲突，写指令减半）；(3) 的 B 操作数 v'[j][v]（k=j，n=v）用 `ldsm_x4_t`（与 (4) 读 skg [j][k] 完全同式）；(4) 的 A 操作数 v'ᵀ[v][j] 用 `ldsm_x4_t` 从 [j][v] 取（lane 地址 = `sv + (ks·16 + ((lane>>4)&1)·8 + (lane&7))·VS + mt·16 + ((lane>>3)&1)·8`，与 opass 第 1364 行读 sv 同族）。smem 不变（[BT][VS] 与 [BV][TS] 同量级）。
| 5 | **ho_mma / hpass 的 h16 B 操作数 4×lds_u32 → ldmatrix.x4**（(1)(2) 两处 + hpass 一处，smem 布局不变）`b10eceb` | 18/57/63/39/**147** · 82/360/384/221/**833** · 82/360/504(bv32)/219/**966** | 4/4 chunk（B300）；3090 过 | **中性**：ncu 计数按预期动了（shared_ld 16.0 → 3.4 M、ldsm 4.3 → 7.5 M、总指令 112.8 → 105.9 M、mio_throttle 0.92 → 0.62），**时钟不动**（ncu 615 → 618 µs；bench ho 380 → 384 / 524 → 525，噪声内）。说明 ho 的"smem 管线 77%"是症状不是瓶颈——减掉 12.6 M 条 lds 一点没省时间，**瓶颈在 mma 依赖链 / 每 block 的并行 warp 数**（short_scoreboard 反升 1.13 → 1.54：少了 lds 的掩护，mma 之间的等待裸露）。保留本轮（指令更少、后续改并行度时少一个变量）。第 6 轮**不做** svT bank 冲突（同理不会动时钟），改做并行度：BV=32 每 warp 只 2 个 V tile + H_TILES=4 个 h tile，mma 链太短；候选 (a) BV=64 一 block 持 8 warp（regs 不增、每 SM warp ×2），(b) 一 block 两条序列 / 两个 hv 交错（独立链填空）。先取 ncu 1×8192 单 launch + Source 级 stall 定位再定 |

### 第 6 轮前置：ho_mma **1×8192 单 launch** ncu（`r10_ncu_ho6.txt.29`，bv32 PF=1，默认路径）

| 指标 | 1×8192 | 8×1024（bv16 PF=0 对照，`.39`） |
|---|---|---|
| grid | **192 block**（1 seq × 48 hv × 4 V 片）；regs 239 → 2 block/SM；148 SM 只有 96 个 SM 有活 | 3072 block |
| `sm__warps_active` | **8.5%** | 12.2% |
| `issue_active` | **23%**（每 4 周期发 1 条） | 25% |
| `pipe_tensor_cycles_active` | **19%** | 17% |
| 每条指令延迟 | 5.8 周期（wait 1.48 / long_sb 1.23 / short_sb 1.16 / mio 0.37） | 7.7 |

结论：ho 是**并行度 / ILP 型**——tensor pipe 五分之一在转，发射槽四分之三空着；不是 smem、不是 HBM（第 5 轮已证）。BV=32 下每 warp 一步只有 2 个 V tile（o / w 各 2 条独立 mma 链）+ 4 个 h tile；mma 之间互相等。两条路：
- **(a) BV=64**：每 warp 8 个 h tile、o / w 各 8 条独立链（ILP ×2），smem 86.5 KB（仍 2 block/SM 档），grid 减半（1×8192 → 96 block，但每 block 8 warp 的 mma 流更满；148 SM 本来就铺不满，再少一半 SM 也只是把两个 block 的活合成一个更满的 block）。regs 风险：PF=1 预取数组随 BV 翻倍 → ~250，要带 PF=0 档一起测；`static_assert` 已允许 BV=64（(MT_V·NT_K) % (2·NW) = 64 % 8 = 0）。
- (b) 一 block 两条独立链（两个 hv 交错）：改动大、regs 翻倍，B300 上不如 (a)。
第 6 轮做 (a)：加 `LAUNCH_HO_MMA(64, K)` 档 + bench `bv64` / `bv64pf` 列；默认档仍 32，B300 出数后再改 `ArchProfile::gdn_chunk_ho_bv` 候选序。
