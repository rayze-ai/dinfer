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
