# Qwen3.5 r11d：GDN prefill 单 kernel（wy 并入 ho，中间量不落显存）

> 状态：**规格已作废并改写（2026-10-21 05:0x）**。原方案「把 wy 熔进 ho」经 FLOP 核算**是亏的**，见下节「作废原因」。
> 现行方向：**ho 单独搬到 tcgen05（BV = V = 128）**，wy 暂留 mma.sync。

## 作废原因（2026-10-21，FLOP 核算）

融合后 grid 只能跟 ho 走 `(V/BV, HV, seq)`，但 wy 的大部分计算**与 V 无关**：

| 分量 | 与 V 有关 | MAC/chunk-head |
|---|---|---|
| kk = k·kᵀ、qk = q·kᵀ | 无关 | 1.05 M |
| 三角逆 A⁻¹ | 无关 | 0.04 M |
| w = A⁻¹(β⊙k·e^gc) | 无关 | 0.52 M |
| u = A⁻¹(β⊙v) | 有关（可切片） | 0.52 M → BV 切片后 0.13 M |

BV=32 → 4 个 v-block 各重算 1.61 M 无关部分：4 × (1.61+0.13+0.91) = **10.6 M** vs 现在 4.2 M = **2.5× FLOP**。
总 51.6 → 65 GFLOP；要赢过 874 µs 需 > 75 TF/s，而 ho 现只有 51 TF/s → **融合版约 1083 µs，更慢**。

而它想省的显存往返只有 117 µs（874 的 13%）。**优先级搞反了**：真正的缺口是利用率——

| kernel | 1×8192 | FLOP | 吞吐 | 占 mma.sync 峰值 |
|---|---|---|---|---|
| wy | 370 µs | 25.8 G | 70 TF/s | 12% |
| ho | 504 µs | 25.8 G | 51 TF/s | **9%** |

**FlashInfer 213 TF/s 已超 48-SM 的 mma.sync 上限（48 × 3.9 = 187）→ 它必然走 tcgen05 且 BV = 全 V。**
r12a 实测 tcgen05 = 2236 TF/s；51.6 GFLOP 按 30% 效率 = **77 µs**。

> 状态（原）：**开工（2026-10-21）**。承接 r11c 的实测结论：FlashInfer GDN 全链 **49 / 156 / 253 µs**（1×1024 / 8×1024 / 1×8192），
> dinfer r10 **208 / 769 / 967**（4.2× / 4.9× / 3.8×）。用户在四组闭环追平后决定：**在现有方案上合并，不推倒重来**。

## 为什么是"合并 wy 进 ho"，不是"A/B/C 三段"

r9 第 2 轮试过 `ho` 两段版（`split`：pass-1 递推写 h16、pass-2 算 o），B300 **全输**：8×1024 555 vs 融合 382、
1×8192 797 vs 504。原因是 h16 每 chunk 写一次（8 K 时 402 MB）再读一遍。**这条路不要再走。**

FlashInfer 的做法也不是在 chunk 维并行——它的 grid 同样是 `(V/BV, HV, seq)`，每 block 串行走完整条序列（和我们 ho 一样），
它是靠 **在一个 block 内把 A⁻¹ / w / u / o 全算完、state 常驻 TMEM** 才快的。所以正确的合并方向是反的：

| | r10 现在 | r11d |
|---|---|---|
| prep | 1 kernel（写 qn/kn/gc） | 并入（qn/kn 的 l2norm、gc 的 cumsum 在 block 内算） |
| wy | 1 kernel，(chunk,hv) 全并行，写 aq/u/w/kg **350 MB** | **删除**，逻辑搬进 ho 的 chunk 循环 |
| ho | 1 kernel，(V/BV,HV,seq) 串行走 chunk，读 350 MB | 唯一 kernel |
| conv | 1 kernel | 保留（短卷积有跨 chunk 状态，另说） |

省的是 **350 MB 写 + 350 MB 读**（8 K，≈117 µs @ 6 TB/s）和 3 次 launch/同步，以及 `wy` 那段孤立的
低效 GEMM（371 µs 只有 29 GFLOP → **78 TF = mma.sync 峰值 575 的 14%**；并入后与 ho 的 GEMM 共用流水）。

## 目标与停止条件（机器可判定）

`crates/kernels/tests/gdn_parity.rs::gdn_chunk_bench`（B300-50 GPU 4/5），新 C 入口默认路径：

- **8×1024 全链 ≤ 300 µs**（现 769；FlashInfer 156）
- **1×8192 全链 ≤ 450 µs**（现 967；FlashInfer 253）
- **1×1024 ≤ 110 µs**（现 208；FlashInfer 49）
- 32×128 ≤ 250 µs（现 421）
- 3090 `ops_parity.sh` + `gdn_parity.rs` 全过（chunk case 数值等价 fp32 递归，门限沿用现值）
- 迭代上限 8 轮；连续 2 轮无进展停下回报

预期不给满：1×8192 只有 `48×(128/BV)×1` 个 block（BV=64 → 96 block < 148 SM），串行链是硬约束，
单 chunk 要做到 ≤ 2 µs 才够 253；mma.sync 大概到 3–3.5 µs，所以 450 是这个结构下的现实值。

## 算法（每 block = (v 块, hv, seq)，串行走 chunk）

每 chunk c 的 7 个步骤，全部在 smem/寄存器里，**不写全局中间量**：

1. `kk = (k⊙β) @ kᵀ ⊙ decay`（BT×BT，下三角）——搬 `wy_mma_kernel` 阶段 1
2. `A⁻¹ = (I + kk)⁻¹`（分块三角逆 `TRI=1`，16×16 对角块 + 非对角块）——搬 wy 阶段 2
3. `w = A⁻¹ @ (β⊙k⊙exp(gc))`（BT×K）——搬 wy 阶段 3
4. `u = A⁻¹ @ (β⊙v)`（BT×BV）——搬 wy 阶段 4
5. `h ← exp(gc_last)·h + kgᵀ @ (u − w·h)`，`kg = k⊙exp(gc_last − gc)`（BV×K，寄存器常驻）——搬 ho
6. `o = (q⊙exp(gc)) @ h_入口 + aq @ u`，`aq = (q⊙exp(gc)) @ (k⊙exp(−gc))ᵀ @ A⁻¹`（BT×BT）——搬 ho + wy
7. 末态回写 `state`（f32）+ 快照（`snap_*`，语义不变）

`h` 的常驻位置分两阶段：
- **阶段 1（本轮）**：`h` 在寄存器 tile（ho 已有，`hacc[H_TILES][4]`），BT=64 / BV=32 或 64。
- **阶段 2（另开 loop，不在本轮）**：`h` 进 TMEM，GEMM 换 tcgen05。

## 设计约束

- **复用现有代码**：`wy_mma_kernel` 的点积 / 三角逆 / u/w 三个阶段与 `ho_mma_kernel` 的 h 递推 / o 阶段都是已验证的，
  搬进同一 kernel 的 chunk 循环里，不重写数学。`gdn_chunk_wy_with` / `gdn_chunk_ho_with` 的**旧路径全部保留**作回退。
- **选路**：`ArchProfile` 新增 `gdn_chunk_fused(arch, ...) -> bool` + `GDN_CHUNK_FUSED=0/1` 覆盖；Sm100 默认 1，
  Sm8x / Sm90 默认 0（先只在 B300 开，3090 只做 parity 不判性能）。
- **kernel 体内不写卡型/形状调优数值**：BT / BV / warp 数由模板参数 + `ArchProfile` 给；smem 预算超了要报错不能静默降级。
- **`prep` 留一个薄壳**：`gdn_chunk_prep` 仍产出 `gc`（cumsum 的 log 衰减）与 `qn/kn` 的 norm 结果给**别的消费方**；
  fused 路径自己算一份进 smem，但不能改 prep 的输出语义（`aq`/`u`/`w`/`kg` 的 scratch 分配在 fused 路径下可跳过，省 350 MB 显存）。
- **快照 / 初始状态**：`state_indices`、`has_initial`、`snap_off/pos/slot` 语义与 ho 一致，一项不许省。
- 新增代码进 `gdn_chunk.cu`（同文件，mma.sync 版）；将来 tcgen05 版另开 `gdn_chunk_sm100.cu`。

## 文件归属

`crates/kernels/cuda/gdn_chunk.cu`（新 `gdn_chunk_fused_kernel` + launcher + C ABI `gdn_chunk_fused`）、
`crates/kernels/src/{ops.rs,ffi.rs,arch.rs,kernels.h}`、`crates/kernels/tests/gdn_parity.rs`（新 case + 计时行）。
**不动 `models/qwen35`**（`gdn.rs` 的调用点在 fused 路径可用后由主会话单独接，避免与 parity 抢工）。

## 环境

- 3090 本机 GPU 0：编译 + `ops_parity.sh` + `gdn_parity.rs` 数值对拍（不判性能）。
- B300-50 GPU 4 或 5：`gdn_chunk_bench` 出性能数（`CARGO_TARGET_DIR=/data/yaochen/infer/target_r11d`，`CUDA_ARCHS=103`）。
- **子代理禁止 `remote_*.sh push`**，只 `put` 单文件；产物落 `bootstrap/r11d/` 由主会话 pull。

## 轮次

| 轮 | 改动 | B300 1×1024 / 8×1024 / 1×8192 全链（µs） | parity | 结论 |
|---|---|---|---|---|
| 0 | 基线 r10 | 208 / 769 / 967 | 过 | 起点；经 r11a/r11b 同口径复测 |

## 设计定稿（2026-10-21，第一版子代理推导，已核对）

**第一版用字面的 `u` / `w` 两段形式**（不改代数），先让数值对上；省 FLOP 的版本留到第二轮。

### 每 block = `(V/BV, HV, seq)`，串行走 chunk

`BV=64`、`BT=64`、`K=128`。`h` 全程在寄存器（沿用 `ho_mma_kernel` 的 `hacc[H_TILES][4]`）。

### smem 预算（98 KB → 2 block/SM）

| 区域 | 大小 | 说明 |
|---|---|---|
| `sM` f32 `[BT][BT]` | 16.4 KB | 逆矩阵；**算完之后与 `sKg` 同址复用** |
| `sM16` bf16 `[BT][TS]` | 9.2 KB | M 的 bf16 影子；**`v'` 算完后被 `sAq` 覆盖** |
| `sAq` / `sU` / `sW` / `sQ` / `sK` | — | 按 `WySmem` 现有布局搬 |
| 合计 | **98 KB** | B300 smem 227 KB，2 block/SM |

`accQ`（即 `Aq`）**全程留在寄存器**（16 regs），到需要落 smem 时才写。

### warp 分工

**8 warp**（`wy` 在 B300 上是延迟型，4 warp 不够）。`ho` 的 `o` / `wh` 相位按 `wcol` 拆列半区，`h` tile 从 16 个/warp 降到 8 个/warp。

### 已被否决的代数简化（第二轮再试，不是第一版）

不做 `w = M(β⊙k e^gc)` 和 `u = M(β⊙v)` 两个 GEMM，改算：

```
v' = M·( β⊙v − (β·2^g_last)·(kg·hᵀ) )
```

数学等价（`w·h = M(β k e^gc) h`），省掉 17.4 KB 的 `sw`，MAC 从 1.31 M 降到 786 K（−40%）。

**陷阱**：`exp(gc)` 与 `exp(g_last)·exp(g_last−gc)` 差一个 `exp(g_last)` 因子，这里最容易错位。**等价性只认对拍不认推导**；对不上就永久放弃这条，不 debug 代数。

**第二处代价**：`kg·hᵀ` 里 `h` 在寄存器且是 mma fragment 布局，当 B 操作数用不了，必须先写 bf16 镜像进 smem（`HoSmem::off_h16`）。那笔 16.4 KB 的寄存器→smem 搬运要算进预算；若它成瓶颈，退路仍是字面的 `w·h`（`w` 已在 smem，直接当 A 操作数）。

### 落地清单

1. `crates/kernels/cuda/gdn_chunk.cu`：`gdn_chunk_fused_kernel` + launcher + C ABI `gdn_chunk_fused`
2. `crates/kernels/src/{ops.rs,ffi.rs,arch.rs}` + `cuda/kernels.h`：`ArchProfile::gdn_chunk_fused()`（Sm100 默认 true，Sm8x/Sm90 默认 false）+ `GDN_CHUNK_FUSED` 覆盖
3. `crates/kernels/tests/gdn_parity.rs`：新 case，fused 输出 vs `wy(Mma)+ho(bv64)` 输出，同输入对齐
4. **旧路径一行不改**（`gdn_chunk_wy_with` / `gdn_chunk_ho_with` / `gdn_chunk_prep`）

### 工作方式（2026-10-21 用户定）

本地 `/data/code/dinfer` 写 → `B300_TAG=50 scripts/dev/remote_b300.sh put` 单文件 → B300 上 `nvcc -arch=sm_103a` 秒级语法检查 → `cargo` + `gdn_parity`。**每轮必须传过去编**。

---

## 里程碑 1（2026-10-21 05:4x，主会话自己写）

**目标**：先把 tcgen05 在这套数据布局下的全部未知点验掉，不写完整 kernel。
新文件 `crates/kernels/cuda/gdn_ho_sm100.cu`，只实现四个 GEMM 中的 **(1) wh = w·hᵀ**。

**结果**（B300-50 GPU 4，`nvcc -gencode=arch=compute_103a,code=sm_103a`）：
- 编译过；`cuobjdump -res-usage`：**REG 26 / STACK 0 / LOCAL 0**（对比 mma.sync 版 ho 的 239）
- 数值：`out[0][0] = 4.404658` vs CPU `4.405192` → **rel 1.2e-4**（bf16 输入 + K=128 累加的正常量级）

**验证过的映射（后续轮直接用）**：
- `tcgen05.alloc.cta_group::1` 分 128 列给 `[BT=64, V=128]` 累加器；`tcgen05.ld.sync.aligned.32x32b.x4` 取回，
  warp w 读 lane 32w..32w+31（128 线程 4 warp 恰好覆盖 128 lane）
- smem 128 B swizzle：`phys = row*8 + ((k>>3) ^ (row&7))`，A=[BT,64]、B=[V,64]，MN 在前、K 连续
- K 轴分块：**第 kb 块的字节偏移是 `kb * rows * 64 * 2`，不是 `kb * 128`**（A 和 B 的块跨度不同，要分开算）
- `make_instr_desc_bf16(M=BT, N=V, a_mn=false, b_mn=false)` + `k_major_128b_desc`

## 踩到的两个坑（都花了轮次，记下来）

1. **smem 布局的元素/字节混算**：`off_tmem = (off_out*2 + out_bytes + 127) & ~127` —— `off_out` 是 bf16 元素偏移，
   `*2` 是字节，再加一个 `out_bytes` 就重复算了。后果是 `tmem_slot` 指到 smem 外，`*tmem_slot` 读到垃圾，
   MMA 写到非法 TMEM 列，**输出全 0 且 `cudaGetLastError()` 返回 no error**。
   教训：**全 0 且无报错 = 大概率是元数据（TMEM 指针 / 描述符）指错，不是计算错**。先单独验证装载/取回通路。
2. **`tcgen05.ld` 是 warp 级指令**，不能让 `if (tid < N)` 只放部分线程进去（同一 warp 内会分歧）。
   正确做法：全部线程参与取回，再由 caller 按行号决定写不写。

## 下一步

(2)(3)(4) 三个 GEMM + 接 C ABI。都是里程碑 1 已验模式的重复，风险低。

---

## 里程碑 2 的设计抉择（2026-10-21 06:0x，未落地）

写完整 ho 之前算了三笔账，结论是先记下来，避免下一轮重走：

### ① `o = o1·dec + Aq·v'` 的 dec 缩放

mma.sync 版是「算完 `q·hᵀ` 再乘 dec」。tcgen05 下累加器在 TMEM，原地缩放要 `ld`+`st` 一整片。
**更好：把 dec 折进 A 操作数**——`(q⊙dec)·hᵀ` 与 `(q·hᵀ)⊙dec` 数学等价（dec 是按 token 的，正好是 q 的行）。
代价是 q 多一次 bf16 舍入（相对误差仍 ~2⁻⁸），换来 (2)(3) 能在同一个累加器上连续累加、中间不取回。

### ② `h = glast·h + v'ᵀ·kg` 的 glast 缩放（真正的难点）

h 若常驻 TMEM，每 chunk 要 `ld → ×glast → st` 一次 128 列，**128 个 chunk 累计可能吃掉整个预算**
（目标 150 µs，这个往返按每 chunk ~1 µs 就是 128 µs）。

**决定：h 留在寄存器（f32 主本），TMEM 只做四个 GEMM 的累加器。**
- (4) 算进临时累加器 → `ld` 一次 → `h_reg = glast·h_reg + tmp`（缩放免费）
- 每 chunk 开头把 `h_reg` 转 bf16 写 smem 当 B 操作数（本来就要写）
- 代价：BV=128 时 h 要 **128 个 f32/线程**（V=128 行 × K=128 列 / 128 线程）。加其余约 180 regs，仍 < 255。
  这也正是 mma.sync 版 BV 卡在 64 的原因（BV=128 要 ~300 regs 且是 mma fragment 布局）；
  走 tcgen05 后 h 不再需要 fragment 布局，只要平铺 f32，才装得下。

### ③ (4) 的 A 操作数转置

`v'ᵀ·kg` 要 A = v'ᵀ（M=v、K=token）。v' 在 smem 里是 `[token][v]`，v 连续 → 对 A 来说是 **MN-major**，
用 `make_instr_desc_bf16(..., a_mn=true, ...)` + `mn_major_128b_desc(lbo, sbo)` 当转置视图用，
与 r12a 在 attn 里对 V 的做法相同（`qwen35-tcgen05.md`：`idesc_o = make_instr_desc_bf16(kBQ, kD, false, b_mn=true)`）。

## 当前状态

- 里程碑 1（GEMM(1) + TMEM/swizzle/取回全链路）：**完成，数值 rel 1.2e-4，REG 26**
- 里程碑 2（四个 GEMM + h 寄存器主本 + 接 C ABI）：**设计完成，未落码**
- 接线依赖 `kernels.h` / `ffi.rs` / `ops.rs` / `build.rs`，r12a 正在改这几个文件，需等它提交后再动

---

## 里程碑 2 进展与卡点（2026-10-21 07:0x，主会话自己写）

### 已完成
- 完整 kernel `gdn_ho_umma_kernel`：4 个 GEMM + chunk 循环 + h 主本 + 末态回写，**能编译、能跑、无 CUDA 错误**
- **REG 168 / STACK 0 / smem 172 KB**（1 CTA/SM）
- 里程碑 1 的 GEMM(1) 数值已验证（rel 1.2e-4）

### 两个已验证的编译期教训
1. **`h` 的 f32 主本必须放 smem**，不能留寄存器：128 线程各持一行 128 个 f32 = 满载，
   实测 **REG 255 + STACK 136 spill**；挪到 smem 后降到 **REG 168 / STACK 0**。
2. `aq` 的形状是 **`[num_chunks, HV, BT, BT]`**（每 chunk 一个 64×64），不是 `[T, HV, V]`。
   按行跨度 `HV*V` 读会静默读到垃圾。

### 卡点：`v'` 的两种用法描述符不一致
`v'` 在 (3) `Aq·v'` 里当 **B**（MN=V、K=token），在 (4) `v'ᵀ·kg` 里当 **A**（M=V、K=token）。
token 在两种用法里都是 K 维，但 A/B 的 MN-major 语义不同，LBO/SBO 不能共用一套。

试过三种布局都没对上参考值（参考值 `out[0][0] = -9.779`，内核给过 3.45 / -0.945 / 3.39）：
`[BT][V]` → `[V][BT]` → `[v 块][token][64 v]`。

### 下一步（不要再盲试）

**拆成两个独立单测，各自验证一个描述符**：
1. 只跑 `Aq·v'`：给已知的 `aq`（64×64）和 `v'`（64×128），核对输出。
   这一步只涉及 B 操作数的 MN-major，与 (4) 无关。
2. 只跑 `v'ᵀ·kg`：给已知的 `v'` 和 `kg`，核对 128×128 输出。
3. 两个都对了再拼回完整 kernel。

**参考实现**（Python，与内核同 bf16 语义）在测试脚本里，注意 bf16 round-to-nearest-even 的写法：
`u = (u + 0x7FFF + ((u>>16)&1)) & 0xFFFF0000`。

### 另一条退路（如果 (4) 的 MN-major 太绕）

(4) 可以不用转置：**先算 `v'ᵀ` 到 smem 的独立缓冲区**（BT×V 的一次转置写，64 个 token × 128 v = 16 KB），
然后 (4) 变成普通的 K-major × K-major。代价是每 chunk 多一次 16 KB 的 smem 写（约 0.1 µs），
换掉 MN-major 的所有不确定性。**推荐先走这条**——转置写是平凡的，(4) 立刻变成和 (1)(2) 一模一样的形状。

### 2026-10-21 07:4x 追加：已定位到 UMMA 描述符层，不是业务逻辑

隔离单测（`o = Aq(64×64) · v'(64×128)`，A/B 都在 smem）的结论：

| 配置 | 结果 |
|---|---|
| A K-major + B K-major，N=128 | `c[0]=3.46`（错，期望 -9.78）；`n<64` 对、`n≥64` 全 0 → **N 只算出一半** |
| A K-major + B **MN-major**（`b_mn=true`, lbo=BT*64*2, sbo=1024） | `c[0] = -9.77879` **正确**；但 `i≥16` 全 0 |
| 同上 + A 的 lbo 改成 2048 / lbo=1024 sbo=0 | `i≥16` 仍然全 0（甚至变成 `i≥8`） |

**确认的事实**：
1. **B 操作数当 K 维在行上时必须用 MN-major**（`b_mn=true`），K-major 只覆盖一半 N。
2. A 在 smem 里的装载**已验证正确**（打印 64 行，每行对角线恰好 1 个非零），所以不是 staging bug。
3. 指令描述符编码**已对着 CUTLASS 核过**（`m_dim = M>>4`，4 对应 M=64；`n_dim = N>>3`），与 `mma_sm100_desc.hpp` 第 431/486 行一致。
4. 剩下的问题在 **A 操作数的 M 方向寻址**：M=64 时硬件只读到第一个 M-tile（16 行）。
   `k_major_128b_desc` 给的 `lbo=0 / sbo=8*128` 是为 `umma_peak` 的 M=128 调的；
   M=64 时 A 需要跨 4 个 M-tile，对应字段应该是哪个还没定。

**下一步（明确的两条）**：
- 读 CUTLASS `make_umma_desc` 对 **M=64 / K-major** 的 canonical layout 推导，别猜 LBO/SBO。
  文件在 B300 上：`/root/develop/vllm-dev/tree/vllm/third_party/deep_gemm/include/cute/atom/mma_traits_sm100.hpp`（需 `sudo -n`）。
- 或者**绕开 M=64**：把 A 的 M 也做成 128（`Aq` 用 64 行 + 64 行零填充），沿用 `umma_peak` 已验证的 M=128 路径。
  GDN 的 BT=64 是因为 chunk 大小，但 GEMM 的 M=128 只是多算一半零行，代价 2× 但立即能用。

---

## 2026-10-21 08:xx M=128 补零改造（进行中）

### 已实测证实的关键结论（勿再推导）

1. **tcgen05 的 A 操作数 M 必须是 128**。M=64 时：每 16 行里只有前 16 行有结果，其余为 0，
   `cudaGetLastError()` 返回 no error。SBO 取 128/256/512/1024/2048 B **完全无影响**。
   证据：身份矩阵 A × 全 1 的 B —— M=64 `good=4096/8192`、M=128 `good=16384/16384`。
   见 `docs/knowledge/kernel-rules.md`。
2. **A 操作数的行数改了，描述符的 K 块跨度必须跟着算**：`(uint32_t)(kb * rows * 64 * 2) >> 4`。
   写死 `(kb*64*2)>>4` 是错的。
3. **B 操作数当「K 维在行上」时必须存成 K 维连续**：`v'` 当 B 要 `[N=V][token]`、`kg` 当 B 要 `[K][token]`，
   都不能用 `[token][V]`。用显式转置写 smem，**不要用 MN-major 描述符**（试了 3 种 LBO/SBO 组合都没对上）。
4. **`v'` 的 token 维只有 64 列**，写它的线程必须 `if (myrow < kBT)`，
   否则 `myrow` 覆盖到 127 会越出 64 列（做过 OOB）。
5. **`aq` 的形状是 `[num_chunks, HV, BT, BT]`**（每 chunk 一个 64×64），不是 `[T, HV, V]`。
6. **`maxabs` 大但 `out[0][0]` 对，不能当成 bf16 累积误差放过** —— 里程碑 1 就是这样把 3/4 行全错解释掉的。

### 当前状态

- (1)(2)(3) 三个 GEMM **数值正确**：`out[0][0] = -9.75000` vs 参考 `-9.75`、`out[0][1] = -4.15625` vs `-4.15625`
- (4) `v'ᵀ·kg` **单独测试完全正确**（`good=16384/16384`，A=`v'ᵀ[V][token]`、B=`kgᵀ[K][token]`）
- 但**拼进完整 kernel 后 (4) 结果不对**：`state[0][0] = -36.29` vs 参考 `-60.94`
  （参考逐 chunk：chunk0 → 0.95790、chunk1 → -60.94093）
- 正在加 chunk 级调试输出定位

### smem 预算（227 KB 上限）

| 区 | 大小 | 说明 |
|---|---|---|
| h（B） | 32 KB | `[V][64]×2` |
| q、w（A） | 各 32 KB | 补零到 M=128 |
| kg（B） | 16 KB | `[BT][64]×2` |
| Aq（A） | 16 KB | 补零到 M=128 |
| v'、v'ᵀ | 各 16 KB | `[V][token]` |
| kgᵀ（B） | 16 KB | `[K][token]` |
| **合计** | **176.5 KB** | h 的 f32 主本放全局 `state`（每 chunk 读写 32 KB） |

寄存器：**REG 168 / STACK 0**。

---

## 2026-10-21 09:xx B300 首轮性能数（UMMA 版已接 C ABI 与 ArchProfile）

### 数字（B300-50 GPU 4，CUDA_ARCHS=103，`gdn_chunk_bench`，连出两次一致）

| 形状 | 现有 ho（bv64）| UMMA 版 | 对比 |
|---|---|---|---|
| 1×1024 | 59 µs | **867** | 15× 慢 |
| 8×1024 | 309 | **3561** | 11× 慢 |
| 32×128 | 180 | **1648** | 9× 慢 |
| 1×4096 | 302 | **3943** | 13× 慢 |
| 1×8192 | 595 | **7893** | 13× 慢 |

全链（prep + wy + ho + conv）不受影响，因为默认路径没切（`ArchProfile::gdn_ho_umma` 只在
`GDN_HO_UMMA=1` 或显式调用时走）。

### 慢的结构性原因（不是我算漏了 2× 补零，是并行度）

UMMA 版 grid = `HV × seq`，**1×8192 时只有 48 个 block**，而 B300 有 148 个 SM —— 只用到 32% 的卡。
现有路径靠 `BV` 切片（V/64 = 2 份）+ mma.sync 多 warp 掩盖延迟。

我在设计阶段算的那笔账（"tcgen05 2236 TF/s，51.6 GFLOP 只要 46 µs"）**漏掉了并行度**：
2236 TF/s 是 148 个 block 全部在跑时的数字，48 个 block 只能拿到约 1/3。
再叠加 M=128 补零的 2× FLOP 浪费（AB 两个操作数各补一半零行），实际效率更低。

### 下一步（必须先定方向，不要再直接改代码）

1. **恢复 V 切片**：BV=64 → grid ×2 = 96 block，BV=32 → 192 block。但 h 的 f32 主本要按 BV 切，
   而 `h[v][k]` 的 v 维正好是 BV 切的维度，切了不影响正确性（每个 block 只管自己那段 v）。
   这是最直接的改法，预期 7893 → ~2600（1×8192）。
2. **或者接受结论**：GDN 的 ho 段在单序列长 prefill 下并行度天然不足（48 个 hv × 1 条序列），
   tcgen05 换不来收益，转回 mma.sync 并把精力放到别处。
3. 需要先算清：BV=64 时 smem 能否装下（现在 176.5 KB 是按 BV=128 算的，BV=64 会减到约 130 KB）。

### 已落地的接口（都过编译 + 3090 对拍 16/16）

- `kernels.h` / `ffi.rs`（含 sm100 条件编译 + NotSupported 桩）/ `ops.rs::gdn_ho_umma`
- `arch.rs::ArchProfile::gdn_ho_umma(arch, k, v)` + `has_sm100()`，**卡型 case 分支**：
  Sm100 且 K=V=128 → true；Sm90/Sm8x → false；`GDN_HO_UMMA=0/1` 覆盖
- `models/qwen35/src/gdn.rs` 调用点（有快照时回退 mma.sync）
- `build.rs` 白名单加 `gdn_ho_sm100`

---

## 2026-10-21 10:xx 根因定位：是我的 kernel 结构，不是 tcgen05 本身

用户批评后重查，前面「并行度不足」的结论**是错的**：

FlashInfer 的 grid shape 是 `(B × H, 1, 1)` 或 persistent `(min(B*H, max_active_clusters), 1, 1)` ——
1×8192 时同样是 48 个 CTA，**并行度和我的完全一样**。它跑 253 µs，我跑 7893 µs，
同样的并行度差 **31 倍**，所以问题在我 kernel 内部效率（1.1% vs 它的 35%）。

### 实测拆解（B300 GPU 4）

| 项 | 实测 |
|---|---|
| 每 chunk 耗时 | **61.3 µs**（1×8192 → 7843 µs）|
| `__syncthreads` | 0.019 µs/次 × 8 次/chunk = 0.15 µs |
| `mbar commit+wait` | 0.050 µs/次 × 3 次/chunk = 0.15 µs |
| 普通 ld/st 装载 144 KB | **6.73 µs/chunk** |
| `cp.async` 装载同量 | **1.10 µs/chunk**（6× 快）|
| `tcgen05.ld` 改单次等待 | 无改善（65 vs 61 µs）|

**同步和 TMEM 取回都不是主因**（合计 < 1 µs）。装载 6.7 µs 也不是。剩下 ~54 µs 在别处。

### FlashInfer 的真实结构（`gated_delta_net_chunked.py`）

**12 个 warp（384 线程）按角色分工**，不是 4 warp 全串行：

| warp | 角色 |
|---|---|
| 0–3 | compute_group_0（T-pairwise / kk_epi / qk_epi / inverse）|
| 4–7 | compute_group_1（kv_decay_v / v·k·state / epi ops）|
| 8 | mma 发射（前 6 个 GEMM）|
| 9 | TMA 装载 q/k/v |
| 10 | mma 发射（后 5 个 state/output GEMM）|
| 11 | epilogue（存 O）+ 预取 gate/beta |

- **两个独立的 MMA 发射 warp**（8 和 10），GEMM 队列始终有两条流在飞
- 寄存器**按角色分配**：CG0 224 / CG1 256 / 其它 24–48
- **mbarrier 流水，不用 `__syncthreads`**（除 TMEM alloc）
- smem stage 数：q 2 / k 4 / v 3 / ainv 3 / qk 2 / o 2 / gate 5

### 下一步

按 12 warp + 双 MMA 发射 + mbarrier 流水重写 `gdn_ho_sm100.cu`。
这是追平 FlashInfer 的必要结构，不是可选项。
