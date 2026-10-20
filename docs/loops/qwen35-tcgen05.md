# Qwen3.5 r12：tcgen05 化（attn 先，GDN 后）——把 mma.sync 的 575 TF 上限换成 UMMA

> 状态：**开工（2026-10-21）**。触发：r11a 收口结论——attn 6 K 单层 1558 µs = **286 TF = mma.sync 实测峰值 575 TF 的 50%**，
> 700 µs 目标物理不可达（需 636 TF）；r11d 的 GDN 同样撞这个墙。用户 2026-10-21 决定两步一起开。

## 为什么必须换指令集

`scripts/dev/mma_peak.cu` 在 B300-50 实测：**m16n8k16 bf16 纯算 537–575 TFLOP/s**（4 / 16 / 32 warp 都一样），
而标称 2.25 PF/s 是 **tcgen05 / FP8** 口径。`mma.sync` 是 Ampere 时代的逐 warp 指令，在 Blackwell 上它只能吃到
标称的 ~25%——这不是调优能突破的，是 ISA 的代差。

tcgen05.mma（UMMA）的三点不同，正是这 4× 的来源：
1. **单发射**：一个 warp 发一条指令算整个 CTA（甚至 2-CTA）的 GEMM，不再有每 warp 的发射率上限；
2. **TMEM 累加器**：C tile 在 Tensor Memory（256 KB/SM）里，不占寄存器、不需要 ldmatrix 反复搬；
3. **异步**：`tcgen05.commit` / mbarrier 完成通知，GEMM 与 softmax / 递推天然重叠。

## 分两阶段，本轮只做阶段 1

### 阶段 1（r12a，本轮）：attn 的 tcgen05 版 + 工具链建立

attn 是干净目标：纯 GEMM + softmax，无跨 chunk 串行依赖，已有 r11a 的 v6 kernel 作数值参考（`GQA_PREFILL_CFG`）。

**先做"工具链自证"，再改业务 kernel**（避免又陷入"照着一个不懂的结构调 8 轮"）：

1. **单 kernel 自证**：新建 `crates/kernels/cuda/umma.cuh` + `crates/kernels/tests/umma_peak.rs`，
   一个最小 tcgen05 GEMM（M=128, N=256, K=64 循环），在 B300 上跑出 **TF/s**。
   **门槛：≥ 900 TF/s**（1.5× mma.sync 的 575）。达不到就停下回报，不要往下走。
   这一步要解决的全部新东西：TMEM 分配（`tcgen05.alloc`）、smem 布局（128 B swizzle，`SWIZZLE_128B` 描述符）、
   UMMA 指令描述符（`tcgen05.mma.cta_group::1.kind::f16`）、`tcgen05.ld` 取回累加器、mbarrier 等待。
2. **attn 移植**：`crates/kernels/cuda/gqa_fa_sm100.cu`（**新文件**，遵守 CLAUDE.md 多卡型规则），
   QK 与 PV 两次 GEMM 走 UMMA，softmax 仍在寄存器；`ArchProfile::gqa_prefill_cfg` 增加 tcgen05 档（如 `5xxx`），
   `GQA_PREFILL_CFG` 可覆盖回 v6。

### 阶段 2（r12b，**不在本轮**）：GDN 的 tcgen05 版

等 r11d 的 fused dataflow 与 parity 落地后，把 `h` 搬进 TMEM、7 个 chunk 内 GEMM 换 UMMA。另行立项。

## 目标与停止条件（机器可判定）

阶段 1：
- `umma_peak`：B300 **≥ 900 TF/s**（硬门槛；不达即停）
- `gqa_fa_bench` 默认路径 **T=6019 ≤ 800 µs**（现 1558；即 ≥ 555 TF 等效）
- 3090 + B300 `ops_parity` gqa 项全过（`GQA_PREFILL_CFG=0` 回退 v5 也要过）
- 16 层 attn：6 K 32.6 → **≤ 17 ms**
- 迭代上限 10 轮；连续 3 轮无进展停下回报

## 硬性约束

- **新代码只进 `*_sm100.cu` / `*_sm100.cuh`**，`gqa.cu` 里的 v5 / v6 一行不改（只加分发分支）。
- kernel 体内不写卡型 / 形状调优数值；BT / warp / stage 由 `ArchProfile` + 自调优给。
- **build.rs**：① `sm100_only` 白名单**显式加项**（`mega_sm100` / `gqa_fa_sm100` / `umma_peak_sm100`），
  **不改成"stem 以 `_sm100` 结尾"**——那会让 `mxfp4_gemm_sm100.cu`（glm53/kimik3 在用）从"永远编"变成"has_sm100 才编"，超范围；
  ② 目标 arch 含 100/103 时**全局**换 `a` 变体（`-gencode=arch=compute_103a,code=sm_103a`）——tcgen05 需要 `a` 后缀，
  而 `sm_103a` cubin 的可运行范围与 `sm_103` 相同，全局换无副作用且不用引入 per-file 旗标。
- **本机编译约束（2026-10-21 实测纠正）**：本机 nvcc 12.4 最高 sm_90a，`MEGA_SM100_DEV=1` **也编不过 tcgen05 语法**
  （PTX 版本 / 指令不支持）。所以本机**不能**做 tcgen05 的语法检查。开发循环是：本地写 → 单文件 `put` →
  B300 GPU 5 上 `nvcc -arch=sm_103a` 直接编（不经 cargo，几秒一轮）→ 通过后走 cargo 全量 + 测试。
- 3090 上不编 sm100 文件（`has_sm100=false`），只需保证 ① `cargo build --workspace` 不破 ② `ops_parity` 不受影响。

## 环境

- **出数：B300-50 GPU 5**（`CUDA_VISIBLE_DEVICES=5`，`CARGO_TARGET_DIR=/data/yaochen/infer/target_r12a`，`CUDA_ARCHS=103`）。
  GPU 0/1 常驻、2/3 主会话、4 r11d、6/7 留用。
- 3090 本机只做 `MEGA_SM100_DEV=1` 语法编译（无 sm100 硬件，跑不了）。
- **子代理禁止 `remote_*.sh push`**；单文件 `put` 到 `/data/yaochen/infer/dinfer_tc05/`；产物落 `bootstrap/r12a/`。
- 杀进程按端口 `pkill -f "dinfer serve .*--port N\b"`，不用全局 `pkill -x dinfer`。

## 参考

- FlashInfer 的 Blackwell GDN（`flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py`，CuTeDSL，4754 行）
  是现成的 tcgen05 用法范例：7 个 tcgen05 GEMM/chunk、state 在 TMEM、TMA + 128 B swizzle、num_warps=10。
  容器在 B300 上：`sudo -n nerdctl -n k8s.io run --rm --net host --gpus device=5 -v /data/yaochen/infer/bootstrap:/b <img>`（`--net host` 必带）。
- CUTLASS 的 `examples/70_blackwell_gemm` / `include/cutlass/arch/` 有描述符构造的权威写法。

## 轮次

| 轮 | 改动 | umma_peak TF/s | gqa T=1024/3009/6019（µs） | parity | 结论 |
|---|---|---|---|---|---|
| 0 | 基线 v6（cfg 3464） | 575（mma.sync） | 78 / 454 / 1558 | 过 | 起点 |
| 1 | **工具链自证**：`umma.cuh`（TMEM 分配 / 128 B swizzle 描述符 / UMMA 指令描述符 / `tcgen05.ld` / mbarrier）+ `umma_peak_sm100.cu`（M=128 N=256 K=64 循环，8 深度 mbarrier 流水）+ `tests/umma_peak.rs`。build.rs：`sm100_only` 加两项、arch 旗标 100/103 全局换 `a` 变体 | **2236**（148 block = 1 CTA/SM，两次一致 2236.0 / 2236.5；296 block 2243.7 / 2233.4；592 block 1809） | — | — | **门槛 900 过**（目标 2.25 PF 的 99.4%）。正确性两处都验：check 模式 k=192 cos=1.0 rel=4.0e-7；峰值模式 4096 轮同块累加 cos=0.99999999（证明 MMA 全执行，不是排空不彻底） |
| 2 | **attn 移植 WIP**：`gqa_fa_sm100.cu`（BQ=128 / NKI=64 / 每线程一行），QK+PV 走 UMMA、S 与 O 在 TMEM、softmax 在寄存器、P 写 smem 当 A；`cfg >= 5000` 分发，默认仍 3464 | — | **651 / 1691 / 5158**（v6：78 / 454 / 1558） | 过（cos 0.9999995，v5 对拍） | **慢 3.3×，未达目标**。插桩：qk_wait 2% / softmax 22% / rescale 5% / **pv+load 72%**（其中"等 K/V 装载"占该段 91%）；每 tile 每 SM 实测 13.5 µs vs 理想 0.56 µs（24×） |
| 3 | **合并 K/V 访存**（相邻线程读同一 key 的相邻片，而非相邻 key）：5158 → 3311（−36%）。插桩变化决定性：**等 K/V 从 91% → 3%**，`pv+load` 段的瓶颈换成 **QK 发射 67%** | — | **442 / 1118 / 3311** | 过（cos 0.9999995） | 装载延迟已解决；仍是 v6 的 2–5.7× 慢。每 tile 8.9 µs vs 理想 0.56（16×）。**主会话判定：3 轮内没进 3000 µs，1 CTA/SM 这条路走不通 → attn 停，转 GDN** |

## 踩坑（r12a 实测，接手先看这一节）

| 现象 | 根因 | 解法 |
|---|---|---|
| 输出全错 / `Invalid __shared__ write ... out of bounds` | `Layout::kv_one` 是**字节数**却当 `bf16*` 的步长用（`sP + slot * Layout::p_one`），偏移放大 2× 越界 | 所有布局偏移一律字节；指针换算显式 `/2`（`kKvSlotElems`）。`compute-sanitizer --tool memcheck` 直接指到行号 |
| 测试挂住不动（>60 s 无输出） | `__syncthreads_or` / `__syncthreads` 放进了 `if (tile > 0)` 之类的条件分支 → 部分线程少一次 barrier，与后面的 `__syncthreads()` 错配 | 全 block barrier 必须无条件执行；用 `const bool need = (cond) && __syncthreads_or(x)` 让 barrier 提到条件外 |
| 数字与基线**一模一样**（tcgen05 被静默跳过） | `gqa.cu` 里的 `extern "C" __attribute__((weak))` 桩 + 真实现的 strong 符号：归档里 `gqa_fa_sm100.o` 排在 `gqa.o` **之后**时，链接器取先看到的弱定义。函数指针常量同样不可靠 | 用 build.rs 传的 host 侧宏分叉（`-DMEGA_SM100` → `GQA_UMMA_RC(...)` 调真实现，否则返回 InvalidValue 回退）。不是 `#if __CUDA_ARCH__`，不在 kernel 体内，合规 |
| UMMA 读到旧的 P（数值轻微偏） | 普通 `st.shared` 写 P 之后没过渡到 async proxy | 写 P 之后、发 PV 之前必须 `fence.proxy.async.shared::cta` |
| 加载看起来"带宽不够" | 实际是**延迟受限**：单 CTA、装载链并发请求太少。K/V 全量只有 24.7 MB（T=6019/Hkv=4/D=256），全在 L2，不是 HBM 问题 | 先查每条装载链的 outstanding 请求数，再看要不要 TMA 多级流水 |

### 另一条：UMMA 的 K 跨度必须等于 swizzle atom
128 B swizzle 下一个 atom 在 K 轴上是 64 个 bf16，而 `make_umma_desc<Major::K>` 要求 `LBO = 0`（K 轴只有 1 个 atom）。
所以 D=256 的点积必须**按 64 列切成 4 块**、每块单独发一组 MMA，不能在 K 轴上跨块。同理 MN-major（PV 的 V）
用 LBO 跨 MN 块、SBO 跨 k-atom（8 行 × 128 B）。

---

## 收口（2026-10-21 06:2x，2 步 / 4 轮）

### 第一步（工具链自证）：**通过，超预期**
- `umma_peak` = **2236 TF/s**（标称 2.25 PF/s 的 99.4%；门槛 900；mma.sync 实测 575）
- 两道正确性：check 模式 cos=1.0；峰值模式 4096 轮同块累加 cos=0.99999999（排除"测的是排空"）
- **负结果**：592 block（4 CTA/SM）掉到 1809 —— TMEM 分配竞争。**1 CTA/SM 就够喂满 tensor core**，不要为 occupancy 加第二块

### 第二步（attn 移植）：跑通数值，**性能未达标，已停**
- 442 / 1118 / **3311** µs（T=1024/3009/6019）vs v6 的 78 / 454 / 1558 → 5.7× / 2.5× / **2.1× 慢**
- 数值全程 cos=0.9999995、maxabs 3.9e-3 / 7.8e-3（bf16 一 ulp）
- 最后一轮找到真根因：**K/V 装载访存不合并**（相邻线程读相隔 16 KB 的行 → 每 16 B 独占一条 128 B line，L2 放大 8×）。
  改成相邻线程读同一 key 的相邻 16 B 片后，"等 K/V" 从 91% → 3%，5158 → 3311（−36%）
- 剩余瓶颈：QK 的 16 条 K=16 全累加到同一 TMEM 累加器 → 串行 → 每 16 条 commit + 等；
  1 CTA/SM 只有一条发射流，填不满间隙（每 tile 8.9 µs vs 理想 0.56）

### 判断：**attn 的 tcgen05 路线停止，转 GDN**
- attn 只占 6 K prefill 的 12%（32.6 / 277 ms），做完美也只省 22 ms
- GDN 占 19%（52 ms），且 chunk 内 GEMM 是 K 方向长、连续的（state/h/v 都在 smem），
  不存在 attn 这种"短 K + 频繁 commit"的发射/依赖比问题 —— 正是 `umma_peak` 跑到 99.4% 的形状
- **留下的资产**：`umma.cuh` 全套原语（描述符 / TMEM / mbarrier / `tcgen05.ld`-`st`）直接被 r11d 复用，不重写

### 踩坑（已在上面「踩坑」节）

### 交接
- 默认 `GQA_PREFILL_CFG` 仍是 3464（v6），tcgen05 路径只有 `=5000` 才走，**对现有路径零影响**
- `crates/kernels/tests/gqa_umma_prof.rs` 是 clock64 分段插桩，留给 GDN 线复用同一套手法
- commits：`905daa3`（第一步）、`c6eec3d` / `3a09385` / `b745adb`（第二步），本地 dev，未 push
