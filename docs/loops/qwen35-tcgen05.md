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
