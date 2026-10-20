# Qwen3.5 r11b：decode 侧 1.2×（小 M GEMM 带宽 6.2 → ≥7 TB/s + split-K 归约融合 + 小算子）

> 状态：**进行中（2026-10-20 开工）**。起点 engine-gap 9f/9g：c32 TPOT 12.54（GEMM 8.2 ms/步 = 69%）、c8-long 10.88（GEMM 7.9 = 73%）。

## 环境

- **出数：B300-50 GPU 5**（`B300_TAG=50`，`CUDA_VISIBLE_DEVICES=5`，`CARGO_TARGET_DIR=/data/yaochen/infer/target_r11b`，构建 `taskset -c 80-95`、serve `taskset -c 56-63`、bench `taskset -c 112-127`，端口 **8051**）。GPU 0/1 常驻（vLLM `:8013` 在 GPU 1 作对手）、2/3 主会话、4 r11a、6/7 留，都不碰。
- 3090 本机只跑 parity（`cargo test -p dinfer-kernels --test ops_parity -- --ignored parity_gemm_small_m`）+ 编译。
- 同步：不 `push`；改动文件 `remote_b300.sh put` 单文件到远端同路径（开工 md5 已核对一致）。产物落远端 `bootstrap/r11b/`，本机 `pull` 回 `test/qwen35/2026-10-20/r11b/`；四组 bench 进 `test/qwen35/2026-10-20/bench_b300_N`（N ≥ 20）。
- 微基准：`CUDA_VISIBLE_DEVICES=5 GEMM_BENCH_M=8,32 GEMM_BENCH_SKIP_LM_HEAD=1 cargo test --release -p dinfer-kernels --test gemm_small_m -- --ignored --nocapture`（B300），每形状 lt_def / lt_best / small_m 三列 µs 与 GB/s。

## 文件归属

`crates/kernels/cuda/gemm_small_m.cu`（新增 mma kernel 同文件）、`crates/kernels/cuda/kernels.h`、`crates/kernels/src/{ffi,ops,gemm,arch}.rs`、`crates/kernels/tests/{gemm_small_m,ops_parity}.rs`、`models/qwen35/src/{gdn,attn,model}.rs` 调用链。**不碰 `crates/kernels/cuda/gqa.cu`**（r11a 在改）。

## 开工前的事实（第 0 轮，B300 GPU 5 微基准 + c8-long nsys sqlite 复核）

- 微基准 M=8 / 32：in_proj 16480 28.1 / 27.0 µs（6.0 / 6.3 TB/s）、gate_up 54.1 / 54.9（6.6 / 6.5）、down 32.6 / 33.0（5.5 / 5.4）、qkv_gate 24.8 / 24.0（5.9 / 6.1）、out_proj 9.6 / 9.8（6.6 / 6.4）。cuBLASLt 8 候选内最优与默认差 ≤ 7%：**候选集里没有更快的 tile，再搜 Lt 没用**。
- nsys c8-long 复核（grid 归类）：`128x8` grid 272 = gate_up 57.6 µs；`64x8` grid 544 **也是 gate_up**（34816/64，另一个 M 桶的步）55.1 µs；in_proj 是 `128x8 4x1` grid 132 **28.5 µs（5.9 TB/s）**。9g 行"in_proj 64x8 2.6 TB/s"是把 gate_up 的 64x8 误归到 in_proj，**实际所有 decode GEMM 都在 5.9–6.2 TB/s，没有单点坏形状**。
- 自写 FMA `small_m` 在 M=8 / 32 只有 1.7 / 0.46 TB/s（每 16 B 权重要 M×8 条 FMA + 转换，算力型），只对 N=48 这类小 N 有用。
- attn.qkv 8192×5120（84 MB）微基准 10 TB/s > HBM：**权重进了 L2（B300 L2 ≈ 126 MB）**，微基准 / 自调优对 ≤ L2 的形状偏乐观；真 decode 每层权重都是冷的。

## 改法（按序）

1. **tensor-core 小 M kernel `small_m_mma`**：每 warp 16 行 W（mma.m16n8k16 bf16，A 片段 = W 直接 16 B 向量读，K 维按 lane 置换，不走 smem / ldmatrix）、A（激活）按 TK 分块进 smem（行 pad 16 B 防 bank 冲突）；M ≤ 32 → 1–4 个 n8 tile，累加器 4×M/8 寄存器；split-K 用"最后到达的 block 归约"（`__threadfence` + 每 N-tile 计数器，自复位，graph 安全）**省掉独立 reduce kernel（106 次 × 3.5 µs = 0.37 ms/步）**。候选 (splits, warps) 由 `ArchProfile` 给、启动自调优选，kernel 不含数值。目标每形状 ≥ 7 TB/s。
2. 自调优计时改冷 L2（多层权重轮转），让表选择与真 decode 一致。
3. 小算子融合（rmsnorm → GEMM A 装载 prologue 等），先量再动。
4. 若仍差：步内 gap（host 段）。

## 停止条件

- `parity_gemm_small_m`（3090）新 case 全过；`gemm_autotuned_matches_cpu` 过。
- `bench_compare.sh`（B300 GPU 5 vs vLLM :8013）：**c32 TPOT p50 ≤ 11.31 且 c8-long ≤ 9.26**；四组不退化。
- 迭代上限 8 轮；卡住 2 轮停下回报。4090 门禁由主会话跑。

## 轮次

| 轮 | 改动 | 微基准 M=8/32（in_proj / gate_up / down / qkv_gate / out_proj TB/s） | bench（c1 / c8 / c32 / c8-long TPOT ms；tok/s ×vLLM） | 结论 |
|---|---|---|---|---|
| 0 | 基线 = bench_b300_19（`70736cc`） | lt：6.0/6.3 · 6.6/6.5 · 5.5/5.4 · 5.9/6.1 · 6.6/6.4；small_m 1.7/0.46 | 9.7 / 10.3 / 13.1 / 10.7；1.09 / 1.11 / 1.04 / 1.04 | GEMM 全部 ~6 TB/s，无单点；换 tensor-core 小 M kernel |
| 1 | **tensor-core `small_m_mma` 初版**（每 warp 16 行 W 直读 16 B、置换 K、A smem pad 16 B、`#pragma unroll 4`、split-K 最后 block 归约）`0047c09`；parity 10/10（3090） | mma M=8/32：4.4/3.0 · 5.4/3.7 · 4.1/2.8 · 3.9/2.6 · 4.3/2.2（lt 不变 ~6.2）；全部形状仍选 lt | 未 bench | 比 FMA 版快 2.5–7×，但远低于 lt：M=32 时 smem B 片段读是瓶颈（pad 16 B 下 128-bit 装载 2 路 bank 冲突）+ 每 warp 在途只 2 × 16 B，unroll 被 `__syncthreads` 分块排空。第 2 轮：W 寄存器预取环 4 级跨 TK 连续 + pad 64 B |
| 2 | W 寄存器预取环 4 级（跨 TK 连续）+ A smem pad 64 B（未 commit，被第 3 轮替代） | mma M=8/32：3.9/2.7 · 4.2/2.9 · 3.6/2.5 · 3.6/2.4 · 4.4/2.5 | — | M=8 反而略降、M=32 不变：**瓶颈不是 W 在途而是 smem 读**——W 当 mma "A"（16 行）时每 16 B W 要配 M/8 × 16 B 的 smem B 片段读，smem 读量 = W 读量 × M/8，M=32 时 4×。第 3 轮换角色：激活当 "A"（寄存器，每 chunk 读一次），W 当 "B" 直读 global，每 warp R 条 8 行 stripe 复用同一 A 片段 → smem 读量 = W 的 1/R |
