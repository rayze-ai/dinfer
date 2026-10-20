# kernel 规则与已验证的结构选择

## 硬规则（CLAUDE.md，用户 2026-10-19）
- kernel 只收运行时参数；splits / BV / tile / warp 数等由 `ArchProfile`（`crates/kernels/src/arch.rs`，按 cc + SM 数）与启动自调优给，环境变量只做覆盖。
- 多卡型实现按 arch **分文件**（`*_sm8x.cu` / `*_sm100.cu`），kernel 体内禁 `#if __CUDA_ARCH__`；有卡型差异才分，没差异一份。
- 预分配缓冲上的算子**显式传 token 数**（`forward_rows(…, t)` / `tokens` 参数），禁止从缓冲长度推（两次踩坑：GEMM M=2112、GDN decode 分发失效）。
- 新加 C ABI 符号：声明（`kernels.h` / `ffi.rs` 含 no-nvcc stub）与定义**同一个 commit**（并行子代理的 commit 曾带走别人的声明，HEAD 链接失败）。

## 已验证的结构（B300 sm_103 实测，3090 多为带宽下限看不出差别）
| 场景 | 选择 | 数字 | 证据 |
|---|---|---|---|
| 递归状态 decode（GDN/KDA，状态 [V,K] f32 每 (seq, 头)） | 寄存器驻留 warp kernel：grid (B, HV, V/BV)，K 维 warp shuffle；多值蝶式归约（`tr`）+ 一遍 FMA；BV 由 SM 数定（Sm100 ≤ 32） | B300 B=32 431 → 32 µs/层（下限 25） | loop 4 / 7a / 7c |
| cp.async 流水预取状态 | **不赢**（占用 2 → 1 block/SM 抵消） | pipe 35.4 vs tr 34.1 | loop 7c |
| chunk prefill（delta rule） | prep / wy / ho 三段，ho 按 (seq, head, V 块) 并行 + mma.sync bf16；conv1d 按 T 分块（含 bos 的块独占 conv_state） | ho 1346 → 204 µs、conv 874 → 76、wy 652 → 221 | loop 5A / 5E |
| 小 M GEMM（M ≤ 8） | cuBLASLt 前 N 候选逐个计时 + 自写 split-K 取优，表键 (M 桶, N, K, bias)，非桶 M 借幂桶（`AlgoCheck`）；默认 algo 按键缓存（否则每次 forward 查启发式） | M=1 大 N：lt 默认 81 → 16 µs（splitK 5） | loop 3 / 5B |
| M ≥ 16 小 M kernel | fp32 FMA 成算力型，不如 lt；留作候选由自调优定 | m32 N=6144 355 vs lt 97 | loop 9 疑点 2 |
| 同输入多投影 | 权重按行拼成一条 GEMM，下游按行距读不拷贝 | 每步少 160 launch，c32 TPOT 13.2 → 12.5 | loop 9 |
| rmsnorm h ∈ (1024, 8192] | 1024 线程 × 1 向量（单次访存等待） | −4–8% | loop 7b / 7c |
| decode GEMM 效率上限 | HBM 72–76%（`gate_up` 58 µs = 356 MB），与 vLLM 同档，不再是差距来源 | — | loop 5G |

## 数值口径
- 硬判决（top-k / 路由 / argmax）只在可判定行断言（ref top-2 gap > 阈值）；bf16 输出链 rel ~5e-3 = 一 ulp，用 cos ≥ 0.999x 而不是 abs。
- 参考侧 rmsnorm 的 bf16 落点要与 kernel 一致（乘权重**后**落 bf16）。
- 运行时下标的寄存器数组会落 local memory（stack frame ≠ spill）：模板递归展开。

## megakernel 第一阶段实测（`docs/loops/mega-sm100.md`，2026-10-20）
- **第 1 轮：persistent kernel 慢于 11 kernel + graph 基线**（B300 单 GDN 层 decode：146/158/190 µs → 188/215/339 µs，0.78/0.74/0.56×）。基线口径公平（现路径也走 graph + 自调优 GEMM）。
- 阶段 prof 显示的真因：① GEMV 阶段只到 4.5 TB/s（cuBLASLt 自调优的 splitK 算法 6+ TB/s）——persistent CTA 内手写 GEMV 的访存并行度不如专用 kernel；② GDN 递归阶段 B=32 80 µs（独立 kernel 32 µs）——任务粒度 / 占用被统一 grid 绑住；③ 依赖等待与小阶段（conv 16、add_norm 10 µs）叠加。
- **结论先记为假设**：消掉 kernel 边界省的 ~2–3 µs/边界 × 11，抵不过每个阶段失去"各自最优 launch 配置"的损失；megakernel 要赢必须每阶段都达到独立 kernel 的效率（OCC / 每阶段任务粒度分开调），第 2 轮验证。vLLM 做不到这条路不等于这条路一定赢。
