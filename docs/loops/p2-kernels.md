# loop: P2 基础算子对拍

## 目标

`kernels` 里每个参考算子（bf16，sm80+）与 PyTorch 逐元素对拍通过；`layers` 的 Linear / RmsNorm / Rope / Embedding / Mlp / Sampler 能在 3090 上跑。

## 环境

- 本机 3090（sm86）。生成期望值用 PyTorch（CPU 即可，用 fp32 算再 cast bf16）。
- 期望值存 `.npy`（numpy 原生格式，Rust 侧用 `npyz` 读）到 `test/<date>/kernels/`。

## 算子清单（P2 范围）

| 算子 | 签名要点 | 对拍容差 |
|---|---|---|
| `rmsnorm` | `[T,H] bf16 × w[H] → [T,H]`，eps；可选 residual add 融合（`fused_add_rmsnorm`：out = norm(x+res)，res 原位更新） | atol 1e-2, rtol 1e-2 |
| `rope` | interleave（非 neox）与 neox 两种；cos/sin 表 fp32；就地作用于 q/k 的 rope 维 | 同上 |
| `silu_and_mul` | `[T,2I] → [T,I]`，out = silu(x[:, :I]) * x[:, I:] | 同上 |
| `embedding` | `ids[T] i32 × table[V,H] → [T,H]` gather | 精确 |
| `gemm_bf16` | cuBLASLt，`C[M,N] = A[M,K] · B[N,K]^T`（权重按 PyTorch `[out,in]` 存） | atol 2e-2（累加顺序差） |
| `argmax` | `[B,V] fp32 → [B] i32` | 精确 |
| `softmax_sample` | temperature + top-k + top-p + Philox 随机；greedy 时退化 argmax | 分布级：10k 次采样的频率与 torch.multinomial 偏差 < 2% |
| `topk_topp_mask` | 对 logits 做 top-k / top-p 置 -inf（参考实现，排序法） | 精确（mask 结果） |

## 观察方法

`scripts/verify/kernels.sh`：
1. `python3 scripts/verify/gen_kernel_refs.py` 生成输入与期望。
2. `cargo test -p dinfer-kernels --features ref-tests -- --ignored` 读 `.npy` 跑 GPU 对拍。

## 停止条件

`scripts/verify/p2.sh` exit 0（= p1 + kernels.sh）。

## 迭代上限

每个算子 3 轮；超过则记进 pitfalls 并标注待办。

## 记录

| 日期 | 算子 | 结果 |
|---|---|---|
| 2026-10-14 | rmsnorm / fused_add_rmsnorm / rope×2 / silu_and_mul / embedding / gemm / argmax | 首轮即过（修了测试读文件的 cwd 问题后） |
| 2026-10-14 | sample（分布级） | 20k 次采样 L1 略超 0.03；采样数提到 100k 后过。kernel 本身正确（小 vocab 逐 token 频率与期望差 < 0.003） |
| 2026-10-14 | topk_topp_mask | 第 1 轮：`k=5,p=0.5` 行保留 5 vs 2 —— top-p 在原始分布而非 top-k 后分布上算。第 2 轮：改为 vLLM 口径（top-p 只统计 ≥ top-k 阈值的元素）后过 |

**loop 结束**：`scripts/verify/p2.sh` exit 0（2026-10-14）。
