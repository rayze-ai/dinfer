# Loop: KDA 算子（Kimi-K3 线）

> 状态：**进行中**。里程碑 2（KDA 对拍通过）是本次交付目标。

## 环境

- 参考源码：`/data/ref/vllm/vllm/third_party/flash_linear_attention/ops/kda.py`（Triton，1576 行）+ `fused_recurrent.py`；
  K3 私有解码 `ops/third_party/kda/fused_recurrent.py`（packed decode，含 gate/beta 融合）。
- 数学参考（不依赖 fla 安装）：`/data/ref/models/kimik3/modeling_kimi_linear.py:479` `KimiDeltaAttention`。
- 对拍环境：**纯 torch CPU/GPU 参考**（`scripts/verify/gen_kda_refs.py`），不需要 fla / vllm 包——
  原因：本机 `/data/ref/venv-tf519` 没有 vllm 也没有 fla，B300 容器有 vllm 但 `vllm.third_party` 的 fla 只随 vLLM 一起可用，
  为了对拍链稳定，参考直接按上面的数学规格用 torch 重写（与 fla 同式），并用 `test/kimik3/<日期>/` 归档一次与 vLLM 数值的三方核对。
- 开发卡：本机 3090（sm86）、4090（sm89）先过；B300（sm_103）后过。B300 只用 GPU 0–5。

## KDA 数学规格（已核对，作为实现与参考的唯一口径）

逐层、逐头（K3：96 头 × head_dim 128，`gate_lower_bound=-5.0`，`use_full_rank_gate`）：

1. `in_proj_qkvgfab` 切出 q/k/v（各 7168→12288）、g（7168→12288）、f_a（7168→128）、b（7168→96）。
2. **短卷积**：q/k/v 各过 depthwise causal conv1d（weight `[dim, 4]`，siLU）：
   `y[t] = silu(Σ_{j=0..3} w[d,j] · x[t-3+j])`，t<0 用 conv_state 的末尾 `W-1=3` 列。
   conv_state 布局 `[slots, dim, 3]`，每步 shift-left 后把新 token 追加到第 2 列。
3. **gate**：`g1 = f_b_proj(f_a)` → `[T, 96, 128]`；`a = exp(A_log[h])`；`x = g1 + dt_bias`；
   `gate = lower_bound * sigmoid(a * x)`（有 lower_bound 时）否则 `-a * softplus(x)`。
   注意：`A_log` 是 f32（`[96]`），`dt_bias` 是 f32（`[12288]`）。
4. **beta**：`beta = sigmoid(b_proj(x))` → `[T, 96]`（**逐头一个标量**，不是 headwise）。
5. **qk l2norm**（kernel 内）：`q /= sqrt(Σq² + 1e-6)`，`k` 同；然后 `q *= head_dim^-0.5`。
6. **递归核心**（decode，逐 token）：
   `S[v,k] *= exp(gate[k]); v' = (v[v] - Σ_k S[v,k]·k[k]) · beta; S[v,k] += v'[v]·k[k]; o[v] = Σ_k S[v,k]·q[k]`
   S 是 fp32 `[96, 128(V), 128(K)]`，按请求一块（每层 ≈ 6.5 MB）。
7. **prefill（chunk 形式，BT=64）**：`g` 先做 chunk 内累加和 `× 1/ln2`（下游用 exp2），
   然后 `A = tril(diag(beta)·K·Kᵀ, -1)` → `A ← (I+A)⁻¹` → `u = A·(beta⊙v)`、`w = A·(beta⊙k⊙exp2(g))`、
   `kg = k·exp2(g_last - g)` → 跨 chunk 扫 h → `o = q·h + tril(q·kᵀ)·v'`。
   **本次先用"逐 chunk 顺序递归"参考实现**（与上式数值等价、慢），块级并行留 v2。
8. **输出门 + 归一化**：`g2 = g_proj(x)` → `[T,96,128]`；`o = RMSNorm(o, head_dim) * sigmoid(g2)`；再 `o_proj` 回 7168。

## 观察方法

1. `scripts/verify/gen_kda_refs.py <out_dir>`：torch 生成全部算子的输入与期望 `.npy`（固定 seed），
   覆盖：单序列 prefill、varlen 两条序列、decode 单步、decode 连续 4 步（状态累积）、
   极端值（gate 极大/极小、beta 接近 0/1）、head_dim 非 2 幂（128 之外的 64 用于回归）。
2. `cargo test -p dinfer-kernels --features ref-tests --test kda_parity`：
   Rust 侧读 `.npy`、跑 kernel、按容差断言。
   容差：bf16 输出 atol 1e-2 / rtol 1e-2；fp32 状态 atol 1e-5；递归状态比输出更严（atol 1e-4）。
   口径见 `docs/pitfalls.md` "硬判决不可逐位复现"——软判决不比逐位。
3. `scripts/verify/k3_kda.sh` 汇总上面两步，**exit 0 = 通过**。

## 停止条件（机器可判定）

- `scripts/verify/k3_kda.sh` exit 0（新增的 KDA 算子全部 case 在容差内）；
- 并且 `cargo build --workspace --features qwen3` 仍 exit 0（公共 trait 无回归）。

## 迭代上限

- 同一错误（同一 case 同一方向偏差）最多 3 轮；超过就停，把现象/已排除的假设写进本节再继续。
- 总轮数上限 8 轮；到上限未过则收功，报告卡在哪一个 case 与最大误差。

## 轮次记录

| # | 日期 | 改动 | 结果 |
|---|---|---|---|
| 1 | 2026-10-19 | 建规格：KDA 数学口径 + 参考脚本方案（纯 torch，不依赖 fla） | 规格定稿 |
