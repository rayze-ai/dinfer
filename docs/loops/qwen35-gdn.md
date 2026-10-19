# Loop: GDN 算子（Qwen3.5 线）

> 状态：**里程碑 2 通过**（2026-10-19，3090 sm86 / 4090 sm89 / B300 sm103，10/10 case）。方法论见 `docs/model-onboarding.md` §3。

## 环境

- 参考源码：`/data/ref/vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`（层与 fused decode）；
  `vllm/third_party/flash_linear_attention/ops/{fused_recurrent,chunk,chunk_delta_h,chunk_o,l2norm}.py`（Triton 口径）；
  `vllm/model_executor/layers/mamba/ops/layernorm_gated.py` / `fused_norm_gate.py`（输出门）。
- 对拍参考：**纯 torch**，`scripts/verify/gen_gdn_refs.py`，不依赖 vllm / fla（本机 venv 没有，B300 容器才有）。
  与 `gen_kda_refs.py` 同构：固定 seed，生成输入 + 期望 `.npy`，bf16 按原始位存（`<u2`）。
- 开发卡：本机 3090（sm86）先过；4090（sm89）、B300（sm103，GPU 6/7，`CUDA_ARCHS=103`）后过。
  本机没有 27B 权重，对拍与结构无关，全部用形状参数（K=V=128, H=16, HV=48 是 Qwen3.5 的真值，另加一个小形状回归）。

## GDN 数学规格（已核对，实现与参考的唯一口径）

逐层、逐 v 头（Qwen3.5：K=128, V=128, H=16 k 头, HV=48 v 头, group = HV/H = 3, conv width 4, silu）：

1. **投影**：`in_proj_qkv` → `[T, 2·H·K + HV·V]`，按 `[q, k, v]` 平铺（non-interleaved，Qwen3.5 口径）；
   `in_proj_z` → `[T, HV·V]`；`in_proj_a` / `in_proj_b` → `[T, HV]`（每 v 头一个标量）。
2. **短卷积**（depthwise causal conv1d，weight `[dim, 4]`，bias=False，**silu**）：
   `y[t,d] = silu(Σ_{j=0..3} w[d,j] · x[t-3+j,d])`；`t<0` 取 conv state 末尾 3 列（列 0 最旧）。
   conv state 布局 `[slots, dim, 3]` bf16，左移后新 token 追加到最后一列。
3. **gate / beta**（逐 v 头标量，注意与 KDA 的区别：`dt_bias` 是 `[HV]`，不是 `[HV,K]`）：
   `x = a[t,hv] + dt_bias[hv]`；`g = -exp(A_log[hv]) · softplus(x)`（softplus 阈值 20：`x<=20 ? log1p(exp x) : x`）；
   `beta = sigmoid(b[t,hv])`。`A_log` / `dt_bias` 均 f32。
4. **q/k l2norm**（kernel 内）：`q /= sqrt(Σq² + 1e-6)`、`k` 同；然后 `q *= scale`，`scale = K^-0.5`。
5. **递归核心**（逐 token，S 为 f32 `[V, K]`，v 头 hv 用 k 头 `h = hv / group`）：
   `S *= exp(g)`；`v' = (v - S·k) · beta`；`S += v' ⊗ k`；`o = S·q`。
6. **输出门 + RMSNorm**（`RMSNormGated(norm_before_gate=True, activation=silu)`）：
   `y[t,hv,d] = rmsnorm(o[t,hv,:], w, eps) · silu(z[t,hv,d])`，`w` 是 `[V]`，per (token, v 头) 一行。
   `silu(z) = z·sigmoid(z)`；`sigmoid` 变体（K3 用）也保留一个参数位。
7. **chunk 形式 prefill**（性能项，v2）：`g` 先做 chunk 内累加和；`A = tril(diag(beta)·K·Kᵀ ⊙ exp(g_i-g_j), -1)`；
   `A ← (I+A)⁻¹`；`u = A·(beta⊙v)`、`w = A·(beta⊙k⊙exp(g_last-g))`；跨 chunk 扫 h；`o = q·h + tril(q·kᵀ)·v'`。
   **本里程碑先走"逐 token 顺序递归"**（与上式数值等价、慢），状态 `[slots, HV, V, K]` f32。

## 观察方法

1. `scripts/verify/gen_gdn_refs.py <out_dir>`：torch 生成输入与期望，case 覆盖
   `conv1d`（varlen 3 序列 + 有/无初始 state，prefill 后再 update 4 步）、
   `gate`（含极端 a）、`recurrent_prefill`（varlen 2 序列、group=3）、
   `recurrent_decode`（4 步状态累积）、`gated_rmsnorm`（silu 与 sigmoid 两个 act）。
2. `GDN_REFS_DIR=<dir> cargo test -p dinfer-kernels --features ref-tests --test gdn_parity -- --ignored --test-threads=1`：
   Rust 侧读 `.npy`、跑 kernel、按容差断言。容差：bf16 输出 atol 1e-2 / cos > 0.999；
   f32 状态 atol 2e-3 / cos > 0.999（递归是累乘，比单步输出敏感）；gate/beta f32 atol 1e-4（exp/log 近似）。
   口径见 `docs/pitfalls.md`"硬判决不可逐位复现"——只比软判决。
3. `scripts/verify/qwen35_gdn.sh` 串起上面两步并加一次 `cargo build --workspace --features qwen3` 回归，**exit 0 = 通过**。

## 停止条件（机器可判定）

- `scripts/verify/qwen35_gdn.sh` exit 0（上述全部 GDN 算子 case 在容差内）；
- 并且 `cargo build --workspace --features qwen3` exit 0（公共 trait 无回归）；
- 3090 跑通后再在 4090 上跑同一个脚本 exit 0；B300（`CUDA_ARCHS=103`）同。

## 环境备注（踩过的坑）

- 4090 的 GPU 3–5 常被他人进程占满（~23 GB / 24 GB）：先看 `nvidia-smi --query-gpu=index,memory.used` 再选卡。
- 4090 / B300 没有 torch：参考 `.npy` 已进仓库，`qwen35_gdn.sh` 默认复用（`FORCE_REF_GEN=1` 强制重生成）。
- B300 kernel 编译要 `CUDA_ARCHS=103`；4090 走远端 `env.sh`（`CUDA_ARCHS=89`）。

## 迭代上限

- 同一错误（同一 case 同一方向偏差）最多 3 轮；超过就停，把现象与已排除的假设写进下面这张表再继续。
- 总轮数上限 8 轮；到上限未过则收功并报告卡住的 case 与最大误差。

## 轮次记录

| # | 日期 | 改动 | 结果 |
|---|---|---|---|
| 1 | 2026-10-19 | 建规格：数学口径（含与 KDA 的三处差异：dt_bias 逐头、输出门 silu、group 映射）+ 参考脚本方案（纯 torch） | 规格定稿 |
| 2 | 2026-10-19 | 写 `gdn.cu` + Rust 封装 + `gen_gdn_refs.py` + `gdn_parity.rs` + `qwen35_gdn.sh`；首次跑 3090：conv/gate/norm 6 项过，4 项递归 case 报 `CUDA_ERROR_ILLEGAL_ADDRESS` | 用 compute-sanitizer 定位：递归 kernel 用动态 shared 但启动未传 smem 字节数（`<<<grid, block>>>` 缺第三参）→ 修 | 
| 3 | 2026-10-19 | 修 launch 参数 | **3090 10/10 过**：conv/norm 0 误差，gate 1.4e-6，递归 out 8e-6 / 状态 1e-7 |
| 4 | 2026-10-19 | 4090（sm89，GPU 7）：首次用 GPU 3 撞上别人的进程（23 GB 占满）→ OOM；换 7 | **10/10 过** |
| 5 | 2026-10-19 | B300（sm103，GPU 6） | **10/10 过**，里程碑 2 完成 |
