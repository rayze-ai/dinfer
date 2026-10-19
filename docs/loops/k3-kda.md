# Loop: KDA 算子（Kimi-K3 线）

> 状态：**进行中**。里程碑 1–3a 完成（五算子 8/8 对拍）；**里程碑 A（chunk 形式 prefill）完成**：
> 两卡对拍通过、3090 7.2×/4090 5.15×（门槛 5×）。下一步 B（MXFP4 grouped GEMM）。

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
| 2 | 2026-10-19 | 实现 `crates/kernels/cuda/kda.cu` 五个算子（conv1d prefill / conv1d update / gate / 顺序递归 / gated RMSNorm）+ Rust 封装 + `gen_kda_refs.py` + `kda_parity.rs` | **8/8 case 通过**（3090 sm86） |
| 3 | 2026-10-19 | 修两处：`concurrent block_reduce_sum` 之间缺 `__syncthreads`（q/k 归一化的两个归约共用静态 shared）；参考侧 `has_state=0` 槽未清零 | 全过，误差见下表 |

### 实测误差（3090 sm86，`scripts/verify/k3_kda.sh` exit 0）

| case | 最大误差 | cos |
|---|---|---|
| conv_prefill / varlen / update ×4（out+state） | 0 | 1.0 |
| gate（lower_bound / softplus） | 1e-6 / 2.3e-5 | 1.0 |
| recurrent_prefill out / state | 3e-5 / 1e-7 | 1.0 |
| recurrent_varlen out / state | 1e-5 / 1e-7 | 1.0 |
| recurrent_decode ×4 步 out / state | 0 / 1e-7 | 1.0 |
| gated_rmsnorm | 0 | 1.0 |

### 已知限制 / v2 待办

1. **顺序递归，不是 chunk 并行**：`recurrent_kernel` 每 block 一个 (序列, 头)，块内时间维串行，
   `head_dim²=16384` 次乘加/头/token。对拍精确，性能远低于 fla 的 chunk 形式（BT=64 矩阵化）。
   prefill 的真实现要做块级：`A=(I+tril(diag(β)KKᵀ,-1))⁻¹`、`w/u` 重算、跨 chunk 扫 h、`o=q·h + tril(qKᵀ)v'`。
2. **`head_dim > 76` 时 shared 超 64 KB**：递归 kernel 用 `cudaFuncSetAttribute` 提到 67 KB（sm86 上限 99 KB），
   到 `head_dim=256` 会顶到 262 KB 上限，需改成分块（每 block 一个 V 分片）。
3. gate 的 fp32 中间量走独立 kernel（`kda_gate`）；K3 的 packed decode 把 gate + conv + 递归融成一个 kernel，
   等模型层接好后按实际瓶颈再定是否合并。
4. sm_103（B300）尚未跑过对拍，等 kernel 进 B300 构建后补一轮。

## v2 规格：chunk 形式 prefill（里程碑 A，2026-10-19）

> 状态：**已实现，两卡通过**（3090 7.2–7.4×、4090 GPU 7 5.15×）。结果见文末「轮次记录」与「加速比」。

### 环境

- 参考：fla `chunk_gated_delta_rule` 的分解（`/data/ref/vllm/vllm/third_party/flash_linear_attention/ops/`）：
  `kda.py::_chunk_kda_fwd_with_cumulative_g:1238` 串起 `chunk_kda_scaled_dot_kkt_fwd`（A/Aqk）→
  `solve_tril`（M）→ `recompute_w_u_fwd`（u/w/kg）→ `chunk_gated_delta_rule_fwd_h`（跨 chunk 扫 h）→
  `chunk_gla_fwd_kernel_o`（输出）。子块消融参考 `kda.py:558` 的 `intra_sub_inter`（16 行一块算衰减）。
- 代数自检探针：`scripts/verify/kda_chunk_probe.py`（纯 torch，`/data/ref/venv-tf519/bin/python -I` 直接跑）。
  它把 chunk 形式与**逐 token 顺序递归**对比，并给出 bf16 消融表；规格是否写错先在这里看，不进 CUDA 调试。
- 开发卡：本机 3090（sm86）；4090 用 **GPU 7**（`scripts/dev/remote4090.sh`，`CUDA_VISIBLE_DEVICES=7`）。
  sm_103 留 B300，本阶段不跑。

### 代数（已与顺序递归数值核对，作为唯一口径）

`gc = cumsum(g)` 是**块内 inclusive**（含本行）；`g` 是自然对数空间的 gate（`kda_gate` 的输出），
下游一律 `exp2(gc · 1/ln2)`（fla 同样把 cumsum 乘 `RCP_LN2`）。块 `c` 覆盖 `[c·BT, c·BT+BT)`：

```
A[i,j]  = β_i · (k_i·k_j) · exp(gc_i − gc_j)     j < i          （严格下三角）
Aq[i,j] = (q_i·k_j) · exp(gc_i − gc_j)           j ≤ i          （含对角线）
M       = (I + A)⁻¹                                             （单位下三角，前代即可）
u       = M · (β ⊙ v)
w       = M · (β ⊙ k ⊙ exp(gc))
v'      = u − w·h_prev                                          （h 是块入口状态 [V,K]）
o       = (q ⊙ exp(gc))·h_prev + Aq·v'
kg      = k · exp(gc_last − gc)
h_next  = h_prev · exp(gc_last) + v'ᵀ·kg                        （按 K 维衰减）
```

- `exp(gc_i − gc_j)` 用 `exp2(gc·RCP_LN2)` 之差算；实际实现按 fla 的子块技巧：每 16 行以该子块首行为参考
  `exp2(gc_i − gc_ref)` 与 `exp2(gc_ref − gc_j)` 相乘，避免逐元素做 64×64×128 次指数。
- 对角线：A **不含**对角线（`tril(...,-1)`），Aq **含**对角线。这正是顺序递归里
  `v' = (v − S·k)·β; S += v'⊗k; o = S·q` 的展开结果，`exclusive` cumsum 会差 6e-2（已实测，错的）。
- `h` 的入口：`h_prev = 0` 或状态的槽值；末态 = 最后一个块的 `h_next`。

### 分解与实现约束

四个 kernel（各自一个 C ABI 入口，与 fla 的分段一致；这样 shared 用量留在 sm86 的 99 KB 内）：

| kernel | 作用 | 并行度 |
|---|---|---|
| `kda_chunk_kkt` | 算 A 与 Aqk（按 16 行子块） | (chunk, 头) |
| `kda_chunk_wy` | M = 前代求逆、u / w / kg | (chunk, 头) |
| `kda_chunk_h` | 跨 chunk 扫 h（h 留 fp32 SRAM），同时产出 v' | (V 分片, 头, 序列) |
| `kda_chunk_o` | o = qg·h + Aq·v' | (chunk, 头) |

- **BT 是模板参数**，host 选（默认 64）；kernel 体内不写死其它调优数值。
- 中间张量（A / Aqk / M / u / w / kg / v' / h）由 host 侧 scratch 分配，布局写进 `ops.rs` 的封装注释。
- mma 操作数 bf16（`mma.sync.m16n8k16`，复用 `gqa.cu` 的 `ldsm_x4` / `mma16816`），
  **`v'` 与 `h` 的累加保持 fp32**（消融表 "+ A" 行的做法）。
- varlen：`cu_seqlens` + host 侧算好的 chunk 索引表（chunk → (序列, 块号)），kernel 内不做除法和分支。

### 观察方法

1. `scripts/verify/kda_chunk_probe.py`：规格自检（fp32 应 ~1e-7；bf16 应看到下表量级）。
2. `gen_kda_refs.py` 新增 `recurrent_chunk` case：varlen 两段 + 非零初态 + T 不是 BT 整数倍，
   期望值由**顺序递归**参考产出（不是 chunk 参考——chunk 是被测实现，参考必须独立）。
3. `kda_parity.rs` 新 case：与上面参考比，另加 T=1024 / 96 头的计时 case。
4. `scripts/verify/k3_kda.sh` 汇总，exit 0 = 通过。

### 停止条件（机器可判定）

- `k3_kda.sh` exit 0，且新 chunk case 满足：
  - out：atol 1e-2 + **cos ≥ 0.9999**；
  - state：**相对 max_err ≤ 5e-3**（相对 `|state|max`）+ cos ≥ 0.9999
    （口径 = `docs/pitfalls.md` 的软判决；coordinator 2026-10-19 确认）。
- **加速比**：3090 上 T=1024 / H=96 / K=128，chunk 全链 ≤ 顺序版的 1/5。
  基线已实测：顺序版 **76.8 ms** → 门槛 **≤ 15.4 ms**（`kda_parity.rs` 的计时 case 打印）。
- `cargo build --workspace --features qwen3` 仍 exit 0。

### 实测精度预算（探针输出，作为 kernel 的预期量级）

T=256 / H=2 / K=128 / BT=64，`|state|max 0.53`：

| 走 bf16 的环节 | out max_err | state max_err | state 相对 | cos |
|---|---|---|---|---|
| 全 fp32 | 2.1e-7 | 6.2e-6 | 1.2e-5 | 1.0 |
| A（KKᵀ/Aqk） | 2.0e-4 | 5.3e-5 | 1.0e-4 | 0.99999+ |
| + u/w | 2.1e-4 | 9.7e-4 | 3.5e-3 | 0.99999+ |
| + o / h（全 bf16） | 2.1e-4 | 1.3e-3 | 3.5e-3 | 0.99999+ |

→ bf16 mma 全开也落在容差内（state 相对 3.5e-3 < 5e-3），与 vLLM fla 自己把 `u`/`w`/`v_new`
存成 `k.dtype` 一致。留 fp32 累加 `v'`/`h` 是为了先拿安全档，速度有余量再放宽。

### 轮次记录

| # | 日期 | 改动 | 结果 |
|---|---|---|---|
| 1 | 2026-10-19 | 规格 + 探针：定 inclusive cumsum 口径、四 kernel 分解、bf16 消融、76.8 ms 基线 | 规格定稿 |
| 2 | 2026-10-19 | 实装 `kda_chunk.cu` 三段（prep / wy / ho），全 fp32；`gen_kda_refs.py` 加 varlen + 短序列 case；`kda_parity.rs` 加对拍 + 计时 | 对拍首次通过；加速 2.38× |
| 3 | 2026-10-19 | `ho` 共享布局补 K 维 1 列（消除列访问 bank 冲突）+ qg/w/kg 复用同一缓冲 | 31.7 → 7.0 ms，整链 2.38× → 6.7× |
| 4 | 2026-10-19 | `wy` 按 16 行子块预计算衰减因子（对齐 fla `intra_sub_inter`），内层只剩乘加 | 去掉逐 (i,j,k) 的 exp2（A/Aq 精度不变） |
| 5 | 2026-10-19 | `wy` 的 u/w 按 4 个连续 j 分块（`sM[i][t]` 一读喂 4 次 FMA）；`ho` 的 K 维模板化 + `#pragma unroll`；`prep` 拆成 q / k 两个 block 并行 | **3090 7.2–7.4×（10.3 ms）**、**4090 GPU 7 5.15×（5.66 ms）** |

### 实测精度（2026-10-19，`k3_kda.sh` exit 0，3090 与 4090 同结果）

| case | out max_err / cos | state max_err / 相对 / cos |
|---|---|---|
| `recurrent_chunk`（varlen 130/100，末块不满，非零初态） | 2.4e-4 / 0.9999947 | 1.39e-3 / 1.6e-3 / 0.9999986 |
| `recurrent_chunk_tail`（varlen 5/3，远小于 BT） | 1.2e-4 / 0.9999937 | 1.88e-3 / 2.5e-3 / 0.9999988 |

均在 stopped 条件内（out atol 1e-2 + cos ≥ 0.9999；state 相对 ≤ 5e-3 + cos ≥ 0.9999）。
此前 8 个顺序算子 case 全部保持原误差（无回归）。

### 加速比（T=1024 / 96 头 / head_dim 128）

| 卡 | 顺序递归 | chunk 全链 | 加速 |
|---|---|---|---|
| 3090（sm86） | 74–76 ms | **10.3 ms** | **7.2–7.4×** |
| 4090（sm89，GPU 7） | 29.2 ms | **5.66 ms** | **5.15×** |
| **B300（sm_103）** | **36.33 ms** | **4.87–4.90 ms** | **7.41–7.46×** |

4090 的顺序递归更快（时钟 / bf16 单元更强），chunk 一侧被共同的 K 维标量累加瓶颈卡住，
所以加速比反而比 3090 小——5× 门槛在两张卡上都过了，但 4090 余量只有 3%。

**B300（2026-10-19，B300-50 GPU 0，sm_103，CUDA 13.1，release 二进制）**：chunk 全链 4.87 ms，
顺序递归 36.33 ms，7.41–7.46×。顺序递归比 4090 慢 24%（它是逐 (序列,头) 块的标量 FMA + shared 复用，
完全不吃 sm_103 多出来的算力），chunk 全链比 4090 快 14%——sm_103 上的绝对瓶颈仍是「K 维标量累加」，
`mma.sync` 提速（下面第 1 条）在 Blackwell 上收益预计比 4090 更大。sm_103 上 11 个 test 全过（exit 0），
误差与 sm86/sm89 同量级（chunk out max_err 2.4e-4 / cos 0.9999951，state 相对 2.68e-3 / cos 0.9999985）。
日志 `test/kimik3/2026-10-19/b300_50/`。

### 实现要点（下一会话接续时先看这几条）

1. **共享内存**：sm86 上限 99 KB。`wy` 用 `(3·BT·BK + 2·BT·BT + BC·BK)·4`（BK=64 时 97 KB），
   `fj`（`[BT][BK]`）借 `sA`（`[BT][BT]`，同尺寸）——A 只在 A/Aq 循环结束后才写，不冲突。
2. `ho` 的 `BV=16`、256 线程；`BV=32` 与 128 线程都实测更慢（sm86 上 8.4 / 9.5 ms vs 7.0）。
3. `cudaFuncSetAttribute` 的 smem 上限**必须按 kernel 地址缓存**（`set_smem` 里的 `smem_cache`）：
   早先只按字节数缓存，换特化后属性没设上，直接 `cudaErrorInvalidValue`。
4. `wy` 的 A/Aq 寄存器累加用**全局 n 序号**（`n ∈ [i0/RPN, (i0+BC)/RPN)`），
   不要用子块内序号——那会让不同子块共用同一组寄存器（会写错且越界）。
5. `kda_chunk_bench` 用真实 gate 量级（`-5·sigmoid(a·x)`，a ∈ [1,15.9]），
   三个 kernel 的迭代次数都已固定，结果在两卡上稳定（3090 复测 3 次 10.30 ms ±0.02）。

### 后续提速（未做，按性价比排序）

1. **K 维标量累加改 bf16 mma.sync**：`wy` 的 A/Aq/u/w 与 `ho` 的 qg·h / w·h / v'ᵀ·kg
   都是 `[BT,K]×[K,·]`，现在全走标量 FMA。换成 `mma.sync.m16n8k16`（复用 `gqa.cu` 的
   `ldsm_x4` / `mma16816`）能把这部分再压 2–3 倍——这是 4090 余量的主要来源。
   精度预算已验（见上表 + 探针的 bf16 消融）：只把 mma 操作数转 bf16、`v'`/`h` 保持 fp32，
   state 相对误差 ~1.6e-3，仍有 3× 余量。
2. `ho` 的 h 更新（`v'ᵀ·kg`）目前每个 (vv,kk) 线程串行扫 BT，可拆成按 i 分块 + 归约。
3. v3 若要支持 BT≠64，只需在 host 侧加特化（kernel 已是模板）。

## 下一步

- MXFP4 专家 dequant 参考（`docs/loops/k3-moe.md`）：`weight_packed` U8 + `weight_scale` e8m0 / group 32 → bf16 回退。

## 阻塞记录

- ~~2026-10-19 收尾时 B300 8 卡全部被别的作业占满~~ → **当晚 21:00 卡已空，sm_103 对拍与计时已完成**（见「加速比」表）。
- 若以后卡又被占（每卡约 248 GB 的 llm-d PD 推理），先找主会话确认归属，**不要杀掉别人的 Pod**。
- 本机 3090 可继续跑全部 CPU/GPU 单测（已验证），4090 GPU 3–7 也可用。
