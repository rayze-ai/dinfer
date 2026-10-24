# loop 规格：kaon3-layer1

- 状态：进行中（2026-10-24 创建）
- 创建：2026-10-24
- 验证脚本：`scripts/verify/kaon3-layer1.sh`
- 迭代记录：`test/kaon3/2026-10-24/kaon3-layer1.md`
- 上游：`docs/loops/kaon3-fp4dense.md`（MoE 段已排除；已有干净起点见下）

## 目标

用**截层对拍**定位 compile 口径下差异从**第几层**开始，并先只做 **1 层**。
1 层通过与否，决定往上一层还是往层内查。

## 为什么用截层而不是 hook（关键）

上一轮所有逐层尝试都坏在同一处：**hook 在 inductor 图里不生效**，
而且 eager 抓的数据不能当 compile 判据（`docs/knowledge/segment-isolation.md` §3）。

截层对拍绕开这个：vLLM 用 `--hf-overrides '{"text_config":{"num_hidden_layers":N}}'`
**真的只建 N 层**，我们侧 `kaon3_probe --layers N` **真的只跑 N 层**。
两侧都是 compile、都是真实代码路径，不需要 hook。

## 起点（2026-10-24 实测，可信）

compile 口径、中文 prompt、`kaon3_oracle_diff`：

| 口径 | flip | \|Δlp\| |
|---|---|---|
| **FLOOR**（同一 oracle 抓两次，BI=1） | **0.00%** | **0.0000** |
| 30 层全量（`MOE_COMBINE_F32=1`） | 14.29% (7/49) | 1.1599 |

**FLOOR = 0** → vLLM compile + BI=1 逐位可复现，判据是"逐位"，没有容差可言。

## 环境（每次从干净状态起）

- 机器 5090，`scripts/dev/remote_5090.sh`。**GPU 1 = 私有 oracle，GPU 7 = 基线禁用**。
- 用前 `nvidia-smi` 核显存，别假设空闲。
- oracle（**compile，不开 `enforce_eager`**）：
  ```bash
  ORACLE_GPU=1 ORACLE_PORT=28101 ORACLE_NAME=kaon-oracle-1 \
    scripts/dev/kaon3_oracle_server.sh 1        # 参数 = 截到 N 层
  ```
- 我们侧（compile 口径的 norm）：
  ```bash
  CUBLASLT_WORKSPACE_SIZE=1 RMSNORM_MODE=0 \
    ./target/release/kaon3_probe $M --batch <oracle目录> --layers 1 --out <ours>
  ```
- 对拍：`./target/release/kaon3_oracle_diff <oracle目录> <ours目录>`

## 观察方法

| 看什么 | 命令 |
|---|---|
| 端到端差值 | `kaon3_oracle_diff <oracle> <ours>` 的 `POOLED` 行 |
| FLOOR（必做） | 同一 oracle 抓两次，互相比；**必须为 0**，否则判据不可用 |
| 每层收敛情况 | `flip_pct` 与 `mean_common_d` 随 N 变化的曲线 |

## 停止条件（可机器判定，见 verify 脚本）

1. **FLOOR 必须为 0**（同一 oracle 两次抓取逐位一致）。不为 0 则整轮作废。
2. 参数逐项核对：block 32 / max-num-batched-tokens 4096 / BI=1 / compile 模式 /
   截层数两侧一致。任何一项不符则整轮作废。
3. 每层记录 `flip_pct`、`mean_common_d`，直到找到**第一个 flip > 0 的层号**。
4. 1 层时若已 flip > 0 → 差异在**第 0 层内部**，转层内定位；
   若 1 层 flip = 0 → 逐层往上加，找到第一层开始偏。

## 上限

每轮 ≤ 10 次迭代；单层定位超过 3 轮仍无结论，停下写卡点。

---

## 数据布局（用户 2026-10-24 要求）

产物按 `用途_日期时间_轮次` 存：`$REMOTE_ROOT/runs/<用途>_<YYYYMMDD-HHMM>_<rNN>/`，
目录内 `README.txt` 记口径与环境变量。工具 `scripts/dev/mkrun.sh`，规范见 `docs/data-layout.md`。

## 已排除（不要重复）

- **dense NVFP4 GEMM**：自洽合成判据，`ndiff=0/11264`，三条独立路径互证。
- **`MOE_COMBINE_F32` 是主因**：compile 口径下两种口径都差 ~1.1，改它方向还反了。
- **eager 抓的子模块数据当判据**：规格禁止；且参照 `L*.bin` 两次抓取自身差 0.0077。

## 第 1 轮结果（2026-10-24）：**差异在 L0 之内**

参数对齐 5/5 通过，FLOOR = 0（参照逐位可复现），判据可信。

```
截到 1 层（我们 vs vLLM compile）：
  decisive=0  flip=0.00%  mean_common_d=0.0085  ΔNLL=+1.5064
```

**关键：`decisive=0`。** 1 层时 top1 margin 全为 0，`flip` 是 0/0，
**不能用 flip 判"一致"**——改成看 `|Δlp|` 与 `ΔNLL`。两者都不为 0 → **L0 已经偏了**。

这与 `docs/loops/kaon3-parity.md` 第 21–22 轮的结论一致（当时用 p02 测得 `|Δlp| 1.4898`），
**互相印证**：不需要再往 30 层二分，范围就在 L0 内部。

### 判据（修正后，后续每层都用这套）

| 情况 | 用什么判 |
|---|---|
| `decisive > 0` | `flip == 0` |
| `decisive == 0`（层数少） | `mean_common_d == 0` **且** `dnll == 0` |

### 下一步

L0 内部已有明确的嫌疑（`kaon3-parity.md` 第 20–22 轮）：
compile 下 inductor 把 `norm → ×(1/scale) → clamp ±448 → fp8` 融成一个 kernel，
**全程 fp32、没有 bf16 往返**；我们多了一次 bf16 舍入。改 `inv` 公式无效（kernel 级 A/B `0/22528`）。
第 22 轮卡在：那个融合 kernel 的 12 个入参（三个输出 `out_ptr4` bf16 / `out_ptr5` bf16 /
`out_ptr7` fp8，`w` 在两处乘的位置不同）还没逐个对上模型参数。

## 第 2 轮：**norm kernel 已逐位正确（mode 2），但端到端反而更差**

### 权威源码（`vllm/ir/ops/layernorm.py:11-25`，从 oracle 容器直接读的）

```python
def rms_norm(x, weight, epsilon, variance_size=None):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x_var = x if variance_size is None else x[..., :variance_size]
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    if weight is not None:
        x = x.to(weight.dtype) * weight     # ← 先舍 bf16，再乘 w
    return x.to(orig_dtype)
```

### 数值验证（把我们的 `input_norm.bin` 与运行时 `rms_norm` 逐位比）

```
nm0（fp32 连乘、末尾舍一次） vs vLLM rms_norm : 17063 / 67584 不同，max|d| = 0.5
nm2（x·inv 先舍 bf16 再乘 w）vs vLLM rms_norm :     0 / 67584 不同，max|d| = 0.0   ← 逐位相同
```

**`RMSNORM_MODE=2` 才是 compile 口径，且我们的 kernel 实现已经完全正确。**

这推翻了 `kaon3-parity.md` 第 20 轮「bf16 往返被 inductor 消除」的判断 ——
那是读了 inductor 生成的低层 Triton 源码得出的，与 ATen 层 FX 图和数值实测都矛盾。

### 但：端到端 A/B 里 mode 2 **更差**

```
截 1 层，cn1：
  RMSNORM_MODE=0 : mean_common_d=0.0085  ΔNLL=+1.5064
  RMSNORM_MODE=2 : mean_common_d=0.0103  ΔNLL=+1.8513   ← 更差
```

norm 本身逐位对了，端到端却更差 → **说明下游还有别的口径用错了，
且它的错正好被 mode 0 的错「抵消」了一部分。** 这是最需要警惕的情况
（两个错误互相掩盖），不能因为"mode 0 数字好看"就选它。

### 下一步

在 mode 2（norm 已逐位正确）的基础上，逐个核对 L0 的后续段：
`input_norm → q/k/v fp8 量化 → attention → o_proj → post_attn_norm → …`
每一段单独与参照比，找出下一处口径错。

## 第 3 轮：L0 的精确数据流（直接从 FX 图读出，不再是推断）

来源：oracle 容器 `/root/.cache/vllm/torch_compile_cache/6c6aae890c/rank_0_0/backbone/computation_graph.py`
（**compile 口径的 ATen 层图，2816 维，单层截断**）。全部节点都标了 dtype，**全是 bf16 张量**。

```python
# ---- attention 前 ----
input_norm  = rms_norm(embed, w_input, 1e-6)          # bf16
qkv_fp8     = quant_fp8(input_norm)                   # bf16 → to(f32) → *recip → clamp(±448) → fp8
attn        = flashinfer_scaled_fp8_mm(qkv_fp8, W_qkv_fp8, ...)
q           = attn[:, :4096].unflatten(16, 256)        # bf16
q           = rms_norm(q, w_q_norm, 1e-6)             # bf16   ← 也是 rms_norm 同一个 op
k           = attn[:, 4096:6144].unflatten(8, 256)
k           = rms_norm(k, w_k_norm, 1e-6)             # bf16
... rope ...
o           = attn_kernel(q, k, v)                     # bf16

# ---- residual + FFN ----
add         = rms_norm(o, w_post_attn, 1e-6) + embed   # bf16   ← 第 301 行
pre_ff1     = rms_norm(add, w_pre_ff, 1e-6)            # bf16   ← 第 307 行
mlp_out     = mlp(pre_ff1)                             # bf16
post_ff1    = rms_norm(mlp_out, w_post_ff_1, 1e-6)     # bf16   ← 第 383 行附近
pre_ff2     = rms_norm(add, w_pre_ff_2, 1e-6)          # bf16   ← 第 383 行（注意输入是 **add**）
router_in   = rms_norm(add, None, 1e-6) * root_size * scale   # 无权重 norm，第 384 行
moe_out     = moe_forward(pre_ff2, router_in, ...)      # bf16
post_ff2    = rms_norm(moe_out, w_post_ff_2, 1e-6)     # bf16   ← 第 424 行
sum2        = post_ff1 + post_ff2                      # bf16   ← 第 424 行 `add_2`
post_ff     = rms_norm(sum2, w_post_ff, 1e-6)          # bf16   ← 第 430 行
out         = (post_ff + add) * layer_scalar           # bf16   ← 第 434/437 行
```

### 关键结构点（与我们的实现逐条对照用）

1. **`pre_ff2` 和 `router_in` 的输入都是 `add`**（= attention 残差流），**不是** `mlp_out`。
   → 我们 `layer.rs` 里 router 用的 `sc.routed` / MoE 用的 `sc.h2`，要确认都来自 `add`。
2. **`pre_ff2` 用 `w_pre_feedforward_layernorm_2`**，`post_ff_2` 用 `w_post_feedforward_layernorm_2`，
   两者是不同权重。
3. `router_in` 是 **`rms_norm(add, None)` 无权重**，然后 `* root_size * scale`（两次 bf16 乘）。
4. 最后 `(post_ff + add) * layer_scalar` —— 残差加在 **post_ff 之后**，与 `add` 加。
5. `post_ff_norm` 的输入是 `post_ff1 + post_ff2`，**不含 residual**；residual 在最外面加。

### 已数值验证（mode 2）

```
input_norm vs 运行时 rms_norm : 0 / 67584 不同（逐位）
q_norm     vs 运行时 rms_norm : 0 / 98304 不同（逐位）
```
→ norm kernel 在 mode 2 下完全正确；**差异在别处**。
