# kaon3-compile 迭代记录

规格：`docs/loops/kaon3-compile.md`　验证脚本：`scripts/verify/kaon3-compile.sh`

## 口径（用户 2026-10-23 定）

**全程 compile。** 参照 = vLLM 默认 torch.compile 服务；我们侧 `RMSNORM_MODE=0`。
eager 抓的中间量（`/w/hid2/*`、`kaon3_vllm_hidden.py`）**不作判据、不当下限分母**。

---

## 第 0 轮：环境盘点（2026-10-23 深夜）

**发现环境不干净**——开始本轮前 `kaon-oracle` 还停在**上一会话遗留的截层实例**：

```
--hf-overrides {"text_config":{"num_hidden_layers":1}}
```

任何人这时读数都会是错的（截 1 层的 logits 全部饱和，见 `kaon3-gelu.md` 第 4 轮）。
已按规格 `docker rm -f` 后由 `kaon3_oracle_server.sh`（无参数 = 全 30 层）重启，核过
`Running=true` / cmd 无 `num_hidden_layers` / env 无 eager / `:28100` 返回 200。

规格里因此把"先拆再起"写成硬步骤，并在 `kaon3-compile.sh` 里做成机器可判定的四项检查。

## 第 1 轮：`RMSNORM_MODE=0` 整跑（交接区点名的唯一未验差异点）

- 改动：无代码改动，只切口径 `PROBE_ENV="CUBLASLT_WORKSPACE_SIZE=1 RMSNORM_MODE=0"`。
- 基线（MODE=1 历史）：flip 16.56% / |Δlp| 1.1994 / FLOOR 0.83% / 0.0276。
- 结果：见下方"第 1 轮结果"。

### 第 1 轮结果

（待整跑完成回填）

## 读参照源码：compile 口径的两个语义点（2026-10-23，读镜像副本 `/data/ref/vllm-img-v0.28.0-pr455/`）

### 1. `fused_add_rms_norm`（`vllm/ir/ops/layernorm.py:44`）——残差**先 fp32 加、舍 bf16 存回**

```python
x = x.to(torch.float32)
x = x + x_residual.to(torch.float32)
x_residual = x.to(orig_dtype)          # ← 舍回 bf16，下游读的是这个
variance = x.pow(2).mean(dim=-1, keepdim=True)
x = x * torch.rsqrt(variance + epsilon)
if weight is not None: x = x.to(weight.dtype) * weight
return x.to(orig_dtype), x_residual
```

注意：**`x_residual`（残差流）与 `x`（norm 输入）不是同一个值**。norm 的方差与归一化用**未舍入的 fp32 和**，
而残差流写回时舍了一次 bf16。我们的 `rmsnorm_vec_kernel<FUSED_ADD>` 写法是
「相加 → 舍 bf16 写回 `rr` → **再把舍入后的值读回 `f[c]` 用于归一化**」，注释说"与标量版逐位一致"。

**这是与参照的语义分歧点**：参照用 fp32 和算方差，我们用 bf16 舍入后的和算方差。
除非 inductor 把那次 store 重排成 f32 直通（需要生成代码验证），否则这里每层都会引入偏差。
→ 若第 1 轮 MODE=0 仍不达标，这是**第一优先**要 A/B 的点（新增 `RMSNORM_ADD_F32` 开关）。

### 2. `rms_norm`（`vllm/ir/ops/layernorm.py:10`）——`rsqrt(mean+eps)`，源码里有 bf16 舍入

源码 `x = x.to(weight.dtype) * weight` 是显式 bf16 舍入；我们 `norm.cu:58` 的注释说 inductor 把它消除了
（A/B 证实 bf16 版与 f32 版逐位相同）。**若第 1 轮读数不对，这条也要重验**，别只信注释。

### 待查（未读）

- `lm_head` 前 `final_norm` 是否与 fp8 量化融合（vocab=262144 大形状）。
- logits softcap `tanh(x/30)·30` 的精度与位置。
