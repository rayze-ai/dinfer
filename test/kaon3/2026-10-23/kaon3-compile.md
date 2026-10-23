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

`PROBE_ENV="CUBLASLT_WORKSPACE_SIZE=1 RMSNORM_MODE=0"`，run `1023-234651-1725776`，30 用例 538 行：

| 项 | 第 1 轮（MODE=0） | 基线（MODE=1） | FLOOR | 判定 |
|---|---|---|---|---|
| 可判定行 top1 翻转率 | **16.98%**（82/483） | 16.56%（82/483） | 0.83% | ❌ 超 1.5×+3 |
| 共有 token \|Δlp\| | **1.1816** | 1.1994 | 0.0276 | ❌ 超 1.5× |
| 短 prompt ΔNLL | **+0.2621** | −0.1625 | −0.0025 | 参考 |
| long_zh ΔNLL | +0.0862 | +0.1466 | −0.0817 | ✅ 通过 |
| 单 token（bos1） | top1 一致 | 一致 | — | ✅ 通过 |

**结论：`RMSNORM_MODE=0`（compile 口径）整跑与 MODE=1 无实质差别**（flip 82 vs 82，Δlp 1.18 vs 1.20）。
交接区把这条列为"唯一还没验过的已知差异点"——**现已验过，不是主因**。
结合第 19 轮观察 7（compile 实例里把 norm 换成 BI 版，`mean|dlp|` 1.4892→0.3741），
说明剩下的大头在 **norm 之外的 compile 融合链**（量化 / MoE / attention / 出口）。

环境检查 8 项中 6 项过、2 项（翻转率、Δlp）不过。

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

## 读参照：logits 出口路径（compile 口径，`vllm/model_executor/layers/logits_processor.py`）

```python
logits = self._apply_head(lm_head, hidden_states, embedding_bias)   # bf16 输出（head_dtype=None）
logits = logits[..., :self.org_vocab_size]
...
if self.soft_cap is not None:                                        # logits_processor.py:209
    logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
```

两个要点：
1. **softcap 在 bf16 logits 上做**（`_apply_head` 返回模型 dtype），但 `tanh` 在 PyTorch 里对 bf16
   是"升 f32 计算 → 存回 bf16"，即 `round_bf16(tanh_f32(bf16(x)/30)*30)`。我们 `softcap_bf16_n`
   注释写"单次舍入 `to_bf16(c*tanhf(v/c))`"——**口径看起来一致**，且曾做过三步 A/B 无差异。
2. **真正的分歧风险在 lm_head GEMM 本身**：`_apply_head` 走 `lm_head.quant_method.apply`，即 fp8 GEMM。
   我们的 `gemm.forward_bf16(normed, embed.table, ...)` 用的是 **bf16 权重**（tied embedding 表）。
   若参照把 lm_head 量化成 fp8，则 vocab=262144 的大 GEMM 每行误差会被 softcap 前的幅度放大。
   **这是下一个要确认的点**（读 checkpoint 的 quant 配置即可判定，不必跑）。

### 待办

- [ ] 确认参照 `lm_head` 是否量化（读远端镜像里加载后的 `lm_head.quant_method`，或 checkpoint config）。
- [ ] 确认 `final_norm` 在 compile 下是与 fp8 量化融合还是分离（fusion 会改变舍入点）。

## 第 1 轮进行中（环境已确认干净）

`kaon3-compile.sh` 前 6 项环境检查全过：
```
kaon-oracle 就绪（GPU 1 :28100 layers=30 block=32 BI=1）
✅ oracle 容器在跑   ✅ 全 30 层   ✅ 未开 eager（compile 口径）
✅ BI=1   ✅ :28100 200   ✅ GPU 无残留
```
oracle 启动日志确认走 compile 且 IR 选 native：
`IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])`，
`init engine … (compilation: 122.05 s)`。整跑进行中。

## 确认：`lm_head` 不量化（排除假设 2 的一半）

远端 `config.json` 的 `quantization_config.quantized_layers` 共 424 项，按后缀统计：

```
Counter({'linear': 189, 'q_proj': 30, 'k_proj': 30, 'o_proj': 30,
         'gate_proj': 30, 'up_proj': 30, 'down_proj': 30, 'experts': 30, 'v_proj': 25})
lm_head quantized: []    embed quantized: []    ignore: []
```

→ **`lm_head` / `token_embd` 都是 bf16**，我们 `gemm.forward_bf16(normed, embed.table, …)` 口径正确。
假设 2「lm_head 量化」**排除**。剩下要查的只有 `final_norm` 是否与量化融合（不进 graph 就看不出来）。

## 更正：Gemma4 不用 `fused_add_rms_norm`，残差是显式 bf16 加法（排除该假设）

前面"compiled `fused_add_rms_norm` 残差语义分歧"的猜测**不适用**——
`vllm/model_executor/models/gemma4.py:698-731` 的 decoder layer 是：

```python
residual = hidden_states
hidden_states = self.input_layernorm(residual)     # 独立 norm，不融 add
hidden_states = self.self_attn(...)
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = hidden_states + residual           # ← 显式 bf16 加法
residual = hidden_states
...
hidden_states = self.post_feedforward_layernorm(hidden_states)
hidden_states = hidden_states + residual           # ← 再一次
```

即 **norm 与残差加分离**，两处加法都在 bf16 里做。我们 `layer.rs:825 / 856 / 860`
用 `add_inplace_bf16_n`，口径一致 → **假设排除**。
（oracle 日志里的 `fused_add_rms_norm=['native']` 只是 IR 优先级注册，本模型没调用它。）

### 层内残差结构（照参照抄，供后续核对）

| 步骤 | 参照 | 我们 |
|---|---|---|
| input_norm(residual) | 独立 | `layer.rs:679` ✅ |
| attn → post_attn_norm → **+residual** | bf16 | `layer.rs:823-825` ✅ |
| pre_ff_norm → dense → post_ff_norm_1 | — | `849` ✅ |
| pre_ff_norm_2(residual) + router(residual) → moe → post_ff_norm_2 | **并联** | `950 / 1011` ✅ |
| (ff1 + ff2) → post_ff_norm → **+residual** | bf16 | `856/860` ✅ |

## router 预处理：参照是**三次独立 bf16 舍入**（`gemma4.py:299-301`）

```python
def forward(self, x):
    x = self.norm(x)                       # 无权重 RMSNorm
    x = x * self.root_size.to(x.dtype)     # bf16 乘 → 存回 bf16
    x = x * self.scale.to(x.dtype)         # bf16 乘 → 存回 bf16
    router_logits, _ = self.proj(x)        # out_dtype=float32
    return router_logits
```

**但编译后 inductor 会把这三步融成一条 fp32 链**（这正是我们加 `rmsnorm_scale2` 的依据，
`ops.rs:603`：「全程 fp32、只舍入一次」）。

⚠️ **这条与"照抄 Python 源码"相反——是推断，不是实测**。判据必须是 inductor **生成的代码**，
不是 `gemma4.py`。`/w/tmp/ind` 已在 2026-10-23 丢失，需要重导。

**重导方法**（`kaon3-gelu.md` 第 2 轮已验证可行）：
```bash
GPU=1 scripts/dev/remote_5090.sh ctr 'cd /w/dinfer && TORCHINDUCTOR_CACHE_DIR=/w/tmp/ind \
  VLLM_BATCH_INVARIANT=1 python3 -I -c "..."'
```
导出后 grep `computation_graph.py` 里 `router` / `root_size` / `scale` 对应的融合段，
看 `x_bf16 * rs * sc` 是**一次舍入还是两次**。这是假设 6 的判据来源。

### `final_norm`（`gemma4.py:1344`）

`residual is None` → `hidden_states = self.norm(hidden_states)`，**独立 norm，不与 lm_head 融合**
（lm_head 是独立的 `ParallelLMHead` 模块，中间没有量化）。假设 2 的"融合"分支**排除**。

## 第 2 轮：直接从 oracle 容器取 FX 图（不用重导）

**方法（省掉重导，已验证）**：oracle 容器刚做过 compile，产物就在容器里：
```bash
docker cp kaon-oracle:/root/.cache/vllm/torch_compile_cache /w/tmp/ind2/
# → /w/tmp/ind2/torch_compile_cache/0642510a5c/rank_0_0/backbone/computation_graph.py（12451 行）
```
**以后要 inductor 产物，先 `docker cp` 现成的容器，别重跑导出。**

### 图内算子清单（全 30 层，`grep -oE` 统计）

```
331 torch.ops.vllm_ir.rms_norm            120 torch.ops.vllm.bmm_fp8
 60 torch.ops.vllm.unified_kv_cache_update   60 torch.ops.vllm.unified_attention_with_output
 30 torch.ops.vllm.moe_forward              30 torch.ops.higher_order.triton_kernel_wrapper_mutation
```
331 = 30×(input_norm + post_attn_norm + pre_ff_norm + post_ff_norm_1 + pre_ff_norm_2 + post_ff_norm_2
+ post_ff_norm + q_norm + k_norm + v_norm) + final_norm 左右 —— 即**所有 norm 都走同一个 op**。

### 逐行读数 1：norm 输出的 dtype 在 FX 图里 **是 bf16**

```
rms_norm_default: "bf16[s59, 2816]" = torch.ops.vllm_ir.rms_norm.default(l_inputs_embeds_, w, 1e-06, None)
view:             "bf16[s59, 2816]" = rms_norm_default.view(-1, 2816)
to:               "f32[s59, 2816]" = view.to(torch.float32)          # ← 量化入口
```

⚠️ `norm.cu:58` 的注释说「`x.to(weight.dtype)` 的 bf16 舍入被 inductor 消除了」。
**FX 图上它没被消除**（签名就是 bf16）。是否在 fuse 后的 Triton kernel 里才消失，**必须看生成的 kernel**，
不能靠 FX 图推定。这是 `NORM_FUSE_QUANT` 该不该开的判据。

### 逐行读数 2：fp8 量化 = `f32 × reciprocal(scale)` → clamp(±448) → fp8

```
to_1:      "f32[]"          = input_scale.to(torch.float32)
reciprocal:"f32[]"          = to_1.reciprocal()        # ← 先求倒数再乘，不是除
mul:       "f32[s59,2816]"  = to * reciprocal
clamp:     "f32[s59,2816]"  = mul.clamp(-448.0, 448.0)
to_2:      "f8e4m3fn[...]"  = clamp.to(torch.float8_e4m3fn)
```
→ 与 `norm.cu:439` 的注释一致（`out * (1.0/scale)`，clamp ±448）✅

### 逐行读数 3：`v_norm` 的 weight 是 `None` ✅

```
rms_norm_default_3: "bf16[s59, 8, 256]" = torch.ops.vllm_ir.rms_norm.default(unflatten_2, None, 1e-06, None)
```
→ 与我们 `head_rmsnorm(v, None)` 一致。

### 逐行读数 4（**重要**）：router 预处理在 FX 图里是**两次独立 bf16 舍入**，没融合

`/w/tmp/cg.py:1400-1412`（layer 0）：

```
rms_norm_default_4: "bf16[s59,2816]" = vllm_ir.rms_norm.default(add, None, 1e-06, None)   # 无权重 norm
to_9:  "bf16[]"           = router_root_size.to(torch.bfloat16)
mul_4: "bf16[s59,2816]"   = rms_norm_default_4 * to_9        # ← 乘完是 bf16（舍入 1）
to_10: "bf16[2816]"       = router_scale.to(torch.bfloat16)
mul_5: "bf16[s59,2816]"   = mul_4 * to_10                    # ← 再乘再舍（舍入 2）
```

⚠️ **这与我们 `rmsnorm_scale2`（「全程 fp32、只舍入一次」，`ops.rs:603`）的口径相反。**
FX 图显示参照是 `bf16(norm) → ×rs → 舍 bf16 → ×scale → 舍 bf16`，**三次舍入**。
我们按"inductor 会融成 fp32 链"的推测只舍一次 → **每层 router 输入都不同 → 可能选不同专家**。

标注为**待验证**：FX 图不等于 fuse 后的 Triton kernel。判据是 `artifact_*_subgraph_*` 里的生成代码
（若有专门的 router 段）或直接 A/B：加 `ROUTER_PRE_3ROUND=1` 开关走三次 bf16，看 flip 是否下降。

### 逐行读数 5：router GEMM 走 **BI Triton persistent matmul**，不是 cuBLAS

```
batch_invariant.py:146  matmul_persistent:  empty "bf16[s59,128]" = torch.empty(...)
batch_invariant.py:151  grid: sym_min(170, ceil(s59/128))
batch_invariant.py:184  triton_kernel_wrapper_mutation(kernel_idx=1, grid=[(sym_min,1,1)], ...)
gate_linear.py:231      to_11: "f32[s59,128]" = empty.to(torch.float32)     # 输出转 fp32
```

→ **即便在 compile 模式下，router 投影仍是 BI 的 Triton persistent matmul**（`VLLM_BATCH_INVARIANT=1` 生效）。
我们 `router_bf16_ab()` 用 cuBLAS bf16 GEMM，**两者是不同的 kernel**。
→ 这解释了为什么必须有 `CUBLASLT_WORKSPACE_SIZE=1` 才能稳定：我们的 router GEMM 受 cuBLASLt 启发式影响，
而参照根本不用 cuBLAS。**router GEMM 要复刻 BI persistent matmul 才可能逐位**（假设 6 升级为主嫌疑）。

### 顺带确认：`moe_forward(rms_norm_default_3, to_11, …)` 的第二个参数是 **f32 router logits**

`to_11: "f32[s59,128]"` → 与我们"logits 保持 f32 进 top-k"一致 ✅

### 参照集口径已验证（重要，排除一个怀疑）

从**当前** compile oracle 重抓 p02，与仓库里的 `test/kaon3/oracle_clean/p02.json` 比：
```
130 项 prompt_logprobs，不同 0
```
→ `oracle_clean` **确实是 compile 口径**，可以放心当靶子。FLOOR 0.83% 是 30 用例聚合的跨进程差，不是口径错。

## 第 3 轮：`RMSNORM_MODE=2`（`x·inv` 后先舍 bf16 再乘 w）

**依据（FX 图 + 源码，两者一致）**：
- FX 图：`rms_norm_default: "bf16[s59,2816]" = vllm_ir.rms_norm.default(...)`——**返回类型是 bf16**；
- 源码 `vllm/ir/ops/layernorm.py:17`：`x = x.to(weight.dtype) * weight`——**先舍 bf16 再乘 w**；
- 第 19 轮的数值表也支持：`(x·rsqrt).to(bf16)·w` vs eager IR **0 差异**；
  不先舍入直接乘 w → **47030/67584 元素不同**。

而我们 `norm.cu` 的 mode 0 是 `x·inv·w` fp32 连乘、末尾舍一次 —— 与上面**不符**。
（第 19 轮注释称"inductor 消除了那次舍入"，但 FX 图和源码都不支持该说法。）

**改动**：`norm.cu` 新增 `norm_mid()`（mode 2 时舍 bf16），作用于三处：vec kernel、warp kernel、标量回退；
`rmsnorm_scale2` 的 mode 2 走 FX 图的三次舍入（norm 出口 / ×rs 后 / ×scale 后）。mode 0 路径**未变**。
`layer.rs` 的 `norm_mode() == 0` 统一改成 `!= 1`，让 mode 2 走同一条 compile 分支。

- 命令：`PROBE_ENV="CUBLASLT_WORKSPACE_SIZE=1 RMSNORM_MODE=2" bash scripts/verify/kaon3-compile.sh`
- 对照：第 1 轮 `MODE=0` → flip **16.98%** / |Δlp| **1.1816**

**结果**（run `1024-000545-1743120`，重编后）：

| 口径 | flip | \|Δlp\| | 短 ΔNLL | 长 ΔNLL |
|---|---|---|---|---|
| MODE=1（eager 口径，历史基线） | 16.56% | 1.1994 | −0.1625 | +0.1466 |
| MODE=0（compile fp32 连乘） | 16.98% | 1.1816 | +0.2621 | +0.0862 |
| **MODE=2（先舍 bf16 再乘 w）** | **15.94%** | **1.2367** | +0.1621 | **−0.1018** |
| FLOOR | 0.83% | 0.0276 | −0.0025 | −0.0817 |

**判读**：三个口径都在 16–17%，差别在噪声量级；MODE=2 的 flip 略低、长文本 ΔNLL 反而最接近 FLOOR（−0.1018 vs −0.0817），
但 |Δlp| 略升。**norm 口径不是主因，三个都是"次 ulp 级别"的贡献。**
保留 `RMSNORM_MODE=2` 作为默认候选（长文本更准），但不指望靠它达标。

### 第 3 轮期间的排除项（读 FX 图）

**已排除：q 的 fp8 量化**——图里 `query_2: "f8e4m3fn[s59, 16, 256]"` 进 attention，
用 `attn_buffers_q_scale_` 的 `reciprocal` 量化、clamp ±448。我们 `layer.rs:753`
`fp8_e4m3_roundtrip_bf16_n(&mut q, …)` 已做（2026-10-22 A/B：关掉 ΔNLL −0.0018→+0.0438 明显变差）✅
**k / v 是 bf16**（`view_6/view_7: "bf16[s59, 8, 256]"`），KV cache 才走 fp8 ✅

**已排除：output 缓冲**——`empty: "bf16[s59,4096]"` 由 attention 原地写，我们 `sc.o` 同口径 ✅

### 攻击面收缩（FX 图全算子清单 vs 已核对）

| 图内算子 | 次数 | 状态 |
|---|---|---|
| `vllm_ir.rms_norm` | 331 | 第 1/3 轮 A/B（mode 0 vs 2） |
| `vllm.bmm_fp8` | 120 | **未核对**（fp8 GEMM 的 scale 应用顺序） |
| `unified_kv_cache_update` | 60 | 未核对（kv fp8 量化点） |
| `unified_attention_with_output` | 60 | q fp8 ✅ / 其余未核对 |
| `vllm.moe_forward` | 30 | **未核对（最大一块）** |
| `triton_kernel_wrapper_mutation` | 30 | = BI persistent matmul（router）+ 其他 |
| `aten` 逐元素（add/mul/split/cat…） | 其余 | 逐元素，口径已基本确认 |

### 第 3 轮第一次作废：编译失败 + 未 sync

第一次跑 `RMSNORM_MODE=2` 时 `dinfer-kernels` 构建失败：
```
cuda/norm.cu(390): error: identifier "mode" is undefined
```
`rmsnorm_scale2_kernel` / `launch_scale2` 没传 `mode`。**但脚本没拦住**——`kaon3-parity.sh` 只 `grep '^error'`，
而 build.rs 的失败是 `cargo:warning=` 前缀的 `nvcc[...] error:`，grep 不到 → 脚本继续用**上一轮的旧二进制**
跑完了整轮（`❌ kernel 对拍：` 那行是空的，其实已经失败了）。

**这类"编译失败但测试照跑"必须堵住**：
- 本地改完必须先 `sync`（我漏了一次，远端还是旧文件，第一次修完再编仍报同一个错）；
- 判据要检查 `cargo:warning=.*error:` 与 `BUILD FAILED`，不能只 grep `^error`。

已修 `norm.cu` 的 kernel/launcher 签名并 sync，重编通过。**下面这条读数必须来自重编后的二进制。**

### 附：`fp8_flashinfer_parity` 目前跑不起来（数据缺失，不是数值失败）

```
thread 'fp8_matches_flashinfer_bitwise' panicked at ...:41:64:
读 meta.txt: Os { code: 2, kind: NotFound }
```
靠 `test/kaon3/d17b/` 的真实入参（`A_fp8.bin` / `B_fp8.bin` / `ref_f32.bin` / `meta.txt`），
这些文件在远端已不在。**要重跑需先用 `scripts/dev/kaon3_vllm_fp8det.py` 重新抓一组。**
在拿到之前，`bmm_fp8` 只能靠端到端 A/B 判，不能靠这个测试。

### 第 3 轮期间的读数：router 归约用 `rcp.approx`，参照是精确除法（**新发现的真差异**）

参照实际执行的是 **Triton kernel** `_gemma4_routing_kernel`（`gemma4.py:100-163`），不是
`gemma4_routing_function_torch`（那个只在非 CUDA 平台用）：

```python
all_raw_exp = tl.math.exp2((all_logits - max_l) * 1.4426950408889634)
renorm_raw  = tl.sum(tl.where(top_mask, all_raw_exp, 0.0), axis=0)
inv_renorm  = 1.0 / renorm_raw                       # ← 精确除法（IEEE div.rn）
all_weights = (all_raw_exp * inv_renorm * all_scales).to(tl.float32)
```

我们 `moe.cu:232-237`：
```cpp
const float inv = route_rcp(renorm);                 // ← rcp.approx.ftz.f32（近似！）
topk_w[..] = __fmul_rn(__fmul_rn(raw[k], inv), sc);
```

| 项 | 参照 | 我们 | 判定 |
|---|---|---|---|
| `exp2` | `tl.math.exp2` → `ex2.approx` | `ex2.approx.ftz.f32` | ✅ 同 |
| `log2e` 常数 | `1.4426950408889634` | 同 | ✅ |
| 归一化倒数 | `1.0 / renorm` **精确** | `rcp.approx` **近似** | ❌ **差 ~1-2 ulp** |
| 乘序 | `exp * inv * sc` | `(exp*inv) * sc` | ✅ 同 |
| top-k 选择 | 按 logits 位模式排序（bitcast 成 int，保序） | 按 logits 直接比 | ✅ 等价 |
| 归约次序 | `tl.sum` 按 BLOCK_E=128 树形 | 32 lane 树 + 组内顺序 | ⚠️ 可能不同（需测） |

**`rcp.approx` 相对精确除法有 1-2 ulp 误差**。128 个专家、每 token 2 个权重——
这直接影响 `topk_w`，进而每条专家路径的输出。**这是目前最具体的一处口径差。**

→ 第 4 轮：把 `route_rcp` 换成 `__frcp_rn`（精确倒数）或 `__fdiv_rn(1.f, renorm)`，整跑比 flip。

## 第 4 轮：router 归约换精确倒数（`ROUTER_RCP_EXACT=1`）

**假设**：参照 Triton 的 `inv_renorm = 1.0 / renorm_raw` 是 IEEE `div.rn`；
我们 `route_rcp` 用 `rcp.approx.ftz.f32`（1–2 ulp 偏差），直接影响每 token 的 top-k 权重。

**改动**：`moe.cu` 加 `route_inv(x, exact)`，`exact_rcp` 由 host 侧读 `ROUTER_RCP_EXACT` 传入
（device 内不能有动态初始化的 static，第一次写成 device 侧 lambda 编译失败——已改）。
默认仍走 `rcp.approx`，A/B 时用环境变量开。

- 命令：`RMSNORM_MODE=2 ROUTER_RCP_EXACT=1 …`（同时保留第 3 轮较优的 norm 口径）
- 对照：MODE=2 无此项 → flip **15.94%** / |Δlp| **1.2367**

**结果：完全无变化，两次输出 md5 相同。**
```
1024-000545（无开关）p02.json md5 = 48751c2a3c356b37581689d1dce61b8a
1024-001230（RCP_EXACT=1）      md5 = 48751c2a3c356b37581689d1dce61b8a
```
直接跑探针两次对比也是同一个 md5（`198becdd…`）。

**为什么应该无效（先验证了环境变量确实生效，再解释）**：
| 量 | 值 |
|---|---|
| `rcp.approx.ftz.f32` 最大相对误差 | ≈ 2⁻²³ |
| bf16 相对舍入步长 | 2⁻⁹ |
| 比值 | **16384 倍** |

`topk_w` 算完后立刻 `bf16(topk_w · per_expert_scale)`（`layer.rs:976` 的默认 vLLM 口径）。
**rcp 的下位误差比 bf16 的半个 ulp 小 4 个数量级，必然被舍掉** → 输出逐位相同是预期结果，不是环境问题。

→ **`ROUTER_RCP_EXACT` 这条假设正式排除**。开关保留（无害，且可用于 f32 combine 口径的实验）。

## 诊断：p02 逐位置差异的**形态**（第 3 轮产物，重要）

不只是看聚合指标，把 p02 的 23 个位置逐个展开（`/tmp/ours_p02.json` vs `oracle_clean/p02.json`）：

| 观察 | 数据 |
|---|---|
| top-1 翻转 | 4/23（第 6/10/14/18 位） |
| **最佳 token 的 logprob 偏差** | 大多数 \|d\| < 0.1，但 4 行到 **0.77 / 0.36 / 0.32 / 0.50** |
| **相邻 top1-top2 间隔比（ours/ref）** | **中位数 0.96**（整体尺度一致！）但 4 行是 **0.22 / 0.50 / 2.2 / 9.7** |
| 完全精确的位 | 第 17、20、21、22、23 位（Δ 全是 0.0000） |

**形态判读（关键）**：
- 是**离散跳变**，不是连续的小偏差 → **不是** ulp / 累加顺序问题（那种会是全面小误差）；
- 后几位（20–23）**完全精确**、整体尺度中位数 0.96 → 主干是对的；
- 个别位置的间隔比差到 9.7 倍 → **那些位置某个专家的权重或专家选择不同**。

→ 这指向 **MoE 路由 / 专家选择** 的分歧：当 top-k 边界附近有两个专家分数接近时，
我们和参照选了不同的专家，该 token 的 logits 就整块不同（表现为"间隔比突变"）。
**这与 `router` GEMM 用不同 kernel（我们 cuBLAS vs 参照 BI persistent matmul）高度吻合。**

### 验证方法（下一步）

单看端到端 flip 无法区分"路由选错"与"专家算错"。要**直接比 top-8 集合**：
`scripts/verify/kaon3-parity.sh` 里已有 `kaon3_router_truth_all.py`（fp64 真值比对），
但那是"谁更准"，不是"我们和参照是否同一集合"。需要补一个：**逐层 dump 我们的 top-k 集合，
与参照同一输入的 top-k 对比**，只在对齐的行上判 logprob。

而参照的 top-k 拿不到（compile 不可 hook）。替代：**用 `ROUTER_F32=1` 让我们的 router 走精确路径**，
若 flip 明显下降，则确认主因在 router 数值。

## 为什么 "ulp 级差异" 能放大成 16% flip：128 选 8 × 30 层

配置（`config.json` text_config）：`num_experts=128`、`top_k_experts=8`、`num_hidden_layers=30`。

**放大机制**：router logits 只要差 1 ulp，在 **第 8 名与第 9 名分数接近**的 token 上就会换专家。
换掉一个专家 = 该 token 的 MoE 输出整块不同（不是小扰动），经过 30 层累积，
**少数层上的少数 token 被换 → 端到端 16% 的 flip**。

这与实测形态完全吻合：
- 大多数位置 Δ 很小（那些位置 top-8 边界不敏感）；
- 少数位置**突变**（间隔比 9.7 倍、top-1 翻转）；
- 后几位**完全精确**（边界恰好不敏感）。

**结论**：只要 router GEMM 与参照不是同一个 kernel，就不可能达到 FLOOR 线。
而参照的 router 是 **BI Triton persistent matmul**（FX 图已证），我们用的是 **cuBLAS bf16 GEMM**
（要靠 `CUBLASLT_WORKSPACE_SIZE=1` 才稳定）——**两者归约次序必然不同**。

→ **假设 6（router GEMM 复刻 BI persistent matmul）升级为 loop 的主线**，
不再是"试试 A/B"，而是"必须做"。规格里的假设清单要按这个重排。

## 第 5 轮：**决定性测量**——cuBLAS 与参照 BI matmul 在 router 形状上差 43%

在 oracle 容器里直接调参照自己的 `matmul_persistent`（`batch_invariant.py`），与 `torch.matmul`（cuBLAS）比：

```
shape      M      N      K    ndiff/total   max|d|
cublas       24    128   2816      1332/3072     7.812e-03
BI-self      24    128   2816         0/3072
cublas       64    128   2816      3352/8192     1.562e-02
BI-self      64    128   2816         0/8192
cublas      512    128   2816       103/65536     7.812e-03
BI-self     512    128   2816         0/65536
```

**读法**：
- 真实 router 形状（K=2816、N=128）上，**43% 的输出元素不同**（M=24 时 1332/3072），最大差 7.8e-3；
- 参照的 BI matmul **自身完全可复现**（0 差异）→ 目标可达；
- 这也解释了之前为什么必须 `CUBLASLT_WORKSPACE_SIZE=1` 才能稳定：cuBLAS 的启发式会换 kernel。

**结论**：**router GEMM 必须复刻 BI persistent matmul，没有替代方案。**
43% 的 logits 元素不同 → top-8 边界必然换专家 → 端到端 16% flip 完全说得通。

**这条把"是否要写这个 kernel"从假设变成了定论**（测量成本：一次容器内 Python，几分钟）。

## 根因定位：router 走 BI matmul 的**真正机制**（我们代码注释写错了）

`models/kaon3/src/layer.rs:908-912` 的注释说：
> vLLM（容器 0.26.1rc1，sm_120）`GateLinear` 落到 Tier 6：`F.linear`（cuBLAS，bf16 输出）→ `.to(fp32)`
> （sm_120 不满足 `allow_specialized_router_gemm`，Tier 5 的 fp32 epilogue 不开）

**前半句对（落到 Tier 6），后半句"cuBLAS"错。** 实际路径：

```python
# vllm/model_executor/layers/linear.py:203-212  (UnquantizedLinearMethod.apply)
if envs.VLLM_BATCH_INVARIANT and current_platform.is_cuda_alike():
    return linear_batch_invariant(x, layer.weight, bias)      # ← 直接走 Triton persistent
return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)
```

```python
# vllm/model_executor/layers/batch_invariant.py:889
def linear_batch_invariant(input, weight, bias=None):
    output = matmul_batch_invariant(input, weight.t())         # → matmul_persistent
    ...
```

**注意**：`batch_invariant.py:911-927` 里那个 `is_device_capability_family(80)` 分支只决定
**是否安装 dispatcher patch**（`aten::linear` 等）；sm_120（12.x）走 else、只设
`CUBLASLT_WORKSPACE_SIZE=1`、**不 patch dispatcher**。但 `linear.py:209` 是**显式调用**，
不经过 dispatcher，所以在 sm_120 上**照样走 BI Triton persistent matmul**。

FX 图里 `batch_invariant.py:146/151/184` 的 `matmul_persistent` + `triton_kernel_wrapper_mutation`
就是这个路径，两者对上了。

**我们的现状**：`gemm.forward_bf16` = cuBLASLt。第 5 轮实测该形状上**43% 元素不同**（max 7.8e-3）。
→ **必须换成 BI persistent matmul 的等价实现。** 这是 loop 的主线，不再是候选假设。

## 第 5 轮成果：`router_gemm_bi_f32` **逐位复刻成功（0/8192）**

新增 `crates/kernels/cuda/router_gemm_bi.cu` + `ops::router_gemm_bi_f32` + kernel 级对拍
`crates/kernels/tests/router_bi_parity.rs`（数据由容器内 `matmul_persistent` 导出到 `/w/tmp/rbi`）。

```
ROUTER_BI [M=64 K=2816 N=128] ndiff=0/8192 max|d|=0.000e0
```

**实现要点（三条，缺一条就全错）**：
1. **A 块 smem 必须每 warp 一份**（`a_tile[BI_WARPS][16][BK+8]`）。
   最初写成 block 共享一份 → 8 个 warp 都在写同一块 smem → 全部 8192 元素错（max|d| 2.455）。
2. **K 方向严格顺序**：BK=64 分块、块内 4 条 `mma.m16n8k16` 顺序累加，全程一个 fp32 累加器
   （参照 `matmul_persistent` 的结构）。
3. **输出先舍 bf16 再转 f32**：`to_f32(to_bf16(acc))`。漏掉这步 → max|d| 3.898e-3、
   **4192/8192 全部不同**（正好一个 bf16 ulp，是这个 bug 的典型指纹）。

### 踩坑记录（值得记）

| 症状 | 根因 |
|---|---|
| `ndiff=8192/8192`，max\|d\| **2.455**（数量级都不对） | smem 只分了一份，8 个 warp 写同一块 |
| `ndiff=8192/8192`，max\|d\| **3.9e-3**（一个 bf16 ulp） | 漏了输出的 `to(bf16)` 舍入 |

"全部元素都不同但只差一个 ulp" = **输出精度不对**，不是算法错——这条以后可以直接当指纹用。

### 第 5 轮（续）：接入后端到端 A/B **完全没有差异**（待解释）

`router_gemm_bi_f32` kernel 级逐位 0/8192，已接入 `layer.rs:918`（`ROUTER_GEMM_BI` 默认开，
诊断输出确认分支确实执行：`[kaon3] router GEMM = BI persistent matmul`）。

但两条路径的整批输出**逐字节相同**：
```
md5 /w/tmp/o0/p02.json = 48751c2a3c356b37581689d1dce61b8a   (ROUTER_GEMM_BI=0, cuBLAS)
md5 /w/tmp/o1/p02.json = 48751c2a3c356b37581689d1dce61b8a   (ROUTER_GEMM_BI=1, BI)
```
`--oracle` 判定也完全一致：`4 行翻转 / top5 1.8089 / 实际 token 2.1794`（旧口径 1.1816 的差异来自
`RMSNORM_MODE` 与 oracle 文件不匹配，不影响本 A/B 的结论）。

**必须解释清楚才能继续**：
1. 若真实 router 输入下两 kernel 输出确实相同 → 我前面 43% 的测量用错了对照
   （那次比的是 `torch.matmul` vs `matmul_persistent`，而我们原来走的 cuBLASLt 启发式**未必**与
   `torch.matmul` 同轨；也可能真实 routed 输入的动态范围小，两者恰好都舍到同一 bf16）；
2. 若两 kernel 输出不同却端到端相同 → 说明 router 之后有某一步把差异抹掉了（量化？bf16 存储？）。

→ 下一步：**抓真实 `sc.routed` 输入**，分别喂两个 kernel，直接比输出。

## 第 5 轮结论：router GEMM **不是**差异来源（我前面的 43% 测量用错了对照）

在**真实 router 输入**（p02 前向时 `sc.routed` 的实际内容）上，把两个 kernel 都跑一遍直接比：

```
[kaon3] DIAG cuBLAS vs BI: ndiff=0/3072 max|d|=0.000e0 md5=53fc67f37d192829/53fc67f37d192829
[kaon3] DIAG cuBLAS vs BI: ndiff=0/3072 max|d|=0.000e0 md5=ab47dc88750f4936/ab47dc88750f4936
```

**逐位相同。** 两份 md5 是 layer 0 和 layer 1 两次诊断（每层各跑一次）。

### 前面那个 43% 的测量错在哪

我用的是 `torch.matmul(a, w.t())` vs `matmul_persistent(a, w.t())`，随机正态输入。
但 **`torch.matmul` 不必然走我们代码里用的 cuBLASLt 启发式路径**：
- `torch.matmul` 在 2D 形状下可能选 split-k / 不同 tile 的 kernel；
- 我们 `gemm.forward_bf16` 是显式 cuBLASLt + `CUBLASLT_WORKSPACE_SIZE=1` 钉死的算法。

**而且随机正态输入的动态范围比真实 `routed` 大得多**，真实输入下两者恰好都舍到同一 bf16。
→ **"43% 元素不同"是真的（对那个对照成立），但它不代表我们的 cuBLAS 路径与参照不一致。**

**教训（可复用）**：A/B 的**对照必须用真实数据 + 真实调用路径**。
用 `torch.matmul` 代替被测代码自己的 GEMM 封装，会得到一个真实但无关的数字。

### 保留的决定

`router_gemm_bi_f32` **保留**（kernel 级逐位 0/8192，已在真实输入上验证与 cuBLAS 同结果）：
- 它不改变数值，但**不依赖 `CUBLASLT_WORKSPACE_SIZE` 环境变量**，比 cuBLASLt 更可控；
- 默认开（`ROUTER_GEMM_BI`），cuBLAS 路径留作 A/B 与回退。
- **但它不是 flip 的来源**，loop 还得往下找。

## 第 6 轮：`MOE_COMBINE_F32=1` 让 p02 的 top-1 翻转归零（**重大**）

同环境（`ROUTER_GEMM_BI=1 CUBLASLT_WORKSPACE_SIZE=1 RMSNORM_MODE=2`），只切一个开关，p02 单例：

| 开关 | top-1 不一致 | top5 \|Δlp\| | top1 \|Δlp\| max | 实际 token \|Δlp\| |
|---|---|---|---|---|
| `MOE_COMBINE_F32=0`（**当前默认**，号称"vLLM cutlass_moe 口径"） | **4 / 22** | 1.8089 | 2.0405 | 2.1794 |
| **`MOE_COMBINE_F32=1`**（f32 权重 + f32 乘积） | **0 / 22** | **1.3760** | **0.5195** | **1.1844** |
| `NORM_FUSE_QUANT=1` | 3 / 22 | 1.8046 | 2.6193 | 1.7220 |

**判读**：
- `MOE_COMBINE_F32=1` 把 **top-1 翻转从 4 降到 0**，top1 |Δlp| max 从 2.04 降到 0.52（**4 倍**）。
  这是本轮所有尝试里唯一产生质变的开关。
- 默认值（`=0`，"vLLM cutlass_moe 口径"）**是错的**：那个口径假设 combine 时权重先落 bf16；
  参照的 Triton/inductor 路径显然不是。
- `NORM_FUSE_QUANT=1` 有轻微改善但不质变，暂不动。

**下一步**：把 `MOE_COMBINE_F32` 默认改成 1（与 `RMSNORM_MODE=2` 一起），跑完整 30 用例验证。

## 第 6 轮结果（全量 30 用例，run `1024-010201-1778220`）

```
OURS POOLED cases=30 rows=538 decisive=483 flips=68 flip_pct=14.08 mean_common_d=1.2847 dnll=-0.3142
```

| 轮次 | 改动 | flip | \|Δlp\| |
|---|---|---|---|
| 基线 | MODE=1（eager 口径） | 16.56% | 1.1994 |
| 1 | MODE=0 | 16.98% | 1.1816 |
| 3 | MODE=2 | 15.94% | 1.2367 |
| **6** | **`MOE_COMBINE_F32=1`（新默认）** | **14.08%** | 1.2847 |
| FLOOR | — | 0.83% | 0.0276 |

**判读**：
- flip 16.56% → **14.08%**（降 2.5 个百分点，约 15% 相对改善），方向正确但**离 FLOOR 还差 17 倍**；
- p02 单例上 top-1 归零、top1 |Δlp| 2.04→0.52 的改善，在 30 用例聚合上被摊薄成小改善
  → **说明 combine 口径只是众多差异之一，不是唯一主因**；
- `|Δlp|` 反而从 1.1816 升到 1.2847：flip 与 |Δlp| 两个指标不同向，说明改动在"让 top-1 对上"上有效，
  但整体分布仍有系统性偏差。

## 附：抓参照中间量的工具坑（2026-10-24）

想抓参照的**真实 MoE 输入**做逐位核验时，连续三次抓不到东西，原因各不相同：

| 尝试 | 现象 | 原因 |
|---|---|---|
| patch `MoERunner.forward(self, layer, x, topk_weights, topk_ids)` | 无输出、无产物 | **签名不对**：实际是 `(hidden_states, router_logits, input_ids, shared_experts_input)` |
| `nohup … | tail` 包远端 `docker exec` | 日志 0 字节 | `nohup` + 管道 + ssh 三层叠加把 stdout 吞了 |
| 后台任务退出码 0 但无产物 | — | 后台 shell 只负责"启动"，不代表远端跑完 |

**规则**：
1. **先 `inspect.signature` 确认被 patch 的方法签名**，别照记忆写；
2. 远端长任务的输出用 `run_in_background` + 前台同步捕获（`timeout … $R run '… 2>&1 | tail -N'`），
   **不要** `nohup … > log &` —— ssh 断开时 stdout 会丢；
3. 判断"跑完没有"要看**产物**（`ls /w/tmp/…`），不看退出码。

### 抓取一直卡住的真正原因：**在 oracle 容器里跑，GPU 被占着**

前几次抓取全部失败（超时 143 / 无产物），根因不是脚本：
`docker exec kaon-oracle python3 …` 跑在 **GPU 1** 上，而 **oracle 服务本身已经占了 27 GB**（`nvidia-smi` 显示 `585, 27386 MiB`）。
再加载一个 LLM 必然卡死或 OOM。

**正确做法**：用 `GPU=<n> scripts/dev/remote_5090.sh ctr '…'`，它 `docker run --gpus device=$n` 起干净的临时容器
（`ctr` 默认 `GPU=0`；GPU 1 = oracle、GPU 7 = 基线，都别用）。

**教训**：抓参照数据时先看那张卡**有没有别的进程占着**（`nvidia-smi --query-compute-apps`），
`docker exec 进服务容器`听起来最省事，实际上是最容易撞上资源冲突的做法。

### 抓取工具链又两个坑（2026-10-24，别重犯）

1. **`pkill -f "moe_capture.py"` 会杀掉自己**：Bash 工具那条命令行本身含这个字符串，
   `-f` 匹配全命令行 → 自我终止（两次 exit 144）。**链内杀进程用 `pkill -x`**（匹配进程名而非命令行）。
2. **脚本要放在容器可见的挂载里**：`ctr` 把远端仓库根挂成 `/w`，本地 `/tmp/*.py` 在容器里不存在。
   → 抓取脚本入库到 `scripts/dev/moe_capture.py`，容器内用 `/w/scripts/dev/moe_capture.py`。
3. **要看实时输出就写远端文件**（`> /w/tmp/cap.log 2>&1; tail`），
   不要靠 `nohup … &` + 本地 log —— ssh 断开时 stdout 会丢，日志 0 字节。

4. **vLLM 脚本必须有 `if __name__ == '__main__':` 保护**：
   vLLM v1 用 `multiprocessing` spawn 起 EngineCore，子进程会 `runpy.run_path(main_path)` **重新执行主模块**。
   没有保护 → 子进程也去 `LLM(...)` → `RuntimeError`（堆栈起点是 `spawn.py:297 _fixup_main_from_path`）。
   看到这个堆栈就是缺保护，不是 vLLM 装坏了。

5. **`LLM(...)` 的模型在 EngineCore **子进程**里，主进程 hook 够不到**：
   `llm.llm_engine.model_executor` → `AttributeError: 'LLMEngine' object has no attribute 'model_executor'`。
   必须 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 走单进程（`kaon3_vllm_hidden.py` 早就这么做了，我没照抄）。
   同时 `VLLM_BATCH_INVARIANT=1` 要与 oracle 口径一致，否则抓到的是 BI=0 的数值。
