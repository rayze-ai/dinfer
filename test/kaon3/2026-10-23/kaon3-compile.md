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
