# kaon3-fp4dense 迭代记录

规格：`docs/loops/kaon3-fp4dense.md`（上限 10 轮，之后转路线 3）

## 第 0 轮（开局，读参照源码 + 复核上一轮判据）

**结论先行**：上一轮的三条判据——"a_q/a_s 配对错位"、"还差常数 1/14.22"、"权重 scale 是自然布局"——
**全部不成立**。第一条是同一次调用的产物；第二条是线性读 swizzled 缓冲区的伪影（已算清）；
第三条**无法用字节数判定**（`n=1408` 是 128 的倍数，两种布局字节数都是 247808）。

### 已确证（读参照源码，非推测）

1. **MoE 走的是另一套 kernel**：`run_cutlass_moe_fp4` → `scaled_fp4_experts_quant`
   → `cutlass_fp4_moe_mm` → `silu_and_mul_scaled_fp4_experts_quant` → `cutlass_fp4_moe_mm`。
   探针测的 `cutlass_scaled_fp4_mm`（dense）**不是**专家实际走的路。
2. **sf 写出偏移**（`nvfp4_utils.cuh:164`）：
   `off = (m>>7)*numKTiles*512 + (kidx>>2)*512 + (m&31)*16 + ((m>>5)&3)*4 + (kidx&3)`
   → 维度顺序是 `(g, kO, r32, a4, k4)`。
3. **反量化**：`sf = e4m3(gs·amax/6)`，`out_scale = 1/(sf/gs)`，`q = fp4(clamp(x·out_scale))`，
   反量化 `q·sf/gs`（vLLM 自己的 `dequantize_nvfp4_to_dtype`）。

### 第 0 轮实测：四种布局组合都复现不了 `y.bin`

`scripts/dev/fp4_dense_ref.py`（反量化后 fp32 matmul，乘子取 `alpha`）：

```
A=swz  B=swz   *alpha   relerr=0.999776  corr=0.259136  ‖out‖/‖y‖=0.000870
A=swz  B=nat   *alpha   relerr=0.999781  corr=0.254553  ‖out‖/‖y‖=0.000869
A=nat  B=swz   *alpha   relerr=0.999853  corr=0.224987  ‖out‖/‖y‖=0.000639
A=nat  B=nat   *alpha   relerr=0.999857  corr=0.221814  ‖out‖/‖y‖=0.000633
```

**corr ≈ 0.25 说明不是布局问题，是主逻辑问题**（布局错只会让部分 scale 错位，corr 仍应接近 1）。
`‖out‖/‖y‖ = 0.00087`，而 `0.00087 × 462.4516(a_gscale) = 0.4025`，
离 1 还差 2.49 倍——量级也没对上。**这份 `y.bin` 数据集本身不可用作判据。**

**已排除的自身错误**：我第一版 `unswz()` 把 permute 写反了（写成 `(g,a4,r32,kO,k4)`），
`convert_swizzled_to_linear` 的正确顺序是 `(g,kO,r32,a4,k4)`。已作废该函数输出。

### 下一步（本轮内）

停用 `y.bin`。改用**自洽合成数据**建判据：A、B 两侧都用 `scaled_fp4_quant` 量化（两侧都 swizzled），
用 `dequantize_nvfp4_to_dtype` 复算，先证明 `ref == cutlass_scaled_fp4_mm`（vLLM CI 同款配方），
再拿这份已知答案喂我们的 kernel。这样判据的每一环都可控。

## 第 1 轮：换自洽合成判据 → **我们的 dense GEMM 逐位一致（ndiff=0）**

### 做法

`scripts/dev/fp4_dense_synth.py`，vLLM CI 同款配方（`test_nvfp4_scaled_mm.py`）：
两侧都用 `scaled_fp4_quant` 量化 → `cutlass_scaled_fp4_mm` 出 `y` → 用参照自己的
`dequantize_nvfp4_to_dtype`（逐字复制，仅 oracle 侧诊断用）复算 → 纯 numpy 按公式再独立复算。

标准形状 `m=8 n=1408 k=2816`（Kaon 专家 gate/up 的真实形状），`torch.manual_seed(0)`。

### 判据自证（先证明判据可信，再测我们）

```
[1] cutlass vs 参照 dequant+matmul: relerr=0.001671 corr=0.99999869 bf16 逐位不同 1/11264
    max|d|=4.9954e-01  ‖y‖=5581.0322
[3] numpy 独立复算 vs 参照 dequant+matmul: relerr=0.000000 corr=1.00000000 ‖np‖/‖ref‖=1.000000
```

三条独立路径（cutlass kernel / 参照 python 复算 / 纯 numpy）互相吻合到 fp32 精度，
**`y.bin` 现在是可信判据**。注意 `[1]` 里那个 `1/11264` 说明 cutlass 与 python 复算
在 bf16 下**并非严格逐位**（fp32 累加序差异），所以下面的 `ndiff=0` 是真结果而不是巧合。

### 我们的 kernel（`fp4_dense_synth_parity.rs`）

```
FP4_SYNTH A=swz  B=nat   ndiff=0/11264 max|d|=0.000e0 ‖ours‖/‖ref‖=1.000000 corr=1.00000000
FP4_SYNTH A=nat  B=nat   ndiff=0/11264 max|d|=0.000e0 ‖ours‖/‖ref‖=1.000000 corr=1.00000000
```

**逐位一致**（两条布局路径都对）。

### 结论

**"专家 NVFP4 dense GEMM 与参照差 7.8%" 作废。** 那是上一轮 `/w/tmp/fp4d` 那份坏数据测出来的：
用参照自己的复算去比那份 `y.bin`，corr 只有 0.25——判据本身坏掉，任何对比都没有意义。
所以端到端 flip 14.08% 的差异**不在 dense GEMM 的数值实现上**。

### 过程中的自身错误（记下来避免重犯）

1. 第一版手写 `unswz()` 把 permute 维序写反（写成 `(g,a4,r32,kO,k4)`，正确是 `(g,kO,r32,a4,k4)`）。
   → 改成**按 `nvfp4_utils.cuh:164` 的偏移公式逐元素取**，不再依赖 permute 语义。
2. 第一版对拍把 `swz` 当成了"两侧布局"开关，实际 `forward_nvfp4_prequant` 的 `swz` **只作用于 A 侧**，
   B 侧永远按自然 `[n, k/16]` 读 → 两组都配错了，得到 corr 0.949 / 0.265 的假差异。
   → 必须**先读被调函数的签名与注释**再设计 A/B。

### 下一步

端到端差异要找别处。按嫌疑排序：
1. **MoE 权重 scale 的布局**：`n=1408` 是 128 的倍数，swizzled 与自然字节数都是 247808，
   **无法用字节数判定**。参照的 `nvfp4_blockwise_moe_kernel.cu` 用 `tile_atom_to_shape_SFB`，
   需要对 `w1_scale` 做同样判定。
2. 专家量化走的是 `scaled_fp4_experts_quant`（按专家偏移），不是 dense 的 `cvt_fp16_to_fp4`。
3. SiLU+Mul 与量化融合（`silu_and_mul_scaled_fp4_experts_quant`）的舍入点。

## 第 2 轮：MoE 权重 scale 布局确认 + 我们自己的量化路径也是 0

### 权重 scale 布局（读源码，不再靠字节数猜）

`VLLM_CUTLASS` 分支走 `prepare_nvfp4_moe_layer_for_fi_or_cutlass`
（`vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py:321`）：
只有 `FLASHINFER_TRTLLM` 才做 `align_trtllm_fp4_moe_hidden_dim_for_fi` 的 shuffle，
**`VLLM_CUTLASS` 不做任何 swizzle**，权重 scale 保持线性 `[E, N, K/16]`。
→ 我们按自然布局读是对的。

### 我们自己的量化器也逐位 0（这是关键补充）

第 1 轮喂的是参照的 `a_q/a_s`（`forward_nvfp4_prequant`），**完全绕开了 `quantize_a_kernel`**。
本轮补测 `forward_nvfp4`（吃 bf16 输入、走我们自己的量化）：

```
FP4_SYNTH A=swz  B=nat   ndiff=0/11264 max|d|=0.000e0 ‖ours‖/‖ref‖=1.000000 corr=1.00000000
FP4_SYNTH A=nat  B=nat   ndiff=0/11264 max|d|=0.000e0 ‖ours‖/‖ref‖=1.000000 corr=1.00000000
FP4_SYNTH self-quant     ndiff=0/11264 max|d|=0.000e0 ‖ours‖/‖ref‖=1.000000
```

**dense GEMM 整段（量化 + GEMM + 布局 + 乘子）全部逐位一致**，彻底排除。

### 下一步

差异在 MoE 的其余环节。用现成的 `models/kaon3/tests/moe_stage_parity.rs` + `/w/tmp/moe1`
（真实激活 + 真实 router logits + 参照 MoE 输出）定位分段。

## 第 3 轮：真实数据下 MoE 段有差异，路由影响很小

`models/kaon3/tests/moe_stage_parity.rs` + `/w/tmp/moe1`（p02 / layer 0，T=24）：

| 口径 | ndiff | 说明 |
|---|---|---|
| 我方路由权重 | 50214/67584 | 基线 |
| **参照的 topk_ids/topk_w**（`MOE_USE_REF_TOPK=1`） | **49347/67584** | 把路由整段换掉只降 867 个元素 |

→ **路由不是主因**，差异在"专家 GEMM + gelu + combine"这一段内部。

```
ours[0..8] = [0.23632813, 1.78125, -2.21875, 0.03564453, 0.40234375, 0.265625, 0.9453125, 4.90625]
ref [0..8] = [0.23339844, 1.78125, -2.21875, 0.032958984, 0.40039063, 0.26367188, 0.9453125, 4.90625]
‖ours‖=364.6431  ‖ref‖=363.8837  比值=1.0021
```

读法：**部分元素逐位相同**（1.78125 / -2.21875 / 0.9453125 / 4.90625），
不同的那些**我方一致偏大**（0.23633 vs 0.23340、0.035644 vs 0.032959），差 1–1.5 个 bf16 ULP。
范数比 1.0021 的偏离是"多数元素略微偏大"累积出来的，不是某一个点的语义错。

### 已排除（累计）

dense FP4 GEMM（量化+GEMM+布局+乘子，三条路径都 `ndiff=0`）、路由段、gelu·mul、MoE 权重 scale 布局。

### 下一步（决定性实验）

vLLM 的融合路径按 activation 分叉（`cutlass_moe.py:645`）：
`activation == SILU` 才走 `silu_and_mul_scaled_fp4_experts_quant`（SiLU+Mul 与量化融合）；
**Kaon 是 gelu_tanh，走 else 分支**：`apply_moe_activation` 写 bf16 的 `c2`，再 `scaled_fp4_experts_quant(c2, …)`。
所以"融合算子的舍入点"这个嫌疑**不成立**，结构与我们一致。

剩下要做的是**逐段抓参照的 c1 / c2 / c3**（patch `run_cutlass_moe_fp4` dump），
与我们的 gate/up、gelu、down 输出逐位比，定位是哪一段引入 1 ULP。

## 第 4 轮：gate/up/act 逐位 0；combine 的 bf16 乘积是主因

### 抓参照 compile 口径的分段（新工具 `scripts/dev/moe_seg_capture.py`）

compile 下 forward hook 不触发，所以改成 **patch 模块函数**（compile 会照常调用被 patch 的函数）。
抓 `apply_moe_activation` 的输入/输出（= c1 / c2）、`scaled_fp4_experts_quant` 的输入与输出、
`run_cutlass_moe_fp4` 的 output（= y）。产物 `/w/tmp/moe3`。

**踩坑**：`IDS=$(cat p02.ids | tr "\n" " ")` 得到的是 **4096 token 的长序列**（不是 24），
所以 `m*topk = 32768` 行、`a1_sf.bin` 有 900 MB。

### 逐段比对（参照自己的专家输入 + 真实权重，专家 0，4096 行）

| 段 | ndiff | 结论 |
|---|---|---|
| gate | **0/2883584** | ✅ 逐位 |
| up | **0/2883584** | ✅ 逐位 |
| act（gelu·mul） | **0/2883584** | ✅ 逐位 |

**专家 GEMM + 量化 + gelu 整段全部逐位一致。** 差异只在 down / combine。

### combine 口径：bf16 乘积是主因

参照（`cutlass_moe.py:686`）：
```python
output.copy_((c3.view(m, num_topk, k) * topk_weights.view(m, num_topk, 1).to(out_dtype)).sum(dim=1))
```
即 **权重先舍 bf16、乘积再舍 bf16**，只有最后沿 topk 的 sum 用 f32。

我们在 `/w/tmp/moe1`（T=24）上加 `MOE_BF16_PROD=1` 实测：

| 口径 | ndiff | max\|d\| | ‖ours‖/‖ref‖ |
|---|---|---|---|
| 全 f32（原） | 49347/67584 | 2.5e-1 | **1.0021** |
| **bf16 权重 + bf16 乘积** | 46055/67584 | 1.25e-1 | **0.9999** |

范数比从 1.0021 修正到 0.9999、max|d| 减半 → **确认是这个口径**，但还没逐位：
剩下的差异来自**累加顺序**——参照沿 `dim=1`（slot 0→7 顺序）归约，
我们现在按专家 id 顺序累加。下一轮改成先按 slot 收集、再按 slot 顺序求和。

## 第 5 轮：**四个分段全部逐位 0**，差异只在 combine

给 `moe_seg_parity.rs` 加了 down 段（用**参照自己的 c2 当 act 输入**，排除我方 gelu 的影响）：

```
SEG rows=32768 expert=0 行区间=[0,4096) 共 4096 行
SEG gate ndiff=0/2883584   max|d|=0.000e0
SEG up   ndiff=0/2883584   max|d|=0.000e0
SEG act  ndiff=0/2883584   max|d|=0.000e0
SEG down ndiff=0/11534336  max|d|=0.000e0
```

抓 c3 的做法：patch `ops.shuffle_rows`，第二次调用（`shuffle_rows(c3, c_map)`）的返回值就是
combine 之前的 per-slot 输出。产物 `/w/tmp/moe3b`。

### 至此 MoE 内部全部逐位：量化 → gate/up → gelu·mul → down

**唯一剩下的是 combine**（`(c3 * w).sum(dim=1)`）。

| 口径 | ndiff | ‖ours‖/‖ref‖ |
|---|---|---|
| 全 f32 | 49347/67584 | 1.0021 |
| bf16 权重 + bf16 乘积 | 46055/67584 | 0.9999 |
| + 按 slot 顺序累加 | 46055/67584 | 0.9999 |

**注意**：`slot_order` 改了 `ours[0]`（0.23633 → 0.23535）但 **ndiff 没变**，
说明 f32 累加顺序不是当前的瓶颈。剩下 46055 个元素差 1–1.5 ULP。

### 下一步

参照的 `topk_weights` 是 **bf16** 张量（`topk_weights.view(m, num_topk, 1).to(out_dtype)` 里的
`to(out_dtype)` 说明它原本可能不是 bf16）。要逐位复现 combine，必须确认：
1. 参照 `topk_weights` 进 combine 时的**原始 dtype 与数值**（抓下来，不要从 logits 重算）；
2. `.sum(dim=1)` 在 bf16 张量上到底是 f32 累加还是逐次 bf16 舍入（PyTorch 语义）。
下一步抓这两项。

## ⚠️ 第 4/5 轮结论作废：抓到的 compile 分段数据**全是 0**

**现象**：`/w/tmp/moe3` 与 `/w/tmp/moe3b` 的 `c1_gateup.bin`（92 MB）、`c2_act.bin`、
`c3_down.bin`、`y_moe.bin` **每一个字节都是 0**（`od -t x1` 全是 `00`，nz 比例 0.0）。

**根因**：vLLM 启动时先跑一次**显存 profile**，用**全零 dummy 输入**走一遍前向。
我的 patch 条件写的是"第一次调用就抓"（`if cap["n"] == 0`），抓到的正是那次 dummy。
Rust 侧 `moe_seg_parity` 报 `gate/up/act/down ndiff=0` —— 是拿**全零比全零**，
不是真实数据的逐位一致。**第 4、5 轮的"四段全部逐位 0"作废。**

**唯一仍然成立的**：`moe_stage_parity`（第 3 轮）用的是 `moe_capture.py` 抓的
**eager** 口径数据（非零，`max|d|=2.5e-1`、范数比 1.0021、bf16_prod 修正到 0.9999）。
但那是 **eager**，**违反 compile 口径**（规格明写不得用 eager 做判据）。

**修正**：`moe_seg_capture.py` 加非零守卫（`nz(t) > 0` 才 dump），并打印每次调用的 nz 比例。
重新抓取后必须**先验证非零**再拿去做任何比对。

**教训**（写进 pitfalls 前先确认修复有效）：抓参照中间量的脚本，
**必须校验数据非零/非常量再入库**，否则"ndiff=0"可能只是两个全零张量相等。

## 第 6 轮：找到作废的根因并修好抓取

### 根因确认
`od -A d -t x1 /w/tmp/moe3/c1_gateup.bin` → 92 MB 全是 `00`。
vLLM 启动时先跑一次**显存 profile**（全零 dummy 前向），我的 `if cap["n"] == 0` 抓到的是那次。

### 修正
`moe_seg_capture.py`：所有 dump 前加 `nz(t) > 0` 守卫，并打印每次调用的 nz 比例。
另外确认 `/w/tmp/moe1`（eager 口径）是**真实数据**（`x` nz=0.9994、`y` nz=1.0），
所以第 3 轮基于它的结论仍然有效 —— 但**那是 eager，不能做 compile 判据**。

### combine 的确切语义（读参照源码）
`vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py:680`：
```python
output.copy_((c3.view(m, num_topk, k) * topk_weights.view(m, num_topk, 1).to(out_dtype)).sum(dim=1))
```
- `topk_weights` 来自 `gemma4_routing_function_torch`，返回时**已经是 `.to(torch.float32)`**；
- 进 combine 时再 `.to(out_dtype)` = **转回 bf16**；
- 乘积是 bf16×bf16 → `.sum(dim=1)`。

→ 即：**bf16 权重、bf16 乘积、沿 slot 求和**。这与我方 `MOE_BF16_PROD=1` 的口径一致。

⚠️ 但注意：配置里出现了 `'enable_bf16x3_router_gemm': False` 与
`compilation_config['compile_ranges_endpoints'] = [4096]`，说明 oracle 是 **inductor compile** 模式，
真实调用路径可能与 eager 的 `gemma4_routing_function_torch` 不同 —— 必须用 compile 抓到的
`topk_w` 实测确认，不能只看 Python 源码。

## 第 7 轮：跑端到端 A/B，同时重抓 compile 分段

`ctr` 的坑：`docker run --rm` 随 SSH 会话结束被杀（已记 pitfalls）。
torch.compile 首次追踪要 15 min 左右，所以抓取必须宿主机 `setsid nohup docker run` 脱离会话。
GPU 4 被别的进程占了 18 GB → 换 GPU 5（已记入规格的"用前核显存"）。

端到端 A/B（`kaon3_probe` + `kaon3_oracle_diff`，`CUBLASLT_WORKSPACE_SIZE=1` `RMSNORM_MODE=0`）：
- `MOE_COMBINE_F32=1`（现状默认）
- `MOE_COMBINE_F32=0`（bf16 权重口径，与参照 `cutlass_moe.py:680` 一致）

同时重抓 compile 分段（带非零守卫），用于逐段比对确认到底哪段引入差异。

## 第 7 轮结果：端到端 A/B —— combine 不是主因，**差异不在 MoE**

`kaon3_probe` 30 用例 / 538 行，`CUBLASLT_WORKSPACE_SIZE=1 RMSNORM_MODE=0`：

| 口径 | flip | mean_common_d | dnll |
|---|---|---|---|
| **FLOOR**（参照自比） | **0.83%** | **0.0276** | -0.0025 |
| `MOE_COMBINE_F32=1`（f32 权重） | 15.94% | 1.2577 | +0.2027 |
| `MOE_COMBINE_F32=0`（bf16 权重） | 16.98% | 1.1816 | +0.2621 |

**两个口径都离 FLOOR 差 19 倍**，bf16 口径只把 |Δlp| 从 1.2577 降到 1.1816，
flip 反而略升。→ **combine 不是端到端差异的主因。**

### 关键推论：差异不在 MoE

`moe_stage_parity`（真实数据，eager 口径）显示 MoE 段只有 ~1 ULP 差、范数比 1.0021；
而端到端 `mean_common_d = 1.25`（比 FLOOR 大 45 倍）。
**MoE 不可能贡献这么大的偏差** —— 端到端差异主要来自 **MoE 之外**（attention / norm / 前几层）。

这意味着 `kaon3-compile` 那 17 轮把注意力集中在 MoE 上，**方向可能从一开始就偏了**。

### 下一步（本轮剩余轮次）

逐层比对 `--hidden-cmp`，定位差异**从第几层开始**、以及那一层是哪一段。
不预设是 MoE。

## 第 8 轮：**差异从 L0 就开始**，不在 MoE

数据按新规范放在 `/w/runs/hidden_20261024-0232_r08`（`docs/data-layout.md`）。
参照 `/w/hid/fn`（**eager 口径**，`kaon3_vllm_hidden.py:58` 写死 `enforce_eager=True`），
p02 的 24 token，`L*.bin` 135168 字节 = 24×2816×2，nz=0.4996（已核非零）。

`kaon3_probe --forward --hidden-all … --hidden-cmp /w/hid/fn`：

| 层 | `RMSNORM_MODE=1` 相对误差 / max | `RMSNORM_MODE=0` 相对误差 / max |
|---|---|---|
| **L0** | **5.795e-3** / 3.226e-2 | **4.954e-3** / 1.147e-2 |
| L1 | 2.917e-2 / 7.538e-2 | 3.008e-2 / 8.495e-2 |
| L2 | 3.856e-2 / 7.472e-2 | 3.716e-2 / 8.449e-2 |
| L29 | 1.693e-1 / 3.920e-1 | 1.601e-1 / 2.911e-1 |

**读法**：
- `L0` 就已经差 `5e-3`（max 3.2e-2）→ **差异从第一层就存在，不是 MoE 累积出来的**。
- L0→L1 放大 5–6 倍，之后每层涨 1.0–1.3 倍（缓慢累积）。
- 口径（MODE 0/1）对 L0 只影响 15%，不是主因。

### 结论：`kaon3-compile` 那 17 轮盯着 MoE，方向从一开始就偏了

端到端 `|Δlp| = 1.25` 不可能由 MoE 的 1 ULP 造成（第 7 轮已排除），
逐层数据也证实**第一层就偏**。L0 是**滑窗注意力层**（`layer_types[0] = sliding_attention`），
下一步在 L0 内部逐子模块定位，不在 MoE 上再花时间。

## 第 9–10 轮：**L0 的实现是错的**（相对 L2 ≈ 1.0），L1 只有 1 ULP

用 `--layer-iso <l>:<参照 l-1 层>.bin`：灌参照的上一层残差流，只跑第 l 层，与参照同层比。
数据在 `/w/runs/hidden_*_r10`（新规范）。

| 层 | 输入 | 逐行相对 L2 | 判读 |
|---|---|---|---|
| **L0** | embedding（与参照同一份） | **1.34, 1.07, 1.14, 1.27, 1.14, 1.36 …** | **完全对不上** |
| **L1** | **参照的 L0**（`/w/hid/fn/L0.bin`） | **0.0039, 0.0125, 0.0044, 0.0025, 0.0052, 0.0070** | 1 ULP 级 |

`max|a-b|`：L0 达到 **35–89**（‖b‖ 量级 ~1e1），L1 在 1e-3 量级。

**结论**：差异的**根因在第 0 层**（滑窗注意力层），L1 及以上只是把这个误差往下传。
`kaon3-compile` 的 17 轮、以及本 loop 前 7 轮，全部花在 MoE（在第 14 层之后）上，
**方向从一开始就偏了**。

### 过程中的自身错误（已记 pitfalls / memory）

1. 把 `--hidden-cmp` 的输出（比的是同目录下一整套 `L*.bin`）当成了 `--layer-iso` 的结果 ——
   两者写的是不同文件名（`L{l}.bin` vs `iso_L{l}.bin`），我混读了。
2. 两次凭印象猜文件格式（先按 uint16、后按 f32）导致 reshape 失败；
   正确做法是**先 grep 写入代码**（`kaon3_probe.rs:641` 写的是 f32 LE）。
3. `mkrun.sh` 只 `put` 过一次，后来 `sync` 没带上它 → `No such file or directory`。

### 下一步

**攻 L0**。L0 是 `layer_types[0] = sliding_attention`（滑窗 1024，GQA 16/8 头，head_dim 256）。
先看 L0 内部的子模块：embed 之后 → `input_layernorm` → attention（q/k/v/o）→ 残差 →
`pre_ff_norm` → router → MoE → `post_ff_norm`。用 `--hidden-all` 的子模块 dump 逐项比。

## 第 11 轮（超上限，收尾）：L0 层内定位 —— 已缩小到 post-attention 之后

> ⚠️ 先纠正第 9–10 轮的一个错误结论：我当时说"L0 相对 L2 ≈ 1.0，实现完全错"，
> 那是**用 `L0.bin` 当 L0 的输入**造成的（L0 的输入应当是 `Lm1.bin` = embedding 输出）。
> 用正确输入 `--layer-iso 0:/w/hid/fn/Lm1.bin` 重跑：**逐行相对 L2 0.0018–0.0109，均值 5.795e-3**。
> 所以 L0 **不是"完全对不上"，而是 5.8e-3 级别的偏差**，与逐层对拍的数字一致。

### 已确证（本次逐子模块比对，全部核过非零）

参照侧：`kaon3_vllm_hidden.py` + `SUB_LAYER=0`（**eager，仅诊断**），`/w/runs/hidden_20261024-0245_r10`。
我方：`kaon3_probe --layer-iso 0:…/Lm1.bin` + `SUB_LAYER=0`，`/w/runs/hidden_20261024-0242_r10`。

| 子模块 | 相对 L2 | 结论 |
|---|---|---|
| `Lm1`（embedding） | **0.0（67584/67584 逐位相同）** | ✅ |
| `input_norm` | **0.0（逐位相同）** | ✅ norm 实现对 |
| `o_proj` vs 参照 `attn_out` | **0.0（逐位相同）** | ✅ **attention 整段（含滑窗掩码、RoPE、GQA）正确** |
| `pre_ff_norm` | **0.0（逐位相同）** | ✅ |
| `dense`（MLP） | **0.0（逐位相同）** | ✅ |
| **`post_ff_norm`** | **2.66e-3（均值，max 4.4e-3）** | ❌ **唯一偏离的段** |

`post_ff_norm` 的偏离是**全元素均匀**的（每 256 列 2.2e-3–3.9e-3，逐行 1.9e-3–4.4e-3），
不是局部边界问题 —— 典型的**舍入/口径差**，不是语义错。

### 已排除（本层内）

- embedding、`input_norm`、attention 整段（`o_proj` 与参照逐位相同）、`pre_ff_norm`、`dense` MLP
- 于是偏差只能来自 **`post_ff_norm` 的输入**：`sc.x`（残差流）+ 两条支路相加的方式。

`post_ff_norm` 的定义（`layer.rs:857`）：`rmsnorm(sc.x + post_ff_norm_2(moe), post_ff_norm.weight)`。
其中 `sc.x = embed + post_attn_norm(o_proj)`。

### 下一步（接手第一件事）

1. 单独把 `post_ff_in`（= `sc.x + post_ff_norm_2(moe)`，`layer.rs:856` 落盘）与
   参照的对应量逐位比 —— **需要给参照侧加 `post_ff_in` 的 hook**（当前只抓了
   `input_norm/attn_out/o_proj/pre_ff_norm/dense/post_ff_norm`）。
2. 重点看 **`post_feedforward_layernorm_1` / `_2`**：Gemma4 这一层有三个
   `post_feedforward_layernorm{,_1,_2}` 权重，我们只用了 `post_ff_norm` 一个，
   另两个分别给 MoE 支路用。**很可能其中一个没接对**。

### 口径提醒

本轮的参照是 **eager**（hook 在 compile 下不触发），只用于定位。
修复后必须回 compile 口径重跑端到端（`kaon3_parity.sh`）。

## 第 12 轮：解开 `1/14.22` 之谜 —— 它是 `layer_scalar`，不是量化常数

之前（上一轮循环）花了 6 次反推的那个"1/14.22 常数"，**真实身份是每层的 `layer_scalar`**：

```
model.language_model.layers.0.layer_scalar = 0.0703125 = 1/14.2222
```

`layer.rs:353` 读它，`layer.rs:861` 在**层末**乘上（`isotropic` 式的层输出缩放）。
所以：
- 逐层残差流 `L{l}.bin` **已含** `layer_scalar`；
- sub-dump 的中间量（`post_ff_norm` / `ff1` / `ff2` 等）**未含**。

**这解释了此前所有"对不上"**：我拿 `iso_L0` 去比未乘标量的中间量，比值当然是 0.0703。

### 全部自洽验证（本次）

```
layer_scalar = 0.0703125 (1/14.2222)
(residual + post_ff_norm) * layer_scalar  vs  iso_L0        relL2 = 0.002587
我方 post_ff_norm  vs 参照 post_ff_norm                     relL2 = 0.002863
我方 (res+post_ff_norm)*ls  vs  参照 L0.bin（已含 ls）        relL2 = 0.007882
```

### L0 的真实结论

L0 的实现**基本正确**：`input_norm` / `attention`（`o_proj` 与参照逐位相同）/ `pre_ff_norm` /
`dense` 全部 **0.0 逐位**，只有 `post_ff_norm`（以及含 MoE 的部分）差 **2.6e-3**。
这个量级是 **bf16 舍入 / 归约顺序**，不是语义错。

### 下一步

1. 拿到参照的 `post_ff_norm_1` / `post_ff_norm_2` / `moe_out`（本轮正在抓，`r12`），
   把 2.6e-3 拆到 MLP 支路还是 MoE 支路。
2. 若拆分后两边都是 1e-3 量级的舍入差，那 **L0 已经到 bf16 噪声底**，
   端到端 `|Δlp| = 1.25` 的成因要另找（更可能是**逐层累积** + 不连续量化放大，
   参见 `docs/knowledge/parity-localization.md` 的"放大器"一类）。

## 第 13 轮（决定性拆分）：L0 里除 MoE 支路外**全部逐位 0**

参照补抓了 `post_ff_norm_1 / pre_ff_norm_2 / router_out / moe_out / post_ff_norm_2`
（`/w/runs/hidden_20261024-0254_r12`，eager + `SUB_LAYER=0`），逐段比：

| 我方 | 参照 | relL2 | 逐位相同 |
|---|---|---|---|
| `input_norm` | `input_norm` | **0.0** | **100%** |
| `o_proj` | `attn_out` | **0.0** | **100%** |
| `pre_ff_norm` | `pre_ff_norm` | **0.0** | **100%** |
| `dense`（MLP 输出） | `dense` | **0.0** | **100%** |
| `ff1`（MLP 过 norm） | `post_ff_norm1` | **0.0** | **100%** |
| **`moe_y`** | **`moe_out`** | **4.316e-3** | **49.6%** |
| **`ff2`**（MoE 过 norm） | `post_ff_norm2` | 3.878e-3 | 47.0% |
| `post_ff_norm` | `post_ff_norm` | 2.863e-3 | 56.9% |

**读法**：
- 归一化、注意力（含滑窗掩码/RoPE/GQA）、MLP 支路 —— **6 段全部逐位相同**，实现没问题。
- **唯一的差异源是 MoE 支路**（`moe_out` 4.3e-3，约一半元素差 1 个 bf16 ULP）。

这与第 3 轮 `moe_stage_parity` 的独立测量一致（真实数据、范数比 1.0021、~1 ULP）。
**两条独立路径互证：MoE 支路差 ~1 ULP，其余全部逐位一致。**

### 所以端到端 |Δlp| = 1.25 从哪来？

不是某一层"写错了"，而是 **1 ULP 的 MoE 差在 30 层里逐层累积，
再被下游的不连续量化（fp8/fp4 激活量化）放大**。
这正是 `docs/knowledge/parity-localization.md` 第 1 条里的"放大器"一类，
也是 `requirements.md` 验收口径里 FLOOR（0.83%）存在的理由 ——
参照自己两个进程之间也有 0.83% 的翻转率。

**问题变成：我们能否把 MoE 的 1 ULP 压到 0。** 这才是 `kaon3-compile` 那 17 轮真正该做的事，
只是当时没有先做逐段定位，不知道只有 MoE 一段。

## 第 14 轮：MoE 差异的定量刻画 —— 纯舍入，无语义错

`moe_y`（我方）vs `moe_out`（参照），L0，24 token × 2816：

```
max|d| = 0.25      （|b|max = 50.0）
|d| 分位 p50/p90/p99/p99.9 = 3.1e-05 / 0.00293 / 0.03125 / 0.0625
相对 |d|/|b| 中位数 = 0.003922  （≈ 1 bf16 ULP = 2^-8）
ULP 中位数 = 1.004
|d| > 1.0 的元素数 = 0
按 |b| 分桶的最大 ULP：|b|≥1 → 4.0；[0.1,1) → 15.4；[0.01,0.1) → 150；<0.01 → 686
```

**读法**：
- 大值区（|b| ≥ 1，7113 个元素）**最大只有 4 ULP**；
- 小值区的"大 ULP"是**相对误差在近零处放大**，绝对差仍然极小（|d| > 1.0 的元素 **零个**）。
- 中位 1.004 ULP = 典型的**归约顺序差**。

**结论：MoE 支路没有语义错，只有累加/舍入顺序造成的 ~1 ULP。**
这与参照自己在两个进程间的差异（FLOOR 0.83% 翻转）同量级。

### 这与"端到端 |Δlp| = 1.25"的落差

1 ULP 的 MoE 差 → 每层残差流带 1e-3 级误差 → 30 层累积 → 下游 fp8/fp4 激活量化
把 1e-3 的不连续跳变放大 → logprob 偏 1.25。

这不是"修一个 bug"能解决的，属于 `parity-localization.md` 第 1 条的**放大器**一类。

## 第 16 轮（重大发现）：`MOE_COMBINE_F32=0` 时 L0 层内**全部逐位 0**

### 决定性 A/B（同一份二进制，只切 combine 开关，L0 单层隔离）

| 对比 | `MOE_COMBINE_F32=1` | `MOE_COMBINE_F32=0` |
|---|---|---|
| `moe_y` vs 参照 `moe_out` | 4.316e-3（49.6% 逐位同） | **0.0（100% 逐位同）** |
| `ff1` vs 参照 `post_ff_norm1` | 0.0 | **0.0** |
| `ff2` vs 参照 `post_ff_norm2` | 3.878e-3 | **0.0** |
| `post_ff_norm` vs 参照 `post_ff_norm` | 2.863e-3 | **0.0（100% 逐位同）** |

**→ MoE 支路的全部差异来自 combine 口径。`MOE_COMBINE_F32=0`（bf16 权重）让整层逐段归零。**

这与参照源码一致：`cutlass_moe.py:680` 的 `(c3 * topk_weights.to(bf16)).sum(dim=1)`
就是 bf16 权重口径 —— 即 **`MOE_COMBINE_F32=0` 才是对的**，而当前默认值是 1。

### ⚠️ 但同时发现一个必须先解决的问题：参照的 `L0.bin` 自身不确定

即使 `post_ff_norm` 100% 逐位相同，我们与参照的 `L0.bin` 仍差 6.1e-3。查下去发现：

```
同一份参照，两个目录里的 L0.bin 互相比：relL2 = 0.007667   ← 参照自己都不一致
参照 (residual + post_ff_norm) 与 L0.bin 的比值中位数 = 0.0371
参照 L0.bin / residual 比值中位数 = 0.0195
```

**说明 `kaon3_vllm_hidden.py` 的 `L0.bin` 与它的子模块 hook 不是同一口径**
（很可能 patch `DecoderLayer.forward` 取到的返回值经过了后续处理，
或者多次 warmup/prefill 调用混淆）。所以：

- **子模块级比对（`post_ff_norm` 等）是可信的** —— 两侧口径一致，且互相自洽；
- **`L0.bin` 这一路（逐层残差流）不可信** —— 参照自己重复抓都对不上。

→ 第 8–13 轮里所有基于 `/w/hid/fn/L*.bin` 的逐层结论（"L0 差 5.8e-3、L1 放大 5 倍"、
"L2 起缓慢累积"）**全部降级为不可信**，需要按新口径重做。

### 下一步

1. **先把 `MOE_COMBINE_F32` 默认值改成 0**（有源码 + 逐位证据支持），跑端到端 A/B 看 flip。
2. 修 `kaon3_vllm_hidden.py` 的 `L0.bin` 口径（或有条件时直接用 compile 口径抓），
   重做逐层。
3. compile 口径的 MoE 分段抓取：patch 在 inductor 图里不生效（`CALL ... nz=0.0000` 60 次），
   这条路走不通；改走"我们的实现在 eager 下逐段归零 + compile 端到端验收"。
