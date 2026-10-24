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
