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
