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
