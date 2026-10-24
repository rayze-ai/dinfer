# loop 规格：kaon3-fp4dense

- 状态：进行中（2026-10-24 创建）
- 创建：2026-10-24
- 验证脚本：`scripts/verify/kaon3-fp4dense.sh`
- 迭代记录：`test/kaon3/2026-10-24/kaon3-fp4dense.md`
- 上游：`docs/loops/kaon3-compile.md`（已到迭代上限，卡在 dense GEMM 7.8%）

## 目标

把专家 **NVFP4 dense GEMM** 的 7.8% 差异消掉，使端到端 flip 落到 FLOOR 线内。
上限 10 轮；10 轮未通过就转**路线 3（直接绑 cutlass，编译 FP4 MoE kernel）**，不再纠缠。

## 口径（用户 2026-10-24 重申）

- **compile 口径**：参照 = vLLM 默认（不开 `enforce_eager`）；我们侧 `RMSNORM_MODE=0`。
  **不得**用 eager 抓的中间量做判据。
- 环境必须干净：oracle 容器先 `docker rm -f kaon-oracle` 再起；启动后核三项
  （端口归它、cmd 里没有 `num_hidden_layers`、`VLLM_*EAGER*` 不存在）。
- 逐层对拍**必须** `CUBLASLT_WORKSPACE_SIZE=1`（否则测出一整套假发散）。
- 数据类型与计算顺序要盯住：bf16/f32 舍入点、fp32 累加序、`rcp.approx.ftz` vs 真除法。
- **不靠猜**：每一轮先读参照实现（源码 → PTX/SASS），再下判断。

## 环境

- 机器：5090（`scripts/dev/remote_5090.sh`，ControlMaster 已 alive）。cwd=`/w/dinfer`，产物落 `/w/tmp`。
- GPU：0 / 2–6 = 我们；1 = 私有 oracle；**7 = 共享基线，禁用**。
- 编译：
  ```bash
  GPU=0 scripts/dev/remote_5090.sh ctr 'cd /w/dinfer && CUDA_ARCHS=120 cargo build --release -p dinfer-kaon3 --bins'
  ```
- 参照量化/GEMM 探针：`scripts/dev/fp4_dense_probe.py`（产物 `/w/tmp/fp4d`）
- 我们侧对拍：`crates/kernels/tests/fp4_dense_parity.rs`（`FP4_DIR=/w/tmp/fp4d`）
- 重置到干净状态：`scripts/dev/remote_5090.sh run 'docker rm -f kaon-oracle'` + 重跑 oracle 脚本

## 观察方法

| 看什么 | 在哪里 | 命令 |
|---|---|---|
| dense GEMM 逐位差 | `fp4_dense_parity.rs` stdout | `FP4_DIR=/w/tmp/fp4d … cargo test --release -p dinfer-kernels --test fp4_dense_parity -- --ignored --nocapture` |
| 量化自洽（参照自己的参考实现） | 探针 stdout | 用 vLLM 的 `recover_swizzled_scales` / `break_fp4_bytes` / `dequantize_nvfp4_to_dtype` 复算 `y` |
| 端到端 flip | 30 用例整跑 | `scripts/verify/kaon3-compile.sh` |

## 停止条件（可机器判定，见 verify 脚本）

1. `fp4_dense_parity` 报 `ndiff == 0`（逐位相同）。
2. 端到端 `scripts/verify/kaon3-parity.sh` exit 0
   （它已按 FLOOR 基线判 flip ≤ 1.5×0.83%+3pp、`|Δlp|` ≤ 1.5×0.0276、长文本 ΔNLL）。

## 上限

**10 轮**。到上限未通过 → 转路线 3，把卡点与已排除假设写进本文件"卡点"区。

---

## 参照实现已确认的事实（2026-10-24 读源码，可直接用）

来源：`/data/ref/vllm-img-v0.28.0-pr455`（Kaon-V3 的 oracle 就是这个镜像）+
`/data/ref/vllm` main 的 `csrc`（镜像里没有 csrc，只能拿 main 的源码当代理，用前先核对镜像行为）。

### 1. MoE 走的不是 dense kernel

`run_cutlass_moe_fp4`（`vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py:560+`）：
`get_cutlass_moe_mm_data` → `shuffle_rows` → **`scaled_fp4_experts_quant`** → `cutlass_fp4_moe_mm`
→ **`silu_and_mul_scaled_fp4_experts_quant`**（SiLU+Mul 与量化融合）→ `cutlass_fp4_moe_mm` → `shuffle_rows`。
我们探针测的 `ops.cutlass_scaled_fp4_mm`（dense）是**另一个 kernel**，量化器也不同
（`cvt_fp16_to_fp4` vs `scaled_fp4_experts_quant` 的按专家偏移版本）。

### 2. `sf` 的字节布局（两处源码互证，已确认）

- 写入：`csrc/libtorch_stable/quantization/fp4/nvfp4_utils.cuh:164 cvt_quant_to_fp4_get_sf_out_offset`
- python 侧：`vllm/model_executor/layers/quantization/utils/nvfp4_utils.py:13 swizzle_blockscale`

按 **128 行一个 mTile**，块内字节偏移：

```
offset_in_tile = kO*512 + (m&31)*16 + ((m>>5)&3)*4 + (kidx&3)
全局偏移       = (m>>7)*numKTiles*512 + offset_in_tile,  numKTiles = ceil(outputCols/64)
```

即块内 dim 顺序 = `[kO(ceil(scale_n/4))][b=m&31 (32)][a=(m>>5)&3 (4)][c=kidx&3 (4)]`。
等价 reshape：`[mTile][4(a)][32(b)][kO][4(c)]` → permute `(0,1,4,3,2,5)` → `[mTile][kO][32(b)][4(a)][4(c)]`。

### 3. 反量化公式（vLLM 自己的参考实现）

`tests/kernels/quantization/nvfp4_utils.py`（main）：

```python
scale = global_scale * (vec_max / 6)          # 再 cast 到 e4m3
output_scale = 1 / (scale / global_scale)
q = fp4_round(clamp(x * output_scale, -6, 6))
反量化 = q * (sf / global_scale)              # dequantize_nvfp4_to_dtype
```

nibble：`break_fp4_bytes` 用 `(low, high)` 交错 → **低半字节 = 偶数列**（与上轮实测一致）。

### 4. 上一轮的 `1/14.22` 常数是**读回 bug**（已算清，不是猜测）

`scripts/dev/fp4_quant_calib.py` 用 `s[:M, :K//16]` 直接从 `sf.view(uint8)` 的 `(128,176)` 视图里
按**线性行优先**读 `(m, kidx)`，但缓冲区是按第 2 条公式 swizzled 写的、且 padding 区是
`torch.zeros` 初始化的。算一下重合度（`k=2816`）：

```
有效 sf 位置 1408 个；线性读覆盖的位置 1408 个；两者重合只有 96 个
→ 线性读到的 1408 个里有 1312 个是 0 → 均值被稀释到 96/1408 = 0.06818
```

而 `a` 全 1、`gs=1` 时：`sf = e4m3(1/6) = 0.171875`，`output_scale = 1/sf = 5.818`，
`q = fp4_round(5.818) = 6.0`，所以 `q*sf = 6 × 0.171875 = 1.03125`。
**1.03125 × 0.06818 = 0.0703125 —— 与实测的 `q*sf mean=0.070312` 逐位吻合。**

`gs=100`：`sf = e4m3(16.667) = 16`（e4m3 尾数 3 位，1.0417 舍到 1.0），
`q = fp4_round(100/16 = 6.25) = 6.0`，`q*sf = 96`，`96 × 0.06818 = 6.545` ✓（实测 6.545455）。
`gs=462.45`：`sf = e4m3(77.08) = 80`，`q = fp4_round(5.78) = 6.0`，`480 × 0.06818 = 32.73` ✓（实测 32.727）。

**结论**：三行全部由"线性读 + 零稀释"解释干净。反量化的语义就是 `q·sf/gs`，**没有** 1/14.22 常数，
反推出来的 `F=7.11` 是伪影 —— 上一轮"还差一个常数"的整条线索**作废**。

**同时确认**：`fp4_dense_probe.py` 里 `a` 与 `a_q/a_s` **是**同一次调用的产物
（`aq, asq = ops.scaled_fp4_quant(a, …)` 之后 dump 的是同一个 `a`），
交接区怀疑的"配对错位"**不成立**，不用再查。

### 5. 判据要换成 vLLM 自己的参考实现（本轮做法）

不再用自己写的 numpy 复算，改用参照自带的三个函数（`tests/kernels/quantization/nvfp4_utils.py`）：
`convert_swizzled_to_linear` / `break_fp4_bytes` / `dequantize_nvfp4_to_dtype`。
它们与被测 kernel 同仓同版本，是这套语义的**权威口径**。
用法只限于 oracle 侧诊断（不进仓库、不作判据），符合"Rust 实现、Rust 验证"。

## 卡点

（待填）
