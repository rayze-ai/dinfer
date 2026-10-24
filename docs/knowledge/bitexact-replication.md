# 逐位复刻参照算子（bit-exact replication）

> 与 `parity-localization.md` 的分工：那页讲"差异在哪"，这页讲"定位到某个算子之后，怎样写到与参照逐位相同"。
> 证据全部来自 Kaon-V3 对 vLLM（`VLLM_BATCH_INVARIANT=1`，sm_120），`test/kaon3/2026-10-23/D-16-REBASE.md` §51–§55。

## 0. 先证明逐层参照与端到端判据是同一口径（2026-10-23，最贵的一课）

**判断**：抓逐层参照的进程，必须先用它直接出 `prompt_logprobs` 对端到端 oracle，**0 差异**才往下做。
**证据**：Kaon-V3 对 eager vLLM 30 层逐位 0，端到端纹丝不动——oracle 服务是 compile 模式，vLLM 自身 eager↔compile 差 |Δlp| 1.57（§58）。
**怎么用**：抓取脚本的 LLM 参数逐项抄服务（含**是否 enforce_eager**、max_num_seqs、batched tokens、block、BI）；hook 依赖 eager 时改用截层 + 二分。

## 1. 真值是参照实际执行的机器码：源码 → PTX → SASS，逐级下钻

**判断**：源码只能用来猜结构，PTX 能看出顺序，**只有 SASS 是真值**。
**为什么**：Kaon-V3 的 RMSNorm 连写四版：照源码 `sum_sq / n_cols` + `1/sqrt` 写（§52）、照 PTX `div.full` + `sqrt.approx` 写（§55 第一版），
都在新数据上被推翻。SASS 才显示 `div.full(ss,n)` 被拆成 `RCP(n)` 并和 `+eps` 融合成一条 `FFMA`，`div.full(1,rms)` 是 `RCP`。
Triton 用的是**自带 ptxas**（`triton/backends/nvidia/bin/ptxas{,-blackwell}`），和我们 nvcc 的 ptxas 不同，**同一段 PTX 会编出不同 SASS**。
**怎么用**：
- `TRITON_CACHE_DIR=<dir>` 跑一次参照算子 → 取 `<dir>/*/<kernel>.ptx` 和 `.cubin`；
- 用 Triton 自带 `nvdisasm`（容器里没有系统版）反汇编 cubin，看 `MUFU` / `FFMA` / `FMUL` 链；
- 我们这边**每一步都写显式 PTX 指令**（`rcp.approx.ftz` / `sqrt.approx.ftz` / `__fmaf_rn` / `__fadd_rn`），不赌编译器怎么合并。

## 2. 编译器调度也会改累加顺序

**判断**：线程内归约的顺序要看 PTX，不能按"元素 0..7"假设。
**证据**：Triton 的 8 元素平方和实际顺序是 `mul(e1)` → `fma(e0)` → `fma(e2..e7)`；按 0..7 写，3 行的 inv 偏 1 ulp。
warp / block 的结构（每 warp 蝶形 → 4 warp 蝶形 → 块间顺序加）也必须照搬，否则只是碰巧对上。

## 3. 观测参照时，探针本身会改变结果

**判断**：在参照 kernel 里多存一个中间量，编译器的合并行为就会变，结果跟着变。
**证据**：复制 Triton RMSNorm kernel，把 `rms` 存出来，`inv` 就从 0.8536648 变成 0.8536649（不存的版本里 RCP 和 SQRT 的合并方式不同）。
**怎么用**：探针只能存**最终用到的量**（ss、inv）；需要看中间量时，以 SASS 为准，不以探针输出为准。

## 4. 用"允许区间"反推，先排除输入问题

**判断**：某行输出不一致时，枚举标量（inv、ss）的 ulp 偏移，看是否存在一个值能让**整行**全部一致。
- 存在 → 输入没问题，差在标量算法；区间端点能告诉你偏差有多大；
- 不存在 → 输入就不一样，先去查数据来源。
**补充**：拼接来的输入（别的脚本抓的层输出、上游 hook 的输出）要和**本算子 hook 的 `inp[0]`** 逐位比一次，再下结论。

## 5. 先确认参照走的是哪条路径，并用参照自己的函数复现

**判断**：把真实输入喂给参照自己的函数（`rms_norm_batch_invariant`、`F.linear`），看能否复现前向的真值。
**证据**：
- 独立进程里 `F.linear` 和前向的 router 差 748，因为没开 BI；按 BI 的设置（`preferred_blas_library("cublaslt")`
  + 禁降精度归约 + `CUBLASLT_WORKSPACE_SIZE=1`）就为 0（§51）；
- 前向的 RMSNorm 是 BI Triton kernel，不是 `forward_native`（后者先舍入 x·inv，会差 719）；vLLM C 版 `rms_norm` 会错在和我们一样的位置。
**怎么用**：不要拿"离线单独调一次 F.linear"当参照，那个进程的全局开关和服务不同。

## 6. 第三方库 algo：短描述不够，要看完整属性

**判断**：cuBLASLt 的 `id/tile/splitk` 相同，累加序仍可能不同；关键是 `REDUCTION_SCHEME`（OUTPUT_TYPE = 部分和先舍入到输出类型）。
**证据**：`id=21 tile=11 splitk=8` 有两个候选，`red=4` 差 801/3072，`red=2` 为 0（§51）。`Gemm::algo_desc_full` 会打印全部属性。

## 7. attention：同一个数学式，指数参数写法不同就会错

**判断**：`ex2(s·log2e − m₂)` 和 `ex2((s − m)·log2e)` 舍入不同；Triton `tl.exp(S − m)` 是后者。
**证据**：v6 改成自然域后，Kaon-V3 L4 隔离从 4457 降到 1（§54）。默认已开（`ATTN_MODE` bit1）。

## 反推参照的数值口径时：先造已知输入，别在真实数据上试错（2026-10-24）

复刻 NVFP4 MoE 时，我为了搞清"块缩放/乘子怎么组合"，在**真实数据**上反复调 numpy 公式，
一共猜错 6 次（permute 方向 4 次、nibble 顺序 1 次、乘子 1 次），每轮都要重新跑一遍验证。

**正确的做法**：构造**已知答案**的极小输入，一次把语义钉死。
```
A 的 fp4 全 1.0（编码 0x22）、A/W 的 scale 全 1.0（e4m3 0x38）
→ 理论输出 = k · 1 · 1 · alpha
只要换几个 alpha 各跑一次，就能看出 alpha 作用在哪一层
```
（`scripts/dev/fp4_calib.py`）

**通用流程**：
1. **先定标**：极小形状 + 全 1 输入 + 已知解析解，确认"谁是乘子、作用在哪一步"；
2. 再用真实数据验证；
3. 最后才改自己的实现。

**判据技巧（本轮用到、有效）**：
| 现象 | 含义 |
|---|---|
| 逐行/逐列相对误差是**同一个常数** + 相关性高 | 乘子错了一个固定因子（最容易修） |
| 相对误差随机分布、量级不大 | 累加次序 / 舍入差异 |
| 量级差几倍 | 语义/公式错 |

**已知易错点清单（NVFP4）**：
- e2m1 的 nibble 顺序：**低半字节 = 偶数列**；
- 块缩放 `a_s` 在 cutlass 里是 **padded+swizzled**（`[round_up(M,128), round_up(K/16,4)]`），
  权重侧 `s1` 却是**自然布局** `[N, K/16]`；
- `alpha = weight_scale_2 / a1_gscale`（`a1_gscale = 1/input_scale`），
  且 `process_weights_after_loading` 会 in-place fuse 激活 scale。

## 极小规模定标的正确姿势（2026-10-24 实测有效）

**问题**：反推参照算子的数值口径（乘子在哪、scale 怎么映射），在真实数据上试错会一直猜错。
本轮在真实数据上猜错 6 次；换成极小规模定标，**一次就钉死了乘子**。

**设计**（以 NVFP4 dense GEMM 为例）：
```python
# A 的 fp4 全 1.0（编码 0x22）、A/W 的块缩放全 1.0（e4m3 0x38）
# → 理论输出 = k · 1 · 1 · alpha
for alpha in (1.0, 2.0, 0.5):
    out = ops.cutlass_scaled_fp4_mm(...)
    print(alpha, out[0,0], out[0,0]/alpha)
```
**结果**：
```
alpha= 1.00  y= 64.0000   y/alpha= 64.0000     (k=64)
alpha= 2.00  y=128.0000   y/alpha= 64.0000
alpha= 0.50  y= 32.0000   y/alpha= 64.0000
```
→ `alpha` 是**唯一乘子、作用在最后**，一次确定，无需再猜。

**第二个探针（排布映射）**：只让第 0 个 16-元素块非零、四个块缩放取 `1/2/4/8`，
读 `y/(16·alpha)` 就知道"第 0 块实际取到哪个 scale"——这能直接验证 swizzle 映射对不对，
比"解析真实数据看像不像"可靠得多。

**API 坑清单（`ops.cutlass_scaled_fp4_mm`，本轮全踩过）**：
| 约束 | 报错 |
|---|---|
| `scale_a` 必须 **padded + swizzled** | `scale_a must be padded and swizzled` |
| scale 的 dtype 必须 **`float8_e4m3fn`**（不能 uint8） | `Inconsistency of torch::stable::Tensor type:scale_a` |
| `out_dtype` 只支持 **bf16 / fp16** | `Unsupported output data type of nvfp4 mm sm120 (Float)` |
| `scaled_fp4_quant` 的 `padded_n` 是**输入列数**（=k） | `padded_n must be >= n` |

## 复刻量化算子前，先把"参照用哪个 kernel"查清楚（2026-10-24）

NVFP4 这条线花了 17 轮，最大的时间浪费来自**没先确认参照走哪个 kernel**就去调公式。

**应该第一件事就做的**：
```python
# 在参照容器里直接问
from vllm.model_executor.layers.fused_moe.experts import cutlass_moe as cm
cm.run_cutlass_moe_fp4          # hook 它，看真实入参形状
# 或看日志：grep "Using .* NvFp4 MoE backend"  → VLLM_CUTLASS
#            grep "Using MoE.*Modular"          → MoEPrepareAndFinalizeNoDPEPModular
```
拿到"参照是 cutlass 融合 kernel"这个事实后，才知道**自己在跟什么比**，
以及"要不要照它的归约结构复刻"还是"直接调用它"。

**本轮在这件事上得到的确定结论**：
- 参照的 MoE 是**一个融合 kernel**（gate/up/SwiGLU/down/combine 全在里面），
  Python 层**没有逐专家的 `nn.Module`** → 单专家 hook 抓不到（试过，一个都没挂上）；
- `ops.cutlass_fp4_moe_mm` 是 **torch op**（`_C_stable_libtorch.abi3.so`，`TORCH_LIBRARY` 注册，
  入参 `at::Tensor`），**不能像 cuBLASLt 那样从 Rust 直接链接**。

**判据层面的收获**（可直接复用）：

| 现象 | 含义 |
|---|---|
| `ndiff` 接近全量、`max|d|` = 一个 bf16 ulp | 输出精度/舍入位置不对（如漏了 `to(bf16)`） |
| `ndiff` 全量、`max|d|` 大一个量级 | smem/寄存器层面的布局错（如每 warp 共享了同一份 smem） |
| 量级比 ≈ 1.00x 但逐元素不同 | 语义对、归约次序不同 |
| 逐行相对误差是**同一常数** + corr 高 | 乘子错了一个固定因子（最好修） |
