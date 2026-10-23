# D-16：重排一次事实（用户 2026-10-23 要求"从头理一次"）

> 本轮**没有产生新读数**，只做三件事：① 复核既有结论的可靠性边界；② 检查此前从未查过的
> "数据类型"一档；③ 指出交接区里"我们另有一个放大器"这条推断是**度量假象**。

## 1. 读 vLLM（镜像内 0.28 源码，非 /data/ref）

本模型的 `self_attn.qkv_proj.quant_method` = **`ModelOptFp8LinearMethod`**（D-11 已实测），
其 `apply()`（`quantization/modelopt.py:437+`）是：

```python
def apply(self, layer, x, bias=None):
    return self.fp8_linear.apply_weights(layer, x, bias)
```

**关键**：`quantization/fp8.py::Fp8LinearMethod.apply` 里那条 "BI=1 且非 cutlass → 反量化成 bf16 再
`F.linear`" 的分支**不适用**——本模型不走 `Fp8LinearMethod`。所以：
- BI=1 时参照侧**仍然跑 fp8 GEMM**（FlashInfer），D-11 成立，无需撤回。
- 若模型换了 quant method（如 `Fp8LinearMethod`），BI=1 会走 bf16 反量化路径，届时对比口径完全不同
  ——记在这里，避免下次接新 checkpoint 时踩。

## 2. "数据类型"一档的排查（此前从未系统查过）

| 项 | vLLM | 我们 | 判定 |
|---|---|---|---|
| 残差流 dtype | bf16（`hidden_states` 全程 bf16）| `Tensor<bf16> sc.x` | 一致 |
| `layer_scalar` | `register_buffer(torch.ones(1))`，checkpoint 载入 | `f32`，`!= 1.0` 才乘 | **待核 dtype**（见下）|
| PLE（`hidden_size_per_layer_input`）| 分支存在 | — | config = **0 → 不启用**，无关 |
| 激活量化 scale | 静态 per-tensor（`dynamic: false`）| 同 | 一致 |

**残差流 dtype 已被现有数据反证**：若两侧 dtype 不同，L0 输出相对差会在 ~1e-3（bf16 步进）量级；
实测 L0 = 4.9e-8。**排除**。

`layer_scalar`：vLLM 是 f32 buffer × bf16 tensor（PyTorch 类型提升会得 f32 风险），我们是 bf16 内乘。
**若 `layer_scalar` 在 checkpoint 里恰为 1.0 则无影响**（我们直接跳过乘法）。→ 下一轮第一件事：读它的值。

## 3. **"我们另有一个放大器"是度量假象**（本轮主要修正）

量化器的放大倍数**依赖于扰动本身的大小**：设量化步长 q（e4m3 相对步长 ~6%），
- δ ≫ q：输出差 ≈ δ，放大 ≈ 1（量化噪声是叠加项，不放大）；
- δ ≪ q：多数元素不动、少数跨档，每次跨档代价 q → **相对放大约 q/δ**，且随 δ 增大而下降并饱和。

按此把两张表的**放大倍数**算出来（同一次运行 p02）：

| 段 | vLLM 自家 fi-vs-cu | 我们 vs vLLM |
|---|---|---|
| L0 | 2.65e-05 | 4.9e-08（小 540 倍）|
| L0→L1 | ×161 | ×26 |
| L1→L2 | ×2.0 | **×1.6e4** |
| L29 | 9.38e-02 | 2.93e-01 |

两边**起点差 540 倍**，而放大倍数在 δ 上强烈非线性（≈ q/δ 或 √(q/δ)）。
**因此"我们从 L2 起放大倍数是它们 2–3 倍 ⇒ 我们另有一个放大器"不成立**——它是"我们的 δ 小得多、
所以仍在跨档区"的直接后果。这句要作废。

**更可靠的两条读数**（同一分母下）：
1. L0 单算子：我们 vs vLLM 前向 = 5/98304 = **10/196608**；vLLM 自己的 flashinfer 算子 vs 它自己的前向
   = **7/196608**（D-12）。**同一量级**——我们的 GEMM 并不比 vLLM 自家算子在"单算子复现前向"上更差。
2. 末层两边都饱和到 O(1e-1)：vLLM 换 kernel 到 9.4e-2，我们到 2.9e-1，**同一饱和区**，不是"多一层放大器"。

## 4. 端到端交叉印证

| 配置 | 翻转率 | Δlp |
|---|---|---|
| 我们（默认 = vLLM 口径）| 14.49% | 1.2302 |
| vLLM 自身换 kernel（BI 开 vs 关）| 13.95% | 1.19 |

两者几乎相同 → **我们已经处在"任何非逐位实现"的饱和水平**。

## 5. 结论：卡在哪一步

- L-1 / L0 / L1：**已 ok**（见下节），残余是 fp8 GEMM 累加序的 1-ulp，属不可口径对齐。
- L2–L29 的 1e-2…3e-1 **不是缺陷**，是 fp8 KV 往返放大器（双方共有）把上游 1-ulp 放大并饱和的结果。
- **真正卡住的一步只有一个**：从外部把 vLLM 自己在 L0 的 fp8 GEMM 前向**逐位复现**。
  D-12 那 7/196608 至今无解释，说明我们还没找准 vLLM 真实执行的 op + 调用形态。
- 在这之前，继续挖"L2 的放大器"是追噪声。

## 6. 下一轮具体动作（一个假设，可判定）

在 vLLM 侧 hook `apply_weights`，**在 kernel 调用前把真正喂进去的 `(x_fp8, x_scale, w_fp8, w_scale,
out_dtype)` 原样存盘**，再把这四个字节串灌进我们的 kernel：
- 我们输出与 vLLM 该层 `qkv_raw` **逐位 0** → 我们的 kernel 已等价，剩余差异全在 vLLM 自己的路径里，
  目标改成"对齐它真实走的那条 op"；
- 仍有 1-ulp → 差异在累加序，才需要复刻 tile/K 拆分顺序。

## 7. 本轮的前置条件纪律（用户 2026-10-23 硬性约束）

- 超过 5 分钟的任务，**每 2 分钟探活**一次（看日志增量 / 进程 / `nvidia-smi`），不静默等。
- 每轮跑前 `cleanrun.sh`，跑前跑后各查一次 8 卡空闲；日志无"写 N 个张量"即判失败重来。
- 每个数值落盘前回答：脚本 sync 了吗？A/B 的 md5 不同吗？环境干净吗？跑了几次？

## 8. D-17 进行中（2026-10-23）

- `layer_scalar` 实测**不是 1.0**（`layers.0=0.0703125`、`layers.1=0.203125`、`layers.10=0.7265625`，
  bf16 精确可表示，共 17 个）。两侧都是 bf16 × 这个标量，且 bf16×bf16 恰为 1 次舍入 → 口径一致；
  **但此前从未核过"它是否真的被应用"**，已记入待核。
- **新线索（很可能就是剩下的口径差）**：`FlashInferFP8ScaledMMLinearKernel.apply_input_quant = False`
  —— vLLM 不预先量化激活，**把 bf16 直接交给 FlashInfer，由算子内部量化**。
  那么内部用的是 checkpoint 的**静态 `input_scale`**（`input_activations.dynamic=false`）还是**输入 amax 的动态 scale**？
  这决定激活量化点在哪、舍入对象是谁。我们侧 `Gemm::forward_fp8(a: bf16, a_scale, ...)` 同样是传 bf16 进 kernel 量化，
  但用的是静态 `input_scale`。**待核**。
- 探针 `scripts/dev/kaon3_vllm_fp8det.py` 已同步（md5 一致）并在 GPU 0 后台运行：
  抓 L0 第一个 fp8 GEMM 的真实入参，并**重复 5 次同入参**判定算子自身是否逐位确定。

## 9. D-17 第一轮结果（抓到真实入参，脚本自身崩在类型上）

`kaon3_vllm_fp8det.py` 在 GPU 0 跑到"抓样本"后就崩了——**是我的脚本 bug**：
`orig_mm` 期望 `scale_a` 是 **tensor**（它内部 `assert scale_a.numel() == 1`），我传了 Python `float`。
抓样本本身成功（打印在崩溃之前）。

**抓到的 L0 第一个 fp8 GEMM 真实入参**（p02，24 token）：

| 项 | 值 |
|---|---|
| `A` | `(24, 2816)` `float8_e4m3fn`，stride `(2816, 1)` → 行主序，**已量化** |
| `B` | `(2816, 8192)` `float8_e4m3fn`，stride `(1, 2816)` → **列主序视图 = 物理 `[N,K]` 行主序** |
| `scale_a` | `0.59375`（静态，= checkpoint `input_scale`）|
| `scale_b` | `1.3166156e-3`（= `weight_scale`）|
| `out_dtype` | `bfloat16` |

**两条此前未记录的事实**：
1. `scale_a = 0.59375` 是**静态**标量，与我们口径一致（`input_activations.dynamic=false` 得到印证）。
   → 激活量化点不在"静态 vs 动态"，D-17 原假设的一半被排除。
2. 权重给 FlashInfer 的是**转置视图**（物理 `[N,K]` 连续）。我们 kernel 收的也是 `[n,k]`，
   而累计是 `D[g0*N+gn]`——布局假设一致。

**剩下的差异只可能在**：GEMM 内部 K 方向累加顺序 / tile 形状（FlashInfer 用 cutlass 类 128×128 tile + f32 累加），
以及激活量化的**舍入实现**（我们 kernel 内 `A * 1/a_scale` 饱和转 e4m3；vLLM 走 `scaled_fp8_quant`，
D-12 已验字节一致 0/67584）。

**下一轮**：① 修脚本的类型 bug；② 断网式验证——把已抓到的 `A_fp8/B_fp8/scale` 直接喂我们和 FlashInfer，
判定"同字节入参下输出是否逐位相同"；③ 同入参重复 5 次判定 FlashInfer 自身是否逐位确定（原 D-17 的核心问题，仍未答）。
