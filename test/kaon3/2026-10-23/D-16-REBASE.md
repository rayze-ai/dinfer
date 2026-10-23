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

## 10. 判据 FLOOR 的数字对不上（待用户裁定的第 2 件事）

`scripts/verify/kaon3-parity.sh` 的注释写"跨进程实测 ~18%"，代码默认 `FLR=test/kaon3/oracle_clean_b`；
交接区与迭代记录里写 **FLOOR 0.83%**。但同一份记录里"P1 真实跨进程下限（`oracle_p1c` vs `oracle_p1`，
同参数两进程）"是 **flip 1.51% / mean_common_d 0.1000**（`kaon3-parity.md:175`）。

**三个数（0.83% / 1.51% / 18%）出处不同，不能混用**：
- `0.83%` 来自哪个 oracle 组合，仓里没有落盘证据；
- `1.51%` 是 P1 那次两进程的实测；
- `18%` 在 verify 注释里、标着"2026-10-22 实测"，与上面两个都不符。

**在拿到 `oracle_clean_b` vs `oracle_clean` 的实际 diff 前，判据的分母是不可信的。**
这不影响"我们的读数 14.49% 远大于任何一个候选下限"，但会让"差多少倍"失真。

→ 下一轮顺带做一次 `kaon3_oracle_diff oracle_clean_b oracle_clean`，把真实 FLOOR 钉死并写进规格。

## 11. **D-17b 决定性结果：vLLM 的前向是逐位可复现的**（推翻 D-12）

同入参、同进程、同 GPU，p02（24 token），L0 第一个 fp8 GEMM（`qkv_proj`，A `(24,2816)` B `(2816,8192)`）：

| 对比 | 差异 |
|---|---|
| flashinfer 同入参重复 6 次（run0 vs run1..5）| **0 / 196608 × 5 次，max 0** |
| **重新调用 flashinfer vs 前向里的真实返回值** | **0 / 196608，max 0** |
| flashinfer vs cutlass（同入参）| 7 / 196608（max 0.125）|
| cutlass vs 前向真实值 | 7 / 196608 |

### 结论

1. **FlashInfer 的 fp8 GEMM 自身是逐位确定的**（6 次完全一致）。
2. **vLLM 的前向可以被逐位复现**：拿真实入参重调同一个算子，输出与前向里那份**完全相同（0）**。
   → **D-12 的"连 vLLM 自己都复现不出自己的前向（7/196608）"是错的**：那 7 处是
   `flashinfer vs cutlass` 的差，被我误读成了"自复现失败"。
   **"逐位复刻不可达"这个持续多轮的前提，到此作废。**
3. 唯一不可消除的是**换 kernel 变体**（fi vs cu = 7/196608）；**同 kernel 下目标就是 0**。

### 因此卡点变成一句可执行的话

> 让我们的 fp8 GEMM 在**同一份字节入参**下产出与 FlashInfer **逐位相同**的结果。
> 目前我们是 10/196608（D-13：5/98304 q 槽），**介于 fi 的 0 与 cu 的 7 之间**——离目标只差"累加顺序"。

### 已存盘（供 Rust 侧灌同一份字节）

`/w/hid/d17b/`：`A_fp8.bin`(24×2816 e4m3)、`B_fp8.bin`(2816×8192 e4m3，物理 [N,K])、
`fwd_ref_bf16.bin`(24×8192 bf16，**vLLM 的真实输出**)、`meta.txt`。

下一轮：用这三份文件做 Rust GPU 测试——我们 kernel 吃**预量化**的 A（不自己量化），比 `fwd_ref_bf16.bin`。
- 0 → 完成，口径完全对齐；
- 非 0 且与"我们自定义量化"同样的差异 → 差异只在激活量化；
- 非 0 且不同 → 纯累加顺序，需复刻 FlashInfer 的 K 分块顺序。

## 12. 踩坑：`/w/hid` 是容器内临时目录，容器退出即丢

D-17b 把 `A_fp8.bin` / `B_fp8.bin` / `fwd_ref_bf16.bin` 写到 `/w/hid/d17b/`，容器一退出全没了。
（同样解释了为什么 `bash /w/hid/cleanrun.sh` 报"不存在"——交接区把它当成常驻脚本，**它不常驻**。）

**规则**：容器内产出一律写 **`/hf_data/yaochen/...`**（宿主持久、rsync 树），不要写 `/w/...`。
日志里打印的数值仍然有效（stdout 已落 `/tmp/d17b.log`）；只有文件产物受影响。

**重跑**：`OUT=/hf_data/yaochen/d17b`。

## 13. 修正 §12：`/hf_data/yaochen` 在容器内**不是**宿主的——唯一持久路径是 `/w`

`remote_5090.sh ctr` 的挂载只有：`-v $REMOTE_ROOT:/w`（宿主 `/hf_data/yaochen` → 容器 `/w`）、
`-v /hf_data/hub:/hf_data/hub:ro`、cargo/rustup。容器内的 `/hf_data/yaochen` 是**另一个独立临时目录**，
容器退出即丢。

**所以**：容器内产出一律写 **`/w/...`**，落到宿主 `/hf_data/yaochen/...`。
（§12 写"写 `/hf_data/yaochen`"是错的，按此更正。）

D-17b 已用 `OUT=/w/d17b` 重跑，**结果与首轮逐位相同**（0/196608 ×6 次 + 自复现 0），
说明该结论可复现、不受路径影响。

## 14. 本轮我自己的失误（第三次同类：改坏代码就直接跑）

给 kernel 加预量化入口时，第一次 Edit 把 `gemm_fp8_kernel` 的声明替换掉了而不是新增，
第二次补回时又在 `conv_a` 后留下 4 行残块（`return *reinterpret_cast...`）。两次都是
**改完没在本地看一眼结构就直接 sync + 远端编译**，各浪费一轮 6 分钟编译。

**规则（新增，写进 pitfalls）**：改 `.cu` / 大段 Rust 后，先 `grep -n '<关键符号>\|^}'` 看结构
（函数数、大括号配对、总行数），再 sync。sed/Edit 的多行替换必须回读确认。

最终形态：`gemm_fp8_kernel<PRE>` 模板，`PRE=false` 走 bf16 激活量化、`PRE=true` 直接读 e4m3 字节，
两条路径共用同一套 smem/ldmatrix/mma/K 累加序 → 对同一 A 的数值逐位相同。

## 15. 测试跑通流程（记录给下次省时间）

`crates/kernels/tests/fp8_flashinfer_parity.rs` 的默认数据目录是 `test/kaon3/d17b/`（仓内，便于入库），
但抓取产物落在宿主的 `/hf_data/yaochen/d17b/`（容器内 `/w/d17b`）。跑法：

```bash
GPU=0 scripts/dev/remote_5090.sh ctr "cd /w/dinfer && D17_DIR=/w/d17b CUDA_ARCHS=120 \
  cargo test --release -p dinfer-kernels --test fp8_flashinfer_parity -- --ignored --nocapture"
```

首次编译约 6 分钟（含 nvcc 编 sm_120），重跑只测不编只需几十秒。
**注意**：`ctr` 的输出别接 `| tail -N`——管道会缓冲，中途完全看不到进度，长任务直接 `> log`。

## 16. 首跑差异 196449/196608 → 是布局，不是精度

第一次用预量化入口跑出来的差异是 **196449/196608（几乎全部）**，max 9.9e2——**不是 1-ulp，是系统性错位**。
最可能的原因：`B` 在 vLLM 里是 `stride=(1,2816)` 的**转置视图**，
而抓取脚本用 `np.ascontiguousarray(B.view(uint8))` 会按**逻辑序**重排成 `[K,N]` 连续——
所以 `B_fp8.bin` 到底是 `[N,K]` 还是 `[K,N]` 取决于 numpy 如何处理非连续视图，**不能靠推断**。

**做法**：测试里两种解释都跑一遍（原样当 `[N,K]`；以及手动转置成 `[N,K]`），
哪个对上就是哪个。这样一次运行同时完成"判定布局"和"判定精度"。
→ 教训：**布局歧义要用实验判定，不要从 stride 推**（跨语言/跨框架时 stride 语义极易读反）。

## 17. 决定性结果：我们 = 8/196608，与 cutlass 的 7 同级（都复现不了 FlashInfer）

同一份字节入参（vLLM 真实喂给 FlashInfer 的 A/B/scale），L0 `qkv_proj` `[24,2816]×[2816,8192]`：

| 实现 | vs FlashInfer 前向输出 |
|---|---|
| **我们（`forward_fp8_prequant`）** | **8 / 196608**，max **1.25e-1** |
| cutlass（D-17b） | 7 / 196608，max 1.25e-1 |
| FlashInfer 重跑 | **0** |

**max = 0.125 恰是 25.6 处的一个 bf16 ulp** → 全是 1-ulp 级，不是口径错、不是布局错（布局错是 196449）。

**布局判定的副产品**：`B_fp8.bin` 落盘是 **`[K,N]` 逻辑序**（`np.ascontiguousarray` 会把转置视图按逻辑序重排）。
按 `[N,K]` 读 → 196449/196608；按 `[K,N]` 读再手动转置 → 8/196608。**布局必须实验判定，不能从 stride 推。**

### 这条数字的含义（把整条线钉死了）

我们的 fp8 GEMM 在数值上与 **cutlass** 同级（8 vs 7），都不等于 FlashInfer。
而实测的端到端对照正是：vLLM 换 cutlass vs vLLM 用 FlashInfer = **13.95% / 1.19**；
我们 = **14.49% / 1.2302**。**两者吻合**——不是巧合，是同一件事的两种测法。

→ **结论：要过 FLOOR，必须让 fp8 GEMM 与 FlashInfer 逐位相同（diff=0）。**
差 7–8 个元素看似微小，但经 fp8 KV 往返放大约 1.5e4 倍后就是 14% 翻转率。**没有第三条路。**

### 剩下的唯一技术问题是"复刻 FlashInfer 的累加序"，难点已定位

`bmm_fp8` 在 sm_120 上**带 JIT autotuner**（容器日志：`Autotuning fp8_gemm: 20 configs`、
`Saved 120 configs to /root/.cache/vllm/flashinfer_autotune_cache/0.6.18/`）。
不同 tile/unroll 配置的 **K 方向累加顺序不同** → 要逐位复现，得先知道**该形状命中的是哪一个配置**，
再让我们的 K 循环与它一致（或直接按它的分块重构）。
下一步：dump 该形状的 autotune 选择（`VLLM_...`/flashinfer 的 tuner 接口），把配置的
`tile_k / num_mma_per_k / split` 读出来，对照我们 `BK=64 + 2×mma(k32)` 的顺序。

## 18. 下一步的两个问题（按可能性排序）

**(A) 目标是否良定义**：FlashInfer 在**跨进程**（各自加载权重、各自 JIT autotune）时是否仍逐位一致？
- 若**是** → 目标良定义，任务变成"复刻它的累加序"（可做，需知道命中哪套 tile 配置）。
- 若**否** → "与 vLLM 逐位相同"本身没有唯一答案，判据必须改成"落在其运行间抖动内"。
这决定后面是继续攻 kernel 还是改判据，**必须先答**。测试已在跑（两个独立进程 dump 同一 GEMM 输出比）。

**(B) 若目标良定义**：FlashInfer `bmm_fp8` 在 sm_120 上带 JIT autotuner
（容器日志：`Autotuning fp8_gemm: 20 configs`、`Saved 120 configs`）。
不同 tile/unroll 的 **K 方向累加顺序不同** → 要复现得先读出该形状命中的配置
（`tile_k` / `num_mma_per_k` / split 方式），再让我们的 `BK=64 + 2×mma(k32)` 与它一致。

## 19. **目标良定义：FlashInfer 跨进程逐位一致**

两个**独立进程**（各自加载模型、各自跑 JIT autotune、各自缓存），跑同一份 p02：

| 项 | 结果 |
|---|---|
| 各自内部重复 6 次 | **0 / 196608**（两次都是）|
| 各自"重调 vs 前向真实值" | **0 / 196608**（两次都是）|
| **跨进程：r1 输出 vs r2 输出** | **0 / 196608** |
| 跨进程：A_fp8 / B_fp8 入参 | 完全一致 |

→ **FlashInfer 的 fp8 GEMM 是可确定复刻的**：跨进程、跨 autotune 缓存都给出同一串字节。
**"让我们的 fp8 GEMM 与它逐位相同"是一个良定义、有唯一答案的目标。**

## 20. 因此子 loop D 的剩余任务是一件事

把 `crates/kernels/cuda/gemm_fp8_sm120.cu` 的 K 方向累加序改成与 FlashInfer 一致。
已知的差异只是 8 个元素的 1-ulp（我们 8/196608、cutlass 7/196608、FlashInfer 0），
不是口径、不是布局、不是量化——**纯粹是累加树形状**。

线索：FlashInfer 的 JIT autotuner（`flashinfer/gemm/gemm_base.py`）对 fp8 GEMM 的候选含
`tile_k ∈ {128, 256}` 等；不同 `tile_k` / `num_mma_per_k` / 是否 split-K 会改变 f32 累加的**分组方式**。
我们当前是 `BK=64 + 2×mma(k32)` 顺序累加。
下一步：读出该形状（m=24, n=8192, k=2816）命中的 tactic，再看是否 split-K。

## 21. FlashInfer 的 `bmm_fp8` 真实签名（读错了两次，记下来）

```python
bmm_fp8(A, B, A_scale, B_scale, dtype, out=None, backend='cublas')
# backend ∈ {'cudnn', 'cublas', 'cutlass', 'cute-dsl', 'auto'}
```

- vLLM 的调用形态是 **3D**：`bmm_fp8(a.unsqueeze(0), b.unsqueeze(0), ...)`，输出也 unsqueeze(0)。
- 没有 `tile_k` / `unroll` 这类可传参数（那些在 `gemm_base.py` 的更底层接口里）。
  所以"枚举 autotune 候选"只能通过 **backend 枚举** 做，不能传 tile 参数。
- `flashinfer.bmm_fp8` 不是模块（`import flashinfer.bmm_fp8` 会 ModuleNotFoundError），是 `from flashinfer import bmm_fp8` 的函数。

## 22. **vLLM 的 fp8 GEMM 后端是 cuBLAS**（不是 cutlass，也不是 cudnn）

同入参（L0 `qkv_proj`，`A(24,2816) B(2816,8192)`，`bmm_fp8` 3D 形态）逐个 backend 跑：

| backend | vs vLLM 前向真实值 |
|---|---|
| `auto` | **0 / 196608** |
| **`cublas`** | **0 / 196608** |
| `cudnn` | **8** / 196608 |
| `cutlass` | **7** / 196608 |
| `cute-dsl` | sm_120 不支持 |

→ `auto` 落到 **cuBLAS**；**cuBLAS 就是这个模型的真实执行路径**，而且逐位可复现（跨进程 0）。

**注意一个巧合**：我们的 kernel 也是 **8**，与 **cudnn** 同数。下一步先验"我们 vs cudnn 是否逐位相同"
（若是，说明我们的累加序 = cudnn 的，要改的目标就从 cudnn 序迁到 cublas 序，路径更明确）。

## 23. 复刻 cuBLAS 的难度评估（必须让用户知道的战略问题）

cuBLAS 是闭源库。要"逐位复刻"，只能靠**实验反推它的 K 方向累加分组**：
- 已知 `k=2816`、`tile_k` 常见 128（2816/128 = 22 整），我们当前是 `BK=64 + 2×mma(k32)`（44 组）；
- 差异只有 8/196608 个 1-ulp，说明**顺序极其接近**，差的可能只是 K 遍历的分组边界或末级归约方式；
- 可测：构造**探针输入**（单位向量 / 幂次权重），从输出反推它的分组边界。

**这条路的尽头是"复刻一个闭源库的舍入行为"**。它是可做的（差异极小、且 cuBLAS 确定），
但代价高且脆弱（换 GPU / 换 cuBLAS 版本可能就变）。
→ **需要用户裁定**：是继续追 cuBLAS 逐位，还是把判据改成"达到 vLLM 自身换 kernel 的水平"
（我们 14.49%/1.2302 vs vLLM 换 cutlass 13.95%/1.19，**已经达到**）。

## 24. **用户方向验证成功：直接调 cuBLAS 可以逐位复现**（2026-10-23）

用户提议"能换成直接调用 cuBLAS，测试一下吗"——**成立，而且是最优解**。

同入参（L0 `qkv_proj`，`A(24,2816) B(2816,8192)`，`A_fp8.bin` + `B_fp8.bin` + `fwd_ref_bf16.bin`）：

| 实现 | vs vLLM 前向 |
|---|---|
| **`torch._scaled_mm(A, B_nk.t(), scale_a, scale_b, bf16)`** | **0 / 196608** ✅ |
| 同上重复 3 次 | 逐位一致（确定性 ✓）|
| 我们自己的 kernel | 8 / 196608 |
| `flashinfer.bmm_fp8(..., backend="cublas")`（显式传参）| 8 / 196608 |

**布局要点（又踩一次）**：`B_fp8.bin` 是 `[K,N]` 逻辑序；喂 cuBLAS 要 `B_nk = B.t().contiguous()`（物理 `[N,K]`）
再 `.t()` 成 `[K,N]` 视图。直接传 `B.t()`（非连续）→ 形状报错；传错序 → 196449。

### 结论：实现路线改变

不再需要反推 cuBLAS 的累加分组——**直接调 cuBLASLt 的 fp8 GEMM 即可**。
我们已有 `cudarc::cublaslt` 的完整基础设施（`LtDescs` / `heuristics` / `algo_valid_for` / 桶缓存，见
`crates/kernels/src/gemm.rs`），只需补一条 fp8 数据类型的 Lt 调用。这条路：
- **精确**（实测 diff=0）；
- **简单**（复用现有 Lt 代码，不用摸 kernel 的 K 循环）；
- **与 vLLM 同源**（它就是 `CublasFp8GemmRunner` → cuBLASLt）。
风险：换 GPU / CUDA 版本时 cuBLAS 内部选择可能变，但那时 vLLM 也一起变，仍是对齐的。

**下一步**：在 `crates/kernels` 里加 `forward_fp8_cublaslt`，用 `fp8_flashinfer_parity` 测试判 diff=0。

## 25. 实现：`Gemm::forward_fp8_cublaslt`（2026-10-23）

在 `crates/kernels/src/gemm.rs` 复用现有 Lt 基础设施新增 fp8 路径：
- 描述符 `make_descs_fp8`：A/B = `CUDA_R_8F_E4M3`，C/D = `CUDA_R_16BF`，
  compute `CUBLAS_COMPUTE_32F` + scale `CUDA_R_32F`；
  `A_SCALE_POINTER` / `B_SCALE_POINTER` 传**设备上的 f32 标量指针**（对照 flashinfer
  `data/include/flashinfer/gemm/bmm_fp8.cuh:171-172`）；
- 布局沿用 bf16 路径的 cuBLAS 视角（A 操作数 = W 物理 `[n,k]` 当 `[K,N]` 列主序，`lda=K`）；
- algo 走与 bf16 相同的 `heuristics()` + `default_algos` 桶缓存（同一形状只查一次）。

## 26. cuBLASLt 首版 = 8/196608（与我自写 kernel 同数）：选错 algo

`forward_fp8_cublaslt` 首版跑出 **8/196608**，与我自写 kernel **完全相同**的 8。

这意味着两件事之一：
1. 启发式第一个 algo 恰好落到与我 kernel 相同的累加序；或
2. 我们两版都命中同一条 cuBLAS 内部路径，而 `torch._scaled_mm` 走的是**另一条**（不同 algo/不同 Lt 配置）。

`torch._scaled_mm` 实测是 **0**，所以 cuBLAS 能做到——差在**algo 选择**（`cublasLtMatmulAlgoGetHeuristic` 的第一个
未必是 PyTorch 用的那个）或**描述符细节**（如 `TRANSA/TRANSB` 方向、`ld` 取值导致走不同内部 tile）。
下一步：把启发式的**全部** algo 逐一试一遍，看有没有能到 0 的；若有，就固定用它。

## 27. 枚举 algo 的实现（配合 §26）

`Gemm::forward_fp8_cublaslt_algo(..., algo_idx)`：用 `cublasLtMatmulAlgoGetHeuristic` 取**前 32 个**候选，
按 `algo_idx` 取其一执行；`fp8_cublaslt_algo_count(...)` 给候选数。
`forward_fp8_cublaslt` 改用共用的 `run_lt_fp8`（去掉重复的 matmul 调用）。

测试里逐个跑并比 `ref_f32`，命中 `diff==0` 即停并打印下标。**只用于定标，不进热路径。**

## 28. **成功：cuBLASLt algo[2] 与 vLLM 逐位一致（0/196608）**

启发式共 **8** 个候选，逐个试（同入参、同描述符）：

| algo | vs vLLM |
|---|---|
| [0]（默认启发式第 1 个）| 8 / 196608 |
| [1] | 8 / 196608 |
| **[2]** | **0 / 196608** ✅ |
| 命中即停 | |

→ **原因确认**：不是描述符/布局/scale 的问题，纯粹是 **algo 选择**。
默认取启发式第 0 个恰好与我自写 kernel 同序（都是 8），而 PyTorch/FlashInfer 的 cuBLAS 路径用的是 **algo[2]**。

**这就把"直接调 cuBLAS"彻底打通了**：`Gemm::forward_fp8_cublaslt` 把默认 algo 从 `h[0]` 改成 `h[2]`
（或按"实测与参照一致"选定并写死，附注释说明来历）。
剩下的端到端问题（翻转率 14%）预计会随这一处对齐而大幅收敛——因为 fp8 KV 放大器不再放大
"我们自己引入的" 1-ulp。

## 29. ✅ 最终验证通过

```
自写 kernel B=[K,N] 正解(已转置): diff 8/196608    max 1.2500e-1
cuBLASLt(默认 algo=2):            diff 0/196608    max 0.0000e0
test result: ok. 1 passed
```

**`Gemm::forward_fp8_cublaslt` 与 vLLM 的 fp8 GEMM 逐位一致。**

### 下一步：接进模型前向（还没做）

`models/kaon3/src/layer.rs` 目前走 `Gemm::forward_fp8`（自写 kernel，8 处 1-ulp）。
要享受这个对齐，`QLinear::Fp8` 的调用点要改成 `forward_fp8_cublaslt`。

**注意性能**：cuBLASLt 对小 M（decode 时 m=1..32）未必比自写 kernel 快。
建议按 M 分流：**prefill（M 大）走自写 kernel，或反之——以 bench 为准**，不要凭直觉选。
接入前先测两条路径在 [1×2816×8192, 24×2816×8192, 512×2816×8192] 上的耗时。

## 30. 接入模型前向（2026-10-23）

`QLinear::Fp8` 增加两块**内部暂存**（用 `Mutex` 只为内部可变性，无并发）：
- `quant`：bf16→e4m3 的激活量化缓冲（按 `tokens×k` 需求扩容）；
- `scache`：两个 per-tensor scale 的**设备副本**（cuBLASLt 要设备指针），首次调用建一次。

`forward` 默认走 `quant_fp8` + `forward_fp8_cublaslt`；
`FP8_SELF_KERNEL=1` 可切回原自写 kernel（`forward_fp8`）做 A/B。
新增 `Gemm::quant_fp8` 与 C 入口 `quant_fp8_bf16`（复用 kernel 里已有的 `cvt16` 逻辑，饱和 ±448 + RNE，
与 vLLM `scaled_fp8_quant` 字节一致——D-12 已验）。

## 31. 端到端 verify（cuBLASLt 接入后）：6 过 2 失

run `1023-125513-1295126`（30 用例，6 卡并行）：

| 项 | 读数 |
|---|---|
| kernel 对拍 | ✅ |
| 参照/下限用例一致 | ✅ 30 个 |
| 分片前向 | ✅ 30/30 |
| 逐层 router（fp64 真值）| ✅ |
| **翻转率** | ❌ **15.73%**（FLOOR 0.83%）|
| **共有 token Δlp** | ❌ **1.2189**（FLOOR 0.0276）|
| 自然长文本 ΔNLL | ✅ +0.0597 |
| 单 token top1 | ✅ |

**FLOOR 之谜解开**：`FLOOR POOLED cases=30 rows=538 decisive=481 flips=4 flip_pct=0.83
mean_common_d=0.0276` —— **0.83% 就是 `oracle_clean_b` vs `oracle_clean` 的实测值**，
之前怀疑的 0.83/1.51/18 三个数里，这个是对的（另外两个是别的 oracle 组合/别的时间点）。

**⚠️ 这个 15.73% 与之前的 14.49% 不可直接比**：分母不同（本次 rows=538 decisive=483；
此前是另一套 oracle，rows=655 decisive=607）。要判 cuBLASLt 是改善还是回退，必须做**同集 A/B**：
`PROBE_ENV="FP8_SELF_KERNEL=1"` 跑同一套用例，比 flip_pct。

**另一个必须查的假设**：`algo[2]` 只在 `m=24,n=8192,k=2816`（L0 qkv）上定标，
120 个 fp8 层的 N/K 各不相同；**不同形状命中的"与 vLLM 一致的 algo"未必都是下标 2**。
若某形状上 algo[2] 反而更差，就解释了翻转率没降。
→ 做法：对每个 (n,k) 组合，同入参下枚举 algo 并与参照比；或先用"层内 1-ulp 计数"看哪层变差了。

## 32. 形状扫描首次无效（我的 bug，与 §14 同类）

读数全是 `m=4096`、四个 backend 全 `=0` —— **假的**。
原因：`seen` 没在 `llm.generate` 前清空，LLM 构造期的 **profiling run** 用
`m=max_num_batched_tokens=4096` 的假批量先占了每个 `(n,k)` 的 key，真实 prefill（`m=24`）被挡在门外。
（同 `kaon3_vllm_hidden.py` 里 `cap.clear()` 的道理。）

→ **规则**：任何"前向里抓 X"的 hook，`LLM(...)` 之后必须清空累积器，否则抓到的是 profile 阶段的假数据。
已修脚本（`seen.clear()`），重跑。

## 33. 用户提问：要不要回退？（2026-10-23）

用户："之前为了排查和 cuBLAS 的问题改动了很多，现在是否要改回去"。**同意先不回退**（用户确认）。

改动可分为三类：

| 类 | 内容 | 建议 |
|---|---|---|
| **A. 纯新增** | `forward_fp8_cublaslt` / `_algo` / `algo_count` / `quant_fp8` / C 入口、`fp8_flashinfer_parity.rs`、6 个诊断脚本 | **保留**（不调用不生效，零风险，是回归与对拍基础）|
| **B. 改行为** | `QLinear::Fp8` 默认走 cuBLASLt，`FP8_SELF_KERNEL=1` 可回退 | **待 A/B 判定** |
| **C. 动既有代码** | `gemm_fp8_kernel` 改 `template<PRE>` + 预量化入口 | 保留（`gemm_lowp_parity` 已验证自写路径行为未变）|

**判定 B 去留需要两块目前缺失的证据**（都在跑）：
1. **同集 A/B**：`PROBE_ENV="FP8_SELF_KERNEL=1"` 跑同一套 verify，比 `flip_pct`；
   现有 15.73%（新，rows=538）与 14.49%（旧，rows=655）**分母不同，不可比**。
2. **形状普适性**：`algo[2]` 只在 `(n=8192,k=2816)` 定标；模型有 6 种 (n,k)
   （2816×2112 / 2816×4096 / 2816×8192 / 4224×2816 / 8192×2816 / 10240×2816），
   其他形状的"正确 algo 下标"未必也是 2。

**"整体重理一遍"放在这两项之后**（用户同意）：届时才有完整事实来定哪些留、哪些撤。

## 34. **同集 A/B 结果：cuBLASLt 不是明确改进**（决定 B 类改动的关键证据）

两轮**同一套用例、同分母**（rows=538 / decisive=483），只差 `FP8_SELF_KERNEL`：

| 配置 | 翻转率 | Δlp | ΔNLL | 判定 |
|---|---|---|---|---|
| **cuBLASLt（新默认）** | **15.73%** | **1.2189** | **+0.0676** | 6 过 2 失 |
| **自写 kernel**（`FP8_SELF_KERNEL=1`）| **14.49%** | 1.2302 | +0.1531 | 5 过 3 失 |

**三个指标方向不一致**：
- 翻转率：cuBLASLt **更差**（15.73 vs 14.49，+1.24pp）
- Δlp：cuBLASLt 略好（1.2189 vs 1.2302，−0.9%）
- ΔNLL：cuBLASLt 明显好（+0.0676 vs +0.1531，减半）

→ **不能宣称 cuBLASLt 是进步。** 之前 14.49% 与 15.73% 的比较（不同分母）作废，这次是可比口径。

**最可能的解释**（待形状扫描验证）：`algo[2]` 只在 `(n=8192,k=2816)` 上定标，
而 cuBLASLt 现在被用到**全部 6 种 (n,k)** 上。若其余形状的"正确 algo 下标"不是 2，
那些层反而从"8 处 1-ulp"变成"更大的偏差"，抵消甚至反超了单形状的收益。

**附带问题**：本轮 `逐层 router 真值` 报 `RT_PROBE_FAIL`（探针崩溃）——两任务并发抢 6 卡所致，需干净环境复测。

## 35. 形状扫描重跑两次都无效 —— 同一类失误第 4 次（未 sync）

第二次运行仍读全 `m=4096`，以为 `seen.clear()` 没生效；实测是**远端脚本还是旧的**
（`md5` local=`b5ebf102…` vs remote=`21abdba…`）。launch 命令里只有 `ctr`，**没先 `sync`**。

**这条已经第 4 次**（改脚本/改代码不先同步或自检就发远端）。已固化到 `pitfalls.md`：
**任何 `ctr` 调用前**：① `remote_5090.sh sync`；② 比 `md5sum` 本地 vs 远端；
③ 不同就停，不要跑。宁可多花 5 秒，不要浪费一轮 6 分钟并产出假读数。

## 36. **形状扫描（md5 已验证，m=24 真实）——cuBLAS 对全部 6 形状都是 0**

| (n,k) | auto | **cublas** | cudnn | cutlass |
|---|---|---|---|---|
| 2816×2112 | 0 | **0** | 0 | 5 |
| 2816×4096 | 0 | **0** | 4 | 5 |
| 2816×8192 | 0 | **0** | 0 | 6 |
| 4224×2816 | 0 | **0** | 0 | 10 |
| 8192×2816 | 0 | **0** | 8 | 7 |
| 10240×2816 | 0 | **0** | 14 | 18 |

**两个结论**：
1. **cuBLAS 是普适正确的**（6/6 形状全 0），不是只在 qkv 上成立。方向对。
2. **我的实现有错**：`forward_fp8_cublaslt` 把 algo **写死 index 2**，而 2 只是 `(8192,2816)` 的答案。
   其他形状的正确 algo 下标不同 → 直接解释 §34 的 A/B（翻转率 15.73% > 14.49%）。

**更关键**：`flashinfer.bmm_fp8(..., backend="cublas")` 用**默认 tactic=-1** 就能对全部形状给 0，
而我们的 Lt 用默认 `h[0]` 在 `(8192,2816)` 上给 8。→ **差异在描述符构造，不在 algo 选择**。
最可能是 cuBLAS 视角的**行列方向 / lda-ldb-ldc 约定**与 flashinfer 不同，导致 cuBLAS 内部挑了不同 kernel。

**下一步**：照 `flashinfer/data/include/flashinfer/gemm/bmm_fp8.cuh` 的
`CuBlasLtMatrixLayout(type, rows, cols, ld, ...)` 与 `setAttribute(TRANSA/TRANSB)` 逐项对齐我们的
`make_descs_fp8`，而不是继续猜 algo 下标。

## 37. 读 flashinfer 源码找到真因：描述符两处与它不一致

`flashinfer/data/include/flashinfer/gemm/bmm_fp8.cuh` 的 `Fp8GemmDescriptors`（第 ~118 行）：

```cpp
a_layout(get_cuda_data_type<AT>(), m, k, k, true),   // A = 激活 [m,k], ld=k
b_layout(get_cuda_data_type<BT>(), k, n, k),         // B = 权重 [k,n], ld=k
d_layout(get_cuda_data_type<DT>(), m, n, m),         // D = [m,n], ld=m
matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, CUBLAS_OP_T);
matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, CUBLAS_OP_N);
int8_t fast_accum = 1;
matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_FAST_ACCUM, fast_accum);   // ← 关键
```

我们的 `make_descs_fp8` 原来有两处不同：
1. **操作数方向反了**：我们让 W 当 A 操作数（沿用了 bf16 路径的写法），flashinfer 是激活当 A。
2. **没设 `FAST_ACCUM`**：flashinfer 显式置 1 —— 这直接决定 f32 累加顺序，**极可能就是我们差的那 8 个 1-ulp**。

**已按 flashinfer 逐项改写**，并**去掉写死的 algo 下标**（改回启发式默认第 1 个 = flashinfer 的 tactic=-1 路径）。
若这条成立，就不再需要"按形状标定 algo"，对全部 6 个形状自然成立。

## 38. 弯路：误读 flashinfer 的 `t` 参数（已回退）

按 `a_layout(m, k, k, true)` **字面**改成 `(m,k)` 后，cuBLASLt 在 fp8 heuristic 上直接
`CUBLAS_STATUS_INVALID_VALUE`（带不带 `FAST_ACCUM` 都一样）。

真因：`CuBlasLtMatrixLayout` 的**第 5 个参数 `t`** 会交换行列：
```cpp
cublasLtMatrixLayoutCreate(&raw, type, t ? cols : rows, t ? rows : cols, ld);
```
所以 `a_layout(m,k,k,true)` → `LayoutCreate(k, m, k)`，**与我们原来的 `(k,m,ld=k)` 等价**。
→ **布局本来就是对的**，差异确实只在 algo 选择。已回退到原方向并把这段写进代码注释。

**教训**：读参照实现要连 wrapper 的构造参数语义一起读，不能只看调用点的字面实参。

## 39. 正确做法：按形状标定 algo，不写死下标

- `algo[2]` 只在 `(n=8192,k=2816)` 上验证过；写死它会让其余 5 个形状变差（§34 的 A/B 就退步在这一点）。
- 新增 `scripts/dev/kaon3_d17_calib.py`：抓 vLLM 前向里**每种 (n,k)** 的真实入参 + 输出，存
  `/w/d17b/shapes/<n>x<k>/{A_fp8,B_fp8,ref_f32}.bin`，供 Rust 侧逐形状标定 algo 下标。
- 标定结果应作为**启动期自调优**写进 `default_algos`（符合 `arch-profile` 规格：调优数值由实测给，
  不写死在 kernel / 代码里）。

## 40. 逐形状标定结果：每个形状都有 0 差异的 algo，但**下标不稳定**

| (n,k) | 启发式候选数 | 0 差异的 algo 下标 |
|---|---|---|
| 10240×2816 | 8 | **[2, 3]** |
| 2816×2112 | 8 | **[0, 1]** |
| 2816×4096 | 8 | **[1]** |
| 2816×8192 | 8 | **[0, 2]** |
| 4224×2816 | 8 | **[0, 1]** |
| 8192×2816 | 8 | **[2, 3]** |

**交集为空** → 不存在通用下标。写死任何一个是错的（这正是 §34 A/B 退步的原因：
写死 2 时，`2816×2112` / `2816×4096` / `4224×2816` 三个形状全变差）。

**下标不是稳定标识**：启发式返回的**顺序**随形状变，同一个"正确 kernel"在不同形状下排在
不同位置。要稳定复现，应该按 **algo 自身的配置属性**（`CUBLASLT_ALGO_CONFIG_ID` /
tile / stages）选，而不是按下标。

下一步：dump 每个候选 algo 的 `CUBLASLT_ALGO_CONFIG_ID`，看"0 差异"是否对应同一个 ID
（若是 → 直接用 ID 选，与形状无关；若否 → 只能按形状建索引表）。

## 41. ✅ 全部 6 个形状逐位一致（默认选择器）

按 **algo 配置属性**（而非列表下标）选择后，`fp8_algo_calib` 全绿：

| (n,k) | 默认选择器 vs vLLM |
|---|---|
| 10240×2816 | **0 / 245760** |
| 2816×2112 | **0 / 67584** |
| 2816×4096 | **0 / 67584** |
| 2816×8192 | **0 / 67584**（需 split-K 的 `id=35 tile=13 splitk=3`）|
| 4224×2816 | **0 / 101376** |
| 8192×2816 | **0 / 196608** |

**根因总结**：与 vLLM 逐位一致的是**具体的 cuBLASLt 配置**
`id=67 tile=13 splitk=1`（K=8192 时为 `id=35 tile=13 splitk=3`），
而这个配置在 `AlgoGetHeuristic` 返回列表里的**位置随形状漂移**（有时第 0、有时第 2）。
→ **必须按配置属性选，不能按下标**。这解释了 §34 写死下标 2 的 A/B 退步。

**实现**：`forward_fp8_cublaslt` 里用 `(n,k) → 配置字符串` 的小表 + 通配回退，
`algo_desc()` 读出候选的 `id/tile/splitk` 做匹配；未命中时退回启发式第 0 个（不会更差）。
符合 `arch-profile` 规格：调优数值来自实测表，不写死在 kernel 里。

## 42. 端到端 verify（全形状逐位一致后）：**只改善 1.45pp**

run `1023-133609-1331717`，同一套用例（rows=538 / decisive=483）：

| 配置 | 翻转率 | Δlp | 判定 |
|---|---|---|---|
| FLOOR（参照自身两进程）| 0.83% | 0.0276 | — |
| 自写 kernel（§34）| 14.49% | 1.2302 | 5 过 3 失 |
| cuBLASLt 写死 algo[2]（§34）| 15.73% | 1.2189 | 6 过 2 失 |
| **cuBLASLt 按配置选（本轮）** | **13.04%** | 1.2847 | 6 过 2 失 |

**关键结论（必须诚实记录）**：
- fp8 GEMM 已做到 **6/6 形状逐位 0 差异**（§41，有断言守护），**端到端只从 14.49% → 13.04%**。
- → **fp8 GEMM 不是主要放大器。** 我此前"D-5 定位到 KV 往返 → GEMM 累加序是根"的因果链
  **不完整**：GEMM 那 8 个 1-ulp 确实存在且已消除，但它在端到端的贡献只有约 1.45pp。
- 剩余 ~12pp **来自别处**，候选（按可能性）：RMSNorm 分母的 f32/bf16 口径、RoPE 表、
  attention 内部（softmax/softcap）、MoE 路由与激活量化、KV cache 的写入/读取路径细节。
- Δlp 反而略升（1.2302 → 1.2847），说明翻转率与 Δlp 两个指标对不同差异的敏感度不同，
  **不能只看翻转率**。

**下一步（用已排除 GEMM 的干净基线重跑逐层）**：GEMM 已在字节级对齐，现在 `--hidden-all` 的
逐层残差流就是"除 GEMM 外所有差异"的干净画像。重跑 D-2 那张表，看 L0 是否已到 0、L2 的跳变是否还在。
