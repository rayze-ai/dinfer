# 逐段隔离与参照自证（Kaon-V3 2026-10-24 沉淀）

> 配套 [parity-localization.md](parity-localization.md)：那篇讲"差异分几类"，
> 本篇讲**怎么把范围钉死到一段**，以及**拿到参照数据后先做什么**。
> 2026-10-24 两轮共 30+ 轮盲试的教训：没有这两步，猜哪一段都会白费。

## 1. 逐段隔离：一次就把范围从"30 层"收敛到"一个支路"

### 工具

- 我们侧：`kaon3_probe --forward --hidden-all <dir> --layer-iso <l>:<参照第 l-1 层输出>`
  + `SUB_LAYER=<l>`（dump 该层每个子模块）
- 参照侧：`kaon3_vllm_hidden.py` 加 `SUB_LAYER=<l>`，在**同一位置**打 hook
  （`language_model.model.layers.N.<子模块名>`）

两者写出的 `.bin` 命名对齐后逐项比 relL2。

### 实测效果（Kaon L0）

```
embedding                     0.0  逐位 100%
input_norm                    0.0  逐位 100%
attention (o_proj vs attn_out) 0.0  逐位 100%
pre_ff_norm                   0.0  逐位 100%
MLP (dense)                   0.0  逐位 100%
MLP + norm (ff1)              0.0  逐位 100%
MoE 支路                     4.3e-3（中位 1.004 bf16 ULP，|d|>1.0 元素为零）
```

一张表就把"整模型都不对"的错觉打破，范围从 30 层收敛到一个支路。

### 三条配套规则（每条都踩过坑）

1. **`--layer-iso` 的输入必须是上一层的输出**：L0 用 embedding（`Lm1`），**不是** `L0`。
   传错层会得到 relL2 ≈ 1.0 的"整层全错"**假象**（实际只有 5.8e-3）。
2. **层输出与层内中间量可能差一个标量**：Kaon 每层末尾乘 `layer_scalar = 1/14.2222`。
   拿"已乘标量的层输出"比"未乘的中间量"，比值恒为 0.0703 ——
   看起来像"神秘的常数因子"，实际只是口径不一致。
3. **hook 的模块名要限定前缀**（`language_model.model.layers.N.`）：
   否则 `vision_tower.encoder.layers.N` 也会命中，抓出 1152 宽的张量、形状对不上。

## 2. 参照自证：先比"参照 vs 参照"，再比"我们 vs 参照"

三项，任一项不过就别往下做：

| 检查 | 怎么查 | 不过的后果 |
|---|---|---|
| **重抓一致** | 同配置抓两次，逐位比 | Kaon `L*.bin` 两次差 **0.0077** → 该路判据作废 |
| **内部恒等式** | 如 `post_ff_norm == norm(residual + ff1 + ff2)` | 对不上说明抓的不是同一口径 |
| **非零 / 非常量** | `od -t x1`、numpy 的 nz 比例、min/max | 抓成 vLLM 显存 profile 的**全零 dummy**，会报出"逐位一致" |

### 两个真实翻车

- **全零 dummy**：vLLM 启动先跑一次显存 profile（全零输入），patch 的 `if cap["n"]==0`
  抓到的正是那次 → 四个文件每个字节都是 `00`，`ndiff=0` 是全零比全零，两轮结论作废。
  → dump 前加 `if nz(t) > 0` 守卫，并打印每次调用的 nz 比例。
- **参照自身不一致**：`L0.bin`（层返回值）与同一次运行的子模块 hook（`post_ff_norm` 等）
  差 0.0077，且 `(residual+post_ff_norm)` 与 `L0.bin` 的比值是 0.0371（既不是 1 也不是 `layer_scalar`）。
  → 层返回值整包 dump 必须自证；子模块级 hook（口径一致、可做恒等式）才可信。

## 3. 口径红线（本项目规格）

- 参照 = **compile**（vLLM 默认，不开 `enforce_eager`）。
- **eager 抓的数据只能诊断，不得写进判据**。hook 在 inductor 图里不生效
  （实测 patch 后 60 次调用 `nz=0.0000`），所以子模块参照只能 eager ——
  它的用途仅限于"定位哪一段"，验收到修复后必须回 compile 口径跑端到端。
- 同一份实现可能"eager 全对、compile 不对"，因为 inductor 会融合掉一些舍入
  （Kaon 的 MoE combine 就是：eager 走 Python 的 bf16 乘积，compile 融合后不做那次舍入）。

## 4. 数据布局（用户 2026-10-24 要求）

产物按 `用途_日期时间_轮次` 存：`$REMOTE_ROOT/runs/<用途>_<YYYYMMDD-HHMM>_<rNN>/`，
目录内 `README.txt` 写口径、环境变量、来源命令、非零校验。
工具：`scripts/dev/mkrun.sh`。详见 [docs/data-layout.md](../data-layout.md)。
