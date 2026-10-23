# loop 规格：kaon3-gelu

- 状态：**进行中**
- 创建：2026-10-23
- 验证脚本：`scripts/verify/kaon3-gelu.sh`
- 迭代记录：`test/kaon3/2026-10-23/kaon3-gelu.md`
- 上游：`docs/loops/kaon3-parity.md`（子 loop D，第 23–24 轮已确认 norm / router 口径）

## 目标

让 `models/kaon3` 的 **gelu_tanh·mul** 在 **vLLM compile 口径**下逐位一致（0/67584 元素不同），
并把 L1 截层的 `|Δlp|(实际 token)` 从 1.5221 压下去。

## 背景（第 23/24 轮已确认的事实）

- vLLM 默认走 **compile**（`enforce_eager=False`），compile 与 eager 的 norm 是**两条实现**；
  我们要对齐的是 compile（判据 `oracle_clean` 来自 compile 服务）。
- L0 的 norm / residual / 无权重 norm / fp8 量化 / router 预处理**已逐条核对**，只剩 gelu 与 MoE 内部未查。
- 现有 `activation.cu::gelu_tanh_mul_kernel` 有 `mode` 参数（0=全 f32 一次舍入，1=gelu 先舍 bf16 再乘 up），
  这一套是**按 eager 的 `act_and_mul_kernel` 定案的**，compile 口径未验证过。

## 环境

- 机器：5090（`scripts/dev/remote_5090.sh`，ControlMaster，断了要新 token 向用户要）
- 跑 GPU：`GPU=0 scripts/dev/remote_5090.sh ctr '<cmd>'`（cwd=`/w/dinfer`；GPU 1 = oracle，7 = 基线，禁用）
- 模型：`/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478`
- oracle：compile 模式、截 1 层已抓在 `/w/tmp/lay/L1.json`（由 `kaon3_oracle_server.sh 1` + `kaon3_oracle.sh` 产生）
- 编译：`GPU=0 scripts/dev/remote_5090.sh ctr 'cd /w/dinfer && CUDA_ARCHS=120 cargo build --release -p dinfer-kaon3'`
- 重置到干净状态：远端二进制重编即可；oracle 容器 `docker rm -f kaon-oracle` 再 `scripts/dev/kaon3_oracle_server.sh 1`

## 观察方法

| 看什么 | 在哪里 | 命令 |
|---|---|---|
| 探针 L1 截层 vs compile oracle | 探针 stdout | `./target/release/kaon3_probe $M --forward --layers 1 --oracle /w/tmp/lay/L1.json $(cat test/kaon3/oracle_clean/p02.ids)` |
| kernel 级逐元素差 | `/w/tmp/*.bin` + Python 比对 | `GPU=4 ... ctr 'cd /w/tmp && python3 gelu_cmp.py'` |
| inductor 生成代码 | `/hf_data/yaochen/tmp/ind/inductor/*/` | `grep -n 'tanh' <file>` |
| FX 图 | `.../backbone/computation_graph.py` | `grep -n 'gelu' ...` |

## 停止条件（可机器判定，`scripts/verify/kaon3-gelu.sh`）

1. **kernel 级**：我们的 gelu 输出 vs compile 参考（`torch.compile(F.gelu(tanh)·up)`）**逐位相同**（0/67584）。
2. **截层级**：`kaon3_probe --layers 1 --oracle /w/tmp/lay/L1.json` 的「实际 token |Δlp| 均值」**≤ 0.30**
   （现状 1.5221；期望 norm+router+gelu 全部对齐后接近 0）。

两条都满足 exit 0。

## 上限

8 轮。每轮只改一个假设（tanh 实现 / 常数 / 舍入次数 / 乘序）。到上限仍不满足就停下，
把已排除的假设写进本文件"卡点"区交给用户。

## 假设清单（按怀疑度排序）

1. **`tanhf` 的实现**：我们调 CUDA 的 `tanhf`（可能是 `htanh` 近似或 libdevice 精确版），
   inductor 生成的是 `libdevice.tanh`。两者在 sm_120 上可能差 ulp。
2. **常数精度**：`0.7978845608028654f` 与 inductor 里的常量是否同一 f32 值（写死的 double→float 舍入）。
3. **`x³` 的算法**：我们是 `x*x*x`，inductor 生成的是 `tmp4=x*x; tmp5=tmp4*x`（同序，但要确认）。
4. **舍入次数**：compile 下 gelu 与 `* up` 是否融成一条 fp32 链（像 router 那样），还是中间落一次 bf16。
5. **`torch._C._nn.gelu` 的 ATen 实现**：`approximate='tanh'` 在 ATen 里是 `0.5*x*(1+erf(x/√2))` 的 tanh 近似式，
   注意其内部可能用 `opmath_t`（f32）但 `tanh` 走 `std::tanh`。

## 卡点

（未开始）

## 第 1 轮（2026-10-23）

**假设 1–5 一次性检验**（kernel 级 `[24,2112]`，`/w/tmp/gelu_cmp.py`）：

| 对比 | 不同元素 | max |
|---|---|---|
| 我们的 `gelu_tanh_mul`(mode 0) vs **compile 参考** | **0 / 50688** | 0.0 |
| 我们的 vs eager（`F.gelu(tanh)*u`） | 12670 / 50688 | 3.125e-2 |
| compile vs eager 参考自身 | 12670 | — |

→ **gelu 在 compile 口径下已经逐位正确**（现有 `mode=0`：全 f32、`tanhf`、最后舍一次）。
且证实 gelu 也是 compile≠eager 的一个点（12670 个元素差），只是我们已经站在 compile 这边。

**结论**：gelu 不是残余差异来源。条件 1 过、条件 2 未过（1.5221）。
**下一假设**：MoE 支路内部（专家 GEMM / combine 加权 / `moe_forward` 的 per-expert scale 折法），
以及 attention 的 `o_proj` 段。见下方新增假设 6–8。

## 新增假设（第 2 轮起）

6. **`moe_forward` 的 combine 口径**：`moe.cu::moe_combine` 有 `mode` 参数（vLLM cutlass_moe vs 旧口径），
   要确认 compile 下走哪条（默认值是否对）。
7. **专家 gate/up/down 的 GEMM 算法**：NVFP4 路径，compile 下是否仍走同一个（custom op 应是同一个）。
8. **attention 输出 → `o_proj`**：`unified_attention_with_output` 是 custom op，但 `o_proj` 前的输入量化点要对。

## 第 2 轮：抓 compile 中间层的三条路都不通（记录，避免重试）

| 方法 | 结果 |
|---|---|
| forward hook（eager 下能用） | compile 下**不触发**（图已 trace） |
| 替换 `torch.ops.vllm.<op>.default` | 破坏 `OpOverloadPacket`（`_dispatch_cache` 缺失），且 dynamo 在建图时已固化句柄 |
| 包 `MoERunner.forward`（Python 层） | compile 后**根本不调用**（直接跑图） |
| `TorchDispatchMode` | 只命中 `unified_attention_with_output`；`moe_forward`/`bmm_fp8` 在 warmup 期已消耗掉首次计数 |
| `.numpy()` 在 fake trace 里 | 崩（`FakeTensor` 无 `numpy`）；需 `torch.compiler.is_compiling()` 守卫 |

**结论**：compile 模式下拿不到层内中间量，只能做**整层输出**级对拍。
可行替代：截层 oracle（`kaon3_oracle_server.sh <N>`）做**逐层**比对；层内只能靠“改一个点 → 看整层是否变好”的 A/B。

## 第 3 轮：关键转向——差异在**层外**，不在 decoder 层内

1 层截断（同一 p02、同一 oracle 参数）：

| 对比 | 实际 token \|Δlp\| | top5 \|Δlp\| |
|---|---|---|
| 我们 vs compile | 1.5221 | 0.0097 |
| 我们 vs **eager** | **0.0564**（MODE=0）/ 0.0243（MODE=1） | 0.5391 |
| eager vs compile | **1.4892** | — |

**推理**：30 层时 eager vs compile = 1.5725（`test/kaon3/2026-10-23/e2e-eager-vs-compile.md`），
1 层时 1.4892。若差异来自层内，1 层应远小于 30 层；实测 1 层已占满 → **差异在层外**：
embedding / final norm / lm_head / softcap / logits。

**这解释了为什么逐层对齐做了 18 轮、端到端不动**：修的层是对的，但差的 1.5 不在层里。

**下一步**：查 compile 下输出路径
1. final norm 与 lm_head fp8 量化是否融合（同 input_norm→qkv 的模式，vocab=262144 大形状）
2. logits softcap 的数值与位置
3. lm_head 权重量化 / dequant 路径

## 第 4 轮（**纠正第 3 轮**）：1 层截断的数据是退化的，不能用来判断

| 层数 | eager top-5 logprob | 是否饱和 |
|---|---|---|
| **1 层** | `[-11.3745, -11.3745, -11.3745, -11.3745, -11.3745]`（五次完全相同） | ❌ **全部饱和到 softcap ±30** |
| 30 层 | `[-0.0188, -6.8938, -7.1438, …]`（有明显 top-1） | ✅ 正常 |

**1 层截断的模型输出幅度过大，几乎所有 logits 被 `tanh(x/30)·30` 压到 ±30 边界**，
top-5 全部相等（并列）→ 排序靠 tie-break 噪声决定 → `|Δlp|` 与翻转率**没有判别力**。

**所以第 3 轮的"差异在层外"结论作废**：那个推理建立在饱和数据上。
同时，前面所有基于 `kaon3_oracle_server.sh 1` 的 L1 对比（含 top5 0.0097/0.0102 这些数）都在退化数据上，
**不能作为判据**；它们能说明的只有"工具链通了"。

**有效的对拍层级**：
- 逐层：用 `--hidden-all` 比**残差流**（不经过 lm_head/softcap，无饱和问题）——这条一直是对的；
- 端到端：**必须 30 层**（`oracle_clean`）。

**下一步**：回到 30 层端到端。已知 `ours vs compile(oacle_clean)` = flip 16.56% / |Δlp| 1.1994，
`eager vs compile`（30 层，p02）= 1.5725。先测 `ours vs eager`（30 层）——若也很小，
说明我们贴 eager、差在 compile；若也大，说明是普遍 bug。
