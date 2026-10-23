# loop 规格：kaon3-compile

- 状态：**进行中**（2026-10-23 创建）
- 创建：2026-10-23
- 验证脚本：`scripts/verify/kaon3-compile.sh`
- 迭代记录：`test/kaon3/2026-10-23/kaon3-compile.md`
- 上游：`docs/loops/kaon3-parity.md`（子 loop D）、`docs/loops/kaon3-gelu.md`

## 口径（用户 2026-10-23 明确）

**全程 compile 模式，环境必须干净，不许拿 eager 当靶子、也不许用 eager 抓的中间量做判据。**

- 参照 = vLLM `torch.compile`（服务实例不设 `enforce_eager`；vLLM 默认即 compile，
  `vllm/config/vllm.py:1379`）。生产也是它。
- 我们侧 = `RMSNORM_MODE=0`（compile 口径的 norm），**探针与整跑一律带这个默认**。
- eager（`RMSNORM_MODE=1` / `enforce_eager=True` / `kaon3_vllm_hidden.py` 抓的 `/w/hid2/*`）
  只能用于"诊断哪种实现更接近"，**不得写进任何判据、不得作为 FLOOR 分母**。
- 已知不变量：compile 逐位确定（同配置两进程 → 0.0000），所以**目标是逐位 0，噪声底为 0**。

## 目标

`models/kaon3` 的 Rust 前向在 **p02 等 30 个用例**上与 **compile 参照**（`test/kaon3/oracle_clean`）
达到 FLOOR 线：可判定行 top1 翻转率 ≤ 1.5×FLOOR + 3pp、共有 token `|Δlp|` ≤ 1.5×FLOOR，
自然长文本 `|ΔNLL|` ≤ max(3×|FLOOR|, 0.01)。现状 **flip 16.56% / |Δlp| 1.1994**（FLOOR 0.83% / 0.0276）。

## 环境

- 机器：5090，`scripts/dev/remote_5090.sh`（ControlMaster；断了向用户要 token）。cwd=`/w/dinfer`。
- GPU 分工：**0 / 2–6 = 我们**，**1 = 私有 oracle**，**7 = 共享基线（禁用）**。
- 编译（每轮都要，sm_120 本机编不了）：
  ```bash
  GPU=0 scripts/dev/remote_5090.sh ctr 'cd /w/dinfer && CUDA_ARCHS=120 cargo build --release -p dinfer-kaon3 --bins'
  ```
- 参照服务（**干净启动 = 唯一允许的起点**）：
  ```bash
  scripts/dev/remote_5090.sh run 'docker rm -f kaon-oracle'      # 先拆，防上次遗留的截层/改参实例
  scripts/dev/kaon3_oracle_server.sh                              # 无参数 = 全 30 层、无 --enforce-eager、BI=1
  ```
  启动后**必须核三项**，任一不符就重来：① 容器在跑且端口是它自己的（脚本已查）；
  ② `num_hidden_layers` 没出现在 cmd 里（= 全 30 层）；③ 环境里没有 `VLLM_*EAGER*`。
- 重置到干净状态（每轮开始前）：
  ```bash
  scripts/dev/remote_5090.sh run 'docker rm -f kaon-oracle' && scripts/dev/kaon3_oracle_server.sh
  scripts/dev/remote_5090.sh run 'nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader'  # 无我们自己的残留
  ```
  （2026-10-23 实测：上一轮会话残留了一个 `--hf-overrides num_hidden_layers:1` 的截层实例，
  任何读数都会是错的——**先拆再起**是硬步骤，不是建议。）

## 观察方法

| 看什么 | 在哪里 | 命令 |
|---|---|---|
| 端到端判定 | `kaon3-parity.sh` 汇总行 | `PROBE_ENV="CUBLASLT_WORKSPACE_SIZE=1 RMSNORM_MODE=0" bash scripts/verify/kaon3-parity.sh` |
| 单点 A/B | 同一份二进制，只换环境变量 | 见下"开关表" |
| 截层（仅当不饱和） | 截层 oracle + `--layers N` | `scripts/dev/kaon3_oracle_server.sh <N>`；**先验 top-5 不并列**，并列即作废 |
| inductor 生成代码 | 远端正向导出目录 | 需重建：`/w/tmp/ind` 在 2026-10-23 已丢失，按 `kaon3-gelu.md` 第 2 轮方法重导 |

**开关表**（一次只改一个，默认值 = compile 口径）：

| 变量 | 默认 | compile 期望 | 作用点 |
|---|---|---|---|
| `RMSNORM_MODE` | 1（待改回 0） | **0** | norm 数值口径 |
| `NORM_FUSE_QUANT` | 0 | 未定（inductor 融合链） | norm→fp8 量化是否融合 |
| `ROUTER_F32` | 1（on = 精确） | 未定 | router bf16 舍入点 |
| `MOE_COMBINE_F32` | 0 | 未定 | 专家 combine 精度 |
| `GELU_F32` | 1 | 已确认对 compile 逐位 0 | gelu·mul |
| `CUBLASLT_WORKSPACE_SIZE=1` | — | **测量前置条件** | 固定 cuBLASLt algo，否则 router GEMM 选错 algo 制造假发散 |

**每轮读数必须自证**：① `sync` 后容器内 `grep` 到新代码；② A/B 两次输出 md5 不同、日志命中标志；
③ 跑前跑后 `nvidia-smi` 无自己的残留进程。

## 停止条件（可机器判定，`scripts/verify/kaon3-compile.sh`）

1. **口径干净**：oracle 容器在跑、cmd 里无 `num_hidden_layers` 截断、env 无 `enforce_eager`；
   探针以 `RMSNORM_MODE=0` 运行（脚本显式注入）。
2. **端到端**：`flip_pct ≤ 1.5×FLOOR_flip + 3` **且** `mean_common_d ≤ 1.5×FLOOR_d`，FLOOR 由
   `kaon3_oracle_diff oracle_clean_b oracle_clean` 现算（不写死 0.83/0.0276，它们会随参照重抓而变）。
3. **长文本**：`|ΔNLL|(long_zh) ≤ max(3×|FLOOR|, 0.01)`。
4. **单 token**：`bos1` top1 翻转数 = 0。

四条全满足 exit 0。

## 上限

- 最多迭代：12 轮
- 到上限仍未通过：停下，把已排除的假设写进"卡点"，交给用户。中途不停下、不中途问。

## 假设清单（按怀疑度排序，只列 compile 口径下**未验证**的点）

1. **`RMSNORM_MODE=0` 整跑**——交接区点名的"唯一还没验过的已知差异点"（现状读数 16.56% 是在 MODE=1 下得的）。
   先跑它，再看要不要往下走。
2. **层外：`final_norm` + `lm_head` fp8 量化**——vocab=262144 的大形状，inductor 可能把 norm 与量化融成链
   （和 `input_norm→qkv` 同一模式）。分层对拍覆盖不到这里。
3. **`logits` softcap**——`tanh(x/30)·30` 的位置与精度（是否 f32 全链、舍入次数）。
4. **MoE combine**——`MOE_COMBINE_F32` 两种口径哪一个是 compile 的真实行为。
5. **attention `o_proj` 前的量化点**——`unified_attention_with_output` 是 custom op，出口是否又量化一次。
6. **router 预处理 `rmsnorm_scale2`**——fp32 融合链目前"kernel 级差半 ulp"，未逐位。

## 卡点（到上限或放弃时填）

（未开始）

## 最新一次结果

- 日期：2026-10-23
- 第 0 轮（基线，MODE=1 得的历史数）：flip 16.56% / `|Δlp|` 1.1994 / FLOOR 0.83% / 0.0276
- 第 N 轮：

## 进度快照（2026-10-24 凌晨，loop 进行中）

- **第 1 轮**（`RMSNORM_MODE=0`）：flip **16.98%** / |Δlp| **1.1816**（基线 MODE=1 是 16.56% / 1.1994）
  → **compile 口径的 norm 不是主因**，交接区点名的"唯一未验差异点"已排除。
- **第 2 轮**（读 FX 图）：从 oracle 容器 `docker cp` 出 `computation_graph.py`（12451 行，**不用重导 inductor**）。
  读出：norm 返回 bf16 / 量化是 `f32×reciprocal(scale)` → clamp ±448 → fp8 / router 预处理三次 bf16 舍入 /
  router GEMM 走 **BI Triton persistent matmul**（不是 cuBLAS）/ `v_norm` weight=None / q 进 attention 前是 fp8。
- **第 3 轮**（`RMSNORM_MODE=2`：`x·inv` 后先舍 bf16 再乘 w）：照 FX 图 + `ir/ops/layernorm.py:17` 源码。
  第一次因 `norm.cu` 签名漏 `mode` 编译失败**且未 sync**，脚本只看 `^error` 没拦住 → **作废**；
  已加**编译门禁**（匹配 `cargo:warning=.*error:` 与 `nvfb failed`）后重跑。结果见迭代记录。

### 已排除的假设（别重复）

| 假设 | 结论 |
|---|---|
| lm_head 量化 | ❌ `quantized_layers` 424 项里没有 lm_head / embed |
| `fused_add_rms_norm` 残差语义 | ❌ Gemma4 用显式 bf16 加法（`gemma4.py:702-731`） |
| q 未做 fp8 量化 | ❌ 我们已做（`layer.rs:753`），A/B 证实必需 |
| `oracle_clean` 口径不对 | ❌ 与当前 compile 服务重抓逐项 0 差异 |
| compile 口径 norm（mode 0） | ❌ 第 1 轮：与 eager 口径无实质差别 |

### 尚未核对（按嫌疑度）

1. `bmm_fp8` ×120（fp8 GEMM 的 scale 应用顺序与 out_dtype）
2. `moe_forward` ×30（combine 口径 / per_expert_scale 折进位置）
3. router 的 BI persistent matmul（我们用 cuBLAS → 需 `CUBLASLT_WORKSPACE_SIZE=1` 才稳）
4. `unified_kv_cache_update` 的 kv fp8 量化点

## 第 5 轮主线：复刻 router 的 BI persistent matmul（规格已从源码读出）

**参照路径**（`vllm/model_executor/layers/batch_invariant.py:55-138` + `:155-165`）：

| 参数 | bf16 取值 |
|---|---|
| `BLOCK_SIZE_M` | **128** |
| `BLOCK_SIZE_N` | **128** |
| `BLOCK_SIZE_K` | **64** |
| `GROUP_SIZE_M` | 8 |
| `num_stages` / `num_warps` | 3 / 8 |

**归约结构（决定逐位结果）**：
```
accumulator = tl.zeros((BM, BN), fp32)
for ki in range(k_tiles):                       # K 方向严格顺序，ki = 0,1,2,…
    accumulator = tl.dot(a[ki], b[ki], accumulator)
c = accumulator.to(bf16)                        # 只在最后舍一次
```
`tl.dot` 在 bf16 上发 `mma.m16n8k16`：每个 `BLOCK_SIZE_K=64` 块内是 4 条 k16 mma **顺序累加**，
块与块之间也是顺序累加。**全程一个 fp32 累加器，K 方向零并行**。

我们的 router 走 cuBLAS bf16 GEMM（要 `CUBLASLT_WORKSPACE_SIZE=1` 才稳定），归约树与上表无关。

**为什么这条必须做（不再是"试试"）**：
- 第 4 轮证明：低于 bf16 ulp 的差异（rcp.approx 2⁻²³ vs bf16 2⁻⁹）**会被舍入吞掉**，
  所以剩下的分歧必然在 **bf16 ulp 级别**；
- `num_experts=128` 选 8、30 层 → bf16 ulp 级的 logits 偏差会在 top-8 边界换专家，
  换一个专家就是该 token 的 MoE 输出整块变化（实测：个别位置间隔比差 9.7 倍、top-1 翻转，其余位置 Δ≈0）。
- **只要 router GEMM 不是同一归约结构，端到端就不可能收敛到 FLOOR。**

### 实现要点

新增 kernel（`crates/kernels/cuda/`），专门服务 router 这种 `N` 很小的形状（`N=128`）：
- 每个 block 负责 `BM=128` 行 × `BN=128`（整块 N），K 方向 64 一块顺序累加；
- fp32 累加器，末尾 `to(bf16)` 一次；
- 由 `ROUTER_GEMM_BI=1` 选（默认关，A/B 后再定默认）；
- 判据：**先做 kernel 级对拍**（同入参 vs 参照 BI matmul 逐位 0），再端到端看 flip。

### 第 5 轮实现计划（进行中）

新增 `router_gemm_bi_f32(x[M,K] bf16, w[N,K] bf16, out[M,N] f32)`，逐位复刻参照归约：
- `BM=128`、`BN=128`（N=128 正好一整块）、`BK=64`，K=2816 → 44 个 chunk；
- 每个 chunk 内 4 条 `mma.m16n8k16` **顺序**累加进同一个 fp32 累加器；
- 输出 `f32(bf16(acc))`（参照是 bf16 输出再 cast f32）；
- 8 warp 各管 16 行，每 warp 16 个 `n8` tile。
- 开关 `ROUTER_GEMM_BI=1`；**先做 kernel 级对拍**（vs 容器内 `matmul_persistent`，要求 0 差异），再端到端。

## 第 6 轮方向：换核查手段（逐层参照在 compile 下拿不到）

**已确认的约束**（交接区 + gelu loop 第 2 轮，四种方法全失败）：
compile 模式下 forward hook 不触发、op 不能替换、Python 层不被调用 → **拿不到参照的层内/逐层中间量**。
`kaon3_vllm_hidden.py` 硬编码 `enforce_eager=True`，它产出的参照是 **eager 口径**，按本项目口径**不能作判据**。

所以剩下的手段只有两条：
1. **端到端单点 A/B**（唯一可靠）：改一个点 → 整跑 → 看 flip/Δlp 是否显著下降；
2. **kernel 级逐位对拍**（若能在容器里直接调参照算子的真实实现）。

**已排除（本轮新增）**：
| 假设 | 结论 | 依据 |
|---|---|---|
| router GEMM kernel 不同 | ❌ 真实输入下 cuBLAS 与 BI matmul **逐位相同**（0/3072） | 第 5 轮 DIAG |
| `rcp.approx` vs 精确除法 | ❌ 误差 2⁻²³ 被 bf16 舍入吞掉 | 第 4 轮 md5 相同 |
| norm 口径（mode 0/1/2） | ❌ 三者 flip 都在 16% | 第 1/3 轮 |

**剩余候选**（按可 A/B 性排序）：
1. `MOE_COMBINE_F32`（combine 时权重是否落 bf16）——已有开关，直接 A/B；
2. `NORM_FUSE_QUANT`（norm 与 fp8 量化是否融合）——已有开关，直接 A/B；
3. 专家 GEMM 本身（`forward_nvfp4` 与参照 cutlass_moe 的归约）——需要新开关；
4. attention 的 `o_proj` 段 / `unified_kv_cache_update` 的 kv 量化点。

## 第 6 轮（进行中）：`MOE_COMBINE_F32` 默认改为 1

**依据**（p02 单例 A/B，同环境只切这一个开关）：

| | top-1 不一致 | top5 \|Δlp\| | top1 \|Δlp\| max | 实际 token \|Δlp\| |
|---|---|---|---|---|
| `MOE_COMBINE_F32=0`（原默认） | 4 / 22 | 1.8089 | 2.0405 | 2.1794 |
| **`MOE_COMBINE_F32=1`（新默认）** | **0 / 22** | **1.3760** | **0.5195** | **1.1844** |

原默认值注释写的是"vLLM cutlass_moe 口径（权重先落 bf16、每项 bf16(w·y)）"——
参照实际跑 Triton/inductor 融合路径，**不做那次 bf16 舍入**。这是本轮唯一产生质变的开关。

已改默认并跑完整 30 用例验证（`RMSNORM_MODE=2`）。

### 第 7 轮候选：专家 GEMM（NVFP4）

若第 6 轮后仍不达标，下一个最有嫌疑的是**专家自身的 GEMM**：
`moe.cu` 里逐专家走 `forward_nvfp4`，而参照是 **cutlass_moe 融合 kernel**（`moe_forward` op）。
两者对 NVFP4 的块缩放与累加次序不同 → 每个专家输出差 ulp 级 → 同样经 8 专家加权后放大。

验证手段与 router 一样：**抓真实专家输入，同输入跑两边逐位比**（不要用随机数据，
第 5 轮的教训：随机输入与真实输入的动态范围差一个量级，结论会跑偏）。

## 进度（2026-10-24 01:0x，loop 进行中，未达停止条件）

**第 6 轮已验证的语法点**（p02 单例，同环境只切一个开关）：

| 开关 | top-1 | top5 \|Δlp\| | top1 max | 实际 token |
|---|---|---|---|---|
| `MOE_COMBINE_F32=0`（原默认） | 4/22 | 1.8089 | 2.0405 | 2.1794 |
| **`MOE_COMBINE_F32=1`（已改默认）** | **0/22** | 1.3760 | **0.5195** | 1.1844 |
| `NORM_FUSE_QUANT=1` | 3/22 | 1.8046 | 2.6193 | 1.7220 |

**下一步（按序）**
1. 第 6 轮全量结果出来后，若 flip 明显下降 → 在其基础上继续 A/B；
2. 若仍 ~16% → 第 7 轮查**专家 NVFP4 GEMM**（抓真实专家输入，同输入两路逐位比，**不要用随机数据**）；
3. 每轮结果写 `test/kaon3/<日期>/kaon3-compile.md`，有效判断提炼进 `docs/knowledge/`。

**已排除（勿重复）**：norm 口径（mode 0/1/2 都是 16%）· `rcp.approx`（被 bf16 吞掉）·
router GEMM kernel（真实输入下 cuBLAS 与 BI matmul 逐位相同，0/3072）· lm_head 量化（未量化）·
`fused_add_rms_norm` 残差语义（Gemma4 用显式 bf16 加）· q 的 fp8 量化（已做且 A/B 证实必需）。

## 第 7 轮（进行中）：专家 NVFP4 GEMM 逐位核验

**方法**（沿用第 5 轮修正后的正确做法：真实数据 + 真实调用路径）：
在 oracle 容器里 patch `MoERunner.forward`，抓**第一个 MoE 调用的真实**
`x`（专家输入）、`topk_weights`、`topk_ids`、`y`（MoE 输出）→ `/w/tmp/moe1/*.bin`。
然后用同一组 `x / ids / w` 跑我们的专家 GEMM + combine，与 `y` 逐位比。

脚本：`/tmp/moe_probe.py`（容器内 `M=<model> IDS=<ids> python3 /tmp/moe_probe.py`）。

**判读规则**：
- 若专家输出逐位相同 → 假设排除，往下查 attention / kv；
- 若有差异 → 差异在 NVFP4 的块缩放或累加次序，需要按参照 cutlass_moe 的归约结构复刻。

## 第 7 轮末：MoE 差异已定量，下一步看参照的专家 kernel

**已定量**（真实数据，`models/kaon3/tests/moe_stage_parity.rs`）：
```
MOE_STAGE [T=24 H=2816 E=128 K=8] 48 个被选专家 | ndiff=50214/67584 max|d|=2.500e-1
‖ours‖/‖ref‖ = 1.0021      ← 量级对，语义/路由/scale 折法都对
前 8 个值里 5 个逐位相同，其余差 1–2 个 bf16 ulp
```
→ 差异在**累加归约次序**，不在语义。

**参照配置**（oracle 日志）：`moe_backend='auto'`、`quantization=modelopt_mixed`、
候选里有 `CutlassBatchedExpertsFp8 / CutlassExpertsFp8 / CutlassExpertsW4A8Fp8 / FusedMoEExpertsModular`。
且 combine 是在**融合 kernel 内部**完成的（`MoERunner` 的 `_fused_output_is_reduced`），
我们是"逐专家 GEMM + 单独 scatter-add"——两者归约树不同。

**下一步**：确定参照实际命中哪个 experts 实现，再决定：
- 若是 `CutlassExpertsW4A8Fp8`（NVFP4 权重 + FP8 激活）→ 需要按它的 tile/归约结构复刻；
- 若是 modular 的 prepare/finalize 路径 → combine 的累加顺序也要照抄。
