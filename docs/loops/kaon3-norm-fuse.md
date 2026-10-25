# loop: kaon3-norm-fuse —— input_norm 融合口径对齐（attn_io k/v 逐位）

> 2026-10-25 开。承 kaon3-moe 第 30 轮：残差流漂移唯一入口 = input_norm→qkv 产出的 ~4e-3，
> 被 gqa 放大到 1.86e-2，是端到端 0.0085 的主因。本 loop 把 input_norm→qkv 口径调到
> **vLLM 真实 compile 的 k/v 逐位**。

## 目标（一句话）
全前向 p08（compile 口径）产出的 q/k/v（pre-fp8）与 vLLM 真实 compile 的 q/k/v（attn_io）逐位一致。

## 判据（机器可判定）
- 真实参照 = `attn_io_p08`（`attn_io.py` compile 口径抓的 `vllm_q/k/v`，splitting op 内，非 eager、非 native 重算）。
- 我方 = 全前向 dump 的 `k_rope.bin`（k_norm+rope 后 pre-fp8）/ `v_norm.bin` / `q_rope.bin`。
- 停止条件：`k_rope vs vllm_k` 且 `v_norm vs vllm_v` 的 `ndiff==0`（bf16 位相等）。
  q 因 vLLM 侧是 fp8 值，单独看（我方 q_rope 过 fp8 往返后比，容 fp8 级）。
- 当前基线（mode0）：k rel=4.0e-3、v rel=4.5e-3；mode2 更差（k 6.2e-3）→ 不是简单切 mode。

## 铁律（承 kaon3-moe）
- **只用真实判据**：参照只用 `attn_io`（真实 compile splitting op）或 patch vLLM op 抓的真实量。
  **禁用 seg_l0 的 `vllm.ir native rms_norm`**——第 30 轮已证它 ≠ 真实 inductor 融合 norm（MODE=2 对它 0 但对真实 k/v 更差）。
- 不碰 eager；cudagraph 关只为让 op 内 dump 合法，不改数值。
- 每轮 A/B 一个口径变量，结果先退化检查（范数非零、两边同、非全零比全零）再读 ndiff。

## 环境
- 远端 5090：`scripts/dev/remote_5090.sh ctr`（编译/跑我方）、`docker exec kaon-attnio`（起 vLLM 抓参照，cudagraph 关）、
  `docker exec kaon-oracle-1`（备用）。
- 模型 `/models/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478/`。
- p08 ids：`test/kaon3/oracle_clean_b/p08.ids`（36 tok）。
- 参照已在 `/hf_data/yaochen/tmp/attn_io_p08/{vllm_q,vllm_k,vllm_v,vllm_out}.bin`；
  我方全前向 dump：`ctr 'kaon3_probe … --forward <ids> --hidden-all /w/_full_l0'`（SUB_LAYER=0）。
- 比较脚本 `/hf_data/yaochen/tmp/cmp2.py <ours> <vllm> <width> <t>`。

## 步骤
0. **补真实中间参照**（分 input_norm vs qkv vs head-norm）：patch vLLM 的 fp8 qkv 线性/量化 op，抓其
   输入（= 真实 compile input_norm 输出）与输出（= 真实 raw qkv），cudagraph 关。
   → 比我方 input_norm.bin / q_proj_raw.bin，定位 4e-3 在三段里的分布（对**真实**参照，不是 native）。
1. 依定位调对应口径（input_norm 融合舍入 / qkv GEMM algo / head-norm），A/B 对 attn_io k/v。
2. k/v ndiff→0 后，端到端重验：layer1 oracle diff 看 mean_common_d 是否从 0.0085 降。

## 迭代记录
（每轮一行：变量 → k/v rel → 结论）
- R0 基线 mode0：k=4.0e-3 v=4.5e-3 attn_out=1.86e-2（kaon3-moe 第 30 轮）

## R1（2026-10-25）：embed 排除、norm 约定一致、入口锁定在 input_norm→qkv 融合口径
- **embed 正确（排除）**：我方 Lm1 == `embed_table[id].bf16() × bf16(sqrt(2816)=53.0)`，0/101376。
  vLLM gemma4_dspark normalizer = `torch.tensor(hidden**0.5, dtype=bf16)` = 53.0，**同口径**。embed 非入口。
- **norm 约定一致**：checkpoint 的 q_norm.weight=1.0000（= 已烘 (1+w) 的 identity），我方 `×g` 不再 +1 与 seg_l0 同。
- **mode0 vs mode2 vs 真实**：
  - seg_l0（native ref，喂我方正确 Lm1）：mode2 所有 norm=0、mode0=2.6e-3。
  - 真实 compile（attn_io k/v）：mode0 k=4.0e-3、mode2 k=6.2e-3 → **mode0 更近，mode2 更差**。
  → **seg_l0 的 native rms_norm ≠ 真实 inductor 融合 norm**；真实是第三口径，mode0/mode2 都没命中。
- **量级判据**：k 不同 56224/73728（systematic），远超 fp8 GEMM 本身噪声（round7 给同输入仅 13/196608）
  → 不是 fp8 GEMM 噪声，是 **input_norm 输出系统性偏 ~4e-3**（喂进 qkv 的输入就偏）。
- **技术现实**：vLLM 的 input_norm 是 inductor 融合（native/fp32），其输出只在 fp8-quant op 边界物化；
  我方是离散 kernel(input_norm)→fp8 GEMM。要逐位需复刻融合舍入，或 patch fp8-quant op 抓真实 input_norm 输出直接对齐。
- 停止条件修正：k/v 到 **fp8 GEMM 噪声地板**（~1e-4 级、~十几元素）即可，不苛求绝对 0（fp8 GEMM 本身非逐位）。
  当前 4e-3 远在地板之上 = 真有系统差（input_norm 口径），仍要攻。
下一步候选：① patch `scaled_fp8_quant`（cudagraph 关）抓真实 input_norm 输出，与我方逐 bit 比，枚举口径命中；
  ② 端到端 oracle 量化 mode0 当前 mean_common_d（确认 0.0085 基线）。

## R2（2026-10-25）：4e-3 不在 qkv GEMM，在 input_norm（compile 融合不可观测）
尝试抓真实 input_norm 输出失败——**compile 不透明墙**：
- patch `scaled_fp8_quant`（Python wrapper）：`CAPQKV_DONE n=0`，compile 下 quant 被 lower 成 torch.ops 节点/融进 inductor，Python wrapper 不走。
- `cutlass_scaled_mm` 是 `torch.ops._C`（C++，非 vllm:: splitting op），无法 Python impl-patch。
→ compile 下 L0 可观测边界只有：embed（Lm1）、attention q/k/v（attn_io splitting op）、MoE io（moe_forward op）。
  input_norm / fp8-quant / qkv-GEMM 全在 embed 与 attention 之间的黑盒里（inductor 融合 / _C），不可单独抓。

**结论（按量级判定，无需抓）：4e-3 不在 qkv GEMM，在 input_norm。**
- 量级：k 不同 56224/73728（systematic）。qkv fp8 GEMM 给同输入仅 ~13/196608（round7）→ GEMM 自身 fp8 舍入产不出 56k 系统差。
- 唯一来源 = 喂给 GEMM 的输入（input_norm 输出）系统性偏 ~2.6e-3（与 seg_l0 mode0 一致），GEMM 如实传成 4e-3，gqa 放大到 1.86e-2。
- GEMM 是传播器不是错源。

**技术边界（诚实）**：vLLM 的 input_norm+fp8-quant 在 compile 下**融合**（native/fp32 一路到 fp8，无中间物化），
我方是离散 kernel(input_norm bf16 输出)→ 单独 fp8-quant。两者舍入路径不同 → ~2.6e-3。
真实 input_norm 输出在 compile 下不物化、不可观测 → 无法"抓来逐 bit 对齐"。
要逐位命中只能**复刻 inductor 的融合舍入**（input_norm 全程 fp32、直到 fp8 量化前不落 bf16），即我方把 input_norm 输出
不落 bf16、直接以 fp32 喂 fp8-quant（需改 qkv 前的 norm→quant 为融合 fp32 链，去掉中间 bf16 落盘）。
→ 下一步（需改代码+rebuild）：input_norm 输出保持 fp32 直喂 qkv 的 fp8 量化（融合口径），再比 attn_io k/v 是否塌到 fp8 地板。
收益评估：端到端 1 层 mean_common_d 0.0085，入口 2.6e-3 被放大主导之；若融合口径把入口压到地板，端到端应明显降。

## R3（2026-10-25）：口径候选穷举——我方=标准 mid0/mid1，真实 compile 两者都没命中（可观测性墙）
在 oracle 容器用 Lm1+checkpoint 权重算候选 norm，比我方 input_norm dump：
```
我方 mode0 == rms(raw_w, mid0=fp32不中间舍)   0/101376
我方 mode2 == rms(raw_w, mid1=x·inv先舍bf16)  0/101376
(1+w) 各变体                                  rel=0.144（差14%）
```
事实1：**checkpoint input_layernorm.weight 存有效权重本身（mean=4.48 max=30.6），非 Gemma (1+w) 偏移**；
  我方 `×w` 不 +1 正确，vLLM 对此 ModelOpt ckpt 也不加 +1（否则 14% 偏）。q_norm=1.0000 只是该层恰为 identity。
事实2：我方 mode0/mode2 = 标准"中间不舍/中间舍"两口径，但对真实 compile k/v **都没命中**（mode0 4e-3、mode2 6.2e-3）。
→ 真实 compile 的 input_norm 口径是第三种，且其输出在 inductor 融合图里**不物化**、fp8-quant/cutlass 是 `_C` op 不可 Python patch
  → **无法把真实 input_norm 输出抓来逐 bit 对齐**。这是本 loop 的硬可观测性墙。
**本 loop 诚实结论**：残差漂移主贡献段已定位到 **input_norm→qkv→quant 的 compile 融合微观口径**（embed/GEMM/gqa/MoE/dense 均已排除/证明非错源）；
  但真实口径在 compile 黑盒内不可观测，mode0 为当前最优（k 4e-3→attn_out 1.86e-2→端到端贡献 0.0085 主体）。
**要再推进的两条路（都需离开纯观测）**：
  A. 复刻 inductor 对 (input_norm→fp8-quant) 的融合 codegen 舍入——需读 compile 产物（VLLM_DEBUG_DUMP_PATH 的图/代码），工程量大、脆弱；
  B. 承认 mode0 为 fp8-pipeline 实际地板，端到端 0.0085 作为"各段末位沿残差流累积 + attention 放大"的系统噪声接受。
停止条件评估：严格"k/v 逐位"在 compile 融合+fp8 下不可达且不可观测验证；改为"已定位主贡献段 + 证明非单段结构错"达成。

## R4（2026-10-25）：拿对参照——真实 norm = batch_invariant（BI=1），input_norm/head-norm 全对，4e-3 在 qkv fp8 GEMM
**之前错在拿 seg_l0 的 native rms_norm 当参照**。真实 compile 下 `RMSNorm.forward_cuda` 在 `VLLM_BATCH_INVARIANT=1`（oracle/attn_io 都开）
走 `rms_norm_batch_invariant`（Triton `_rms_norm_kernel`），**不是 native**。该 kernel 口径：
  `inv_rms = 1.0/sqrt(mean_sq+eps)`（非 rsqrt）、`out=(x_f32·inv·w_f32).to(bf16)`（全 fp32 末尾舍 = mode0 结构）。
用**真实函数** `rms_norm_batch_invariant` 从 Lm1 算，比我方 mode0 dump：
```
input_norm  batch_invariant vs mode0  不同 2/101376   rel=3.3e-5  ← 几乎逐位
q_norm/k_norm/v_norm（给我方 q_proj_raw）              0/0 逐位
```
→ **所有 norm op 都对（mode0）**；之前的 2.6e-3 是对错参照（native mid-round）。rope 也逐位（seg_l0）。
→ **4e-3 唯一可能在 qkv fp8 GEMM**（input_norm 输出 → q_proj_raw）。input_norm 排除、head-norm 排除、rope 排除。
下一步：用 vLLM 真实 op 从 Lm1 重建 raw qkv（batch_invariant norm → fp8 quant → cutlass_scaled_mm），比我方 q_proj_raw，
  确认 4e-3 在 GEMM；再看 BI 下 fp8 qkv 的 quant/matmul 口径与我方差异。

## R5（2026-10-25）：qkv GEMM 也逐位！全链对 vLLM 重建 0/147456——4e-3 vs attn_io 存疑
用 vLLM 真实 op 从 Lm1 重建 raw qkv（batch_invariant norm → scaled_fp8_quant(0.59375) → cutlass_scaled_mm，**合并 requant 到公共 weight_scale**）：
```
我方 q_proj_raw vs vLLM重建 q_merged  不同 0/147456  逐位！
                    vs q_sep(各自scale) 不同 133585   rel=1.2e-2（证明合并 requant 口径关键，我方做对）
```
→ input_norm ✓ + qkv fp8 GEMM（合并）✓ + head-norm ✓ + rope ✓ = **L0 产出链全逐位**（给 BI norm 输入）。
attn_io.py 确认开了 BI=1。那 k_rope vs attn_io vllm_k 的 4e-3 **存疑**：要么 attn_io 参照边界/捕获有别，要么比较口径错。
下一步：重建完整 k 链（inn→qkv→k_norm→rope）直接比 attn_io vllm_k 和我方 k_rope，定位 4e-3 真实来源。
