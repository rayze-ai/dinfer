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
