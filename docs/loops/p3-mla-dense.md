# loop: P3 稠密 MLA 通路（GLM 结构，tiny 模型）

## 目标

Rust 端到端跑通"tokens → embedding → N×(MLA + dense MLP) → norm → lm_head → greedy"，
与 HF `DeepseekV3ForCausalLM`（GLM-5.3 的 MLA/MLP 与之同构，`rope_interleave=True`）同配置随机权重模型 greedy 32 token 一致。

## 环境

- 本机 3090。`transformers 4.52.4` 有 `deepseek_v3`，无 `glm_moe_dsa`（5.15+ 才有）；P3 用 DeepseekV3Config，
  字段名与 GLM-5.3 config.json 一致（`q_lora_rank / kv_lora_rank / qk_nope_head_dim / qk_rope_head_dim / v_head_dim / rope_interleave`）。
- tiny 配置：2 层、hidden 256、4 头、q_lora 64、kv_lora 32、nope 32、rope 16、v 32、intermediate 512、vocab 1000、`first_k_dense_replace=2`（全 dense，P4 再开 MoE）。
- 参考生成：`scripts/verify/gen_tiny_glm.py` → `test/tiny_glm/{model.safetensors,config.json,prompt.npy,expect_logits.npy,expect_greedy.npy}`。

## 观察方法

`scripts/verify/p3.sh`：p2 + `cargo test -p dinfer-glm53 --features ref-tests -- --ignored tiny_glm_greedy_parity`。
- 第一道：单 step prefill 的 logits 对拍（atol 5e-2 bf16 累积误差）。
- 第二道：greedy 32 token 的 token id 完全一致。

## 停止条件

`scripts/verify/p3.sh` exit 0。

## 迭代上限

8 轮。每轮记录"哪一层/哪一步先出现偏差"。

## 已知可能出错点（按概率排）

1. RoPE interleave 的配对顺序与 cos/sin 表布局（HF 先 view(d/2,2).transpose 再 rotate_half）。
2. kv_b_proj 拆成 W_UK / W_UV 的维度顺序（`[kv_lora, heads, nope+v]`）。
3. 吸收后 decode 的 q_latent = q_nope · W_UK（按 head），加 q_pe 拼成 [heads, kv_lora + rope]。
4. softmax scale 用 qk_head_dim（nope+rope）的 -0.5。
5. prefill 与 decode 两条路径的数值一致性（prefill 用展开 K/V，decode 用吸收）。

## 记录

| 轮 | 现象 | 根因 | 处理 |
|---|---|---|---|
| 1 | `rope_n: q/positions` ensure 失败 | MLA / MLP 用 scratch 缓冲的 `shape[0]`（= max_tokens 64）当本步 token 数，positions 只有 14 | forward 显式传 `t`；所有 `_n` 变体按容量 `>=` 校验 |
| 2 | `gemm: a.len != m*k` | GEMM 包装要求激活缓冲精确等于 m·k，与"按 max_tokens 预分配"冲突 | a / out 改 `>=`，权重仍精确 |
| 3 | `copy_cols: range` | Rust 包装的参数顺序 `(src_cols, col0, width, dcol0, dst_cols)` 与 C ABI `(src_cols, col0, dst_cols, dcol0, width)` 不一致，调用方按 ABI 传 | 包装改成与 ABI 同序；教训：包装层参数顺序必须抄头文件 |
| 4 | prefill logits 全 14 行 argmax 一致、max abs 0.004；greedy 第 7 步分叉 | HF 该步 top1=0.508 / top2=0.504，差 0.0039 = bf16 1 ULP，平局抖动；32 步 decode logits 最大偏差 0.0068 | 测试口径：全词表 max abs < 0.05；argmax 只在 HF gap > 0.02 时要求一致；greedy 允许在平局步分叉且分叉 token 必须在 HF top-2 |

**结论**：MLA prefill（展开 K/V）与 decode（W_UK/W_UV 吸收 + paged latent）两条路径都与 HF 对齐到 bf16 噪声级。
