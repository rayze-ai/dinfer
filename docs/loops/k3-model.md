# Kimi-K3 里程碑 C-3：MLA 层 + 多层主干真权重对拍

> 状态：**进行中（2026-10-20，B300-50）**。crate 侧 `models/kimik3/src/{mla.rs,model.rs}`（`layer.rs` 的 `Attn::Mla` 接上），
> 测试 `models/kimik3/tests/model_parity.rs`，外壳 `scripts/verify/k3_model.sh`（exit 0 = 过），参考 `scripts/verify/gen_k3_model_refs.py`。
> 前置：C-1（`k3-kda.md`）、C-2（`k3-moe.md`「C-2」）。

## 与 HF 代码核对（`modeling_kimi_linear.py`，唯一口径）

### MLA（`KimiMLAAttention`）

| 项 | HF | 任务书 / 我们 |
|---|---|---|
| RoPE | `self.rotary_emb = None`，`assert self.use_nope`；`q_rot` / `k_rot` **存在但不旋转**，直接 `cat` 进 q / k | 任务书问"rope 64 维存在但不转？"→ **是**：q 每 head 尾 64 维 + `kv_a_proj_with_mqa` 尾 64 维（所有 head 共享）原样参与点积。KV latent 仍是 `512 + 64 = 576` 维，吸收路径与 GLM 相同，只少旋转 |
| scale | `q_head_dim ** -0.5 = 192^-0.5` | 同 |
| v pad | flash_attention_2 路径把 v 从 128 pad 到 192 再截回 | 等价于 `dv = 128` 直接算（pad 的列全 0，不影响 softmax·v） |
| 输出门 | `attn_output * g_proj(x).sigmoid()`，`g_proj: [H·v, hidden]`，**o_proj 之前** | `mul_sigmoid_bf16`（Qwen3.5 同 kernel）→ `o_proj` |
| 投影融合 | HF 分片 `q_a_proj [1536,7168]` / `kv_a_proj_with_mqa [576,7168]` / `g_proj [12288,7168]` 分开存；vLLM 合成 `fused_qkv_a_g_proj` | 装载时按行拼成一条 `[14400, 7168]` GEMM（顺序同 vLLM），输出 `copy_cols` 拆四段 |
| a_layernorm | `KimiRMSNorm(q_lora)` / `KimiRMSNorm(kv_lora)`，eps 用 **默认 1e-6**（`KimiRMSNorm(hidden_size)` 没传 eps！） | **待核**：我们传的是 `rms_norm_eps=1e-5`；差异在 `rsqrt(mean+eps)` 的 1e-5 vs 1e-6，对 |x|~1 的向量影响 ~1e-6 相对，远低于 bf16，但口径上应改成 1e-6（见下方 TODO） |
| prefill | 展开 k = `[kv_b(kv_c).nope \| k_pe 广播]`，v = `kv_b(kv_c).v` | `mla_prefill_bf16`（dq 192 / dv 128，`mla.cu` 的 `MAX_DQ 320 / MAX_DV 256` 够） |
| decode | — | 吸收：`q_lat = [q_nope·W_UK \| q_pe]`，`mla_decode_bf16` 读 paged latent，`o = o_lat·W_UV^T`（每 head 一次小 GEMM × 96 × 2，GLM 线同样待 batched） |

### 主干（`KimiLinearModel.forward`）

- `block_residual` 起始 `[T, 0, hidden]`（空）；每层 `_forward_attn_residual` 返回 `(prefix_sum, block_residual)`；写块层
  （`layer_idx % 12 == 0`）把**本层输入**追加进列表 → 第 k 个块 = 第 `12k` 层的输入（k=0 = embedding 输出）。
- 层循环后 `_apply_output_attn_res(hidden, block_residual)`（候选 = 全部块 ∪ {最终 prefix_sum}，`output_attn_res_{proj,norm}`）→ `norm`。
  我们：`AttnRes::load_output` 把同一组权重装进两个槽，`apply(which=0)`。
- 只加载前 N 层时，`output_attn_res + norm + lm_head` 接在第 N-1 层之后（参考脚本同口径：等价于 `num_hidden_layers = N` 的截断模型）。
- `embed_tokens / norm / output_attn_res_* / lm_head` 全在 `model-00094-of-000096`（`weights::shards_for_top`）；每层恰好一片（`model-{N+1:05d}`）。

## 环境

- B300-50，权重 `/data/models/moonshotai/Kimi-K3`（物理路径）。Rust 测试 GPU 2，容器参考 GPU 3（0/1 Qwen3.5 占用）。
- 参考：`vllm/vllm-openai:nightly` 容器（无 fla 包 → 不能实例化 HF 类），`gen_k3_model_refs.py` 纯 torch 按 HF 逐层：
  KDA 复用 C-1 的 fla 路径（预 sigmoid beta），MoE 复用 C-2 的 MXFP4 解包路径，MLA 本文件 f32 causal attention，AttnRes / RMSNorm 同 C-2。
  真 token：HF tokenizer（`trust_remote_code`，容器里有 tiktoken）编码一句中英混合文本，T ≈ 32；tokenizer 不可用时退回固定伪随机 id（meta.json 记 `tokenizer`）。
- 前 N 层默认 **N = 5**：第 0 层 KDA+稠密 MLP（写块）、1/2 KDA+MoE、**3 MLA+MoE**、4 KDA+MoE，三种层都覆盖。显存估算：
  顶层 embed + lm_head 2 × 2.35 GB，第 0 层 ~2.3 GB，MoE 层各 ~17 GB（专家 packed 15.72 + 其余 ~1.3）→ 5 层 ≈ 75 GB；14 层 ≈ 230 GB 上限。
- 跑法：`B300_TAG=50 scripts/dev/remote_b300.sh run 'cd /data/yaochen/infer/dinfer && source /data/yaochen/infer/env.sh && scripts/verify/k3_model.sh'`；
  `K3_SKIP_GEN=1` 复用已生成参考；`K3_LAYERS=N` 改层数（参考与 Rust 同时生效）。

## 观察方法

`model_parity.rs` 打印：装载秒数 / 显存；每层 `rel max_err / mean_abs / cos` + 末 token 范数（ours / HF）；`output_attn_res`、`final(norm)`、
末 token logits（rel / cos + top-5 双方）；decode（prefill T-1 + decode 1）的 final 与 logits 对整段 prefill；最后汇总表。`RUST_LOG=info` 时 `model.rs` 每层装载一行。

## 停止条件（机器可判定，`k3_model.sh` exit 0）

- 每层 `cos ≥ 0.999 && rel ≤ 5e-2`；`output_attn_res` / `final` / 末 token logits 同门槛；末 token **top-1 一致**；
- decode final vs prefill[-1] `cos ≥ 0.999 && rel ≤ 5e-2`；decode logits vs HF `cos ≥ 0.999`。
- 容差口径同 C-1 / C-2（bf16 链路 + MXFP4 专家 + 路由平局行；见 `docs/pitfalls.md`「硬判决不可逐位复现」）。

## 迭代上限

参考生成 + 对拍一轮约 10 min（5 个 MoE 层各解包 ~500 专家）。上限 6 轮；不过则按层定位（哪一层先掉 cos）→ 子块（attn / mlp 门 / MoE）单独对拍。

## 轮次记录

（见下方「结果」，完成后填）

## 结果

（待填：装载时间 / 显存 / 每层数字 / decode / commit）

## TODO（C-3 之后）

- `q_a_layernorm / kv_a_layernorm` 的 eps：HF `KimiRMSNorm(q_lora_rank)` 用默认 1e-6，不是 `rms_norm_eps`；vLLM 用 `config.rms_norm_eps`。
  两边口径不一致，差异远低于 bf16，先记着。
- 吸收路径 96 头 × 2 条小 GEMM → batched / 一条 kernel；`fused` GEMM 输出的四段 `copy_cols` 可由 kernel 吃 stride 省掉。
- AttnRes 融合 kernel（C-2 TODO，现在每层 2 次 × 候选数的 D2H）；`model.rs` 的 `blocks` 拷贝可改成指针轮换。
- 全 93 层：单卡放不下（专家 packed 15.72 GB × 92 = 1.45 TB）→ 8 卡 EP / TP 切分是 C-4 之后的事。
