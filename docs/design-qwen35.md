# Qwen3.5-27B（`qwen3_5`）设计

> 用户 2026-10-19："0.6B 太小会掩盖或放大问题，用 27B 测一组"；"和 qwen3 有共同之处，按方法论完成 qwen35"。方法论见 `docs/model-onboarding.md`。

## 0. 状态 / 接入清单

- 2026-10-19 结构事实核对完成（下）。实现由独立子代理推进，loop 文档 `docs/loops/qwen35-gdn.md`。
```
[x] §1 结构事实   [x] 差距三分类   [x] gdn loop 文档   [x] GDN kernel 对拍（3090/4090/B300 三卡 10/10）   [ ] config/weights + tiny golden
[ ] prefill logits 对拍   [ ] greedy 32 / 混批   [ ] chat_parity   [ ] 真权重 e2e vs vLLM   [ ] bench CSV   [ ] §11 台账
```

## 1. 结构事实

来源：B300 `/data/models/Qwen/Qwen3.8-27B/config.json`（`text_config`）+ 分片 header（bf16 52 GB / 18 分片；FP8 版 29 GB / 66 分片 `fmt e4m3` dynamic，抽样分片 header 解析失败，先用 bf16）。vLLM 参考 `vllm/model_executor/models/qwen3_5.py`（继承 `qwen3_next.py`），GDN 层 `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`。

- 多模态壳 `Qwen3_5ForConditionalGeneration`，我们只做 `text_config`（`language_model_only`）；视觉不做。
- 64 层，`layer_types` = 48 × `linear_attention` + 16 × `full_attention`，`full_attention_interval 4`（每第 4 层全注意力）。
- hidden 5120、inter 17408（SwiGLU gate/up/down，与 Qwen3 同）、RMSNorm eps 1e-6、`input_layernorm` / `post_attention_layernorm`。
- **全注意力层**：24 q 头 / 4 kv 头（GROUP 6）、**head_dim 256**；`q_proj [12288, 5120]` = 24 × 256 × **2**（`attn_output_gate: true`，一半是 q 一半是门，拼接顺序以 vLLM `qwen3_next.py` 为准）；`k_proj / v_proj [1024, 5120]`；`q_norm / k_norm [256]` per-head；`o_proj [5120, 6144]`；RoPE `partial_rotary_factor 0.25`（只转前 64 维）、mrope `mrope_section [11,11,10]` interleaved（纯文本三路位置相同 → 退化为普通 1D RoPE，**需对拍确认**）。
- **GDN 层**（Gated DeltaNet）：`in_proj_qkv [10240, 5120]`（q 16 × 128 + k 16 × 128 + v 48 × 128 = 2048 + 2048 + 6144）、`in_proj_z [6144, 5120]`（输出门）、`in_proj_a / in_proj_b [48, 5120]`（每 v 头的 α / β 标量）、`conv1d [10240, 1, 4]`（对 qkv 做 causal depthwise conv，kernel 4）、`A_log / dt_bias [48]`、`norm [128]`（输出 RMSNorm over value head_dim，带 z 门 swish）、`out_proj [5120, 6144]`；`mamba_ssm_dtype float32`（状态 f32）。状态：每层每请求 conv 状态 [10240, 3] + 递推状态 [48, 128, 128] f32 → `KvCacheSpec::Mamba`（design.md §1.4 KV group 机制，与全注意力层的 paged KV 是两个 group）。
- vocab 248320、`tie_word_embeddings false`（lm_head 独立）、eos / bos 248044、`max_position_embeddings 262144`。
- MTP 1 层（`mtp_num_hidden_layers 1`，不共享 embedding）：第一版不做。
- chat 模板：待从 `tokenizer_config.json` 读；思考段 / tool_call 格式待核对（大概率同 Qwen3）。

### 1.1 与现有代码的差距（三分类）

| 类 | 项 |
|---|---|
| 直接复用 | RMSNorm / SwiGLU / GEMM / embedding / 采样 / scheduler / runner / engine / server / tokenizer / minijinja / TP 机制 / CUDA graph |
| 扩模板 / 加参数 | gqa prefill & decode：**D=256**（现在 `PF_D = 128` 硬编码，模板化）、**GROUP 6**（v3 加 case，v2 不支持）；qkv_post：partial RoPE（rope_dim 64 ≠ D）+ 输出门拆分（新的融合 kernel 或拆成两步）；KV spec 两个 group（core 已有设计，未必已实现：核实 `KvCacheCoordinator`）；lm_head 不 tie（qwen3 已支持文件里有 lm_head.weight） |
| 全新 | **GDN**：causal conv1d prefill / update、chunk delta-rule prefill、fused recurrent decode、gated RMSNorm 输出（`crates/kernels/cuda/gdn.cu`，参考 vLLM `layers/mamba/gdn/` 与 fla 的 `chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule` Triton）；Mamba 状态的分配 / 回收 / 混批索引 |

与 K3 的 KDA 同族（都是 gated delta rule），区别在 K3 多了 β 的结构与 f_b 门；kernel 先各写各的，收口时合并成一份 `delta_rule.cu`（§11 记一行）。

## 2. crate 模块

```
models/qwen35/src/
  config.rs      Qwen35Config ← text_config；check()：layer_types 只含两种、head_dim 256、group 6、partial 0.25、GDN 头数整除
  weights.rs     名字映射（language_model.model.* 前缀剥离）、融合（gate_up 合并；q_proj 拆 q / gate；qkv 不拆，按偏移读）
  gdn.rs         GdnLayer：in_proj → conv1d(prefill/update) → q/k L2 norm（如 vLLM）→ delta rule（chunk | recurrent）→ gated RMSNorm(z) → out_proj
  full_attn.rs   FullAttention：q/k/v proj → q_norm/k_norm → partial RoPE → 输出门 → gqa(D=256, GROUP 6) → o_proj
  layer.rs       DecoderLayer{attn: Gdn|Full, mlp}
  model.rs       embed → 64 层 → norm → lm_head（独立）
  executor.rs    与 qwen3 同骨架；两个 KV group（paged + mamba 状态）
  parser.rs      思考段 / tool_call（核对后决定能否复用 qwen3 的）
  service.rs     ChatBackend
tests/           golden_parity.rs（tiny golden）、real_parity.rs、chat_parity.rs
```

## 3. 风险 / 待定

- 输出门拼接顺序（q 与 gate 在 12288 维里是交错还是前后半）：以 vLLM `qwen3_next.py` `Qwen3NextAttention` 为准。
- mrope 纯文本退化：对拍确认。
- GDN 的 q/k 归一化、A = -exp(A_log)、dt = softplus(a + dt_bias) 等约定：以 vLLM / fla 源码为准，对拍 Triton 参考。
- 两个 KV group 的 core 支持程度：核实后决定是在 core 加还是 qwen35 内自管。
- FP8 版：bf16 跑通后，FP8 权重 dequant 到 bf16 加载是第一步；原生 FP8 GEMM 是性能项。

## 4. 参考材料

- vLLM：`models/qwen3_5.py`、`models/qwen3_next.py`、`layers/mamba/gdn/qwen_gdn_linear_attn.py`、`layers/mamba/ops/`（Triton）、`model_executor/layers/mamba/mamba_utils.py`。
- fla：`chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule`（vLLM 内置副本或 `pip show fla` 路径）。
- HF transformers `modeling_qwen3_5.py`（本机 `/data/ref/venv-tf519` 若有）。
