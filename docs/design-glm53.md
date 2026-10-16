# 设计：GLM-5.3（`dinfer-glm53`）

> 本文件只讲这个模型的结构事实、crate 模块与实现状态。整体架构、core / kernels / layers / server、阶段计划、性能分析见 `design.md`。

## 0. 状态

**P3–P6 在 tiny 随机权重 fixture 上全部通过**（`scripts/verify/p6.sh` exit 0，2026-10-15）：MLA（展开 / 吸收 / paged）、MoE 参考路径、fp8 128×128 块量化加载期解量化、DSA 索引器 + 稀疏 attention、tokenizer + chat 模板（真 `tokenizer.json` / `chat_template.jinja`，HF golden 15/15 逐字符）、流式解析器、engine + OpenAI server、MTP nextn=1。
**真权重验收挂起**（755 GB fp8 需 ≥ 8×H200）；待办：`layers.78` MTP 与 fused `wk_weights_proj` 权重名核对、1M rope 参数、按显存算 `num_blocks`、TP/EP。
对拍口径（重要，所有模型沿用）见 `docs/pitfalls.md` 2026-10-15 三条与 `docs/loops/p5-dsa-indexer.md`：硬判决（索引器 top-k、MoE 路由）bf16 下不可逐位复现 → 注入参考 top-k 逐层隔离、用实测 hidden 漂移过滤行、HF 全序列重算不前缀稳定。
已知简化：索引器 K cache bf16（真机 fp8）；索引分数 / top-k / 稀疏 attention / MoE 逐专家循环都是参考 kernel；MTP 只 nextn=1 每步 1 draft。
loop 记录：`docs/loops/{p3-mla-dense,p4-moe-runner,p5-dsa-indexer,p6-glm-serve}.md`；测试归档 `test/2026-10-14/`、`test/2026-10-15/`。

## 1. 结构事实

| 项 | 值 |
|---|---|
| 规模 | safetensors 755 GB（FP8），141 shards；≈ 740B 总参数 |
| 主干 | 78 层，hidden 6144，vocab 154880，1M 上下文，bf16 计算 / FP8 权重 |
| 稠密层 | 0,1,2 层 MLP（intermediate 12288），3–77 MoE |
| MLA | q_lora 2048，kv_lora 512，64 头，nope 192，rope 64，v 256；`fused_qkv_a_proj` = q_a(6144→2048) ∥ kv_a_with_mqa(6144→576)；`q_a_layernorm`、`kv_a_layernorm`；`q_b_proj` 2048→64×256；`kv_b_proj` 512→64×(192+256)；`o_proj` 64×256→6144；RoPE theta 8e6 default 型，`rope_interleave=True`（非 neox）；KV 每 token 576 维 |
| DSA 索引器 | 32 头 × 128；`wq_b` 2048→4096，`wk_weights_proj` 6144→(128+32)，`k_norm` LayerNorm fp32；索引 K cache 为 fp8 `128 + 4` 字节/token；`index_topk=2048`；**有索引器的层**：0,1,2 及 6,10,…,74（`skip_topk = max(l-3+1,0)%4 != 0`），其余层 `shared`，复用 `topk_indices_buffer[max_num_batched_tokens, 2048]`；MTP 层总带索引器 |
| MoE | 256 routed / top-8 / 1 shared（2048）；sigmoid + bias + noaux_tc，1 组，`norm_topk_prob`，`routed_scaling_factor=2.5`，router fp32；SiLU |
| 量化 | FP8 e4m3，块 128×128 `weight_scale_inv`，激活 per-token-group(1×128) 动态量化；`modules_to_not_convert` 列出若干 norm / bias / `kv_a_layernorm` |
| MTP | `num_nextn_predict_layers=1`，权重在 `model.layers.78.*` |
| Tokenizer | HF `tokenizer.json`；chat 模板：`[gMASK]<sop>` 前缀，`<|system|>` 注入 `Reasoning Effort: Low/High/Max`，角色 `<|user|> <|assistant|> <|observation|>`，`<think>…</think>`，工具调用 `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`；eos `[154820,154827,154829]` |

vLLM 的 GLM 特判只有一处：`model_type == "glm_moe_dsa"` 时启用 `glm52_low_latency_gemm`（`deepseek_v32/nvidia/model.py:437`），结构上与 DeepSeek-V3.2 完全一致。

## 2. crate 模块


```
glm53/src/
  config.rs      GlmMoeDsaConfig ← config.json（含 indexer / rope_parameters / quantization_config）
  model.rs       embed(+fused 首层 norm) → 78×DecoderLayer → norm → lm_head
  layer.rs       DecoderLayer{attn: DsaMla, mlp: Dense(0..3)|Fp8Moe}
  dsa.rs         DsaMlaAttention：fused_qkv_a → q_a_norm/kv_a_norm/rope → (indexer: wq_b, wk_weights_proj, k_norm, fp8 logits, topk→buffer) → 稀疏 MLA；shared 层读 buffer
  indexer.rs     Indexer 权重与 fp8 索引 K cache（KvCacheSpec::Indexer）
  moe.rs         FP8 块量化 MoE：router(sigmoid+bias, top8, norm, ×2.5) → grouped fp8 gemm → shared
  mtp.rs         GlmMtp（DraftModel，layer 78，带索引器）
  tokenizer.rs   tokenizers::Tokenizer 加载 tokenizer.json；eos 集合
  chat.rs        minijinja 跑 chat_template.jinja；reasoning_effort、clear_thinking
  parser.rs      <think>…</think> 与 <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call> 流式解析（vLLM 对 GLM 走通用 hf 模式，无专用 parser；对照 vllm/reasoning、vllm/tool_parsers 中 glm4_moe 系列实现）
  weights.rs     名字映射：fused_qkv_a=[q_a, kv_a_with_mqa]、gate_up=[gate,up]、wk_weights_proj=[wk, weights_proj]、experts w13/w2 + weight_scale_inv；跳过 layers.78 交给 MTP
```

实际文件（2026-10-15）：`src/{config,model,executor,mtp,parser,service,tokenizer,chat}.rs`（tokenizer / chat 是 `dinfer-text` 的薄封装）；`tests/{tiny_parity,runner_parity,dsa_parity,dsa_diag,mtp_parity,chat_parity}.rs`。
与上表的差异：`layer.rs / dsa.rs / indexer.rs / moe.rs / weights.rs` 的内容分别落在 `model.rs`、`layers::mla::SparseCtx`、`layers::indexer`、`layers::moe`、`layers::weights`（通用层放 layers crate，模型 crate 只做组装）。

## 3. 模型特有的风险 / 待定

- fp8：sm86 无 fp8 张量核，本机只能加载期解量化；1×128 激活量化 + 真 FP8 GEMM 留 4090 / sm90。
- DSA 快路径（DeepGEMM `fp8_mqa_logits`、radix top-k、FlashMLA 稀疏）要 sm90。
- `index_share_for_mtp_iteration=True`：MTP 层若为 shared 索引器层复用主干最后一个 full 层的 top-k；真模型第 78 层按 freq/offset 规则是 full。
- `indexer_types` 只有 78 项，第 78 层（MTP）落到 freq/offset 回退规则，Rust `has_indexer` 与 HF 一致。
## 4. 参考材料

- `/data/ref/models/glm53/`：config.json、tokenizer.json（20 MB，2026-10-15 下载）、tokenizer_config.json、chat_template.jinja、generation_config.json、safetensors index
- HF 参考实现：transformers 5.19 `GlmMoeDsaForCausalLM`（`/data/ref/venv-tf519`，源码抽到 `/data/ref/hf_tf519/`）；注意 ≥ 5 的专家权重是融合 3D 张量（pitfalls 2026-10-15）
- vLLM：`vllm/models/deepseek_v32/`（attention、indexer、MTP）、`parser/glm47_moe.py`
- fixture：`scripts/verify/gen_tiny_glm.py`（`--first-k-dense 1 --fp8 --dsa --mtp --vocab`）
