# 设计：Kimi-K3（`dinfer-kimik3`）

> 本文件只讲这个模型的结构事实、crate 模块与实现状态。整体架构、core / kernels / layers / server、阶段计划、性能分析见 `design.md`。

## 0. 状态

**挂起（2026-10-15）→ 解挂条件满足（2026-10-19）**：B300 8 卡 NVLink 到手，真权重在 B300 `/data/models/moonshotai/Kimi-K3`（1.5 TB、96 分片、专家 MXFP4 `weight_packed`）；等用户定优先级后按 §2 重启。原因记录：暂无 ≥ 8×H200 级别机器，真权重跑不了。crate 只有 `config.rs`（解析官方 config.json 并断言结构事实）+ `lib.rs`。

**2026-10-19 第二阶段进展（KDA / MoE kernel 先行，crate 骨架未动）**：
- KDA **chunk 形式 prefill 完成**（`crates/kernels/cuda/kda_chunk.cu`，里程碑 A）：对拍 out 2.4e-4 / state 相对
  1.6e-3，加速 3090 7.2×、4090 5.15×（T=1024/96 头/128 维，门槛 5×）。细节 `docs/loops/k3-kda.md`。
- **MXFP4 grouped GEMM 完成**（`crates/kernels/cuda/mxfp4_gemm_sm8x.cu`，里程碑 B）：按专家分组、权重 kernel
  内解包 + bf16 乘加，rel 6e-3 / cos 0.9999967；sm100 原生路径留空壳（`mxfp4_gemm_sm100.cu`）。`docs/loops/k3-moe.md`。
- 待做：crate 骨架（`weights.rs` / `kda.rs` / `latent_moe.rs` / `layer.rs` 的 load 路径，里程碑 C），
  以及 KDA（K 维改 bf16 mma）与 MXFP4（tile 改 mma、GPU 侧 tile 计划）的 v2 提速。
对应阶段 P7 KDA / P8 K3 特有 / P9 多模态 + tokenizer（见 `design.md` §5 阶段表，均标 ⏸）。重启时按 §2 的模块表从 `kda.rs` 开始。

## 1. 结构事实

| 项 | 值 |
|---|---|
| 规模 | 2.8T 总 / 104B 激活；safetensors 1.56 TB（MXFP4 专家 + bf16 其余）；96 shards |
| 文本主干 | 93 层，hidden 7168，vocab 163840，1M 上下文，bf16，RMSNorm eps 1e-5 |
| 层类型 | `linear_attn_config.kda_layers` 为 1-based；0-based 第 3,7,11,…,91,92 共 **24 层 MLA**，其余 **69 层 KDA** |
| 稠密层 | `first_k_dense_replace=1`：第 0 层 MLP（intermediate 33792），其余 92 层 MoE |
| KDA | 96 头 × 128；`in_proj_qkvgfab` = q,k,v,g（各 7168→12288）+ f_a（→128）+ b（→96）；`f_b_proj` 128→12288；`q/k/v_conv1d` 短卷积核 4；`A_log[96]`、`dt_bias[12288]`；`o_norm`（gated RMSNorm）；`o_proj` 12288→7168；`gate_lower_bound=-5.0`，`use_full_rank_gate` |
| KDA 状态 | 每层每请求：conv state `[3*12288, 3]` bf16 + 递归状态 `[96,128,128]` fp32 ≈ 6.5 MB |
| MLA | q_lora 1536，kv_lora 512，nope 128，rope 64，v 128，96 头；**`mla_use_nope=True` → 无 RoPE**；`mla_use_output_gate=True` → sigmoid 门 `g_proj`，与 q_a/kv_a 合并为 `fused_qkv_a_g_proj`；KV 每 token 576 维（512+64）bf16 |
| MoE | 896 routed / top-16 / 2 shared（shared intermediate 2×3072）；**latent MoE**：`routed_expert_down_proj` 7168→3584 → 专家在 3584 维工作（w1/w3: 3584→3072, w2: 3072→3584）→ `routed_expert_norm` → `routed_expert_up_proj` 3584→7168；路由 sigmoid + `e_score_correction_bias` + noaux_tc，1 组，renormalize，scaling 1.0，fp32 logits |
| 激活 | `situ`（beta 4.0，linear_beta 25.0）：`SituAndMul`，专家与 shared/dense 均用 |
| 量化 | compressed-tensors `mxfp4-pack-quantized`，group 32，scale uint8(e8m0)，**仅 routed experts** 的 `w1/w2/w3.weight_packed/weight_scale`；attn、shared、dense MLP、lm_head、vision 均 bf16 |
| AttnRes | `attn_res_block_size=12` → 8 个块（写入层 0,12,…,84）；每层 `self_attention_res_proj`/`mlp_res_proj`（7168→1）+ 对应 RMSNorm；模型级 `output_attn_res_proj/norm`。融合算子 `models/kimi_k3/nvidia/ops/attn_res.py` 需逐行移植 |
| 视觉 | MoonViT 27 层，hidden 1024，patch 14，`patchmergerv2` 投影到 7168；占位 token 163605；复用 `kimi_k25_vit.py` |
| MTP | `KimiK3MTP`：enorm/hnorm/eh_proj(2h→h) + 一个无 AttnRes 的 `KimiDecoderLayer`（MLA）+ shared head |
| Tokenizer | tiktoken BPE（`tiktoken.model`），自定义 `pat_str`，256 保留特殊 token；chat 为 XTML 格式（`<|open|> <|close|> <|sep|> <|end_of_msg|>`，`encoding_k3.py` 750 行需移植） |
| 推理参数 | 始终 thinking，`reasoning_effort ∈ {low,high,max}`；多轮必须回传 `reasoning_content` |

权重名（层内）：`language_model.model.layers.N.{input_layernorm, post_attention_layernorm, self_attention_res_norm/proj, mlp_res_norm/proj, self_attn.*, block_sparse_moe.{gate.weight, gate.e_score_correction_bias, experts.E.w1/w2/w3.weight_packed/weight_scale, shared_experts.gate_proj/up_proj/down_proj, routed_expert_down_proj/up_proj/norm}}`；
顶层 `language_model.model.embed_tokens / norm / output_attn_res_norm / output_attn_res_proj`、`language_model.lm_head`、`vision_tower.*`、`mm_projector.*`。
vLLM 的融合规则（`model.py:1110` `packed_modules_mapping`）：`gate_up_proj=[gate,up]`、`in_proj_qkvgfab=[q,k,v,b,f_a]`、`conv1d=[q,k,v conv]`、`fused_qkv_a_g_proj=[q_a, kv_a_with_mqa, g]`、专家 `w13=[w1,w3]`。

## 2. crate 模块


```
kimik3/src/
  config.rs      KimiK3Config{text: KimiLinearConfig, vision}  ← configuration_kimi_k3.py
  model.rs       KimiK3Model：embed → 93×KimiDecoderLayer(AttnRes 状态 prefix_sum) → output_attn_res → norm → lm_head
  layer.rs       KimiDecoderLayer{attn: Kda|Mla, mlp: Dense|LatentMoe, attn_res: Option<AttnRes>}
  kda.rs         KdaAttention：in_proj_qkvgfab 拆分、conv、f_b/beta/gate 计算、chunk prefill / recurrent decode、o_norm、o_proj；KvCacheSpec::Mamba
  attn_res.rs    块前缀和管理（8 块）、调 kernels::attn_res
  latent_moe.rs  down_proj → router ∥ → experts(mxfp4, situ) → norm → up_proj(+shared 残差)
  vision/        MoonViT3d、PatchMergerV2、图像预处理（kimi_k3_vision_processing.py 移植）、token 数公式
  mtp.rs         KimiK3Mtp（DraftModel）
  tokenizer.rs   tiktoken BPE + 自定义 pat_str + 256 特殊 token
  xtml.rs        encoding_k3.py 移植：build_chat_segments → EncodeSegment{text, allow_special}
  parser.rs      思考段 / 工具调用流式解析（对照 vllm/parser/kimi_k3.py、reasoning/kimi_k3_reasoning_parser.py、tool_parsers/kimi_k3_tool_parser.py；格式见 §1.4b）
  weights.rs     名字映射与融合规则（见 2.1）；MXFP4 专家权重按 [E, 2*I, H/2] 打包
```

## 3. 模型特有的风险 / 待定

- **KDA kernel 移植量大**：vLLM 的 KDA 是 Triton（`third_party/flash_linear_attention/ops/kda.py` ~1500 行 + K3 私有 `kda.py` 1372 行 + `kda_metadata.py` 752 行）。两条路：手写 CUDA；或用 Triton AOT 编译成 PTX 嵌入 Rust（保留算法，省掉翻译）。P7 开始前定。
- **MXFP4 格式**：compressed-tensors 的 `weight_packed`（两 fp4 一字节）+ e8m0 scale（group 32）；vLLM 在非 sm100 上的路径（`quantization/mxfp4.py`）要确认是否有 Marlin 风格回退可参考。
- **K3 输出格式**（`reasoning/kimi_k3_reasoning_parser.py`、`tool_parsers/kimi_k3_tool_parser.py`）：思考段 `<|open|>think<|sep|>…<|close|>think<|sep|>`，回复段 `<|open|>response<|sep|>…`，工具调用 `<|open|>tools<|sep|><|open|>call tool="…" index="…"<|sep|><|open|>argument key=… type=…<|sep|>…`；`type="string"` 原文，其他 JSON 解码，属性做 `&quot;`/`&amp;` 反转义。输出可能一开始就在 think 段内。
- **tokenizer / renderer / parser 的模式选择**按架构名（`config/model.py:736`）：`KimiK3ForConditionalGeneration → kimi_k3`（HF tokenizer 类 + `KimiK3Renderer` + `KimiK3Parser`）；`GlmMoeDsaForCausalLM` **不在特判列表**，走 `auto/hf`：标准 HF tokenizer + jinja 模板 + 通用 reasoning/tool parser（GLM 的 `<think>` 与 `<tool_call>` 格式）。`deepseek_v32` 模式的 DSML parser 与 GLM 无关。
- 1M 上下文：KDA 状态按 `max_model_len` 预留，先按 128K 开发。
- design.md §9 待定 2：KDA kernel 手写 CUDA 还是 Triton AOT。
## 4. 参考材料

- `/data/ref/models/kimik3/`：config.json、configuration_kimi_k3.py、modeling_kimi_k3.py、modeling_kimi_linear.py、encoding_k3.py（XTML）、tokenization_kimi.py、kimi_k3_processor.py、safetensors index
- vLLM：`vllm/models/kimi_k3/`、`reasoning/kimi_k3_reasoning_parser.py`、`tool_parsers/kimi_k3_tool_parser.py`、`third_party/flash_linear_attention/ops/kda.py`
