# 设计：Kimi-K3（`dinfer-kimik3`）

> 本文件只讲这个模型的结构事实、crate 模块与实现状态。整体架构、core / kernels / layers / server、阶段计划、性能分析见 `design.md`。

## 0. 状态

**挂起（2026-10-15）→ 解挂条件满足（2026-10-19）**：B300 8 卡 NVLink 到手，真权重在 B300 `/data/models/moonshotai/Kimi-K3`（1.5 TB、96 分片、专家 MXFP4 `weight_packed`）；等用户定优先级后按 §2 重启。原因记录：暂无 ≥ 8×H200 级别机器，真权重跑不了。crate 只有 `config.rs`（解析官方 config.json 并断言结构事实）+ `lib.rs`。

**2026-10-19 第二阶段进展（KDA / MoE kernel 先行，crate 骨架未动）**：
- KDA **chunk 形式 prefill 完成**（`crates/kernels/cuda/kda_chunk.cu`，里程碑 A）：对拍 out 2.4e-4 / state 相对
  1.6e-3，加速 3090 7.2×、4090 5.15×（T=1024/96 头/128 维，门槛 5×）。细节 `docs/loops/k3-kda.md`。
- **MXFP4 grouped GEMM 完成**（`crates/kernels/cuda/mxfp4_gemm_sm8x.cu`，里程碑 B）：按专家分组、权重 kernel
  内解包 + bf16 乘加，rel 6e-3 / cos 0.9999967；sm100 原生路径留空壳（`mxfp4_gemm_sm100.cu`）。`docs/loops/k3-moe.md`。
- **里程碑 C-1 完成（2026-10-19）**：`weights.rs`（名字表 + §2.1 张量表）+ `kda.rs`（`KdaLayer` prefill chunk / decode 递归）+
  B300 真权重第 1 层单层对拍（`models/kimik3/tests/kda_layer_parity.rs`，`scripts/verify/k3_kda_layer.sh`；数字见 `docs/loops/k3-kda.md`）。
- 待做：`latent_moe.rs` / `layer.rs` / `model.rs` 的 load 路径（里程碑 C-2+），
  以及 KDA（K 维改 bf16 mma）与 MXFP4（tile 改 mma、GPU 侧 tile 计划）的 v2 提速。
对应阶段 P7 KDA / P8 K3 特有 / P9 多模态 + tokenizer（见 `design.md` §5 阶段表，均标 ⏸）。重启时按 §2 的模块表从 `kda.rs` 开始。

## 0.1 vLLM 基线（追赶目标，B300-50 8 卡 tp8，2026-10-19）

vLLM nightly 0.26.1rc1（`vllm/vllm-openai:nightly`），Kimi-K3 MXFP4 1.56 TB，`--tensor-parallel-size 8 --max-model-len 8192 --trust-remote-code`，
prefix caching 默认开且 `Mamba cache mode = align`（KDA 递归状态按块对齐缓存）；KV 78 万 token；init 624 s（权重 ~18 min + graph 捕获 + flashinfer 自调优）。
tp6 不可用（视觉塔 4096 维列切不整除）。`test/kimik3/2026-10-19/bench_b300_vllm1/`：

| 组 | TPOT p50 / p99 | TTFT p50 / p99 | out tok/s |
|---|---|---|---|
| c1 p128 | 8.70 / 8.70 ms | 193 / 200 ms | 98.7 |
| c8 p128 | 13.09 / 14.52 | 368 / 402 | 503.0 |
| c32 p128 | 19.15 / 20.80 | 421 / 462 | 1432.4 |
| c8 p1024 | 13.11 / 15.65 | 508 / 536 | 469.4 |

与 Qwen3.5-27B 单卡（TPOT 10.3–12.9）比：K3 c1 更快（8.7，tp8 分摊）但 c32 慢（19 vs 12.9，MoE 896 专家 all-to-all + MXFP4 解包）。dinfer K3 的目标口径就是这四行。

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


### 1.6 参考 `/data/code/llm-cache-prediction`（用户 2026-10-20 指定；采纳口径，不搬代码）

该项目是网关侧 K3 前缀缓存命中率测算 sidecar（纯 CPU Rust），其 `crates/kimik3/{tokenizer.rs,xtml.rs,pyjson.rs}` 已与 vLLM 服务端逐 token 对齐过 100+ 轮，`docs/kimik3/{README,other}.md` 记了四处对齐偏差。对 dinfer 的价值与取舍：

| 项 | 它的结论 | 我们采纳 | 不采纳 / 原因 |
|---|---|---|---|
| 渲染参照物 | **vLLM 服务端 Rust renderer**（`rust/src/chat/src/renderer/kimi_k3/`），不是 HF `encoding_k3.py`：浮点序列化走 serde ryu（`100.0`→`100`、`1e2`→`100.0`、`1e-6` 保留）、tool_calls 参数 serde 完整解析后重序列化、消息 `name` 字段被前端丢弃、thinking 对象形式 | ✅ C-4a 的 golden 生成脚本按它的 `tools/gen_goldens.py` 方式 monkeypatch `encoding_k3` 成服务端语义；我们对拍目标是 vLLM 的 token 序列 | — |
| tiktoken | special 空间 `[len(ranks), +256)`、未命名 `<|reserved_token_{i}|>`；`allow_special` 分段；**400k 字符窗口 + 25k 连续(非)空白 run 切分后再 BPE**（边界影响结果）；BOS 不前置；`thinking_effort` 默认 `max` | ✅ 全部复刻进 `models/kimik3/src/tokenizer.rs` / `xtml.rs` | 代码独立写，不依赖该 crate |
| 命中粒度 | vLLM 对 K3 的 `--prefix-match-unit 128` + `--mamba-cache-mode align` 下，**页大小配平把注意力 block 抬到 1536 = 12×128**，KDA 状态只在 1536 边界有快照；全命中至少留末块重算（`hit_tail_tokens` 按部署实测，三个模型各不同） | ✅ 作为 dinfer 做递归模型 prefix cache（design.md §11 待做项）的设计输入：GDN / KDA 状态快照粒度 = 注意力 block 的整数倍，由 KV 页配平决定，不是独立常量；`cached_tokens` 口径要留末块 | 容量 / LRU / Mooncake 对账是网关侧需求，dinfer 不做 |
| 图片 | `navit_resize_image` token 数公式、`mm_hash` | 暂不（视觉塔 C-6 以后） | — |
| 对账方法 | 100 轮逐 token 对比落盘两列（simulator / vLLM）、边界 ±1 case、非 ASCII 函数名被 vLLM 400 | ✅ C-4a 的 case 清单与"两列落盘"方式 | — |

**结论**：它解决的是"和 vLLM 字节一致地算 token 与命中"，与我们 C-4a 的目标完全重合，它踩过的四处偏差是我们最该借的；命中粒度 1536 的事实直接影响 Qwen3.5 / K3 重开 prefix cache 的设计。

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

### 2.1 KDA 层权重表（B300 真分片 header 核对，2026-10-19，第 1 层 `model-00002-of-000096`）

前缀 `language_model.model.layers.N.self_attn.`；`weights.rs` 的 `Table` 剥掉 `language_model.model.`，`lm_head` 在 `language_model.lm_head.weight`。

| 张量 | dtype | shape | 装载 |
|---|---|---|---|
| `q_proj` / `k_proj` / `v_proj` / `g_proj.weight` | BF16 | `[12288, 7168]` | 四条 `Linear`（HF 分开存；vLLM 融合 `in_proj_qkvgfab`，我们 C-2 再融） |
| `f_a_proj.weight` / `f_b_proj.weight` | BF16 | `[128, 7168]` / `[12288, 128]` | gate 低秩两段，Rust 侧两条 GEMM |
| `b_proj.weight` | BF16 | `[96, 7168]` | beta（逐头标量），输出 bf16→f32 给 kernel |
| `{q,k,v}_conv1d.weight` | **F32** | `[12288, 1, 4]` | 转 bf16 `[12288, 4]`（HF 以 bf16 模型 dtype 跑） |
| `A_log` | F32 | **`[128]`** | 前 96 有值、尾 32 为 0 填充；只取前 96（`load_a_log` 断言尾部全 0；vLLM loader 同样 `narrow`） |
| `dt_bias` | F32 | `[12288]` | 逐 k 维 |
| `o_norm.weight` | **F32** | `[128]` | 均值 0.034，普通 `norm(x)·w·sigmoid(g)`，**非** zero-centered；转 bf16 给 `kda_gated_rmsnorm` |
| `o_proj.weight` | BF16 | `[7168, 12288]` | |
| `input_layernorm` / `post_attention_layernorm.weight` | BF16 | `[7168]` | `KimiRMSNorm`：`w·norm(x)`，非 zero-centered（与 Qwen3.5 不同） |

每层每 slot 状态：conv `3 × [12288, 3]` bf16（221 KB）+ 递推 `[96, 128, 128]` f32（6.3 MB）。

### 2.2 MoE / AttnRes 层权重表（C-2，第 1 层，形状由 `latent_moe.rs` / `layer.rs` 装载时断言）

前缀 `layers.N.block_sparse_moe.`（AttnRes 在 `layers.N.`）。装载实现 `models/kimik3/src/latent_moe.rs`、`layer.rs`。

| 张量 | dtype | shape | 装载 |
|---|---|---|---|
| `gate.weight` | BF16 | `[896, 7168]` | 路由；`moe_gate_f32` 内升 f32（HF 把 x 与 w 都 cast f32 再 linear） |
| `gate.e_score_correction_bias` | F32 | `[896]` | 只参与选择，不参与权重（`moe.cu::route_kernel` 同口径） |
| `routed_expert_down_proj.weight` | BF16 | `[3584, 7168]` | latent 投影，普通 `Linear` |
| `experts.E.w1/w3.weight_packed` | U8 | `[3072, 1792]` | MXFP4 两值一字节；装载时按输出维拼成 `w13 [896, 6144, 1792]`（gate 在前），**packed 常驻** |
| `experts.E.w1/w3.weight_scale` | U8 | `[3072, 112]` | e8m0，group 32 → `[896, 6144, 112]` |
| `experts.E.w2.weight_packed` / `weight_scale` | U8 | `[3584, 1536]` / `[3584, 96]` | → `[896, 3584, 1536]` / `[896, 3584, 96]` |
| `routed_expert_norm.weight` | BF16 | `[3584]` | `KimiRMSNorm`，专家加权和之后、up_proj 之前 |
| `routed_expert_up_proj.weight` | BF16 | `[7168, 3584]` | |
| `shared_experts.{gate,up,down}_proj.weight` | BF16 | `[6144, 7168]` ×2 / `[7168, 6144]` | `dinfer_layers::Mlp`（gate\|up 拼接）+ `Situ` 激活 |
| `self_attention_res_proj.weight` / `mlp_res_proj.weight` | BF16 | `[1, 7168]` | AttnRes 标量门（`moe_gate_f32` E=1 取 f32 分数） |
| `self_attention_res_norm.weight` / `mlp_res_norm.weight` | BF16 | `[7168]` | AttnRes 的 RMSNorm（HF 公式里 norm.w 与 proj.w 相乘后再与 norm(v) 点积） |

专家 packed 常驻字节：每专家 `w13` 11,010,048 + scale 688,128 + `w2` 5,505,024 + scale 344,064 = **17.55 MB**；
每层 896 专家 = **15.72 GB**（bf16 解包要 ~45 GB/层，放不下——这就是保持 packed 的原因，见 k3-moe.md）。

### 2.3 MLA 层权重表（C-3，第 3 层 `model-00004-of-000096`，形状由 `mla.rs` 装载时断言）

前缀 `layers.N.self_attn.`（N ∈ {3,7,11,…,91,92}，24 层）。装载实现 `models/kimik3/src/mla.rs`；与 HF 核对表见 `docs/loops/k3-model.md`。

| 张量 | dtype | shape | 装载 |
|---|---|---|---|
| `q_a_proj.weight` | BF16 | `[1536, 7168]` | 与下面两条按行拼成 `fused [14400, 7168]`（顺序 q_a / kv_a_with_mqa / g，同 vLLM `fused_qkv_a_g_proj`），输出按列拆 |
| `kv_a_proj_with_mqa.weight` | BF16 | `[576, 7168]` | 前 512 = kv latent（过 `kv_a_layernorm`），尾 64 = `k_pe`（**不旋转**，所有 head 共享） |
| `g_proj.weight` | BF16 | `[12288, 7168]` | 输出门：`attn *= sigmoid(g)`，o_proj 之前 |
| `q_a_layernorm.weight` / `kv_a_layernorm.weight` | BF16 | `[1536]` / `[512]` | `KimiRMSNorm`；HF 用默认 eps 1e-6，vLLM 用 `rms_norm_eps` 1e-5（我们目前 1e-5，差异远低于 bf16，见 k3-model.md TODO） |
| `q_b_proj.weight` | BF16 | `[18432, 1536]` | 96 head × (128 nope + 64 "rope")；无 RoPE，尾 64 维原样参与点积 |
| `kv_b_proj.weight` | BF16 | `[24576, 512]` | 96 head × (128 nope + 128 v)；prefill 展开用；decode 拆成每 head `W_UK^T [512,128]` / `W_UV [128,512]`（吸收） |
| `o_proj.weight` | BF16 | `[7168, 12288]` | |

KV cache：每 token latent `[kv_c(512) | k_pe(64)]` bf16 = 1152 B，paged `[num_blocks, block_size, 576]`，每层独立（`MlaLayer.kv`）。
scale `192^-0.5`。顶层：`embed_tokens [163840, 7168]`、`norm [7168]`、`output_attn_res_proj [1, 7168]`、`output_attn_res_norm [7168]`、`lm_head [163840, 7168]` 全在 `model-00094-of-000096`（`weights::shards_for_top`）。

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
