# Qwen3.5-27B（`qwen3_5`）设计

> 用户 2026-10-19："0.6B 太小会掩盖或放大问题，用 27B 测一组"；"和 qwen3 有共同之处，按方法论完成 qwen35"。方法论见 `docs/model-onboarding.md`。

## 0. 状态 / 接入清单

- 2026-10-19 结构事实核对完成（下）。实现由独立子代理推进，loop 文档 `docs/loops/qwen35-gdn.md`。
```
[x] §1 结构事实   [x] 差距三分类   [x] gdn loop 文档   [x] GDN kernel 对拍（3090/4090/B300 三卡 10/10）
[x] 全注意力 kernel 扩模板（D=256 / GROUP 6 / partial RoPE / 输出门，3090+4090 对拍）   [x] config/weights + tiny golden
[x] prefill logits 对拍   [x] greedy 16 / 混批（tiny golden 4/4）   [ ] chat_parity   [ ] 真权重 e2e vs vLLM   [ ] bench CSV   [ ] §11 台账
```

## 0.1 vLLM 基线（追赶目标）

B300 单卡、vLLM nightly 0.29.1rc1、bf16、max-model-len 8192（`test/qwen35/2026-10-19/bench_b300_vllm1/`，2026-10-19）：

| 组 | TPOT p50 | TTFT p50 | out tok/s |
|---|---|---|---|
| c1 p128 | 10.24 ms | 45 ms | 95 |
| c8 p128 | 11.15 | 99 | 675 |
| c32 p128 | 12.87 | 263 | 2148 |
| c8 p1024 | 11.20 | 413 | 557 |

### 0.2 dinfer 首轮（里程碑 5，2026-10-19，`test/qwen35/2026-10-19/bench_b300_{1,2,3}/`；bench_b300_3 为正确性修完的基线：关 prefix cache、slot 归还，c8 TTFT 1218 / c8-long 8138、TPOT 59.7 / 157.5）

同机同 seed，dinfer GPU 0 eager 单流（无 CUDA graph / 跨流 / chunk GDN prefill）：

| 组 | dinfer TPOT p50 | vLLM | 倍率 | dinfer TTFT p50 | vLLM |
|---|---|---|---|---|---|
| c1 p128 | 24.1 ms | 10.3 | 0.43× | 323 ms | 47 |
| c8 p128 | 28.4 | 11.2 | 0.39× | 349 | 101 |
| c32 p128 | 54.8 | 12.9 | 0.24× | 2200 | 264 |
| c8 p1024 | 119.0 | 11.2 | 0.09× | 7147 | 408 |

正确性：3 条 greedy prompt 与 vLLM 逐字一致（修了 zero-centered RMSNorm 与引擎线程 ctx 绑定，见 `docs/loops/qwen35-model.md` 第 12–16 轮）。
性能差距的来源按优先级：① GDN prefill 走逐 token 顺序递归（c8-long TTFT 7 s）→ 接 chunk kernel；② decode 无 CUDA graph（c1 24 ms，64 层 × ~10 launch）；③ GDN slot 复用 bug（正确性，先修）。

### 0.3 里程碑 6 第 1 轮：GDN chunk prefill（`bench_b300_4`，2026-10-19）

| 组 | dinfer TPOT | vLLM | dinfer TTFT | vLLM | 吞吐倍率 |
|---|---|---|---|---|---|
| c1 p128 | 24.1 ms | 10.3 | 45 ms | 46 | 0.44× |
| c8 p128 | 28.3 | 11.2 | 169 | 105 | 0.41× |
| c32 p128 | 44.5 | 12.9 | 357 | 261 | 0.32× |
| c8 p1024 | 36.3 | 11.2 | 470 | 403 | 0.36× |

prefill 问题解决（`gdn_chunk.cu`，3090 T=1024 顺序 41.5 → 4.2 ms）；剩余是 decode：eager 单流 64 层无 CUDA graph，c1 TPOT 24 ms。

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
tests/           golden_parity.rs（tiny golden：4 个用例，`scripts/verify/qwen35_golden.sh`）、real_parity.rs、chat_parity.rs
```

### 2.1 权重映射表（里程碑 3 已核对的语义，写代码前先看这张表）

| HF 名字（`language_model.` 前缀剥离后） | 我们存成 | 备注 |
|---|---|---|
| `model.embed_tokens.weight` | embedding | `[248320, 5120]` |
| `model.layers.{l}.input_layernorm.weight` | `input_norm.w` | 普通 RMSNorm（`x·w`） |
| `model.layers.{l}.post_attention_layernorm.weight` | `post_norm.w` | 同上 |
| `model.layers.{l}.mlp.{gate,up,down}_proj.weight` | `gate_up`（前两者拼接）+ `down` | 同 Qwen3 的 `Mlp::load_tp` |
| `model.layers.{l}.self_attn.q_proj.weight` | **拆成两段** `[q(H·D)]`、`[gate(H·D)]` | **per-head 交错**：vLLM `q_gate.view(..., num_heads, 2*head_dim)` 后 `chunk(2, -1)`，即第 h 头在行 `h·2D .. h·2D+D` 是 q、`+D .. +2D` 是 gate。**不能整段前后切** |
| `model.layers.{l}.self_attn.k_proj / v_proj` | `[k | v]` 拼接 | 同 Qwen3 |
| `model.layers.{l}.self_attn.o_proj.weight` | 原样 | 输入维 = `H·D`（不带 gate） |
| `model.layers.{l}.self_attn.q_norm / k_norm.weight` | 原样 | **zero-centered**：norm 里用 `(1 + w)`，即 `qkv_post_ext` 的 `norm_beta = 1.0`；权重初值全 0 |
| `model.layers.{l}.linear_attn.*` | 见 `docs/loops/qwen35-gdn.md` | GDN 已对拍 |
| `lm_head.weight` | 独立 `lm_head` | `tie_word_embeddings false` |

拼接顺序：kernel 支持 `[q(H·D) | gate(H·D) | k | v]`（`qkv_post_ext(qmul=2)` 只处理 q/k/v 段，gate 段由 `mul_sigmoid_bf16`
在 attention **之后**消费），但**模型侧改用两条紧凑 GEMM**（`qkv [q|k|v]` + 独立 `gate`，`qmul = 1`）——FLOPs 相同、
少一次按行搬运，见 §2.2。

## 2.2 实现决定（里程碑 4，2026-10-19，与 vLLM 对照后落定）

| 项 | 决定 | 原因 |
|---|---|---|
| q_proj 输出门 | **拆成两条紧凑 GEMM**（`qkv [q|k|v]` + 独立 `gate`），不用 `qkv_post_ext(qmul=2)` 的行内 gate 段 | 行内 gate 段带 stride，attention 之后要 `mul_sigmoid` 就得按行搬一次；两条紧凑 GEMM 的 FLOPs 相同、少一次搬运。kernel 的 `qmul=2` 路径仍保留（对拍用） |
| `qkv_post_ext` | `rope_dim = 64`、`norm_beta = 1.0`（`q_norm/k_norm` zero-centered） | 与 HF `Qwen3_5RMSNorm((1 + w))` 一致 |
| GDN 状态 | crate 内 `SlotTable`（`RequestId → slot`）+ 每层 `conv [slots, 10240, 3]` bf16 / `rec [slots, 48, 128, 128]` f32；"首 token 位置 == 0" 的请求清零 | 不改 core 的 KV group；slot 复用发生在"本步不在场且最久未见"的请求上 |
| GDN prefill | 逐 token 顺序递归（与 chunk 形式数值等价） | 里程碑 4 先对拍；chunk 形式是 perf 项 |
| executor | eager 单流，无 CUDA graph / 跨流 pair / host tier / PD / TP 通信 | 里程碑 4 目标是对拍；这些接过来是一轮独立工作（Qwen3 的对应实现可直接搬） |
| mrope | 纯文本三路位置相同 → 退化为 1D RoPE（同 vLLM `qwen3_next.py`） | tiny golden 已验证（partial RoPE 前 64 维） |
| `conv1d` | 权重 `[dim, 1, width]` → `[dim, width]`，无 bias | HF `bias=False` |

## 3. 风险 / 待定

- ~~输出门拼接顺序~~：**已定**（见 §2.1，per-head 交错，vLLM `qwen3_next.py:436-443` 与 HF `modeling_qwen3_5.py:779-782` 一致）。
- mrope 纯文本退化：对拍确认。
- GDN 的 q/k 归一化、A = -exp(A_log)、dt = softplus(a + dt_bias) 等约定：以 vLLM / fla 源码为准，对拍 Triton 参考。
- ~~两个 KV group 的 core 支持程度~~：**已定**（见 §2.2）——全注意力层用 core 的 paged `FullAttention` 组；GDN 状态
  由 crate 内 `SlotTable` 自管。接 `KvCacheSpec::Mamba` 是后续项（要动 core；收益是前缀缓存 / 抢占时状态可随块回收）。
- **TP**：`Qwen35Model::load_with_budget` 在 `world > 1` 时直接报错（里程碑 4 没接 all-reduce；静默跑会少一半结果）。
  接法照 Qwen3：`crates/core` 不动，模型侧加 `comm`、qkv / gate_up 按头切（`weights::load_qkv` / `load_gate` 已支持）、
  o_proj / down 各一次 all-reduce；GDN 的 `in_proj_*` / `out_proj` 同样按头切。
- 真权重（52 GB bf16）需要 ≥ 60 GB 显存：本机 3090 / 4090 装不下 → 里程碑 5 上 B300（用户 2026-10-19 "B300 先不用"，
  待指示）。tiny golden 与真权重无关（形状参数驱动），已通过。
- FP8 版：bf16 跑通后，FP8 权重 dequant 到 bf16 加载是第一步；原生 FP8 GEMM 是性能项。

## 4. 参考材料

- vLLM：`models/qwen3_5.py`、`models/qwen3_next.py`、`layers/mamba/gdn/qwen_gdn_linear_attn.py`、`layers/mamba/ops/`（Triton）、`model_executor/layers/mamba/mamba_utils.py`。
- fla：`chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule`（vLLM 内置副本或 `pip show fla` 路径）。
- HF transformers `modeling_qwen3_5.py`（本机 `/data/ref/venv-tf519` 若有）。
