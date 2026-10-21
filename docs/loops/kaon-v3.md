# Kaon-V3 适配（5090 ×8）——先把基线摸清，再谈超越

> 状态：**开工（2026-10-21）**。目标：在 RTX 5090 上把 Kaon-V3 跑起来，性能 ≥ 基线容器 `kaon-vllm-pegaflow` 的 **1.2×**，
> 达不到就继续优化直到满足。每轮干净编译 → 实测 → 有提升才 commit。

## 机器（2026-10-21 首次连通确认）

| 项 | 值 |
|---|---|
| 主机 | `host-172-16-20-59`（JumpServer 资产 172.16.20.59，用户标识"瑞航"） |
| 跳板 | `jump.aicloud.paigod.work:2222`，**token 一次性**，用完即失效；过期时间见用户给的表格 |
| 账号 | root |
| GPU | **8 × NVIDIA GeForce RTX 5090，32607 MiB each**，driver 显示 814 MiB 常驻 |
| 占用 | **GPU 7 已占 30.3 GB**（疑似基线容器），**0–6 空闲** |
| 连接 | `scripts/dev/remote_5090.sh {up,status,ssh,run,push,pull,put}`，凭据 `~/.ssh/jms_5090_{user,pass}` |

**token 纪律（重要）**：token 是一次性凭据，**第一条命令必须直接 `up` 建常驻 ControlMaster**，
不能先"探活"再建——探活就把 token 用掉了。之后所有操作走 mux。

## 模型

- 名称：**Kaon-V3**，路径 `/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed`
- 目录名说明是 HF cache 布局（`models--<org>--<name>`），**nvpf4 = NVFP4 量化**，`mixed` 表示部分层未量化
- **架构未知**（待读 `config.json`）：需确认 hidden / layers / 注意力类型（是否有线性注意力 / MoE）、
  量化 scheme（NVFP4 block scale 格式）、tokenizer、chat template
- 基线容器：`kaon-vllm-pegaflow`（pegaflow = B300 上见过的那个 KV/专家缓存服务）

## 待摸清（第一步，不改任何代码）

1. `config.json` 全文 → 架构参数、是否 MoE、`quantization_config`
2. 目录结构与权重文件清单、总大小、dtype
3. 基线容器怎么起的（镜像、命令、端口、`--max-num-batched-tokens` 等），**只读不改**
4. 基线实测：TTFT / TPOT / 吞吐 / 并发曲线，用统一的 `ttft_probe.py` + bench 口径
5. dinfer 现有模型 crate 里哪个最接近 → 决定是新建 `models/kaon3` 还是扩展现有

## 停止条件（机器可判定）

1. `scripts/verify/kaon_v3.sh` exit 0：服务起来、`/v1/models` id 正确、greedy 输出与基线逐 token 一致（或 perplexity 同档）
2. 性能：同并发 / 同 prompt / 同 out_len，**out tok/s 与 TTFT 双双 ≥ 基线 1.2×**
3. 不退化：c1 与 c32 两端都要达标（只赢高并发不算）
4. 迭代上限：30 轮；每 5 轮无进展回退审视路线

## 轮次

| 轮 | 改动 | 结果 | 结论 |
|---|---|---|---|
| 0 | 连通 + 环境勘察 | 8×5090 确认，GPU 7 跑基线 | token 一次性（见卡点） |
| 1 | 模型 / 基线勘察 | 见下「勘察结果」 | 架构 = Gemma4 MoE 多模态，量化 = MIXED(NVFP4 专家 + FP8 其余) |

## 勘察结果（2026-10-21，轮 1）

### 模型（`/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed`，16 GB，2 个 safetensors）

- `architectures = ["Gemma4ForConditionalGeneration"]`，`model_type = gemma4` → **多模态**（文本 + 视觉塔 + 音频塔）
- **text_config**：`hidden_size 2816`，`num_hidden_layers 30`，`num_attention_heads 16`，
  `head_dim 256` / `global_head_dim 512`，`num_key_value_heads 8` / `num_global_key_value_heads 2`，
  `intermediate_size 2112`（dense MLP），`moe_intermediate_size 704`，`num_experts 128`，`top_k_experts 8`，
  `enable_moe_block true`，`sliding_window 1024`，`max_position_embeddings 262144`，`vocab_size 262144`（tied），
  `hidden_activation gelu_pytorch_tanh`，`final_logit_softcapping 30.0`，`attention_k_eq_v true`（**K 与 V 共享**）
- **layer_types**：30 层里 **5 层 full_attention**（下标 5、11、17、23、29，即每 6 层一次），其余 25 层 `sliding_attention`（窗口 1024）
- **rope 两套**：full_attention 用 `partial_rotary_factor 0.25` + `rope_theta 1e6` + `proportional`；
  sliding_attention 用 `rope_theta 1e4` + `default`
- **权重命名特殊点**（与标准 HF 不同，直接影响加载器）：
  - `model.language_model.layers.N.mlp.*` 是 **dense MLP**（gate/up/down），MoE 在 `layers.N.experts.N.{gate,up,down}_proj`
  - **每个专家独立一套权重**（3840 = 30 层 × 128 专家），不是合并的 `experts.gate_up_proj`
  - 有 `router.proj.weight` / `router.scale` / `router.per_expert_scale`，还有 `layers.N.layer_scalar`
  - 四个 layernorm：`input_layernorm`、`post_attention_layernorm`、`pre_feedforward_layernorm`
    （+`_2`）、`post_feedforward_layernorm`（+`_1`/`_2`）→ **GEMMA 式前后双 norm**
  - `self_attn.k_norm.weight` 存在 → **QK-norm**
- **量化 `hf_quant_config.json`**（modelopt 0.45.0）：`quant_algo = MIXED_PRECISION`，`kv_cache_quant_algo = FP8`
  - **NVFP4**：恰好 30 项，全是 `layers.N.experts`（即 **MoE 专家走 NVFP4**）
  - **FP8**：394 项 = 视觉塔 189 + 文本层（attn q/k/v/o + dense mlp gate/up/down）
  - 每个量化权重配 `input_scale` / `weight_scale` / `weight_scale_2`（NVFP4 还有二级 scale）

### 基线容器 `kaon-vllm-pegaflow`（GPU 7，`:28067`，已运行 3 周）

```
vllm serve <snapshot> --port 28067 --tensor-parallel-size 1 --pipeline-parallel-size 1   --gpu-memory-utilization 0.90 --dtype auto --served-model-name flowgpt_vllm --max-num-seqs 64   --max-model-len 16384 --block-size 32   --kv-transfer-config '{"kv_connector":"PegaKVConnector","kv_role":"kv_both","kv_connector_module_path":"pegaflow.connector"}'
```
- 镜像 `image.paigpu.com/library/vllm-openai:v0.28.0-pegaflow-pr455-51261c6-cu130`（torch 2.13 cu130）
- env：`CUDA_VISIBLE_DEVICES=7`、**`VLLM_BATCH_INVARIANT=1`**、`PYTHONHASHSEED=0`
- 同镜像还有个 `pegaflow-server-test` 容器（KV 缓存服务）

### 基线实测（2026-10-21，`/tmp/kaon_bench.py`，流式 / ignore_eos / temp 0 / 随机 prompt）

| 并发 | prompt | out | TTFT p50 / p99 (ms) | TPOT p50 (ms) | out tok/s | req/s |
|---|---|---|---|---|---|---|
| 1 | 128 | 128 | 122.1 / 122.1 | 8.34 | 108.3 | 0.85 |
| 8 | 128 | 128 | 175.4 / 184.4 | 9.11 | 758.2 | 5.92 |
| 32 | 128 | 128 | 372.6 / 451.4 | 11.15 | 2261.3 | 17.67 |

**1.2× 门槛（out tok/s）**：c1 ≥ 130、c8 ≥ 910、c32 ≥ 2714；TTFT 不高于上表。

### dinfer 现状差距（必须补齐的）

| 能力 | dinfer 现在 | Kaon-V3 需要 |
|---|---|---|
| 架构 | qwen3 / qwen35 / glm53 / kimik3 | **Gemma4（无）** |
| MoE 路由 | `moe.cu` 有 sigmoid top-k + 分桶（DeepSeek 口径） | Gemma 路由 + `per_expert_scale` + `layer_scalar` |
| NVFP4 GEMM | 只有 `mxfp4.cu` 的**反量化参考**（块 32、e8m0 缩放） | **NVFP4 = 块 16 + e4m3 缩放 + 二级 global scale**，格式不同 |
| FP8 GEMM | **无** | attn 与 dense MLP 全是 FP8 |
| 滑窗注意力 | 无（全是全注意力） | 25 层窗口 1024 |
| QK-norm / K==V | 无 / `attention_k_eq_v` | 需要 |
| 多模态 | 无 | 视觉塔 + 音频塔（**首轮先跳过**，走纯文本路径） |
| arch | 86/89/90/100/103 | **120（RTX 5090，sm_120）** |

### 机器环境

- **宿主机没有 nvcc / rustc / torch / pip**；只有 python3
- 容器 `kaon-vllm-pegaflow` 里有 `nvcc 13.0`、`torch 2.13.0+cu130`
- 128 核 / 503 GB 内存；`/` 439G（剩 332G）、`/hf_data` 1.4T（剩 1.3T）

## 卡点

- **token 一次性**：每次拿到的 token 只能建一次连接。mux 断了就要新 token，所以要把"一次建成、长期复用"做对。
