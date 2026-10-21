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
| 2 | 工具链打通 | **sm_120 编译通过**（dinfer-kernels 38 s） | 宿主机无 nvcc，容器内编；cargo 需走代理 |
| 3 | 参考实现 | 30 层纯 torch 前向跑通，NVFP4 反量化验证 | 见下「参考实现」；与基线对拍未完成 |

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

## 工具链（轮 2，已验证）

| 项 | 结论 |
|---|---|
| 宿主机 | Ubuntu 22.04，gcc 11.4，**无 nvcc / 无 rust**，python3 无 torch |
| Rust | 已装 `~/.cargo/bin`，**rustc 1.99.1**（stable） |
| nvcc | **只在容器** `image.paigpu.com/library/vllm-openai:v0.28.0-pegaflow-pr455-51261c6-cu130` 里，CUDA 13.0（Ubuntu 24.04 基础镜像） |
| **代理** | **`http://172.17.0.1:1081`**（用户提供；`127.0.0.1:1081` 不通，是 docker bridge IP）。rustup / crates.io 都必须走它 |
| 工作目录 | **`/hf_data/yaochen`**（用户 2026-10-21 指定），仓库在 `/hf_data/yaochen/dinfer` |
| sm_120 | **`nvcc -arch=sm_120` 编译通过**；`CUDA_ARCHS=120 cargo build --release -p dinfer-kernels` **38 s 通过** |

**构建命令（宿主机跑，容器内编）**：
```bash
docker run --rm --entrypoint bash --gpus device=0 \
  -v /hf_data/yaochen:/w -v /root/.cargo:/root/.cargo -v /root/.rustup:/root/.rustup \
  -w /w/dinfer -e PATH=/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
  -e CUDA_ARCHS=120 -e https_proxy=http://172.17.0.1:1081 -e http_proxy=http://172.17.0.1:1081 \
  image.paigpu.com/library/vllm-openai:v0.28.0-pegaflow-pr455-51261c6-cu130 \
  -lc 'cargo build --release ...'
```
注意：镜像的 entrypoint 是 `vllm`，**必须 `--entrypoint bash`**，否则参数被 vllm CLI 吞掉。

## 参考实现（轮 3，`/hf_data/yaochen/work/kaon_forward.py`）

纯 torch 版全量前向，用于钉死语义 + 作为 Rust 实现的基准。关键实现点（都已核实）：

| 项 | 结论 |
|---|---|
| **K==V** | **只对 full_attention 层成立**（layer 5 无 `v_proj`；sliding 层 layer 0 有独立 `v_proj`，与 k_proj 数值不同，maxdiff 470） |
| `v_norm` | **无权重参数**，但**仍要算 RMSNorm** |
| `q_norm`/`k_norm` | 有权重，bf16，逐头（`head_dim` 维） |
| V 线性/输出 | 层 0（sliding）：q[4096,2816] k[2048,2816] v[2048,2816] o[2816,4096]，即 NH=16×256、nkv=8×256；层 5（full）：q[8192,2816] k[1024,2816] o[2816,8192]，即 16×512、nkv=2×512 |
| 权重落盘格式 | FP8 层是 `weight[F8_E4M3] + weight_scale[标量F32]`（per-tensor）；NVFP4 层是 `weight[U8] + weight_scale[F8_E4M3] + weight_scale_2[标量F32] + input_scale[标量F32]` |
| NVFP4 布局 | `gate_proj: weight[704,1408]U8, scale[704,176]F8_E4M3` → cols=2816，**块 16**（2816/176）；`down: weight[2816,352], scale[2816,44]` → cols=704，块 16 |
| router | `proj.weight` 是 **BF16**（不是 FP8）；`scale` BF16 [2816]；`per_expert_scale` BF16 [128] |
| `layer_scalar` | BF16 [1] |
| `embed_tokens` | BF16 [262144, 2816]，前向里乘 `sqrt(hidden_size)` |
| PLE | `hidden_size_per_layer_input = 0` → **本模型不启用**，参考实现已跳过 |

### 基线服务的行为陷阱（重要）

- `/v1/completions` 裸 prompt **输出退化**（"HelloHelloHello"、logprob 呈 -1.3865/−1.7615 等差数列）
  → Gemma4 只在 chat 模板下正常，**bench 与对拍都必须走 chat 模板**。
- chat 路径 `logprobs=True, echo=True` 只返回生成部分，**拿不到 prompt 各位置的 logprobs**。
  → 语义对拍改用：模板文本 → `/v1/completions` + `echo=True` + `logprobs`，取 prompt 最后位置 top-k。

## NVFP4 原生 GEMM 的硬件基础（轮 4，**决定性**）

用户 2026-10-21 明确：**不能反量化**（c32 时 128 专家全覆盖，bf16 展开 45.6 GB 单卡放不下，
且带宽翻 4×）。所以必须原生 NVFP4 GEMM。ISA 探测结论：

| 编译方式 | 结果 |
|---|---|
| `-gencode=arch=compute_120a,code=sm_120a` | ✅ **编译通过** |
| `-gencode=arch=compute_103a,code=sm_103a` | ❌ `Instruction 'mma with block scale' not supported on .target 'sm_103a'` |
| `-arch=sm_120a`（隐式） | ❌ 生成的 PTX 仍是 `.target sm_120`，ptxas 报同样错误 |
| `-arch=sm_120` | ❌ 同上 |

**必须显式写 `-gencode=arch=compute_120a,code=sm_120a`**，`-arch=sm_120a` 不会生成 `a` 变体 PTX。
→ `crates/kernels/build.rs` 的 arch 映射要把 `120` 也加进 `→ {a}a` 的分支（现在只有 100/103）。

### 正确的 PTX 形式（抄 CUTLASS `cute/arch/mma_sm120.hpp:3216`）

```ptx
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
  {d0,d1,d2,d3},          // f32 累加器
  {a0,a1,a2,a3},          // A: 4×u32（8 个 fp4 每 reg，k=64）
  {b0,b1},                // B: 2×u32
  {c0,c1,c2,c3},          // C 输入（f32）
  {sfa}, {bidA,tidA},     // scale A + 块坐标（h 约束 = u16）
  {sfb}, {bidB,tidB};     // scale B + 块坐标
```

与直觉不同的三点：① `.kind::` 与 `.block_scale` 在 `.m16n8k64` **之前**；② scale 后面还要跟
`{bid, tid}` 两个 u16 定位块；③ 末尾 `.ue4m3` 是 scale 的格式（不是 e8m0）。

- `scale_vec::4X` → 每 4 个 k-block 一个 scale（k=64 覆盖 4 个 16 元素块）
- 另有 `scale_vec::2X`（`mma_sm120.hpp:3159`）与 `m16n8k32` 形式待测
- 基线走的正是这条路：`FlashInferB12xNvFp4LinearKernel`（**SM120+ 专用**，
  `Sm120BlockScaledDenseGemmKernel`），失败才回落 fbgemm/cutlass/emulation

## NVFP4 MMA 数值验证进度（轮 4b，**未通过**）

| 项 | 状态 |
|---|---|
| 指令可编译（compute_120a） | ✅ 已确认 |
| 指令可执行（kernel 无 CUDA 错误） | ✅ 已确认，`kernel: no error` |
| **寄存器映射正确** | ❌ **未通过**：实测 D 全为 -8，CPU 参考 -32，`bad=128/128` |
| scale 语义（4X 是每 4 块共用还是 4 个独立） | ❌ 未确定 |

**CUTLASS traits 实测布局**（`show_layout.cpp` 打印，`SM120_16x8x64_TN_VS<...,VS=16>`）：

```
ALayout  ((_4,_8),(_8,_2,_2)):((_128,_1),(_16,_8,_512))
BLayout  ((_4,_8),(_8,_2)):((_64,_1),(_8,_256))
SFALayout ((_2,_2,_8),_64):((_8,_0,_1),_16)
SFBLayout ((_4,_8),_64):((_0,_1),_8)
CLayout  ((_4,_8),(_2,_2)):((_32,_1),(_16,_8))
```

A 的 t=0/1、v=0..31 线性偏移已打印（见 `/hf_data/yaochen/work/show_layout.cpp` 输出），
在 (M16,K64) 列主序下换算得 `(m,k) = (lin%16, lin/16)`；按此拼装后结果仍不正确
（所有 D 相等，说明 A/B 里有系统性错位，不是随机误差）。

**当前判断**：映射还需一至两轮才能钉死。可行的替代路线是直接用 CUTLASS 的
`make_fragment` + `copy` 把 smem 搬进寄存器（它内部就是这套布局），避免手推。

## 卡点

- **token 一次性**：每次拿到的 token 只能建一次连接。mux 断了就要新 token，所以要把"一次建成、长期复用"做对。

## Gemma4 前向语义（读 vLLM `gemma4.py` 1708 行，2026-10-21）

### 单层结构（`Gemma4DecoderLayer.forward`）

```
residual = x
h = input_layernorm(residual)                    # 有权重
h = self_attn(h)                                  # 内部顺序见下
h = post_attention_layernorm(h)                   # 有权重，作用于 attn 输出
h = h + residual                                  # ← 残差加在 attn 之后
residual = h

h = pre_feedforward_layernorm(h)                  # 有权重
h = mlp(h)                                        # dense MLP，**无条件执行**
h1 = post_feedforward_layernorm_1(h)
h2 = moe(pre_feedforward_layernorm_2(residual), router(residual))
h2 = post_feedforward_layernorm_2(h2)
h = h1 + h2                                       # ← dense MLP 与 MoE 并联
h = post_feedforward_layernorm(h)
h = h + residual
h = h * layer_scalar                              # ← 每层一个标量
```

**关键：dense MLP 和 MoE 是并联的两条支路，不是二选一。**

### 注意力（`Gemma4Attention.forward`）

```
qkv = qkv_proj(h)                                 # k_eq_v 层：K 权重同时填 K 和 V 槽 → V == K
q,k,v = split(qkv, [q_size, kv_size, kv_size])
q = q_norm(q)                                     # RMSNorm(head_dim)，有权重，逐头
k = k_norm(k)                                     # RMSNorm(head_dim)，有权重
q,k = rope(positions, q, k)                       # is_neox_style=True
v = v_norm(v)                                     # RMSNorm(head_dim)，**无权重**
o = o_proj(attn(q,k,v))                           # scaling = 1.0（QK-norm 隐式承担）
```

- `scaling = 1.0`，不是 `1/sqrt(head_dim)`
- sliding 层 `head_dim = 256`、`num_kv_heads = 8`；full 层 `head_dim = 512`、`num_kv_heads = 2`
  （来自 `gemma4_layer_config`，即 `head_dim`/`global_head_dim` 与 `num_key_value_heads`/`num_global_key_value_heads`）
- sliding 层窗口 1024；full 层无窗口
- `attention_k_eq_v = true` → **K 与 V 是同一份权重**，加载时 K 权重复制到 V

### MoE 路由（`Gemma4Router` + `Gemma4MoE`）

```
# router 输入预处理（只作用于路由，不作用于专家输入）
r = rmsnorm_no_weight(x) * hidden_size**-0.5 * scale      # scale 是学出来的 [hidden]
logits = proj(r)                                          # → fp32, [T, 128]

# 路由
p = softmax(logits, over ALL experts)
topk = top8(p)                                            # 按 p 选
w = p[topk] / sum(p[topk])                                # renormalize
out = Σ_e  expert_e(x) * w_e * per_expert_scale[e]
```

**注意**：`per_expert_scale` 折进路由权重里（`Σ (expert * w * scale)`）。
专家激活是 **`gelu_tanh`**（不是 silu）。

### 其他

- `final_logit_softcapping = 30.0`：`logits = tanh(logits/30) * 30`
- `tie_word_embeddings = true`：embed 与 lm_head 共用
- `hidden_size_per_layer_input = 0` → **PLE（Per-Layer Embedding）本模型不启用**，可跳过
- `num_kv_shared_layers = 0` → **无 KV 共享**，可跳过
- `use_double_wide_mlp = false` → dense MLP 宽度不翻倍
- `embed_input_ids` 里有个 `normalizer` 标量（=`sqrt(hidden_size)`），embedding 后要乘

## 权重显存账（决定实现路线）

| 部分 | 参数量 | 存储 |
|---|---|---|
| 128 专家 × 30 层 × 3 矩阵 (704×2816) | **22.8 B** | NVFP4 = **11.4 GB** |
| dense MLP × 30 (2112×2816×3) | 0.54 B | FP8 = 0.54 GB |
| attention × 30 | 1.04 B | FP8 = 1.04 GB |
| embed / lm_head（tied） | 0.74 B | bf16 = 1.48 GB |
| **合计** | **~25 B** | **~14.5 GB**（与 16 GB 实测吻合） |

**去量化成 bf16 = 约 48 GB，单张 5090（32 GB）放不下** → 必须保持压缩态、在 GEMM 内反量化。
（除非用 TP=2 分摊到两张卡；基线是 TP=1，为可比性优先单卡。）

### roofline（单卡 1.79 TB/s）

- **c1 decode**：每 token 读 8/128×11.4 = 0.71 GB 专家 + 1.6 GB 固定 = **2.3 GB → 1.3 ms**
  基线实测 TPOT **8.34 ms** → 基线只到 roofline 的 **16%**
- **c32 decode**：32 token 覆盖约全部 128 专家 → 11.4 + 1.6 = **13 GB → 7.3 ms/step → 4380 tok/s**
  基线实测 **2261 tok/s** → 只到 **52%**
- **1.2× 目标**：c1 ≥ 130 tok/s（TPOT ≤ 7.0 ms）、c8 ≥ 910、c32 ≥ 2714 tok/s

结论：**空间存在且不小**，瓶颈是基线的 kernel 效率，不是硬件。
