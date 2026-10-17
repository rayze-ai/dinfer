# 整体设计与实现计划

> 2026-10-16 拆分：模型相关内容（结构事实、模块表、状态、风险）移到 `design-kimik3.md` / `design-glm53.md` / `design-qwen3.md`，本文件只放整体架构、共性模块、阶段计划、性能分析、决策。

> 依据：vLLM main `de8291e`（2026-10-13，对应 0.31.0）源码，位于 `/data/ref/vllm`；
> 两模型 HF 仓库文件位于 `/data/ref/models/{kimik3,glm53}`。本文所有行号 / 文件名均指这两处。

## 0. 一句话

用 Rust 重写 vLLM v1 引擎的"主干"（启动、调度、KV 管理、执行、采样、HTTP），
抛掉通用插件层；每个模型是一个独立 crate，只实现自己的层结构、权重映射、tokenizer 与 chat 格式。
构建时 `core + server + 一个模型 crate`。

## 1. vLLM v1 核心分析

### 1.1 进程与数据流

```
客户端 ──HTTP──▶ entrypoints/launchers (FastAPI)
                  │ ChatCompletionRequest → renderer(chat 模板/多模态占位) → tokenizer
                  ▼
              AsyncLLM (前端进程)
                  │ InputProcessor: 分词、mm hash、校验 → EngineCoreRequest
                  │ OutputProcessor: 增量 detokenize、stop 判定、logprobs、SSE
                  ▼ ZMQ(msgpack)
              EngineCore (独立进程)              ← vllm/v1/engine/core.py:115
                  │ Scheduler.schedule()  → SchedulerOutput
                  │ Executor.execute_model(SchedulerOutput) → ModelRunnerOutput
                  │ Scheduler.update_from_output()
                  ▼
              Worker → GPUModelRunner (每 GPU 一个)   ← vllm/v1/worker/gpu_worker.py:192
                  │ InputBatch(持久化 batch) + BlockTable → 扁平 input_ids/positions/slot_mapping
                  │ 各 attention backend 构建 metadata
                  │ CudagraphDispatcher.dispatch(num_tokens, uniform_decode) → 回放/直跑
                  │ model.forward → logits → Sampler → (spec decode: drafter + rejection)
```

### 1.2 启动阶段（Worker 侧顺序，`gpu_worker.py` / `engine/core.py:261`）

| 步骤 | vLLM 位置 | 做什么 | Rust 对应 |
|---|---|---|---|
| 参数解析 | `engine/arg_utils.py` → `config/*` | CLI → `VllmConfig`（Model/Cache/Parallel/Scheduler/Load/Speculative/Compilation） | `core::config` |
| init_device | `gpu_worker.py:436` | 选卡、初始化分布式、记录内存快照 `MemorySnapshot` | `core::device` |
| load_model | `gpu_worker.py:587` → `model_runner.load_model` | 构建模型、流式读 safetensors、按名映射到参数、`process_weights_after_loading`（量化权重重排/吸收） | `core::loader` + 模型 crate `load_weights` |
| determine_available_memory | `gpu_worker.py:634` | 用 `max_num_batched_tokens` 的 dummy batch 跑一次 `profile_run`，测峰值；`gpu_memory_utilization * total - peak` = KV 可用 | `core::runner::profile` |
| KV spec → config | `core.py:268` `get_kv_cache_specs` → `get_kv_cache_configs` | 每层给出 `KVCacheSpec`（见 1.4），按 page 大小分组，算 `num_gpu_blocks` | `core::kv::planning` |
| initialize_kv_cache | `model_runner.initialize_kv_cache` | 一次性分配大 buffer，按 layout 切成每层 view，注册到 attention 层 | `core::kv::alloc` |
| warmup | `worker/gpu/warmup.py`, `kernel_warmup` | 预触发 kernel JIT / autotune | `core::runner::warmup` |
| capture_model | `worker/gpu/cudagraph_utils.py`（V2）/ `cudagraph_dispatcher.py`（V1） | 按 `cudagraph_capture_sizes` 捕获 FULL（decode）/ PIECEWISE 图，先 PIECEWISE 再 FULL，大到小 | `core::cudagraph` |

CUDA graph 要点：key 是 `BatchDescriptor{num_tokens(padded), num_reqs, uniform}`；
运行时把 batch 的 token 数向上取整到最近的 capture size，uniform decode（每请求 `1+num_spec` token）走 FULL 图，否则走 PIECEWISE 或直跑。两目标模型走 breakable 模式（见 1.4b），piecewise 图是"attention 处断开的若干图段"，不是编译器切分。

### 1.3 推理阶段（每个 step）

1. **Scheduler.schedule()**（`v1/core/sched/scheduler.py`）
   - 先遍历 `running`：每请求分配本 step 的 token 预算（decode 1 个 + spec token；chunked prefill 则按剩余预算切块），向 `KVCacheManager.allocate_slots` 要块，不够则**抢占最后一个**（释放其块、回 waiting）。
   - 再遍历 `waiting`：`get_computed_blocks` 做 prefix cache 命中，剩余 token 按预算切块；多模态输入受 encoder 预算与 `EncoderCacheManager` 限制。
   - 产出 `SchedulerOutput`（`sched/output.py:232`）：新请求全量数据 `NewRequestData`、老请求增量 `CachedRequestData`（只传新 block id 与 num_computed_tokens）、`num_scheduled_tokens`、spec tokens、要处理的 encoder 输入、finished ids、需要清零的新块。
2. **Worker 执行**：`InputBatch` 增删请求，`BlockTable` 更新；构建 `input_ids/positions/slot_mapping/query_start_loc/seq_lens`；各 attention backend 的 metadata builder 产出 kernel 参数；`CudagraphDispatcher.dispatch`；forward；取 `logits_indices` 行做采样。
3. **Sampler**（`v1/sample/sampler.py`，`SamplingMetadata` 字段见 `metadata.py:15`）：logits → 惩罚（frequency/presence/repetition）→ 温度 → top-k/top-p/min-p → 采样/argmax → logprobs；spec decode 用 `RejectionSampler` 校验 draft。
4. **update_from_output**：追加 token、判 stop（eos/max_tokens/stop 串）、释放完成请求的块、触发 drafter 下一轮 draft。
5. **OutputProcessor**（前端进程）：增量 detokenize、流式输出。

### 1.4 KV cache 模型（`v1/kv_cache_interface.py`）

- `KVCacheSpec{block_size}` 子类：`FullAttentionSpec`、`MLAAttentionSpec`（每 token 一个 latent 向量，head_size = kv_lora_rank + rope_dim）、`SlidingWindowSpec`、`MambaSpec`（递归状态，每请求固定若干 slot，不按 token 增长）、`SparseCacheRole.INDEXER`（DSA 的 fp8 索引 K cache）。
- **混合模型**（Kimi-K3 = KDA + MLA）：按 spec 相同性分成多个 **KV group**；每组独立 block pool / block table；`KVCacheCoordinator` 保证各组对同一请求分配一致的逻辑块数。Mamba 类层以 `tokens_per_state` 把一个"块"等价为固定 token 数，便于与 attention 组统一调度。
- Prefix cache：按 `hash_block_size` 对 prompt 做链式 hash（含 mm hash、cache_salt），`BlockPool` LRU 淘汰；Mamba 层只能在 checkpoint 位置命中（`get_mamba_prefill_checkpoint_position`）。

### 1.4b 对本项目有直接影响的几个事实（后台分析核实）

- **Model Runner V2 是默认**（`config/vllm.py:703` `use_v2_model_runner`，有 Triton 且无不支持特性即 True）。V2 在 `vllm/v1/worker/gpu/model_runner.py`，CUDA graph 由 `gpu/cudagraph_utils.py` 的 `ModelCudaGraphManager` 管理（`BatchExecutionDescriptor{cg_mode, num_tokens, num_reqs, uniform_token_count, …}`）。旧的 `gpu_model_runner.py` + `cudagraph_dispatcher.py` 是回退路径。Rust 以 V2 为参照。
- **两模型都不走 torch.compile**：`GlmMoeDsaForCausalLM`、`KimiK3ForConditionalGeneration` 及其 MTP 都在 `DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES`（`config/vllm.py:74-101`）→ compilation mode NONE + `breakable cudagraph`：捕获时遇到 `@eager_break_during_capture` 的 attention/KV 写入就断开图段、eager 执行、再续捕获（`compilation/breakable_cudagraph.py`）。对 Rust 是好消息：只需实现 **FULL decode 图 + 分段 piecewise 图**，不需要图编译器。
- **KV 规划**在 `v1/core/kv_cache_planning.py`（`get_kv_cache_configs:1481`，`num_blocks = available // bytes_per_block:638`，跨 rank 取最小）。
- **内存公式**：`available = gpu_memory_utilization × total − non_kv_cache_memory(profile) − cudagraph_estimate`（`gpu_worker.py:740`）。
- **默认 batch 预算**按显存：≥160 GB → 16384 tokens / 1024 seqs；≥70 GB → 8192 / 1024；否则 2048 / 256（`arg_utils.py:2910`）。
- **capture sizes**：`[1,2,4] + 8..256 step 8 + 256..max step 16`，`max = min(max_num_seqs × (1+k) × 2, 512)`（sm100 为 1024）（`config/vllm.py:2292`）。
- **tokenizer / renderer / parser 的模式选择**按架构名（`config/model.py:736`）：`KimiK3ForConditionalGeneration → kimi_k3`（HF tokenizer 类 + `KimiK3Renderer` + `KimiK3Parser`）；`GlmMoeDsaForCausalLM` **不在特判列表**，走 `auto/hf`：标准 HF tokenizer + jinja 模板 + 通用 reasoning/tool parser（GLM 的 `<think>` 与 `<tool_call>` 格式）。`deepseek_v32` 模式的 DSML parser 与 GLM 无关。
- **服务器代码位置**：`entrypoints/openai/api_server.py` 已是废弃转发；真实现在 `entrypoints/launchers/`（路由表 `launchers/api_server/routers.py`，chat 处理 `openai/chat_completion/serving.py`）。

### 1.5 可以丢掉的东西

LoRA、Pooling/Embedding、Encoder-Decoder、CPU/XPU/ROCm/TPU、Ray、PP、EPLB、KV connector / P-D 分离、sleep mode、plugins、torch.compile、通用 HF 模型加载、PromptAdapter、Diffusion scheduler。
**保留**：TP（后期）、EP（Kimi 896 专家几乎必须）、chunked prefill、prefix cache、spec decode（MTP）、structured output（可后置）、多模态（仅 Kimi 图像）。

## 2. 模型

每个模型一个 crate、一份设计文档。结构事实、模块表、实现状态、模型特有风险都在各自文件里，本文件只放共性。

| 模型 | crate | 设计文档 | 状态（2026-10-15） |
|---|---|---|---|
| Kimi-K3 | `dinfer-kimik3` | `design-kimik3.md` | ⏸ 挂起（只有 config），等 ≥ 8×H200 |
| GLM-5.3 | `dinfer-glm53` | `design-glm53.md` | ✅ P3–P6 tiny fixture 全过；真权重挂起 |
| Qwen3-0.6B | `dinfer-qwen3` | `design-qwen3.md` | ✅ Q1 真权重通过、✅ Q2 采样/请求/资源/metrics/健壮性通过（2026-10-16，4090）；主流程验证模型（Q3–Q5） |

### 2.1 模型间共享的部分

MLA 投影与吸收（W_UK 吸进 q、W_UV 吸进 o）、paged MLA decode kernel、分块 prefill、sigmoid+bias 分组 top-k 路由、shared expert、MTP 结构（enorm/hnorm/eh_proj）、1M 上下文、`reasoning_effort` 三档。
差异：KDA（仅 K3）、DSA 索引器（仅 GLM）、MXFP4 vs FP8、latent MoE（仅 K3）、AttnRes（仅 K3）、视觉（仅 K3）、tokenizer 与 chat 格式。

Qwen3 与上面两者的交集更小：只共享 RMSNorm / RoPE（neox）/ SwiGLU / GEMM / scheduler / runner / engine / server / tokenizer / 模板层；
attention 是标准 GQA（`kernels/cuda/gqa.cu`，`KvCacheSpec::FullAttention`），没有 MLA、MoE、索引器、MTP。

## 3. Rust 工作区

```
dinfer/
  Cargo.toml                   # workspace
  crates/                      # 公用基础设施（模型无关）
    core/        dinfer-core   # 配置、调度、KV、runner、采样、metrics、engine、trait
    kernels/     dinfer-kernels# CUDA/C++ 算子 + FFI（build.rs 调 nvcc），按 feature 裁剪
    layers/      dinfer-layers # 基于 kernels 的 Rust 层：Linear/RMSNorm/RoPE/MLA/MoE/GQA
    text/        dinfer-text   # tokenizer、chat 模板、StopChecker
    server/      dinfer-server # axum HTTP：OpenAI 兼容接口、SSE、/metrics、健康检查
  models/                      # 每个模型一个 crate（2026-10-16 起从 crates/ 挪出，便于区分公用与模型私有）
    qwen3/       dinfer-qwen3  # Qwen3-0.6B：模型、executor、service（主流程验证模型）
    glm53/       dinfer-glm53  # GLM-5.3：模型、DSA 索引器、FP8 MoE、MTP、chat 模板
    kimik3/      dinfer-kimik3 # Kimi-K3：模型、KDA、latent MoE、AttnRes、ViT、XTML（骨架）
  bins/
    dinfer/                    # 单一二进制；feature `qwen3` / `glm53` / `kimik3` 决定链接哪个模型
  tests/                       # 跨 crate 集成 / 数值对拍
  scripts/verify/              # exit 0 = 通过
```

需求里只列了 core / server / kimik3 / glm53 四个 crate。这里多拆出 `kernels` 与 `layers` 两个，原因：MLA、MoE 路由、RMSNorm、采样等算子两个模型都要用，放进 core 会让 core 依赖 CUDA 工具链；单独成 crate 便于 CPU-only 单测 core 的调度逻辑。构建仍是 `cargo build -p dinfer --features kimik3`。

### 3.1 关键依赖（固定版本）

| 用途 | crate | 备注 |
|---|---|---|
| CUDA driver/runtime/cuBLAS/NCCL | `cudarc` | 流、事件、graph capture、模块加载（PTX/cubin） |
| 自写 kernel | `cc` + `build.rs` 调 `nvcc` | `-gencode` 按目标架构；fatbin 嵌入 |
| safetensors | `safetensors` + `memmap2` | mmap 分片，零拷贝按名取 tensor |
| tokenizer | `tokenizers`（GLM）、`tiktoken-rs` 或自实现 BPE（K3，需自定义 pat_str，用 `fancy-regex`） |
| chat 模板 | `minijinja`（GLM 的 jinja 模板直接跑） |
| HTTP | `axum` + `tokio` + `serde`；SSE 用 `axum::response::sse` |
| 并发 | `tokio`（前端），引擎线程用 `std::thread` + `crossbeam-channel` |
| 哈希 | `xxhash-rust`（prefix cache 块 hash） |
| 指标/日志 | `tracing`、`metrics-exporter-prometheus` |

## 4. 模块划分（按 crate）

### 4.1 `dinfer-core`

```
core/src/
  config/        EngineConfig{model, cache, scheduler, parallel, spec, cudagraph}；CLI 用 clap
  model/         trait Model, DraftModel, MultiModalEncoder；ModelConfig(hf config.json 解析)
  tokenizer/     trait Tokenizer{encode, decode, decode_incremental, eos_ids, special}
  chat/          trait ChatRenderer{render(messages, tools, params) -> Vec<u32> + mm placeholders}
                 trait OutputParser{feed(token) -> Delta{reasoning, content, tool_call}}  (流式)
  request/       Request{id, prompt_ids, sampling, mm_features, status, num_computed, output_ids, spec_ids, block_hashes}
                 RequestStatus{Waiting, Running, Preempted, Finished(reason)}
  sched/         Scheduler{running: VecDeque, waiting: Queue(FCFS|Priority), budget}
                 schedule() -> SchedulerOutput；update_from_output()
                 SchedulerOutput{new_reqs, cached_reqs, num_scheduled_tokens, spec_tokens, encoder_inputs, finished, blocks_to_zero}
  kv/            KvCacheSpec{Mla{head_size, dtype}, Mamba{shapes, dtypes}, Indexer{bytes}}
                 KvCacheGroup, BlockPool(LRU + hash map), KvCacheManager{get_computed_blocks, allocate_slots, free}
                 BlockTable(GPU 侧 i32 表，按组)
                 planning: specs + available_bytes -> KvCacheConfig{num_blocks, groups, tensors}
  loader/        WeightSource(safetensors 目录 mmap，按名迭代)；WeightMapper(名字重写/融合/分片规则)
  runner/        ModelRunner{input_batch, block_tables, attn_meta_builders, cudagraph, sampler}
                 execute(SchedulerOutput) -> RunnerOutput{sampled, logprobs, draft}
                 profile_run()、initialize_kv_cache()、capture_graphs()
  attn/          CommonAttnMetadata{query_start_loc, seq_lens, slot_mapping, block_table, num_decodes/prefills}
                 trait AttnMetadataBuilder（MLA / Mamba / Indexer 各一个实现，放在 layers 或模型 crate）
  sample/        SamplingParams, SamplingMetadata(设备侧张量), Sampler（调用 kernels）, RejectionSampler
  spec/          MtpProposer：用 DraftModel 产 K 个 draft；接受率统计
  cudagraph/     Dispatcher{capture_sizes, keys}；GraphRunner(cudarc graph capture/replay)
  engine/        Engine：拥有 Scheduler + Runner，step loop 线程；AsyncEngine：mpsc 入口 + 每请求输出流
  output/        OutputProcessor：增量 detok、stop 串、finish_reason、logprobs 组装
  mm/            图像预处理抽象、EncoderCache（按 mm hash）、embedding 合并
```

核心 trait（草案）：

```rust
pub trait Model: Send + Sync {
    fn config(&self) -> &ModelConfig;
    /// 每层的 KV/状态规格，runner 据此规划 group 与 block 数
    fn kv_cache_specs(&self) -> Vec<(LayerId, KvCacheSpec)>;
    fn bind_kv_cache(&mut self, caches: &KvCacheTensors);
    fn load_weights(&mut self, src: &mut dyn WeightSource) -> Result<()>;
    fn embed(&self, ids: &DevTensor, mm: Option<&MmEmbeddings>) -> Result<DevTensor>;
    fn forward(&self, x: DevTensor, positions: &DevTensor, ctx: &ForwardCtx) -> Result<DevTensor>;
    fn logits(&self, hidden: &DevTensor, rows: &DevTensor) -> Result<DevTensor>;
    fn draft(&self) -> Option<&dyn DraftModel>;
    fn vision(&self) -> Option<&dyn MultiModalEncoder>;
}
pub struct ForwardCtx<'a> { pub attn: &'a dyn AttnMetadata, pub slot_mapping: &'a BTreeMap<LayerId, DevTensor>, pub graph_mode: GraphMode, pub stream: &'a CudaStream }
```

### 4.2 `dinfer-kernels`（CUDA/C++ + FFI）

按功能列，标注来源策略（自写 / 移植 / 绑定第三方）：

| 组 | 算子 | 来源 | 两模型 |
|---|---|---|---|
| 基础 | rmsnorm、fused_add_rmsnorm、gated_rmsnorm、layernorm(fp32)、embedding gather、residual add、cast | 自写 | 共用 |
| RoPE | 非 neox（interleave）、fp32 cos/sin 表、融合进 q/k 投影后 | 自写 | GLM（K3 MLA 无 RoPE；K3 DSpark draft 有） |
| GEMM | bf16 GEMM（cuBLASLt）、FP8 块缩放 GEMM（DeepGEMM 绑定，sm90+；sm8x 回退 dequant+bf16）、MXFP4 权重 GEMM（sm100 原生 / 其他架构 dequant 回退） | 绑定 + 自写回退 | 共用 |
| 激活 | silu_and_mul、situ_and_mul（含 linear_beta 项） | 自写 | GLM / K3 |
| MoE | sigmoid+bias 分组 top-k（noaux_tc）、token 按专家排序/对齐、grouped GEMM（bf16 / FP8 block / MXFP4）、权重求和归约 | 移植 vLLM triton `fused_moe` 思路，自写 CUDA；FP8 用 DeepGEMM grouped | 共用，量化不同 |
| MLA attention | prefill：q·k^T 分块 flash（含 chunked context merge）；decode：W_UK 吸收后的 MQA paged decode（FlashMLA 绑定，sm90+；sm8x 自写参考版） | 绑定 FlashMLA + 自写参考 | 共用 |
| MLA 辅助 | kv 拼接写 cache（bf16/fp8）、W_UK/W_UV bmm、输出门 sigmoid 乘 | 自写 | 共用 / K3 门 |
| KDA | causal_conv1d prefill/update、KDA chunk prefill（chunk_intra/inter）、fused_recurrent decode、输出门 | 移植 `third_party/flash_linear_attention/ops/kda.py`（Triton → CUDA，或 Triton AOT 出 PTX 由 cudarc 加载） | K3 |
| AttnRes | attn_res 融合算子（块前缀和 + per-token 标量门 + RMSNorm） | 移植 `kimi_k3/nvidia/ops/attn_res.py` | K3 |
| DSA | indexer fp8 logits（index_q·index_k 加权求和）、top-k 2048 选择、索引 K cache fp8 写入、稀疏 MLA decode（按索引 gather 的 MQA） | 移植 `sparse_attn_indexer.py`/`indexer_topk.py`；稀疏 attention 绑定 FlashMLA sparse | GLM |
| FP8 | per-token-group(1×128) 动态量化、块 scale 处理 | 自写 | GLM |
| MXFP4 | weight_packed/e8m0 scale 解包、与 MoE grouped GEMM 融合 | 自写 | K3 |
| 采样 | 惩罚、温度、top-k/top-p/min-p、argmax、多项式采样（Philox）、logprobs gather、rejection sampling | 自写 | 共用 |
| 视觉 | ViT attention（flash，可复用 MLA prefill 的 dense 路径）、GELU-tanh、patch merge、双线性位置插值 | 自写 | K3 |

FFI 约定：每个 kernel 一个 `extern "C"` 入口，接收裸指针 + 维度 + `cudaStream_t`；Rust 侧 `unsafe` 封装在 `dinfer-kernels`，`dinfer-layers` 之上不再出现 `unsafe`。

### 4.3 `dinfer-layers`

`Linear{Bf16|Fp8Block|Mxfp4}`、`MergedLinear`（融合多投影 + 分片加载）、`RmsNorm`、`Rope`、`Embedding`、`LmHead`、
`MlaAttention{q_lora, kv_lora, nope, rope, v, use_rope, output_gate, sparse: Option<Indexer>}`（含 W_UK/W_UV 吸收、prefill/decode 分流、cache 写入）、
`MoeRouter`、`FusedMoe{experts: ExpertWeights(Bf16|Fp8|Mxfp4), shared, pre/post transform}`、`Mlp`、`Sampler`。
每个层实现 `fn kv_spec(&self) -> Option<KvCacheSpec>` 与 `fn load(&mut self, name, tensor)`。

### 4.4 `dinfer-server`

```
server/src/
  main.rs            clap 参数 → EngineConfig → 按 feature 构造模型 → AsyncEngine → axum
  routes/            /v1/chat/completions  /v1/completions  /v1/models  /health  /tokenize  /detokenize  /metrics
  openai/            请求/响应类型（含 reasoning_content、tool_calls、stream_options、reasoning_effort）
  chat.rs            messages → ChatRenderer（模型 crate 提供）→ token ids + mm 输入
  stream.rs          引擎输出流 → OutputParser（reasoning / tool_call 拆分）→ SSE chunks
  metrics.rs         TTFT/TPOT/队列长度/KV 利用率/prefix 命中率
```

无鉴权，默认只监听 127.0.0.1；对外暴露需反代加鉴权（在 requirements 约束里写明）。

### 4.5 模型 crate

`dinfer-kimik3` / `dinfer-glm53` / `dinfer-qwen3` 的模块表见各自的 `design-<model>.md` §2。
模型 crate 只做组装（config → layers 组件 → Model/Executor → Service）；可复用的层（MLA、MoE、Indexer、GQA attention、Mlp、RmsNorm、Rope、WeightSource）都在 `dinfer-layers`，
tokenizer / chat 模板在 `dinfer-text`。

## 5. 分阶段计划

每阶段都有 `scripts/verify/<阶段>.sh`，exit 0 为通过；数值对拍以 HF transformers（PyTorch）同配置 **tiny 随机权重模型** 为基准。

| 阶段 | 产出 | 验收 |
|---|---|---|
| P0 骨架 ✅ 2026-10-14 | workspace、CI（`cargo build/test/clippy`）、`cudarc` 能建流/分配/launch 一个 add kernel | `scripts/verify/p0.sh` |
| P1 core 纯逻辑 ✅ 2026-10-14 | config、Request、Scheduler、KvCacheManager（含 prefix cache、混合 group、抢占）、SamplingParams；全部 CPU 单测，用 mock Model | `scripts/verify/p1.sh`：27 单测 + 1024 decode step < 200 µs（实测 86 / 138 µs） |
| P2 基础算子 ✅ 2026-10-14（layers 包装层移到 P3 随 MLA 一起做） | rmsnorm / fused_add_rmsnorm / rope（interleave+neox）/ silu_and_mul / embedding / cuBLASLt bf16 GEMM / argmax / topk_topp_mask / sample（Philox）；全部 sm80+ 参考版 | `scripts/verify/p2.sh`：10 个算子与 PyTorch 对拍全过 |
| P3 稠密 MLA 通路 ✅ 2026-10-14（runner 未做，用测试直接驱动模型；runner 移到 P4 与 MoE 一起） | MLA prefill（展开 K/V causal）/ decode（W_UK/W_UV 吸收 + paged latent cache）参考 kernel；layers：Tensor / WeightSource(safetensors mmap) / Linear / RmsNorm / Rope / Embedding / Mlp / MlaAttention；`glm53::Glm53Dense`（2 层 tiny，HF DeepseekV3 同构）| `scripts/verify/p3.sh`：prefill 14 行 logits max abs 0.004；32 步 decode logits max abs 0.007；greedy 仅在 bf16 平局步分叉 |
| P4 MoE + runner + 混合 batch + FP8 加载 ✅ 2026-10-15 | MoE 参考路径（fp32 gate、sigmoid+bias top-k noaux_tc、专家分桶 + 逐专家 MLP + fp32 加权归约 + shared expert）；`glm53::Glm53Model`（`FeedForward::{Dense,Moe}`）；`core::runner`（`ReqState / BatchPlan / BatchMode / ModelExecutor / Runner`）+ `Glm53Executor`（greedy）；`mla_prefill_paged`（吸收路径、任意 query 长、历史在 cache）→ chunked prefill 与 prefill/decode 混合 batch；fp8 e4m3 128×128 块量化权重加载期解量化（`weights_fp8.rs`，sm86 无 fp8 张量核） | `scripts/verify/p4.sh`：p3 + tiny MoE GLM 3 对拍 + runner 2 单测 + Scheduler+Runner+Executor 双请求对拍（plain 32 步 / chunked+mixed 35 步）× dense/MoE + tiny fp8 GLM 同套对拍；logits worst ≤ 0.008，greedy 仅平局步分叉 |
| ✅ P5 DSA（2026-10-15，`dsa_parity` 3/3 PASS） | 索引器参考 kernel（`dsa.cu`：layernorm / 索引分数（paged 索引 cache）/ top-k / row_vis）、`layers::Indexer`（wq_b / wk / k_norm / weights_proj、前 rope 维 interleave RoPE、索引 K cache）、三条 MLA attention 路径加 `topk_indices` gather 模式、跨层共享 top-k 缓冲（full 写 / shared 读）；对拍基准换 transformers 5.19 `GlmMoeDsa`（隔离 venv）；对拍口径：注入 HF top-k 逐层隔离比每层自算 top-k（差异限于 HF k-th 分数 ± 实测分数误差）；行过滤用实测 hidden 漂移（> 0.02）而非路由 margin；HF 全序列重算在 bf16 下不前缀稳定，fixture 导出每步前缀漂移、命中的 decode 步不断言；全程自算只在可判定行断言（见 pitfalls 2026-10-15 三条） | `scripts/verify/p5.sh`：p4 + tiny DSA GLM（3 层 full/shared/full，top-k 8，长 prompt 256 + 32 步 decode）3 道测试；注入 HF top-k 时三层 attention 输出与 HF 逐层 max abs ≤ 0.0007 |
| ✅ P6 GLM-5.3 完整（2026-10-15，tiny fixture + 真 tokenizer/模板） | tokenizer（`tokenizers`）、chat 模板（minijinja 跑官方 jinja，HF 15/15 逐字符一致）、流式解析器（think / tool_call）、engine 循环、OpenAI server（stream/非 stream、usage、tool_calls、reasoning_content）、MTP nextn=1（draft logits 对拍 + oracle-draft 验证 verify 路径）；真权重验收待 ≥ 8×H200 机器 | `scripts/verify/p6.sh`：chat_parity / parser 单测 / serve_e2e（真进程 HTTP）/ mtp_parity |
| ⏸ P7 KDA（挂起，等资源） | conv1d、chunk prefill、recurrent decode、Mamba 类状态管理、混合 group；tiny Kimi-Linear 对拍 | 同 P3 口径 |
| ⏸ P8 K3 特有（挂起） | AttnRes、latent MoE + situ、MXFP4 解包/GEMM、无 RoPE MLA + 输出门；tiny K3 文本对拍 | 同上 |
| ⏸ P9 K3 多模态 + tokenizer（挂起） | MoonViT、投影、图像 token 计数、XTML 渲染、tiktoken；图文对拍 | 图像请求端到端 |
| ⏸ P10 性能（真机部分挂起；单卡部分并入 Q3） | CUDA graph（decode FULL 图含 attention 与采样）、FlashMLA/DeepGEMM 快路径、TP/EP、EP 下 all-to-all、§8.2 算子融合、draft+target 同图 | §8.4 五个场景与 vLLM 同机对比，达到 requirements 口径 |

性能相关的早期门禁（不等到 P10）：
- P1：scheduler 微基准，1024 请求 × 1 step 的 `schedule() + update_from_output()` < 200 µs（CPU，criterion）。
- P3：runner 每 step 不出现 `cudaStreamSynchronize`（nsys 检查）；输出拷贝走 copy stream。
- P4/P5/P7/P8：每新增一类层，记录 decode 路径 launch 数，超过 §8.2 目标即在该阶段内融合，不留到 P10。

建议 GLM 先于 K3：GLM 结构 = 标准 MLA + MoE + 一个索引器，覆盖了 core 的全部路径；K3 再叠 KDA/AttnRes/latent MoE/视觉四个增量。

## 5b. 方向调整（2026-10-15）：先用 Qwen3-0.6B 把主流程做实

**背景**：P0–P6 在 tiny 随机权重上验证了 GLM-5.3 整条链路，但真权重（GLM 755 GB / K3 1.56 TB）需要 ≥ 8×H200 级别机器，
暂时没有。继续在 tiny 模型上堆 K3 特性（P7–P9）或做性能（P10）都缺少真实反馈。

**决定（用户）**：先实现 **Qwen/Qwen3-0.6B**（dense GQA，bf16 1.5 GB，本机 3090 能跑真权重）作为第三个模型 crate，用它验证
**与模型无关的主流程**：真权重端到端、采样、性能（CUDA graph / 融合 / pinned / 双缓冲）、**PD 分离（prefill-decode disaggregation）**。
kimik3 / glm53 只是"不同模型的实现"，当前进度记为阶段性成果，等有资源再做 P7–P9 与真权重验收。

**Qwen3-0.6B 结构事实与模块表**见 `design-qwen3.md`。


**新阶段编号用 Q 前缀**（与 P7–P10 并存，P 系列挂起）：

| 阶段 | 产出 | 验收 |
|---|---|---|
| ✅ Q1 Qwen3 crate + 真权重（2026-10-15） | `models/qwen3`：config、权重加载（tied lm_head）、GQA attention（新 kernel：`gqa_prefill / gqa_decode / gqa_prefill_paged`，K/V paged cache）、Qwen3Model、Executor、tokenizer（Qwen2Tokenizer = BPE，`tokenizers` 加载）、chat 模板（含 `enable_thinking`）、parser（`<think>` + JSON tool_call）、Service；`--features qwen3` 构建 | `scripts/verify/q1.sh`：真权重 prefill logits 与 HF max abs < 0.05（bf16，同机 3090）；greedy 32 token 与 HF 逐个一致（允许 bf16 平局步分叉，口径同 P3）；chat 模板 golden 逐字符；server e2e 用真模型回答一个问题且 `<think>` 段被正确拆出 |
| Q2 采样 + 请求功能 | temperature / top_p / top_k / min_p / repetition / seed（接 P2 的 `topk_topp_mask` + Philox `sample`）、stop 字符串、logprobs、`n>1`、prefix cache 开启、`/metrics`、按显存算 `num_blocks` | 采样分布与 PyTorch 同 seed 对拍；prefix cache 命中率；OpenAI SDK 全字段 |
| Q3 单卡性能 | CUDA graph（decode FULL 图）、pinned + 双缓冲 + 异步 D2H、算子融合（§8.2）、GQA flash-style kernel（或接 FlashAttention/FlashInfer）、异步调度 | §8.4 场景 ①②③ 与 vLLM 同机（3090）对比：TPOT P50 优于 ≥ 20% |
| Q4 PD 分离 | prefill 实例与 decode 实例分进程 / 分卡，KV 传输（本机先 CUDA IPC / NCCL，跨机 RDMA 留接口），proxy 路由，与 vLLM `KVConnector` 语义对照 | 两进程 PD 跑通，输出与单机一致；TTFT / TPOT 对比单机 |
| Q5 多卡（可选） | TP（GQA 按头切）、两卡 3090 或 4090 Pod | 输出一致 + 吞吐 |

PD 分离的详细设计写进 §10（待补），动代码前先定：KV 传输介质、块粒度、谁持有 block table、失败回退。

## 6. 关键决策

| 日期 | 决策 | 备选 | 理由 |
|---|---|---|---|
| 2026-10-13 | 以 vLLM main `de8291e`（0.31.0）为分析基线，放在 `/data/ref/vllm`，不进本仓库 | 固定某个 release tag | 用户要求最新版；两模型实现（`vllm/models/kimi_k3`, `deepseek_v32`）只在 main 上 |
| 2026-10-13 | 在需求的 4 个 crate 之外增加 `kernels`、`layers` | 全塞进 core | core 不依赖 CUDA 工具链，调度逻辑可 CPU 单测；算子两模型共用 |
| 2026-10-13 | 单二进制 + cargo feature 选模型 | 每模型一个 bin crate | 构建命令最简单，符合"core+server+模型"的组合要求 |
| 2026-10-13 | GLM-5.3 先做，Kimi-K3 后做 | 并行 | GLM 是 K3 的子集路径；先打通 core |
| 2026-10-13 | 每类算子先写 sm80+ 可跑的参考版，再接 FlashMLA/DeepGEMM 快路径 | 直接只做快路径 | 本机 3090（sm86）无 FP8/MXFP4 张量核、FlashMLA 需 sm90；否则本地无法验证正确性 |
| 2026-10-13 | 数值验收用 tiny 随机权重模型对拍 HF | 真权重 | 真权重 0.75–1.56 TB，本机不可能；正确性与规模无关 |
| 2026-10-13 | 第一版不做 PP、LoRA、P-D 分离、structured output | — | 需求未提；减小面 |

## 7. 约束与风险

- **硬件**：本机 RTX 3090 24 GB、CUDA 12.4、cargo 1.89。两模型真权重需多卡 H100/H200/B200 级别（GLM FP8 755 GB → ≥ 8×H200 141 GB；K3 1.56 TB → 16 卡以上或 EP 多机）。本机只能做 tiny 模型正确性与 kernel 对拍。
- **sm86 限制**：无 FP8 张量核（sm89+）、无 MXFP4（sm100）；FlashMLA 要 sm90。所以 P3–P9 的本地验证全走参考 kernel，性能阶段 P10 必须在目标机器上做。
- **4090（sm89）的增量**：只多 FP8 张量核，GLM 的 FP8 块 GEMM / FP8 KV cache / FP8 激活量化可本地真算（`fp8_utils.py:961` Triton 路径、`triton_mla.py:200`）。FlashMLA、DSA 稀疏 attention、索引器 top-k 快路径、原生 KDA decode、MXFP4 原生仍需 sm90/sm100；显存同为 24 GB。值得加，但不改变"参考 kernel 先行"的策略。
- **开发是否需要常驻 sm90：不需要**。P0–P9 的全部工作（core、server、模型结构、参考 kernel、CUDA graph、算子融合）在 sm8x 上可完成。只有两件事依赖 sm90：第三方快路径库（FlashMLA / DeepGEMM / FlashKDA）的集成验证，以及 P10 性能对比。做法：日常在 3090/4090；在 P3 和 P6 两个节点各租一次单卡 H100（数小时到一天）验证快路径集成；P10 在目标机器。规避"参考版与快路径布局不一致"风险的关键：参考 kernel 的 KV page 布局、fp8 scale 格式（ue8m0、1×128 组）、conv state 维度顺序，从一开始就按快路径库的要求定，接口签名对齐，后期只换实现不换数据结构。
- **chat 格式正确性**：每个模型的模板渲染都要和 HF `apply_chat_template(tokenize=False)` 逐字符一致（golden 由 `gen_*_golden.py` 生成，`chat_parity` 验证）；已对 GLM-5.3（15 case）、Qwen3（9 case）做到。
- **1M 上下文**：block table、索引 top-k 缓冲、KDA 状态都按 `max_model_len` 预留；默认先按 128K 开发，参数可调。

## 8. vLLM 性能瓶颈分析与本次重构的规避策略

> 目标之一是性能超越 vLLM。本节列出 vLLM 当前架构中的开销来源，标注"源码核实"（有文件:行号）与"经验判断"（社区 profiling 共识，需在 P10 实测确认），并给出 Rust 侧对策。

### 8.1 开销清单

| # | 瓶颈 | 性质 | 证据 | 影响场景 | Rust 对策 |
|---|---|---|---|---|---|
| B1 | **Python 解释器在每 step 热路径上** | 结构性 | `core.py:657 step()`、`scheduler.schedule()`、`model_runner.gather_batch_req_state:1272`、`prepare_inputs:1354` 全是 Python/numpy；V2 runner 用 numpy 向量化 + Triton kernel 搬运（`prepare_pos_seq_lens`、`combine_sampled_and_draft_tokens`）来压缩 Python 行数，说明这里是已知瓶颈 | decode 阶段，batch 越大、每 token GPU 时间越短，CPU 占比越高；对 K3/GLM 这类 decode 单 token 耗时 ~20-40 ms 的大模型，单机影响小，但 MTP 投机 + 高并发时 CPU 侧每 step 要处理 ~1k 请求 × 多 token | 调度、batch 组装、block table 更新全部是 Rust 原生结构（`Vec<u32>`、arena、位图），目标每 step CPU 开销 < 200 µs @ 1024 并发 |
| B2 | **前端 ↔ EngineCore 跨进程**：msgpack 序列化 + ZMQ | 结构性 | `core.py:1217-1239` 两个 IO 线程；`core_client.py:713` `MsgpackEncoder/Decoder`；`SchedulerOutput`/`EngineCoreOutputs` 每 step 序列化一次 | 每 step 两次序列化 + 两次 IPC；流式输出每 token 一次 | 单进程：HTTP 层与引擎在同一进程，用 `crossbeam` 无锁队列传 `Arc<Request>`；引擎线程与 tokio runtime 分离即可，不需要进程边界。保留"多进程"为 DP 场景的可选项 |
| B3 | **GC 停顿** | 结构性 | `core.py:253 freeze_gc_heap()`、`model_runner.py:65 freeze_gc_for_cudagraph_capture`：vLLM 必须手动冻结堆来压 GC 尾延迟 | P99 TPOT 抖动 | Rust 无 GC；注意避免热路径上的堆分配（请求对象池、预分配 batch 缓冲） |
| B4 | **GIL 下的多线程**：IO 线程、输出处理、KV connector 线程争 GIL | 结构性 | `core.py:1616` 显式 `time.sleep(0.001)` 让出 GIL；`async_llm.py` 输出 handler 是 asyncio 协程 | 高并发流式输出时 detokenize 与 step 争 CPU | tokio 多线程 runtime 处理 HTTP/SSE；detokenize 在独立线程池；引擎 step 线程独占一核 |
| B5 | **每 step 的 D2H 同步**：采样结果必须回 CPU 才能调度下一步 | 结构性 | V2 用 `AsyncOutput`（`async_utils.py:115`）在 copy stream 上异步拷贝 + `async_scheduling`（`config/vllm.py:1585` 默认开）用 placeholder 提前调度下一步，才把这个同步从关键路径挪开 | 无 async scheduling 时每 step 多一次 GPU 空转 | 从第一天就按"调度领先执行一步"设计：scheduler 用 `num_output_placeholders` 乐观推进，采样结果到达后再修正；输出拷贝走独立 stream + event，不 `synchronize()` |
| B6 | **CPU 侧 block table / slot mapping 构建** | 结构性 | V1 runner 在 numpy 里算 slot_mapping 再 H2D；V2 改为 GPU kernel（`block_tables.compute_slot_mappings`、`model_runner.py:1568`），并用 UVA 零拷贝缓冲（`buffer_utils.py:29 UvaBuffer`、`states.py:37 StagedWriteTensor`） | 长上下文（1M）block table 行很长，CPU 算 slot 是 O(tokens) | 同 V2：block table 常驻 GPU，CPU 只传"增量 block id"；slot_mapping、positions、seq_lens 由一个 kernel 从 `(idx_mapping, query_start_loc, num_computed)` 算出；请求级元数据用 pinned + UVA |
| B7 | **CUDA graph 覆盖不全** | 结构性 | 两模型走 breakable 模式（§1.4b）：attention、KV 写入、EP all-to-all 处必须断图 eager 执行；每层一次断点 = 93 层 × 若干次 launch 回到 CPU | decode 小 batch 时 launch 开销占比高（K3 每层 KDA+MoE+AttnRes ≈ 15-25 个 kernel，93 层 ≈ 2k launch/step） | 1) decode FULL 图：attention metadata 全部做成"固定形状 + 设备侧长度"，让 MLA decode / KDA recurrent / 索引 top-k 都可被捕获，不断图；2) 层内算子融合（见 9.2）把 launch 数压到每层 ≤ 6；3) prefill 不追求图，靠大 batch 摊销 |
| B8 | **MoE 路由与 dispatch 的多 kernel 链** | 算子级 | `fused_moe` 路径：gate GEMM → sigmoid/bias/topk → 排序/对齐 → grouped GEMM×2 → 加权归约；K3 latent MoE 再加 down_proj/norm/up_proj；EP 下还有 all-to-all（`all2all_utils.py`） | 每 MoE 层 8-12 个 kernel；K3 896 专家 top-16 时 token 排序成本高 | 路由 + 排序 + 对齐融合为一个 kernel（输出 permuted 索引与 expert offsets）；grouped GEMM 的 epilogue 直接做加权累加，去掉独立归约；K3 的 down_proj 与 gate 并行（vLLM 已用双 stream，`model.py:760`），我们直接融合成一个 GEMM（gate 与 down_proj 共享输入，拼成一个 N=896+3584 的 GEMM） |
| B9 | **采样器**：top-k/top-p 对 16 万词表做 sort | 算子级 | `topk_topp_sampler.apply_top_k_top_p` 默认 torch sort；FlashInfer 采样可选 | 每 step 对 [batch, 163840] 排序；batch 1024 时 ~0.5-1 ms | 用 rejection-based top-p（FlashInfer 思路）或分块 radix select，不做全排序；greedy 路径只做 argmax；logprobs 只 gather 需要的 token |
| B10 | **prefix cache 哈希默认 sha256 + pickle** | 配置级 | `cache.py:64` 可选 `xxhash`，默认 `sha256`（文档注明 xxhash 更快但需手动开） | 长 prompt 到达时每 16 token 一次 sha256，1M 上下文 = 6.5 万次 | 默认 xxh3-128，对 token 序列直接按字节哈希，无序列化；多租户安全靠 `cache_salt` 混入 |
| B11 | **tokenizer / chat 模板在 Python 层** | 结构性 | K3 的 `encoding_k3.py`（750 行 Python 字符串拼接）与 renderer 在 asyncio 事件循环里跑；GLM 的 jinja 模板每请求渲染一次（`chat_template.jinja` 256 行，含多重 for 循环） | 长多轮对话 + 工具列表时模板渲染可到数 ms，阻塞事件循环 | XTML 渲染用 Rust 直接产 token（`EncodeSegment` 直接编码，省掉字符串再分词）；jinja 用 `minijinja` 预编译；渲染在 tokio blocking 线程池，不占 HTTP 线程 |
| B12 | **增量 detokenize 每 token 一次 Python 调用** | 结构性 | `detokenizer.py:120` 逐 token `decode_next` + 字符串拼接 + stop 串检查 | 高并发流式 | `tokenizers::DecodeStream` 原生调用；stop 串用 Aho-Corasick 增量匹配；输出文本按 chunk 而非 Python str 拼接 |
| B13 | **多模态预处理在 CPU（PIL）且在前端进程** | 结构性 | `renderers/base.py:894 _process_multimodal` 走 HF processor | K3 图像请求 TTFT | 图像解码用 `image` crate + SIMD resize；或直接上 GPU（nvjpeg + resize kernel）；encoder cache 按 mm hash 复用（同 vLLM） |
| B14 | **structured output 的 bitmask 在 CPU 算再 H2D** | 结构性 | `core.py:669 get_grammar_bitmask` 在 step 内；V2 用 batch_queue 把它和 forward 重叠（`:737-749`） | 开 JSON schema 时 | 不在第一版范围；若做，bitmask 计算放独立线程，与 forward 重叠，H2D 走 copy stream |
| B15 | **内存 profiling 偏保守** | 配置级 | `gpu_memory_utilization` 默认 0.92；profile_run 后再扣 cudagraph 估算（`gpu_worker.py:740`） | KV 容量 = 并发上限 | 精确记账：权重、激活峰值（按 max_num_batched_tokens 解析式算，不靠 dummy run）、图内存（捕获后实测）；默认 utilization 0.95 |
| B16 | **投机解码的 draft 与 verify 分开 launch，且 draft 结果回 CPU** | 结构性 | `core.py:693 take_draft_token_ids`（非 async 时）；async 时在 worker 内 | MTP 每 step 两次 forward | draft 与 target 共享 CUDA graph（target forward → sampler → MTP forward → draft sample 全部在一张 FULL 图内）；draft token 留在 GPU，CPU 只拿接受数 |

### 8.2 算子融合清单（decode 路径，每层目标 launch 数）

**GLM-5.3 MoE 层**（现状约 14 launch）→ 目标 5：
1. `fused_qkv_a_proj` + `wk_weights_proj` 合并为一个 GEMM（vLLM 已部分做：`fused_norm_rope`）
2. norm + rope + 索引 K 量化写 cache + MLA KV 写 cache：一个 kernel（vLLM 的 `fused_norm_rope` 已是此形态，直接对标）
3. `q_b_proj` + W_UK 吸收 + 索引 q + top-k：GEMM 后一个 epilogue kernel
4. 稀疏 MLA decode（FlashMLA sparse 或自研）
5. `o_proj` + residual add + post norm + router gate：GEMM epilogue 做 add+norm，gate 并入下一 GEMM
6. MoE：排序 kernel → grouped GEMM（含 silu）→ grouped GEMM（epilogue 加权累加 + shared 累加）→ 与下一层 input norm 融合

**Kimi-K3 KDA 层**（现状约 18 launch）→ 目标 6：
1. AttnRes 前缀和 + norm + `in_proj_qkvgfab` GEMM（AttnRes 作 GEMM prologue 不现实，保留为 1 个 kernel）
2. conv1d update + f_b/beta/gate 计算 + L2 norm：一个 kernel（vLLM `fused_kda_decode` 已融合 conv+recurrent，对标）
3. KDA recurrent decode（含 o_norm 门控）
4. `o_proj` + AttnRes 写回 + post norm
5. latent MoE：`down_proj ∥ gate` 合并 GEMM → 排序 → grouped GEMM ×2（mxfp4）→ norm + `up_proj` + shared 累加（GEMM beta-add，vLLM 已做 `routed_output_transform:331`）

**跨层**：相邻层的"residual add + RMSNorm"全部融进前一个 GEMM 的 epilogue（vLLM 的 `fused_allreduce_rms_norm` 在 TP 下做了同类事）。

### 8.3 与 vLLM 的关系：哪些直接对标，哪些另起

- **直接对标**（vLLM V2 已是最优形态，照搬设计）：GPU 常驻 block table + kernel 算 slot mapping；UVA/pinned 请求元数据；async scheduling 的 placeholder 机制；AsyncOutput copy stream；`fused_norm_rope` / `fused_q` 这类模型内融合算子；MegaMoE 的 symm buffer EP。
- **另起**（Python 架构决定的，Rust 不必背）：进程边界（B2）、GC（B3）、GIL（B4）、解释器开销（B1）、Python 层 tokenizer/模板/detokenize（B11、B12）。
- **vLLM 也没解决、我们要试的**：breakable cudagraph 的断点（B7）；MoE 路由链融合（B8）；K3 gate+down_proj 合并 GEMM；draft+target 同图（B16）。

### 8.4 性能验收与基准方法

- **基线**：同机、同权重、同 `max_num_seqs / max_num_batched_tokens / 量化 / 投机设置` 下跑 vLLM（固定到 `de8291e`），用同一套 benchmark 脚本（`scripts/bench/`，Rust 写，OpenAI 兼容接口打流量）。
- **指标**：TTFT P50/P99、TPOT P50/P99、单卡 tokens/s（decode）、prefill tokens/s、每 step CPU 时间（引擎内部计时导出到 `/metrics`）、KV 利用率。
- **场景**：① 128 并发短对话（512 in / 256 out）；② 32 并发长上下文（64K in / 1K out）；③ 单请求 1M 上下文 TTFT；④ MTP 开 / 关；⑤ K3 图文混合。
- **阶段目标**：P3 起每阶段记录 tiny 模型下的"每 step CPU 开销"与"每层 launch 数"（nsys 统计），作为 B1/B7 的早期信号；P10 做真模型对比，目标 decode TPOT 优于 vLLM ≥ 20%，TTFT 不劣。
- **归因方法**：nsys 时间线看 GPU 空洞；`tracing` span 看 CPU 各阶段；二者对齐找关键路径。

## 9. 待定问题

1. 目标生产机器是什么（卡型、卡数、单机/多机）？决定 TP/EP 设计与快路径 kernel 的优先级。
2. KDA kernel：手写 CUDA 还是 Triton AOT？
3. structured output（JSON schema / grammar）是否在范围内？
4. 是否需要兼容 vLLM 的 `/metrics` 指标名，便于复用现有看板？
5. 性能对比的基线配置（§8.4）：vLLM 侧是否开 MTP、EP、FP8 KV cache？需要和目标机器一起定，否则对比口径不一致。
6. 是否接受"decode FULL 图把 attention 也捕获"带来的约束：所有 attention metadata 必须固定形状，`max_model_len` 变化需重新捕获。

## 10. 扩展点设计（用户 2026-10-17 提出，待细化）

### 10.1 对接新算子（kernel 插拔）

现状：`crates/kernels` 的 `.cu` 由 build.rs 编 nvcc 静态链接，Rust 侧 `ffi.rs` 声明 + `ops.rs` 安全封装，模型 crate 直接调 `ops::*`。换 kernel = 改 `.cu` + 重编。
方案（按侵入度从小到大，先做 A）：
- **A. 算子 trait + 注册表**：为每类热点算子定义 trait（`AttentionPrefill`、`AttentionDecode`、`Gemm`、`Norm`、`Sampler`），`ops.rs` 现有实现是默认 impl；模型 crate 通过 `OpSet` 结构取 `Box<dyn Trait>`，启动时按配置 / 环境变量（如 `ATTN_IMPL=fa_v5|ref|ext:<so>`）选择。零运行时开销的做法：泛型参数 `M: Qwen3Model<Ops: OpSet>`，默认 `DefaultOps`。
- **B. 外部 .so 动态加载**：`libloading` 按约定的 C ABI（现有 extern "C" 签名就是 ABI，已约定不带项目前缀）加载第三方 kernel 库（FlashInfer / 自研 CUTLASS / Triton AOT 产物），满足 trait 即可。需要一张"ABI 版本 + 形状约束（D=128、block_size 16 倍数）"的能力表供启动期校验。
- **C. PTX/cubin 热加载**：cudarc `load_ptx` 运行时装 kernel，用于 Triton AOT。与 B 的区别是不需要 host 侧 C 封装，但要自己处理 launch 参数布局。
对拍：每个 trait 实现都要能跑 `kernels.sh` 的 ref 测试（测试按 trait 写，不按具体函数写）。

### 10.2 KV cache 对接第三方（分层 / 外置）

现状：`core::kv` 的 `PagedKv` 是 GPU 显存里 `[num_blocks, block_size, Hkv, 2, D]` bf16 单层；prefix cache 块哈希在 scheduler；无 CPU / 远端层。
目标接口（对齐 vLLM `KVConnectorBase_V1` 的事件点，但不照搬其 Python 结构）：
- `trait KvStore`：`alloc / free / block_table`（GPU 层，现有）；`trait KvTier`：`offload(block_ids) -> Future`、`fetch(hashes) -> Future<block_ids>`、`lookup(hashes) -> hit_len`。
- 调度器钩子（三处）：① `schedule()` 新请求入队前 `lookup` 外部命中长度，命中的块异步 `fetch` 到空闲 GPU 块，请求先标 `waiting_for_kv`；② `update_from_output` 后对本 step 完整写满的块触发 `offload`（可按策略：全部 / 只在 GPU 块压力高时）；③ `free` 时若块已 offload 则只释放 GPU 侧。
- 传输层：先做 **CPU pinned 内存层**（`cudaMemcpyAsync` 走独立 copy stream，与计算流用事件同步；块粒度 16 token × 8 head × 128 × 2 × 2 B = 64 KB，PCIe 4 每秒 ~40 万块），这一层本身就能把 prefix 命中率从显存上限扩到主存上限（4090 24 GB → 主机 256 GB）。再做远端：Mooncake / LMCache / NIXL 任一，接口是 `(hash, bytes)` KV 对象存储 + RDMA get/put；dinfer 侧只需实现 `KvTier` 的异步 fetch/offload，块布局与 bf16/fp8 dtype 写进对象元数据。
- PD 分离（Q4）复用同一 `KvTier`：prefill 节点 `offload` → decode 节点 `fetch`，加一个"请求迁移"消息（prompt ids + 块哈希列表 + 采样参数）。
- 风险：块哈希要与 dtype / 模型版本 / block_size 一起哈希；fetch 在飞时该块不能被 GPU 侧回收；fp8 KV 时 scale 随块存。

## 11. 采用 / 改进台账（对 vLLM）

目标是性能超越，不是重写。vLLM 做得对的直接用，能改进的改进；每项记一行，便于回看哪些是"借来的"、哪些是我们的。

| 项 | 采用 vLLM 的 | 我们改进的 | 收益 / 状态 |
|---|---|---|---|
| 调度模型 | v1 的 "无 prefill/decode 之分，只有 num_computed 追赶 num_tokens"、chunked prefill、FCFS + 抢占末尾 | `decode_exempt_from_budget`（decode 行不占 token 预算，每步装下全部 decode + 预算内 prefill） | 调度结构零成本沿用；免预算让混批可能，但闭环 bench 下看不出（q3-xstream） |
| KV 分页 + prefix cache | 块哈希、按块命中、`block_size=16` | 无 | 一致 |
| CUDA graph | decode-only 按 batch 桶捕获（FULL_DECODE_ONLY） | 按桶**懒捕获**（启动 -210 ms；`GRAPH_CAPTURE=eager` 恢复） | 启动 2.9 → 0.6 s 的一部分 |
| 异步调度 | placeholder 机制、stop 检查晚一步、下一步在上一步结果回来前调度 | 下一步的 input_ids 在**设备上**从上一步 `chosen` gather（图内 `row_gather_i32`，不经 host 往返）；D2H 用 pinned 双槽 + 事件；engine 单线程双步流水（pending/sync_done 两槽），不需要 vLLM 的多进程 | bench24：c32 1.05 → 1.08×、c8 1.06 → 1.08×；代价 TTFT +1 步（c8-long 39 → 54 ms） |
| attention | FA2 的算法结构（online softmax、P 片直接当 PV 的 A 片） | 自写 mma.sync 版 v4/v5（Ampere/Ada 无 FA3），paged/prefill 同一模板，混批 decode 行分流到 split-K | 4090 T=1514 0.14 ms/层，TTFT 领先 vLLM 2× |
| decode attention | split-K | v2：8 lane/key、两 q head 共享 K 读、splits 按 batch 自适应 | 与 FA2 decode 持平 |
| 采样 | top-k/top-p/min-p/penalties 语义、Philox | kernel 待融合（4 → 1） | 待做 |
| 权重加载 | safetensors mmap | bf16 直传不经 Vec、设备拼接 qkv/gate_up、并行填页、tokenizer 独立线程 | 2.9 → 0.6 s |
| bench 口径 | `benchmarks/serve.py` 的 TTFT/TPOT/ITL 定义、ignore_eos | 每轮换 seed（避免常驻侧 prefix cache 全命中）；开环泊松 + 长短混合 | 发现并修正了 bench11 之前的口径偏差 |
| 进程结构 | — | 单进程（API + engine 同进程不同线程），无 ZMQ / 多进程 IPC | TTFT 低一半的主要来源之一 |
| 跨流 prefill/decode 并行 | 无（vLLM 单流） | 实现并证实真并行（nsys 39% 重叠） | 闭环 bench 下混批步 <5%，TPOT 无感；留给 prefill 比例高的负载 |

