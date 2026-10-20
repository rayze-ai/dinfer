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

## 10. 扩展点设计：算子插拔与 KV cache 分层（2026-10-17 细化）

两个扩展点都遵守同一条原则：**默认路径零开销**（泛型单态化或一次性查表，不在热路径上做动态分派），扩展实现与默认实现跑**同一套 ref 测试**。

### 10.1 算子插拔

#### 现状（调用面盘点）
模型 crate 与 `layers` 直接调 `dinfer_kernels::ops::*` 共 26 个入口 + `Gemm::forward_bf16`。按类归并为 7 组：

| 组 | 现有入口（`ops.rs`） | 热路径 |
|---|---|---|
| GEMM | `Gemm::forward_bf16`（cuBLASLt） | 每层 4 次 |
| Attention | `gqa_prefill_bf16` / `gqa_paged_bf16` / `gqa_decode_bf16` / `gqa_mixed_bf16`、`gqa_decode_scratch_len` | 每层 1 次 |
| QKV 后处理 | `qkv_post_bf16`（norm + rope + cache write）、`rope_bf16`（独立版，GLM 用） | 每层 1 次 |
| Norm / 残差 | `rmsnorm_bf16(_n)` / `fused_add_rmsnorm_bf16(_n)` / `gather_rmsnorm_bf16` / `add_inplace_bf16_n` | 每层 2 次 + 尾部 1 |
| 激活 | `silu_and_mul_bf16_n` | 每层 1 次 |
| 嵌入 / 杂项 | `embedding_bf16(_n)`、`bf16_to_f32_n`、`row_gather_i32` | 每步 1 次 |
| 采样 | `sample_fused_f32`（图路径）、`penalties_f32` / `temperature_f32` / `min_p_mask_f32` / `topk_topp_mask_f32` / `sample_f32` / `argmax_f32_n` / `logprobs_f32`（eager 路径） | 每步 1–6 次 |

MLA / MoE / DSA（GLM、K3 线）另成一组，接口形态相同，本节以 Qwen3 的 7 组为准。

#### 方案 A（先做）：`OpSet` trait + 编译期单态化 + 启动期选择
```rust
// crates/kernels/src/opset.rs
pub trait Attention {
    fn prefill(&self, s: &Stream, q: &CudaSlice<bf16>, k: .., v: .., out: .., cu_seqlens: .., num_reqs, max_seqlen, h, hk, d, scale, q_stride, kv_stride) -> Result<()>;
    fn paged(&self, ...) -> Result<()>;
    fn decode(&self, ...) -> Result<()>;
    fn mixed(&self, ...) -> Result<()>;          // 默认实现 = decode(rows==1) + paged(skip_single)
    fn decode_scratch_len(&self, b, h, d) -> usize;
    fn supports(&self, head_dim: usize, group: usize, block_size: usize) -> bool;   // 启动期校验
}
pub trait Gemm { fn forward_bf16(&self, a, w, bias, out, m, n, k) -> Result<()>; /* 后续：forward_bf16_f32out、epilogue(silu_mul) */ }
pub trait Norm { fn rmsnorm(..); fn fused_add_rmsnorm(..); fn gather_rmsnorm(..); }
pub trait QkvPost { fn apply(..); }
pub trait Act { fn silu_and_mul(..); }
pub trait Sampler { fn fused(..); fn penalties(..); /* eager 分步同现有签名 */ }
pub trait OpSet { type Attn: Attention; type Gemm: Gemm; type Norm: Norm; type QkvPost: QkvPost; type Act: Act; type Sampler: Sampler;
                  fn attn(&self) -> &Self::Attn; /* ... */ }
pub struct DefaultOps { .. }   // 现有 ops.rs 函数的薄包装；impl OpSet
```
- 模型 crate 改为 `Qwen3Model<O: OpSet = DefaultOps>`，`forward` 里 `k::xxx(...)` → `self.ops.attn().decode(...)`。泛型单态化后与现在的直调零差别（可用 `cargo asm` 核对一次）。
- **启动期选择**：`--ops <name>` / 环境变量 `OPS=default|flashinfer|...`，在 `service.rs` 里 `match` 到具体类型构造 `Qwen3Executor<O>`（枚举分派只在启动期发生一次）。
- **能力校验**：`OpSet::check(cfg) -> Result<()>` 在加载后调一次：head_dim / group / block_size / dtype 不支持直接报错退出，不进运行期。
- **对拍**：`kernels/tests/ref_tests.rs` 改成 `fn attention_refs<A: Attention>(a: &A)` 这样的泛型函数，`DefaultOps` 与任何新实现都调同一组；`kernels.sh` 加 `OPS=...` 循环。

落地步骤（约 1 天）：① 加 `opset.rs` 与 `DefaultOps`（纯搬运，不改 kernel）；② `layers::nn` 与 `qwen3::model` 换成泛型；③ ref 测试泛型化；④ bench 对比 `DefaultOps` 与直调（预期 0 差异）。GLM/K3 线暂不动（它们直调的 MLA/MoE 入口另起 trait）。

#### 方案 B：外部 `.so`（第三方 kernel 库）
- 约定 C ABI = 现有 `kernels.h` 的 extern "C" 签名（已约定不带项目前缀），**加一个 `abi_version()` 与 `capabilities()`**（返回支持的 head_dim 列表、block_size 约束、dtype）。
- `ExternOps` 实现 `OpSet`：`libloading::Library::new(path)`，按符号名取函数指针，缺的符号回落到 `DefaultOps`（逐组回落，不是整库）。
- 典型用途：FlashInfer 的 decode attention（它的 paged layout 与我们一致：`[num_blocks, block_size, Hkv, 2, D]` 需确认 K/V 交错方式，不一致则 `.so` 侧做一次 layout 适配）；自研 CUTLASS GEMM（epilogue 融合）。
- 风险：`.so` 内部用的 CUDA runtime / 流语义必须接受我们传入的 `cudaStream_t`；不能有自己的全局流或隐式同步。校验：在 `check()` 里跑一次 capture 探针（begin_capture → 调一次 → end_capture），不能 capture 的实现只允许用在 eager 路径（标记 `graph_safe=false`，执行器据此关掉该算子所在步的图）。

#### 方案 C：PTX / cubin 热加载（Triton AOT）
- cudarc `CudaContext::load_ptx` + `get_func`，`TritonOps` 按 kernel 名 + 签名表 launch。参数布局由 Triton AOT 的 C 头生成；grid 计算在 Rust 侧按 Triton 的 `num_warps/num_stages` 元数据。
- 只在 B 不可行（没有 host 侧 C 封装）时用；优先级最低。

#### 不做的
- 运行期按形状动态选 kernel（autotune）：先不做；需要时在 `OpSet::check` 阶段对本机形状跑一次微基准选实现，结果缓存到磁盘。

### 10.2 KV cache 分层 / 外置

#### 现状（钩子点盘点）
- GPU 层：`core::kv::KvCacheManager`（`block_pool.rs` LRU 空闲链 + `(group, hash) → block` 缓存表；`hash.rs` 链式 xxh3-128，salt 只混第一块）。
- scheduler 调用点：`get_computed_blocks(req)`（新请求首次调度，查 prefix 命中）、`allocate_slots`（分新块，失败则抢占末尾）、`cache_blocks`（update_from_output 后，对新凑满的块登记哈希）、`free`（完成 / 抢占）。
- 设备层：`qwen3::model::PagedKv { cache: Tensor<bf16> [num_blocks, block_size, Hkv, 2, D] }` 每层一块。

#### 接口
```rust
// crates/core/src/kv/tier.rs
pub struct BlockKey { pub model_id: u64, pub layer_group: u8, pub hash: BlockHash, pub dtype: KvDtype, pub block_size: u16 }
pub trait KvTier: Send {
    /// 这些块在外层有吗？返回命中的前缀长度（块数）——只认连续前缀。
    fn lookup(&self, keys: &[BlockKey]) -> usize;
    /// 把 GPU 块 `src` 的内容拉到外层（异步）；完成后 `poll_done` 可见。调用方保证 src 在完成前不被回收。
    fn offload(&mut self, keys: &[BlockKey], src: &[DeviceBlockRef]) -> Result<TransferId>;
    /// 把外层内容装进已分配好的 GPU 块 `dst`（异步）。
    fn fetch(&mut self, keys: &[BlockKey], dst: &[DeviceBlockRef]) -> Result<TransferId>;
    fn poll_done(&mut self) -> Vec<TransferId>;
    fn stats(&self) -> TierStats;   // 命中 / 字节 / 在飞
}
pub struct DeviceBlockRef { pub layer: u8, pub block: BlockId, pub ptr: u64, pub bytes: usize }
```
`KvCacheManager` 持 `Option<Box<dyn KvTier>>`（这里允许 dyn：每 step 最多调几次，不在 kernel 热路径）。

#### 分层边界：core 放容器，模型放"命中单元"（2026-10-18 用户指正后补）
- **core**：`HostTierIndex`（(group, hash) → 槽，LRU）、`kv_fetch/kv_offload` order 类型、调度钩子、`HostArena`/copy stream/pack 搬运。这些与模型无关。
- **模型 crate 注入 `HostTierPolicy`**：① 什么算一个单元、多组如何合成前缀命中（attention 组取最短前缀；Mamba/KDA 状态块只在块边界有效，与 GPU 侧 `get_computed_blocks` 同一规则）；② 一个单元覆盖多少 token；③ 拉回时填哪些 GPU 块、登记哪些哈希。单元的字节布局 / 竞技场切槽留在模型 executor（`enable_host_tier` 已在那里算 `per_block_all`），trait 不重复暴露。Qwen3 = `FullAttentionPolicy`（一块 × 全层）；GLM（MLA 潜变量 + 索引器 K 两组）、K3（KDA 状态 + 全注意力）各自实现，未写。
- 判据：core 里出现 `groups.len() != 1` / `is_mamba[0]` / 具体 block_size 比较，就是策略泄漏，该抽到 trait。

#### 调度器接入（4 处，都在 `sched/mod.rs`）
1. **首次调度查命中**：`get_computed_blocks` 之后，若 GPU 命中长度 `h_gpu < num_tokens-1` 且有 tier，`tier.lookup(keys[h_gpu/bs..])` 得外层命中 `h_ext`。若 `h_ext > 0`：按 `h_gpu + h_ext` 分配块（`allocate_slots` 的 `new_computed` 参数扩展成 "GPU 命中块 + 待填块"），对待填块发 `fetch`，请求状态置 `WaitingForKv(transfer_id)`、**不进本步**。
2. **到达检查**：每次 `schedule()` 开头 `poll_done()`，把对应请求从 `WaitingForKv` 放回 waiting 队头（它们的 `num_computed_tokens` 已含外层命中，下一步只算剩余 prompt）。
3. **卸载**：`cache_blocks` 登记哈希的同时，若策略允许（默认：全部新凑满的块；可选：只在 `pool.num_free < 低水位` 时），对这些块发 `offload`。GPU 侧照常保留；块被 LRU 淘汰时若 offload 还在飞 → 该块暂不淘汰（`BlockPool::free` 跳过 `in_flight` 标记的块）。
4. **释放**：`free` 不变；外层由自己的容量策略淘汰。

#### 设备层
- `fetch/offload` 的拷贝走**独立 copy stream**；与计算流的依赖用事件：fetch 完成事件 → 计算流在该请求首次 prefill 前 `wait_event`；offload 读块前等计算流上 "该块写完" 的事件（= 该 step 结束事件，已有 `main_done`）。
- 每块 bytes = `block_size × Hkv × 2 × D × 2 B`（Qwen3-0.6B：16×8×2×128×2 = 64 KB/层，28 层 1.75 MB/块-全层）。一个请求 1024 token = 64 块 = 112 MB；PCIe 4 ×16 实测 ~11 GiB/s（pageable）/ ~24（pinned）→ 5–10 ms，与一次 1024 prefill（~20 ms）同量级，**fetch 比重算快 2–4×**，值。
- 块内 layout 与设备一致（直接 memcpy，不转置）；fp8 KV（将来）每块附 scale 向量，`BlockKey.dtype` 区分。

#### 分层实现顺序
1. **`PinnedHostTier`**（先做，1–2 天）：一块大 pinned host 内存（容量 `--kv-host-gib`，如 64 GiB），`HashMap<BlockKey, HostSlot>` + LRU；`offload/fetch` = `cuMemcpyDtoHAsync / HtoDAsync` 在 copy stream 上 + 事件。单机把 prefix 命中容量从显存 17 GiB 扩到主存。验收：`serve_e2e` 加 "超过 GPU 容量的多轮对话前缀复用" 用例；bench 加 "prompt 前缀重复率 50% 的开环负载" 看 TTFT。
2. **`RemoteTier`**：对象存储语义 `(BlockKey → bytes)`，Mooncake / LMCache / NIXL 任一后端；先经 host pinned 中转（复用 1 的路径），再做 GPUDirect RDMA。
3. **PD 分离（Q4）**：prefill 节点算完 → `offload` 到 RemoteTier → 发 "迁移消息"（prompt ids、块 keys、采样参数、已采样的首 token）到 decode 节点 → decode 节点 `fetch` 后直接进 running。调度器只多一种请求来源，`KvTier` 接口不变。

#### 风险与对策
- `BlockKey` 必须含 model_id（权重版本哈希）、dtype、block_size、layer_group：不同模型 / 量化的块绝不能互相命中。
- fetch 在飞时该请求可能被 abort：`free` 时若有在飞 transfer，等它完成再放块（否则 DMA 写入已被复用的块）。
- 外层命中但 GPU 块不够：按现有抢占逻辑处理，fetch 推迟。
- 与异步调度的交互：`WaitingForKv` 的请求不会进任何步，插队 prefill 也看不到它，无冲突。

### 10.3 PD 分离（Q4，2026-10-18 定稿，动代码前的四个决策）

**目标**：prefill 实例 P 与 decode 实例 D 分进程 / 分卡；proxy 一个入口；输出与单机逐字一致；TTFT 不劣于单机、D 的 TPOT 不受 prefill 干扰。Qwen3-0.6B 只验流程，参数与协议按生产模型（GLM / K3，多卡、跨机）可演进来定。

**对照 vLLM**（`kv_connector/v1/nixl/{push,pull}_*`、`examples/disaggregated/disaggregated_serving/disagg_proxy_pushconnector_demo.py`）：vLLM 有 pull（D 用 `remote_block_ids` 经 NIXL READ 拉）和 push（P 算完直接写进 D 事先分好的块）两种；proxy 先打 P（`max_tokens=1` + `kv_transfer_params`），拿回 `remote_*` 元数据再打 D。我们**采用**这个"P 先出首 token、D 接力"的请求流和 `kv_transfer_params` 的信息集合；**改进**：不引入 NIXL / side channel，v1 用自己的流式协议，块搬运复用 host tier 的 pack / pinned 路径。

| 决策 | 选择（v1） | 为什么 | 留的接口 |
|---|---|---|---|
| ① 传输介质 | **P 侧 `kv_pack` → pinned HostArena 槽 → TCP 流 → D 侧 pinned 槽 → `kv_unpack`**。P、D 各起一个 `kv-port` 监听 | 复用 Q3-r 已验证的 pack / arena / copy stream 全套；跨机天然可用；4090 测试机 GPU 0/3 两进程即可验 | `trait KvTransport { send(slots, to) / recv(into slots) }`，v2 同机换 CUDA IPC（`cuIpcGetMemHandle` 直接 D2D），v3 跨机 RDMA（UCX / NIXL 后端） |
| ② 块粒度 | **一个单元 = 一块（16 token）× 全部层**，哈希 = 现有 `BlockHash`（含 model_id / dtype / block_size） | 与 HostTierPolicy 单元一致，D 侧登记直接走 `register_fetched`；逐层流式（vLLM `save_kv_layer`）对 28 层 0.6B 没意义，大模型 TP 下再按层组拆 | 迁移消息里 `units: Vec<(group, hash)>`，层组拆分只改 policy |
| ③ 谁持 block table | **各自持有，不共享**。P 算完即 offload 到自己的 arena 槽并释放 GPU 块；D 收到消息后自己 `allocate_slots` + `register_fetched`（与 host tier 命中同一条路） | scheduler 不需要知道对端；P 的 GPU 块周转快（prefill 实例 KV 常驻没意义） | push 模式（P 直写 D 的块）需要 D 先分块再告知 P，留给 v2 CUDA IPC |
| ④ 失败回退 | **D 等 KV 超时（默认 2 s）或校验失败 → D 本地重算 prefill**（请求已带完整 prompt ids，只丢 TTFT）；P 宕 → proxy 直接把请求发 D 单机跑；D 宕 → proxy 返回 503，不重试到别的 D（KV 已丢） | 任何一方挂都退化成单机，不丢请求、不产生错误输出 | 指标 `pd_fallback_total{reason}`；多 D 时 proxy 按 `running` 最少路由 |

**请求流（v1）**
1. proxy 收 `/v1/chat/completions` → 转发 P，附 `max_tokens=1`、`kv_transfer: {decode_addr}`；P 正常 prefill + 采样首 token，**不释放**该请求 KV，`kv_pack` → arena 槽，开 TCP 推到 D 的 `kv-port`（消息头：request_id、units、首 token、采样参数、prompt ids；体：每单元 `unit_bytes`）。
2. P 向 proxy 返回首 token（流式第一 chunk）+ `request_id`；proxy 把原请求 + 首 token + request_id 发 D。
3. D 的 `kv-port` 线程把块收进 arena 槽，登记 `IncomingKv { request_id → slots }`；D 的 scheduler 对带 `request_id` 的新请求：若 KV 已到 → 当作 host 命中走 `register_fetched` + `kv_fetch`，`num_computed = prompt_len`，直接 decode；未到 → `WaitingForKv`，到达后放回 waiting 队头；超时 → 普通 prefill。
4. D 流式返回后续 token；P 在 D 确认收齐后释放槽（或超时释放）。

**接口形状（2026-10-18 实现，为 NIXL 预留）**：`core::pd::KvTransport { name(); register(MemDesc) /*= NIXL registerMem，一次注册 HostArena::region*/; send(&MigrationHeader, MemDesc) -> Box<dyn Transfer> /*= xfer handle*/ }`，`Transfer::poll() -> InFlight | Done`。控制面（header）始终 TCP 小消息，数据面（payload）由后端决定：TCP 后端 write 同一条连接；NIXL 后端发 WRITE 描述符到 D 已注册的槽、header 走 notif 或本 TCP 通道。executor 只认 `kv_export -> Vec<u8>` / `kv_import(&[u8])`，v2 改为导出到已注册槽后把 `MemDesc` 直接交给 transport（省一次 host 拷贝）。

**v1 实测（2026-10-18，4090 GPU 3 同卡 P+D，c8 (1024,128)）**：功能全过（输出逐字 = 单机，三故障回退正确），性能 TTFT 256 vs 13.7 ms、TPOT 5.5 vs 3.6 ms。代价来自同步导出 / 导入（135 MB/请求走主流 + host 中转）、回环 TCP、proxy 两跳；0.6B 同卡本来就不是 PD 的收益场景。v2 顺序：导出 / 导入异步化 → NIXL / CUDA IPC 去 host 中转 → P 流式首 token → 跨卡重测。**v2 (a) 异步化已做（2026-10-18）：TTFT 256 → 97 ms、TPOT 5.5 → 4.4 ms、吞吐 1000 → 1480 tok/s。** 机制：导出 = 主流 pack → 事件 → copy stream D2H 进 pinned 池 → 发送线程等 done 事件；导入 = 接收线程零拷贝读进 pinned 池 → copy stream H2D → 事件 → 主流 unpack；engine 线程不等 GPU。

**后端 b：共享槽区 shm / cuda-ipc（2026-10-18 实现，loop 第 17 轮）**。先用探针（`scripts/dev/probe_ipc.cu`）在 4090 上定事实：同卡 `cudaIpcOpenMemHandle` 可用、64 MB D2D 0.15 ms（446 GB/s）；**跨卡 open 报 `peer access is not supported`**（`nvidia-smi topo -p2p r` 全 CNS：消费卡无 PCIe P2P，GPU3↔4 还跨 NUMA）。结论：这台机器上 NIXL 也只能走 UCX 的 host 路径，引入 C++ 依赖换不到带宽，**不接 NIXL**；自己做两档共享槽区，接口形状仍按 NIXL（register / xfer handle）留着，真有 RDMA 网卡时只换 `KvTransport` 实现。
机制：D 启动建区（`--kv-transport shm|cuda-ipc --kv-slots N`，每槽 = max_model_len 全层 KV ≈ 448 MB@4k；shm = `/dev/shm/kv-slots-<pid>` mmap + `cuMemHostRegister`，cuda-ipc = `cuMemAlloc` + `cuIpcGetMemHandle`）；P 连上持久 TCP 先收 hello(`RegionInfo`)，映射对端区（shm：mmap + register；cuda-ipc：open，要求 uuid 同卡）；之后每次导出 `acquire_slot` 拿 credit → **P 的 copy stream 直接写 D 的槽**（shm D2H / ipc D2D）→ 等 done 事件 → TCP 发 header{slot}（无 payload）；D 接收线程构造 `SlotPayload`，executor 从槽 H2D / D2D 进 staging → unpack，H2D 完成事件过后 drop payload → 槽号作 credit 回 P（同一连接，4 B）。与 v2(a) 比省掉：P 侧 TCP write 的内核拷贝、D 侧 read 进 pinned 的拷贝、loopback 协议栈；c8 (1024,128) 每请求 ≈ 115 MB。回退链完整：握手失败 / 映射失败（跨卡 ipc）/ 无 credit / 槽导出失败 → 原 TCP payload 路径；D 超时本地重算不变。P 跟随 D 的区类型，两边 `--kv-transport` 不一致也能工作（P 端只决定用 `SharedTransport` 还是 `TcpTransport`）。
**后端 b 实测（2026-10-18，4090，c8 (1024,128) 经 proxy；单机 13.5 / 3.53 ms / 2216 tok/s）**：同卡 shm 71 / 4.12 / 1702，同卡 cuda-ipc 70.5 / 4.10 / 1719（TCP v2c 67 / 4.70 / 1464 → TPOT −12%、吞吐 +16%）；**跨卡 shm（P GPU3 / D GPU4）58.7 / 3.40 / 2029**（TCP 跨卡 57.7 / 3.75 / 1786 → TPOT −9%、吞吐 +14%，**D 侧 TPOT 低于单机**）。三组 `pd_export_slot_total = pd_export_total`，槽路径 100% 命中。同卡 shm ≈ cuda-ipc 说明同卡瓶颈是两个进程抢同一张卡，不是介质；TTFT 60–70 ms 四组不动，剩余是 P 侧 c8 排队 + proxy 两跳 HTTP（loop 第 15 轮分析），与传输无关。**默认仍是 tcp**（零依赖、跨机可用）；同机部署推荐 `--kv-transport shm`（跨卡 / 同卡都行），同卡可选 cuda-ipc（省 host 内存，收益相同）。

**不做（v1）**：P 侧 prefix cache 跨实例共享；D→P 回传（多轮续写走 D 本地 prefix cache）；TP 下的分片传输（Q5）。

**验收**：`scripts/verify/q4.sh` exit 0 = ① 两进程（GPU 0 P / GPU 3 D）+ proxy，greedy 输出与单机逐字一致（e2e 用例 `e2e_qwen3_pd`）；② 杀 P / 杀 D / 掐断 kv-port 三种故障各一次，请求全部正常返回（走回退）；③ bench c8 (1024,128) 经 proxy：TTFT ≤ 单机 1.3×（多一跳 + 传输），TPOT ≤ 单机（D 无 prefill 干扰），进 CSV（`pd1`）。

**PD bench 口径（用户 2026-10-18 确认）**：同机 Qwen3-0.6B 下 PD 经 proxy 的 TTFT 高于单机是预期内（prefill 本身 15 ms，任何中转都比它贵），不再纠结、不作为门槛；PD 线以功能正确 + 故障回退 + 跨卡 TPOT 接近单机（4090：3.75 vs 3.53）为收口标准。真正的收益评估等生产级模型 + 跨机。


### 已做（2026-10-20，用户要求记录）：executor 公共件抽取——"调度相关的东西不应该每个模型实现一遍"

公共件在 **`crates/layers/src/exec.rs`**（`dinfer_layers::exec`；放 layers 是因为它同时依赖 core 的 `BatchPlan` 与 kernels 的
`PinnedSlot` / `Graph`，三个模型 crate 都依赖 layers，不引入循环）。runner / engine 侧的 `ModelExecutor::{supports_async, begin,
finish, stash_sync_result}` 与 pinned 原语 `PinnedSlot` 本来就是公共的，这里补的是把它们接起来的胶水：

| 公共件 | 接口 | 谁在用 |
|---|---|---|
| `AsyncResultRing` | `new(stream, depth, rows)`、`enqueue(stream, chosen, n) -> Ticket`（异步 D2H 到 pinned 槽 + 事件，不等 GPU）、`wait(Ticket) -> Vec<i32>`（FIFO）、`inflight()`；`async_sched_enabled()` 读 `ASYNC_SCHED` | qwen3、qwen35 |
| `DecodeGraphs` | `new(stream, enabled, max_rows)`、`get(n) -> (桶, &Graph)`、`ensure(n, body)`（懒捕获）、`capture_all(body)`（`GRAPH_CAPTURE=eager`，`graph_capture_eager()`）；`body(nb, CapturePhase)`：**`Prepare`（上传假 plan / 采样参数，capture 外）→ `Body` 预热 → 同步 → capture(`Body`)**；`decode_capture_plan(nb)`、`graph_buckets(max_rows)` | qwen3、qwen35 |
| `StateSlotTable` | `new(cap)`、`begin_step(req_ids) -> fresh`、`indices(req_ids)`、`release(finished)`（LRU 兜底） | qwen35 `gdn::SlotTable`、kimik3 `kda::SlotTable`（均为 re-export） |
| `PlanBuffers` | 固定地址列 ids / pos / slots / rows / cu / seq / bt / max_blocks + 异步 `gather_idx` / `prev_chosen`；`upload(stream, uploader, plan, pad_to) -> (tt, nb, nrows)`（pad 到桶，不 flush，模型特有列之后追加）、`gather_inflight(stream, t)`（图内 `row_gather_i32`）、`save_chosen(stream, chosen, n)` | qwen3、qwen35（追加 state_idx / has_init） |

顺序已按计划走完：qwen3 接回（4090 `bench_common1` 门禁，见 `test/qwen3/perf-trend.md`）→ qwen35 接 `AsyncResultRing`
（性能 loop 第 8 轮）→ kimik3 接 `StateSlotTable`（kimik3 尚无 executor，`DecodeGraphs` 等它有了再接）。
踩坑：H2D 不能进 capture（图会记下 host 指针，重放读已释放的 Vec → `row_gather` 越界），所以捕获体分 `Prepare` / `Body`
两阶段（pitfalls 2026-10-20）。

## 11. 采用 / 改进台账（对 vLLM）

目标是性能超越，不是重写。vLLM 做得对的直接用，能改进的改进；每项记一行，便于回看哪些是"借来的"、哪些是我们的。

| 项 | 采用 vLLM 的 | 我们改进的 | 收益 / 状态 |
|---|---|---|---|
| decode 小 M GEMM | vLLM 直接用 torch/cuBLAS，让 cuBLASLt 启发式选 algo（M=1 在 sm_103 上被选成 2-CTA 大 tile，43% 带宽） | 启动自调优（`Gemm::autotune`）：对每个 (N,K) × M 桶实测 cuBLASLt 前 8 候选 + 自写访存型 `gemm_small_m` kernel（split-K 份数一起搜）取最快，查表执行；搜索参数在 `ArchProfile`，`GEMM_TUNE=off` 回退 | 3090：默认→所选 −9%（已近峰值带宽）；B300 待 bench_b300_6（`docs/loops/qwen35-perf.md` 第 3 轮） |
| 调度模型 | v1 的 "无 prefill/decode 之分，只有 num_computed 追赶 num_tokens"、chunked prefill、FCFS + 抢占末尾 | `decode_exempt_from_budget`（decode 行不占 token 预算，每步装下全部 decode + 预算内 prefill） | 调度结构零成本沿用；免预算让混批可能，但闭环 bench 下看不出（q3-xstream） |
| KV 分页 + prefix cache | 块哈希、按块命中、`block_size=16` | 无 | 一致 |
| TP 切分与通信点 | GQA 按头切 qkv、o_proj/down 行并行、每层 2 all-reduce、每 rank 都采样 | 一进程多线程 lockstep（`BatchPlan` clone 直传，无 shm 队列 / 序列化）；lm_head 复制省 all-gather | T1 功能完成，4090 无 P2P 性能待 NVLink 机器 |
| DP 路由 | 前端按在飞 / 队长选 rank；MoE 波次同步 | 同进程原子计数，无 coordinator 进程；dense 不同步 | T2 |
| DCP | block 交错分卡 + lse 归并 | 复用 split-K decode 的 partial/lse 归并，kernel 不改 | 设计 §13.5 |
| CUDA graph | decode-only 按 batch 桶捕获（FULL_DECODE_ONLY） | 按桶**懒捕获**（启动 -210 ms；`GRAPH_CAPTURE=eager` 恢复） | 启动 2.9 → 0.6 s 的一部分 |
| 异步调度 | placeholder 机制、stop 检查晚一步、下一步在上一步结果回来前调度 | 下一步的 input_ids 在**设备上**从上一步 `chosen` gather（图内 `row_gather_i32`，不经 host 往返）；D2H 用 pinned 双槽 + 事件；engine 单线程双步流水（pending/sync_done 两槽），不需要 vLLM 的多进程 | bench24：c32 1.05 → 1.08×、c8 1.06 → 1.08×；代价 TTFT +1 步（c8-long 39 → 54 ms） |
| 异步调度（Qwen3.5 / 多模型） | 同上（vLLM 每个 model runner 都走同一套 async 路径） | 胶水抽成公共件 `dinfer_layers::exec::{AsyncResultRing, DecodeGraphs, PlanBuffers, StateSlotTable}`，模型 executor 只提供捕获体与模型特有列；Qwen3.5 接入后纯 decode 步不再每步 `dtoh_sync` 等 GPU（GDN 模型同样在图内 gather 在飞 token） | 第 8 轮 `bench_b300_15`（主会话跑）；golden `golden_greedy_overlap` 5/5 |
| attention | FA2 的算法结构（online softmax、P 片直接当 PV 的 A 片） | 自写 mma.sync 版 v4/v5（Ampere/Ada 无 FA3），paged/prefill 同一模板，混批 decode 行分流到 split-K | 4090 T=1514 0.14 ms/层，TTFT 领先 vLLM 2× |
| GDN chunk prefill（Qwen3.5） | fla `chunk_gated_delta_rule` 的 WY 形式（块内 `(I+A)⁻¹`、跨块扫 h）与 BT=64 | 与 KDA chunk 同一 CUDA 实现骨架（prep / wy / ho 三段，全 fp32，无 Triton），切块表 `KdaChunkPlan` 两模型共用；中间缓冲全模型一份按需增长；输入直接吃 `mixed_qkv` 不再拆 q/k/v | 3090 T=1024 27B 形状 顺序 41.5 → chunk 4.2 ms（9.96×）；对拍见 `docs/loops/qwen35-perf.md` 第 1 轮 |
| decode attention | split-K | v2：8 lane/key、两 q head 共享 K 读、splits 按 batch 自适应 | 与 FA2 decode 持平 |
| 采样 | top-k/top-p/min-p/penalties 语义、Philox | 四 kernel 融一个（`sample_fused_f32`，共用 row_* 原语，逐位同语义）；vLLM 走 PyTorch 多个 op | bench25：c1 TPOT 2.0 → 1.9，图内 -3 节点 |
| 受限解码（2026-10-19） | `llguidance` 后端（vLLM `guidance` 同源，crates.io 1.9；CPU 算 bitmask、JSON schema → 文法、`{"type":"object"}` 做 json_object）、mask / logit_bias 在 penalties 之前、tool_choice 用 lark 包 tool_call 文本 | ① 文法状态挂在 `SamplingParams.grammar`（Arc<Mutex>），runner 建 `SampleBatch` 时按输出序列追赶 consume + 算 mask，无 vLLM 的 structured_output_manager 线程 / 跨进程 bitmask 传输；② mask + logit_bias 一个 kernel（`logits_mask_f32`：mask 行号 -1 跳过、bias CSR），CUDA graph 内常驻、无约束行零工作、无约束步零额外 H2D（`mask_dirty` 只在有→无时清一次）；③ 错误文法不卡请求：退化为无约束 + warn | 代码在 `crates/guided`（core / kernels 不依赖 llguidance）；e2e `e2e_qwen3_guided`；思考段不约束（vLLM reasoning-aware）留 TODO |
| 权重加载 | safetensors mmap | bf16 直传不经 Vec、设备拼接 qkv/gate_up、并行填页、tokenizer 独立线程 | 2.9 → 0.6 s |
| bench 口径 | `benchmarks/serve.py` 的 TTFT/TPOT/ITL 定义、ignore_eos | 每轮换 seed（避免常驻侧 prefix cache 全命中）；开环泊松 + 长短混合 | 发现并修正了 bench11 之前的口径偏差 |
| 进程结构 | — | 单进程（API + engine 同进程不同线程），无 ZMQ / 多进程 IPC | TTFT 低一半的主要来源之一 |
| 小 kernel 融合 | vLLM 有 fused_add_rms_norm、silu_and_mul（CUDA op） | 我们多了 qkv_post（norm+rope+cache write 一个 kernel）、gather+末层 norm、采样四合一；silu·mul 进 GEMM epilogue 与 lm_head f32 直出留给 megakernel 阶段 | decode 步 kernel 数 17 → 7/层 + 4 尾部 |
| 插队 prefill | 无（vLLM 异步调度下新请求固定多等一步） | 在飞 decode 步未完成时，对 waiting 单独 `schedule_waiting_only` 并同步跑 prefill，结果立刻回填再调度下一步；负载门槛（在飞 ≤ 8 行） | bench28：c8 TTFT 14 → 8 ms、c8-long 54 → 35 ms；代价 c8-long TPOT +0.2 |
| 算子插拔 | vLLM：PyTorch custom op + `@register_oot` / attention backend 枚举（运行期 Python 分派） | `OpSet` trait 泛型单态化（零运行期开销）+ 启动期 `OPS=` 选择（未知名字 fail-fast）；`.so` 按 C ABI 加载并带 capture 探针（待做）；泛型 ref 测试（待做） | 方案 A 已实现（`kernels/src/opset.rs`，Attention/Norm/QkvPost/Act 四组）；Gemm/Sampler 待并入 |
| KV 分层 / 外置 | `KVConnectorBase_V1` 的事件点（scheduler 侧 get_num_new_matched_tokens / update_state_after_alloc；worker 侧 start_load / wait_for_save） | scheduler 直接下发 `kv_fetch/kv_offload` order（无 connector 对象层）；host 索引 LRU；executor pinned 竞技场 + copy stream；**搬运用 pack/unpack kernel + 每块一次拷贝**（第一版每块 28 次小拷贝 TPOT +0.5 ms）；PD / Remote 复用同一 order 形态 | host 层已实现（`--kv-host-gib`，默认关），e2e 过；真挤出 A/B：命中 8840 块、抢占 -50%，但吞吐 -5%（fetch 同步 + offload 占 GPU 抵消了 5% 的额外命中）；留 低水位 offload / 异步 fetch 两项优化 |
| 跨流 prefill/decode 并行 | 无（vLLM 单流） | 实现并证实真并行（nsys 39% 重叠） | 闭环 bench 下混批步 <5%，TPOT 无感；留给 prefill 比例高的负载 |
| 2026-10-18 | 改进 | vLLM PD 用 NIXL（UCX / RDMA）搬 KV，同机也经 UCX；我们按机器事实分档：同卡 cuda-ipc D2D、跨卡无 P2P 走 shm pinned 共享区，P 直写 D 的槽 + credit 环，TCP 只传 header+槽号 | design §10.3 后端 b；不引入 NIXL C++ 依赖，接口形状保留 |
| PD 异构 tp（2026-10-18） | vLLM NixlConnector / Mooncake 的"按 KV 头切片、P 与 D tp 可不同"思路 | 不走 connector 插件，wire 固定为 head-major canonical `[block][layer][kv_head][slot][K/V][D]`，pack kernel 顺手转置，各 rank `cuMemcpy2DAsync` 一次列拷贝即完成分片 / 重排，没有额外 gather kernel 与元数据协商 | 4090 P tp1 → D tp2：TTFT 616 → 60 ms，TPOT 3.17 < 单机；DCP 可复用同一布局 |
| lm_head 列切 | vLLM `ParallelLMHead`（vocab 按 tp 切，`gather` logits 到全 vocab 再采样） | 开关默认关（vLLM 默认开）：小模型 lm_head GEMM 省的时间 ≈ all_gather 代价，bench 说话；gather 传 bf16 + 一趟 `gather_blocks_bf16_to_f32` 修布局升精度，vLLM 是 `tensor_model_parallel_gather` f32/bf16 后 `.float()` 两步 | 2026-10-19 实现，待 B300 bench（loop b-nvlink B2） |
| decode attention v3（2026-10-19 B4，设计定稿待实现） | FlashInfer `decode.cuh`：16 lane/key + GQA 组内 q 头共享 K/V 读（bdy=group）、2 级 cp.async 流水、(m,l,o) 循环外一次合并、host 侧按 occupancy×SM 定 kv_chunk 并把 (请求, chunk) 展平成 work-item（graph 下 grid 固定 + valid mask） | 不引库：自写 `gqa_decode_v3.cu` 读我们的 `[slot][Hkv][2][D]` 布局，不做 kernel 内 RoPE；chunk 大小进 ArchProfile | 目标 B300 c8-long decode 21.6 → ~10 µs/层（`docs/loops/b4-sm100.md`） |
| one-shot all-reduce + 融合 add-rmsnorm（2026-10-19 B2） | vLLM `csrc/custom_all_reduce.cuh` 协议原样采用：每 rank cuda-ipc 区 = `Signal{start/end[36][8], _flag[36]}` + 镜像区，`barrier_at_start / barrier_at_end` volatile flag、单调计数器（图内重放不重置）、固定 rank 顺序 fp32 累加后一次 bf16 round（逐位一致）、阈值 world≤4 <512 KB / world 8 <256 KB | ① handle 交换走进程内 `TpExecutor::dispatch/collect`（vLLM 走 torch.distributed all_gather），任一 rank 失败全组一致回退 NCCL；② 把 all-reduce 与紧随的 residual add + RMSNorm 融成一个 kernel `ar_oneshot_add_rmsnorm_bf16`（和不落回 attn_out / mlp_out，省一次 t×h 读写 + 一次 launch；vLLM 需 flashinfer `trtllm_allreduce_fusion` 才有）；③ 跨层 `pending_mlp` 让第 ② 处 all-reduce 并进下一层 input_norm 的融合 kernel | 测试 `crates/kernels/tests/allreduce.rs`（one-shot == CPU 参考逐位；world 2 == NCCL 逐位）；bench 见 `docs/loops/b-nvlink.md` 第 3 轮 |
| GDN 混合模型（Qwen3.5，2026-10-19 里程碑 4） | vLLM `qwen3_next.py` / `qwen_gdn_linear_attn.py` 的层语义（in_proj → conv1d(silu) → gate/beta → delta rule → gated RMSNorm(z)）、per-head 交错 q/gate 输出门、partial RoPE + zero-centered q/k norm（`fused_qk_rmsnorm_rope_gate`）、`chunk_gated_delta_rule` 的数学口径 | ① q_proj 的 q/gate 拆成**两条紧凑 GEMM**（省 attention 后的按行搬运，FLOPs 相同）；② GDN 状态由模型 crate 自管 per-request slot（新增一个模型不需要动 core 的 KV group 抽象）；③ prefill 先用顺序递归（数值等价），chunk 形式留作 perf 项 | 里程碑 4：tiny golden 4/4（`scripts/verify/qwen35_golden.sh`） |
| 2026-10-19 | 采用（待做） | vLLM 对 Mamba/GDN/KDA 混合模型的 prefix cache：`mamba_cache_mode=align` + `mamba_block_size`，递归状态按块对齐快照才能跳过已缓存 prompt | Qwen3.5 真权重 bench 发现命中前缀后 GDN 状态从零起；现强制关 prefix cache（`models/qwen35/src/service.rs`），K3（KDA）同样 | 待实现后重开 prefix cache，c8-long TTFT 可回到命中档 |
| 2026-10-20 | 采用（口径） | `/data/code/llm-cache-prediction`（用户指定参考）：K3 渲染参照物是 vLLM 服务端 Rust renderer 而非 HF python（浮点 / tool 参数 / name 字段 / thinking 对象四处偏差）；tiktoken 400k/25k 切分；vLLM K3 前缀命中粒度 1536 = 12×128（mamba align 下状态快照在注意力 block 配平边界） | K3 C-4a golden 按服务端语义生成；递归模型 prefix cache 设计按 block 配平粒度 | 避免重踩 4 处偏差；prefix cache 粒度不再拍脑袋 |
| 2026-10-20 | 采用 | vLLM `qwen3_next.py` 的 `in_proj_qkvz`（GDN 四投影一条 GEMM）与 `qkv_proj`（q/gate/k/v 一条 GEMM）：`models/qwen35` 第 9 轮改为融合 `in_proj [qkv|z|a|b]` 与 `qkv [q|gate|k|v]`，下游 kernel 按列段 + 行距读（`x_stride / ab_stride / z_stride / mul_sigmoid_rows`），不拷贝 | 每层 GEMM GDN 5→2、attn 3→2，B=32 一步少 160 次 launch（含 96 次 splitK reduce）；撤回里程碑 4 的"两条紧凑 GEMM"决定（当时为省一次按行搬，现在 kernel 直接带 stride） |

## 12. 整体功能待办路线图（2026-10-18 盘点，不分模型）

> 用户 2026-10-18 要求从整体功能角度总结剩余工作（不考虑 Kimi-K3 / GLM-5.3 的模型实现）。按模块列，状态以本文件与 `docs/progress.md` 为准；完成一项就在这里打勾。

| # | 模块 | 待办 | 状态 / 依赖 |
|---|---|---|---|
| 1 | PD 分离（Q4，§10.3） | d. 跨卡重测（P GPU 3 / D GPU 2，q4.sh 加分卡参数）；b. NIXL / CUDA IPC 传输后端替代 TCP + host 中转；D 侧 unpack 移到 copy stream；多 P / 多 D 路由与健康探测 | v1 + v2 (a)(c) 完成；d 进行中 |
| 2 | KV 分层 / 外置（§10.2） | PinnedHostTier 默认开（需 offload 低水位 + 异步 fetch 把 -5% 吞吐收回）；RemoteTier（Mooncake / LMCache / NIXL） | HostTier 已实现默认关；Remote 未开始 |
| 3 | 算子插拔（§10.1） | 方案 B 外部 `.so`（FlashInfer 等）；方案 C PTX | 方案 A 泛型 OpSet 已完成 |
| 4 | 服务可生产 | 优雅停机（SIGTERM 排空在飞请求）；API key 鉴权；限流 / 并发上限；多卡 TP（当前 `--device` 单卡，无 NCCL）；>8k 长上下文验证（服务预算 2048、bench 最长 1024）；`/metrics` 指标名是否对齐 vLLM（§9-4） | 全部未开始；多卡依赖 §9-1 目标机器 |
| 5 | API 完整度 | `/v1/completions`、`/v1/embeddings`；structured output（`response_format` / JSON schema / grammar，§9-3 待决策）；`tool_choice` 强制；`logit_bias`；投机解码（MTP / n-gram）从 GLM 线抽到通用层；量化（fp8 KV / int8 / AWQ） | 仅 `/v1/chat/completions`（流式 / 非流式 / reasoning / tool_calls / logprobs / n / stop）|
| 6 | 单卡性能（Q3 已收口） | 逐层 megakernel（用户 2026-10-18 决定暂不做）；lm_head f32 直出 / silu·mul epilogue（依赖自写 GEMM）；开环 24 req/s TPOT 落后 vLLM 7% | 四组吞吐 ≥ vLLM、TTFT 领先 2–4×、启动 0.6 s |
| 7 | 待用户决策（§9） | 目标生产机器（卡型 / 卡数 / 单机多机）；structured output 是否在范围；对比基线配置（vLLM 开 MTP / EP / FP8 KV？） | 阻塞 #4 多卡与 #5 部分项 |

建议顺序：① #1 d → b 收口 PD；② #4 可生产四件（停机 / 鉴权 / 限流最小，多卡 TP 最大）；③ #5 API 补齐（completions、structured output）；④ #2 RemoteTier；⑤ 量化 / megakernel 等到生产级模型再做。

## 13. 多卡并行设计：TP / DP / EP(MoE) / DCP / PCP（2026-10-18）

目标机器：4090 Pod 8 卡（两 NUMA 各 4 卡，**无 PCIe P2P**，`nvidia-smi topo` 全 NODE/SYS），NCCL 2.27+ 走 SHM；生产卡型未定（§9-1）。
原则同 §11：vLLM 的并行语义直接采用（切分方式 / 通信点 / DP 波次），进程模型与调度改进。

### 13.1 进程模型（与 vLLM 的主要差别）

| | vLLM | dinfer |
|---|---|---|
| TP 的 rank | 每 rank 一个进程（`MultiprocExecutor` + shm 消息队列广播 `SchedulerOutput`） | **一进程多线程**：每 rank 一个线程 + 自己的 `Device`/`Stream`/`Comm`；rank 0 线程就是 engine 线程，`BatchPlan` 通过 `mpsc` 直接 clone 给其它 rank（无序列化、无 shm 队列） |
| 采样 | 每 rank 都算 lm_head + 采样，driver 出结果 | 同（NCCL ring 保证各 rank 的 all-reduce 结果逐位一致，Philox 采样同 seed → 同 token；省掉 broadcast）。lm_head / embedding **复制不切**（Qwen3 词表 151k × 1024 = 300 MB，切了要 all-gather logits；K3/GLM 可改列切 + all-gather，留开关） |
| DP | 每 DP rank 一个 EngineCore 进程 + 独立 `DPCoordinator` 进程 + 前端按负载选 rank | 每 DP 副本一个 `Qwen3Service`（含自己的 engine 线程与 TP 组），`server::dp::DpBackend` 在同进程按在飞数路由；RequestId 高 8 位编码副本号；**dense 模型 DP 副本完全独立，无波次同步** |
| CUDA graph | 每 rank 独立捕获，集合通信入图 | 同；因各 rank 执行同一 `BatchPlan` 序列，懒捕获在同一步触发，`ncclAllReduce` 入图（要求 NCCL ≥ 2.9） |

多机：TP 组跨机不做；DP 跨机 = 每机一个进程 + 外部 LB（proxy 已有骨架 `server/proxy.rs`），MoE 的 EP 跨机需要 NCCL 跨机 id 分发（后续用 TCP store）。

### 13.2 TP：权重切分与通信点（Qwen3，GQA）

| 权重 | 切分 | 说明 |
|---|---|---|
| q_proj / k_proj / v_proj（拼成 qkv） | 行切（dim 0，按头） | 每 rank `H/tp` 个 q 头、`Hkv/tp` 个 kv 头；要求 `Hkv % tp == 0`（Qwen3-0.6B Hkv=8 → tp ≤ 8；vLLM 的 kv 头复制 `Hkv < tp` 情况暂不做）|
| q_norm / k_norm | 复制 | 每头 D 维，与切头无关 |
| o_proj | 列切（dim 1） | 输入 `H·D/tp`，输出部分和 → **all-reduce ①** |
| gate_proj / up_proj | 行切（dim 0） | `I/tp` |
| down_proj | 列切（dim 1） | 部分和 → **all-reduce ②** |
| embed / norm | 复制 | 见 13.1 |
| lm_head | 复制（默认）/ 列切（开关 `LMHEAD_COLSPLIT=1`） | 列切 = 行切 dim 0 按 vocab，各 rank 算 `vocab/world` 列 logits → bf16 **all_gather** → `gather_blocks_bf16_to_f32` 一趟修正 `[world][rows][vl]` 块布局 + 升 f32（`Qwen3Model::lm_head_logits`）。只在 world>1 且文件有独立 `lm_head.weight` 时生效（纯 tied 不切，embed 要整份 gather）。收益 = 省 (world-1)/world 份 lm_head 权重与 GEMM 读，代价 = 一次 t×vocab bf16 all_gather；小模型收益小，bench 决定是否默认开（loop b-nvlink B2） |
| KV cache | 每 rank 只存自己的 kv 头 | `kv_bytes_per_block` 用本地 Hkv；block 数由 rank 0 决定并广播，各 rank 一致 |

每层 2 次 all-reduce（`t × hidden` bf16），与 vLLM 一致；vLLM 的 `fused_allreduce_rms_norm`（all-reduce + 残差 + RMSNorm 融合，Hopper custom allreduce）是后续改进项（**2026-10-19 B2 已做**：`crates/kernels/cuda/allreduce.cu` one-shot all-reduce + `ar_oneshot_add_rmsnorm`，`comm.rs::all_reduce_add_rmsnorm_bf16` 按消息大小分流，无 P2P 自动回退 NCCL，`CUSTOM_AR=0` 关；见 §11 "one-shot all-reduce"）。实现：`Qwen3Model.comm: Option<Comm>`，`forward` 在 o_proj / down 之后原地 all-reduce；`TpShard{rank,world}` 传给 `Attention::load` / `Mlp::load_tp`；本地 `cfg` 的 `num_attention_heads / num_key_value_heads / intermediate_size` 除以 tp，`vocab / hidden` 不变。

**实测（2026-10-18，4090 GPU 5/6，loop q5 第 3 轮）**：tp2 vs vLLM 单卡 c1 1.14× / c8 0.92× / c32 0.64× / c8-long 0.68×；dp2 c32 1.20× / c8-long 1.31×（≈1.9× 单卡 dinfer）。

**小模型 TP 的预期**：Qwen3-0.6B 单卡 decode 一步 ~3.5 ms；无 P2P 时 NCCL SHM all-reduce 单次 ≥ 30–50 µs，56 次/步 ≈ 2–3 ms，**TP=2 吞吐大概率低于单卡**。TP 在这台机器上是功能正确性验证（输出与单卡一致），性能结论要等 NVLink / P2P 机器（design §9-1）。

### 13.3 DP

- `--dp N --tp M`：副本 i 用卡 `device + i·M .. device + (i+1)·M`。每副本独立 `Qwen3Service`（独立 scheduler / KV / prefix cache）。
- 路由：`DpBackend` 选在飞请求最少的副本（vLLM 前端同策略，但 vLLM 需 coordinator 进程广播队长；我们同进程原子计数）。prefix cache 亲和（同前缀打同副本）留作改进：按 prompt 前 N token 哈希选副本，再按负载兜底。
- **MoE + DP 的波次同步**（vLLM `DPEngineCoreProc._has_global_unfinished_reqs`）：EP 下 all-to-all 要求所有 DP rank 每步都参与，空闲 rank 要跑 dummy step。dense 模型不需要；K3/GLM 上线 EP 时在 `Engine::run` 的 step 循环前加 `dp_group.all_reduce(has_work)`，并复用 vLLM 的"wave"语义（一个 engine 有请求即唤醒全组）。

### 13.4 EP（MoE，K3 / GLM-5.3，设计，未实现）

- 切分：路由专家按 `num_experts / ep` 分卡（每 rank 一组完整专家），shared expert 复制或 TP 切；attention 部分走 TP（MLA 的 kv_b/o_proj 按头切，§design-glm53）。
- 通信：dispatch / combine 两次 all-to-all（NCCL `ncclSend/Recv` 组或 DeepEP 风格 IBGDA；4090 无 P2P/IB，只能 NCCL SHM，吞吐差，仅验功能）。
- 负载均衡：vLLM EPLB（专家副本 + 周期重排）作为改进项；先做静态切分 + 每层专家命中直方图指标（`moe_expert_hits`）。
- 与 DP：EP 组 = DP × TP 全体 rank（vLLM 默认），所以需要 13.3 的波次同步。

### 13.5 DCP / PCP（长上下文 KV 切分，设计，未实现）

- **DCP（decode context parallel，vLLM `decode_context_parallel_size`）**：KV cache 按 **block 交错**（`cp_kv_cache_interleave_size`，默认 1 block）分到 dcp 组各 rank；decode 时每 rank 对自己持有的 KV 做局部 attention 出 `(out, lse)`，再 `all_gather` lse / out 做 log-sum-exp 归并（vLLM `dcp_a2a_lse_reduce`）。Q 头在 dcp 组内复制（每 rank 算全部头 → 显存换 KV 容量）。适用：单请求 KV 超单卡、或 Hkv 小切不动 TP（GQA Hkv=8 时 tp>8 就靠 DCP）。
  - 我们的 kernel 已有 split-K decode（`gqa_decode v2`，每 split 出 partial + lse 再归并），DCP 只是把 split 维度放到卡间：**局部 kernel 不改**，加一个 `merge_lse` 跨卡步骤。
  - block 分配：scheduler 分 block 不变，`BlockId → (rank, local_block)` 由交错函数静态决定；slot_mapping 每 rank 过滤自己的块。
- **PCP（prefill context parallel）**：长 prompt 按 token 切到 pcp 组各 rank（zigzag / ring attention），每 rank 算自己的 Q 块对全部 K/V（ring 传 KV 或 all-gather KV）。4090 无 P2P 下 ring 代价大；**优先用 chunked prefill + DCP 代替 PCP**（vLLM 也只在 FlashAttn/FlashInfer MLA 支持 PCP）。
- 共同前提：attention metadata 加 `cp_rank / cp_world / context_len_local`；`KvCacheSpec.dcp_sharded` 对应我们的 group spec 加字段。

### 13.7 TP + PD（2026-10-18 晚，用户建议"P / D 只是角色不同，一套实现"）

- **一套方法**：`TpExecutor` 实现 `kv_export / kv_export_slot / kv_import / pd_map_region / pd_poll`，P 用前两个、D 用后面的，engine 不区分 TP 与否（`spawn_with_pd` 照用）。
- **分片布局（2026-10-18 深夜改为 canonical，支持异构 tp）**：每 rank 只持自己的 KV 头分片；wire 上每块单元 = **全部 KV 头**的 canonical 布局 `[block][layer][全局 kv_head][slot][K/V][D]`（`kv_pack / kv_unpack` kernel 把 cache 的 `[slot][head]` 转成 head-major，TP 分片就成了每层里连续一段）。导出 / 导入时各 rank 对**同一基址**用 `cuMemcpy2DAsync`（源行距 = 本地层字节，目标行距 = 全局层字节，列偏移 = rank × 本地层字节）只写 / 读自己头段（`Qwen3Executor::pd_export_addr / pd_import_addr` 内部按 `model.tp` 算，`TpExecutor` 不再算偏移；单卡 = world 1 退化为线性拷贝）。
- **约束**：P 与 D 的 tp **可以不同**（`model_tag` = `qwen3:L{层}:bs{块}:kvh{全局头}:hd{D}:bf16`，不含 tp；两侧 canonical 单元字节相等即可）；**cuda-ipc 不可用**（区是 rank 0 那张卡的设备内存，其它 rank 无 P2P 读不到；service 启动时拒绝，`pd_map_region` 兜底再拒一次）；非 0 rank `enable_pd(0, world)` 只建 staging + copy stream，不建 pinned 池。
- **调用序**：PD 调用走与 `Step` 同一条 rank 消息队列（`Msg::Call` 闭包），所以每 rank 上导出排在上一步 forward 之后、下一步之前，与 rank 0 的流序一致；槽 / 缓冲由 `Arc` 保活到最后一个 rank 的 copy stream 读完（`pd_push_inflight` 各 rank 一份 clone）。
- **4090 实测（2026-10-18 晚，loop q5 第 4–5 轮）**：P GPU 3,4 / D GPU 5,6 各 `--tp 2`，tcp 与 shm 两轮 e2e + 三故障 + 流式全 PASS，输出与单机逐字同；c8 (1024,128) TPOT **3.30 / 2.75 ms（单机 3.53）**，TTFT **616 / 795 ms**（单卡 PD 58）。TTFT 退化全在 P 自身：TP prefill 每步 56 次 all-reduce × 4 MB 走 NCCL SHM ≈ 11 ms/步（plain tp2 无 PD 的 c8-long TTFT 同样 231 vs 39），不是 PD 开销。**部署口径：无 P2P 机器上 P 不要开 TP（P 单卡 / D 多卡要异构 tp，见下）；有 NVLink 的机器再评。**
- 异构 tp（P tp1 → D tp2，无 P2P 机器上 P 不开 TP 保 TTFT、D 开 TP 降 TPOT 的实际需求）：上面 canonical 布局已覆盖，q4.sh `Q4_TP_P / Q4_TP_D`；**4090 实测（loop q5 第 6 轮）P GPU 3 tp1 → D GPU 5,6 tp2：tcp / shm e2e + 三故障全 PASS，TTFT 60 / 59 ms（同 tp2/tp2 时 616 / 795），TPOT 3.54 / 3.17（单机 3.56 / 3.52）** ⇒ 无 P2P 机器上 PD 推荐 P 单卡 + D TP。DCP 留 T4（head-major wire 布局可直接复用）。

### 13.6 分阶段

| 步 | 内容 | 验收 |
|---|---|---|
| ✅ T1（2026-10-18） | NCCL 封装 `kernels::comm`、权重分片 `load_bf16_shard`、Qwen3 TP（qkv/gate_up 行切、o_proj/down 列切、2 all-reduce/层）、`TpExecutor` 多线程 lockstep、`--tp` | 4090 `--tp 2` 与单卡 greedy 输出逐 token 一致；bench 记 CSV（预期慢于单卡，记原因） |
| ✅ T2（2026-10-18） | `--dp N` + `DpBackend` 路由 | `--dp 2` c32 10001 tok/s ≈ 1.9× 单卡 |
| ✅ T2b（2026-10-18 晚） | TP + PD：§13.7，`q4.sh Q4_TP=2 Q4_GPU_P=3,4 Q4_GPU_D=5,6` | e2e + 三故障 PASS（tcp / shm），CSV `ab_pd1_tp2*`：TPOT 2.75–3.30 < 单机，TTFT 受无 P2P TP prefill 拖累（非 PD） |
| ✅ T2c（2026-10-18 深夜） | 异构 tp：wire 布局 canonical 按头 + `copy2d` 按 rank 列读写，`model_tag` 不含 tp；`q4.sh Q4_TP_P=1 Q4_TP_D=2` | P 单卡 → D tp2 tcp / shm 全 PASS，TTFT 60 ms（tp2/tp2 616），TPOT 3.17 < 单机 3.52 |
| T3 | TP 下 CUDA graph 含 all-reduce；`fused_allreduce_rmsnorm`；lm_head 列切开关 | TPOT 开销 ≤ 单卡 + 56 × all-reduce |
| T4 | DCP（Qwen3 先做，复用 split-K 归并） | 单请求 context 超单卡 KV 仍可跑 |
| T5 | EP（GLM/K3 真权重到位后） + DP 波次同步 | 功能 |

## 14. 超越 vLLM 的质变路径（2026-10-20，用户："大部分还在照着 vLLM 实现，有没有自己的想法能够实现质的提升"）

### 14.1 为什么"继续抠 kernel"到不了质变

Qwen3.5-27B bf16 单卡 B300（`test/qwen35/2026-10-19/bench_b300_15`）：decode 每步读权重 52 GB，8 TB/s 的物理下限 **6.5 ms**；我们 TPOT c1 10.3 ms = vLLM 10.3 ms，都在 63%。剩下 3.8 ms 不在任何一个 kernel 里：每步 ~900 个 kernel 的边界效应（尾部 SM 空转、launch 间隙、延迟型小算子各 2–3 µs）。nsys 已证实 GPU 100% 忙（async 后 span < kernel 之和），GEMM 在 HBM 72–76%。同精度同硬件、"一算子一 kernel"结构下，上限约 1.1×。vLLM 基于 PyTorch 算子，torch.compile 只融 elementwise，跨 GEMM 融不了——**它结构上到不了下一档，这是我们唯一能拉开差距的地方**。

### 14.2 三条有质变潜力的路（按"是否只有我们能做"排序）

| 路 | 做法 | 预期 | 为什么 vLLM 做不到 / 做不好 |
|---|---|---|---|
| **A. decode 整步 megakernel** | 64 层的 GEMV / norm / GDN 状态更新 / attention 编成静态任务图，一个 persistent grid 跑完整步；SM 间用全局计数器做细粒度依赖；权重 TMA 预取进 smem；B 维请求在 SM 间分配 | B=1：10.3 → ~7.5 ms（1.4×，边界效应消失）；B=32 的 +3 ms B 维开销吃掉大半；Hazy Research 在 Llama-1B 上对 vLLM/SGLang 做到 1.5× | 需要整个前向在自己手里（我们是 Rust + 手写 CUDA，已满足）；PyTorch 算子图做不了 |
| **B. NVFP4 权重（B300）** | 权重 52 → 13 GB，下限 6.5 → 1.6 ms；按层混精度（敏感层 bf16）+ 校准 | 与 A 叠加后 TPOT 进入 2–3 ms 档 | vLLM 也支持，但按层混精度 + 自调优是可以做得更好的地方；不是独有 |
| **C. K3 MoE 融合路径** | 路由 → 按专家聚合 → `tcgen05` MXFP4 grouped GEMM → situ → 散回 + AttnRes 标量门，一个 kernel；`mxfp4_gemm_sm100.cu` 现在是空壳 | K3 92 层 MoE，收益比 Qwen3.5 大一个量级 | vLLM 用通用 Marlin / FlashInfer 类 kernel，路由 / GEMM / 激活 / AttnRes 分开 |

判断为**非质变**的：调度 / 引擎层（已零 Python、TTFT 领先一倍，但 TPOT 由 GPU 决定）；投机解码（两边都有，加法不是乘法）；PD / TP（能力项不是倍率项）。

### 14.3 第一阶段实验（先要数字，不直接重写）

- **范围**：Qwen3.5 的**一个 GDN 层** decode（B ∈ {1, 8, 32}）：rmsnorm → in_proj（合并后 1 个 N=16480 GEMV）→ conv1d_update → gate → GDN 递归 → gated_rmsnorm → out_proj → 残差+rmsnorm → gate_up → silu → down → 残差，现在是 ~11 个 kernel，改成 1 个 persistent kernel。
- **结构**：block = 一个 SM 常驻 CTA；任务 = (阶段, N 分片 / 请求分片)；依赖用全局 `ready[stage][shard]` 计数器 + `ld.acquire`；GEMV 权重分片按 N 切给 CTA，A 行广播；GDN 状态更新复用 `recurrent_decode_kernel` 的寄存器方案；阶段间的激活走全局（L2 常驻，一层的激活 B=32 × 16480 × 2 B = 1 MB）。
- **门禁**：单层 B300 耗时 vs 现有 11 kernel 之和（nsys node 级，现在 B=1 ≈ 0.14 ms/层、B=32 ≈ 0.21 ms/层）；赢 ≥ 25% 才扩到整步；数值与现有路径 bf16 逐位或 cos ≥ 0.9999；代码放 `crates/kernels/cuda/mega_sm100.cu`（Blackwell 专用，TMA / cluster），sm8x 不做（没有 TMA，收益不同）。
- **预算**：一周；失败即止，记入本节。
- **顺序建议**：A 第一阶段 → 若成立，B（NVFP4）与 A 的 GEMV 一起做（FP4 解包就在 GEMV 片段里）→ C 复用 A 的任务图框架做 K3 MoE 层。

### 14.4 与 §11 台账的关系

A / C 是"改进"而非"采用"；做成后在 §11 各记一行。本节由用户决定是否开工。

### 14.5 落地顺序（2026-10-20，用户："先超过 vLLM，做的过程中积累经验；是局部还是整体架构"）

两阶段，第二阶段按整体架构设计、局部切入：

1. **阶段一（到四组 ≥ 1.1×）：纯局部。** GEMM 合并 / small_m 错选 / M=32 splitK 归约 / gated_rmsnorm、qkv_post 融进相邻算子 → 减 launch 数。不动架构，按现有 loop 节奏。
2. **阶段二（质变）：架构级 = "静态任务图"抽象。** 现在模型 crate 的 `forward` 是命令式 launch 序列，每个算子自管缓冲；megakernel 需要加载期定好的任务图（阶段 / 分片 / 依赖 / 缓冲地址），运行时只把图交给 persistent grid。模型 crate 改为**声明层结构**，公共层生成图；**同一张图既可喂 megakernel，也可退化成逐 kernel launch**（图的每个节点 = 一次 launch），现有 kernel 一个不浪费，新旧路径共用模型描述、对拍天然成立。这是第 8 轮 `DecodeGraphs`（捕获体闭包）再往前一步。
3. **阶段二的第一步仍是局部实验**：§14.3 的单层 persistent kernel 只验证收益（≥ 25% 才继续），不改架构；成立后才把任务图引进 core——先让 qwen35 走图描述 + 逐 kernel 路径跑通不退化，再把 GDN 层节点换成 megakernel 实现逐层推；K3 MoE 层直接按图写。

经验积累不依赖赌对结果：第 1 步学 Blackwell persistent kernel 的依赖 / TMA / 占用配法，第 2 步学模型描述该长什么样；失败只损一周，架构不动。
