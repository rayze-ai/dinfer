# 设计：Qwen3-0.6B（`dinfer-qwen3`）

> 本文件只讲这个模型的结构事实、crate 模块与实现状态。整体架构、core / kernels / layers / server、阶段计划、性能分析见 `design.md`。

## 0. 状态

**Q2 通过（2026-10-16，4090 / CUDA 13.0，`scripts/verify/q2.sh`）**：per-request 采样全链路（penalties / temperature / min_p / top-k/p / Philox seed / logprobs）、stop 字符串、n>1、自动 num_blocks、prefix cache 默认开、`/metrics`、32 并发压测无 5xx。实现口径与已知简化见 `docs/loops/q2-serving.md`、`test/qwen3/2026-10-16/q2-serving.md`。

**Q1 通过（2026-10-15）**：真权重（bf16 1.5 GB）在本机 3090 上 prefill logits / greedy 32 / 双请求混批 / 真模型 e2e 全部与 HF 一致（`scripts/verify/q1.sh` exit 0；`test/qwen3/2026-10-15/q1-qwen3.md`）。
这是项目第一个真权重模型，**用它验证与模型无关的主流程**（design.md §5b：Q2 采样 → Q3 单卡性能 → Q4 PD 分离 → Q5 多卡）。
对拍口径：真权重 bf16 下 HF 自己 eager vs sdpa 的 logits 差 0.68、第 27 层 hidden 差 9.0，我们的偏差同量级；验收以 "argmax 在 HF top-2 gap > 1.0 的行一致 + greedy 逐 token 全对" 为准。
性能快照（参考 kernel，未优化）：加载 17 s、prefill 13 token 7 ms、decode ≈ 7.5 ms/步 —— Q3 的起点。
loop 记录：`docs/loops/q1-qwen3.md`（4 轮）。

## 1. 结构事实

来源 `/data/ref/models/qwen3-0.6b/config.json`：28 层、hidden 1024、16 q 头 / 8 kv 头、head_dim 128、
q_norm / k_norm（per-head RMSNorm，在 RoPE 之前）、neox RoPE θ=1e6、SwiGLU intermediate 3072、tied embeddings、vocab 151936、
eos {151645, 151643}、无 sliding window、无 bias。chat 模板：`<|im_start|>role\n…<|im_end|>`，`enable_thinking` 控制 `<think>` 段，
tool_call 为 JSON `{"name","arguments"}`。
与现有代码的差异：需要 **GQA paged attention kernel**（prefill / decode / paged，K/V 分开存，`[num_blocks, block_size, kv_heads, head_dim]`）
与 `KvCacheSpec::FullAttention`；其余（RMSNorm、neox RoPE、SwiGLU、GEMM、scheduler、runner、engine、server、tokenizer、minijinja 模板）全部复用。

## 2. crate 模块

```
qwen3/src/
  config.rs      Qwen3Config ← config.json（head_dim 显式给，≠ hidden/heads；tie_word_embeddings；eos 数或数组）；check() 断言支持范围
  model.rs       Qwen3Model：embed → 28×(RMSNorm → Attention → +res → RMSNorm → Mlp → +res) → norm → lm_head（tied：文件里有 lm_head.weight 就读，否则用 embed）
                 Attention：融合 qkv GEMM → 拆 q/k/v → per-head q_norm/k_norm（RMSNorm over head_dim，RoPE 之前）→ neox RoPE（整个 head_dim）→
                 gqa_cache_write → gqa_prefill（无历史）| gqa_paged（decode / 带历史）→ o_proj；PagedKv [num_blocks, block_size, kv_heads, 2, D]
  executor.rs    Qwen3Executor（ModelExecutor）：上传 BatchPlan → forward → 采样管线（Q2：penalties→temperature→min_p→top-k/p→sample→logprobs；全 greedy 走 argmax 快路径）（无投机）
  parser.rs      <think>…</think> → reasoning；<tool_call>\n{"name","arguments"}\n</tool_call> → tool_calls（JSON，vLLM hermes 风格）；
                 初始态 Content（生成提示不以 <think> 结尾）；标签前缀与 </think> 前的 \n 都做流式保留
  service.rs     Qwen3Service（ChatBackend）：dinfer-text 的 HfTokenizer + ChatTemplate（模板在 tokenizer_config.json 的 chat_template 字段）；
                 enable_thinking 默认 true，可由请求 chat_template_kwargs 覆盖；KvCacheSpec::FullAttention
tests/
  chat_parity.rs  真 tokenizer / 模板 vs HF golden（9 case 逐字符、encode、增量 decode）
  real_parity.rs  真权重 prefill logits / greedy 32 / 双请求混批（chunked prefill + no_think）vs HF
  diag.rs         逐层 bisect（需 HF 侧 /tmp/qwen3_hf dump）
```
复用：`layers::{Embedding, Linear, Mlp, RmsNorm, Rope, WeightSource}`、`kernels::gqa_*`、core 的 Scheduler / Runner / engine、server、`dinfer-text`。

## 3. 模型特有的风险 / 待定

- ~~采样只有 greedy（Q2）~~ Q2 已完成；无投机解码（Qwen3 无 MTP 层；若要做可用 EAGLE/独立 draft，不在计划内）。
- 权重加载 17 s 里大部分是 safetensors mmap 首次缺页 + 逐张量 H2D（Q3 并行 / pinned）。
- `max_position_embeddings` 40960：rope 表按它预留；YaRN 等 rope_scaling 未实现（config 为 null）。
## 4. 参考材料

- `/data/ref/models/qwen3-0.6b/`：config.json、generation_config.json、tokenizer.json、tokenizer_config.json（含 chat_template）、vocab.json、merges.txt、model.safetensors（1.5 GB）
- HF：transformers 5.19 `Qwen3ForCausalLM`（`/data/ref/venv-tf519`）；golden：`scripts/verify/gen_qwen3_golden.py` → `test/qwen3_golden/`
- vLLM：`vllm/model_executor/models/qwen3.py`

## 5. Attention kernel：prefill_tc 与 paged_tc 是什么、为什么服务路径选 paged

### 5.1 两个 kernel 分别是什么

| | `gqa_prefill_bf16` → `gqa_fa_v5_kernel<PAGED=false>`（代码里叫 prefill_tc） | `gqa_paged_bf16` → `gqa_fa_v5_kernel<PAGED=true>`（paged_tc） |
|---|---|---|
| K/V 从哪读 | 本 step qkv GEMM 的输出缓冲（连续内存，按 `kv_stride` 跨行读） | **paged KV cache**：按 `block_table[b][pos / block_size]` 找到物理块，再按 `pos % block_size` 定位（间接寻址） |
| 能看到的 key | 只有本 step 的 token（`[0, row]`，纯 causal） | 历史 + 本 step：`[0, hist + row]`，hist = `seq_len − q_len` |
| 适用的 step | 纯 prefill 步：所有请求都是新 prompt、都没有历史（`BatchMode::Prefill`） | 有历史的任何 step：prefix cache 命中后的续算、chunked prefill 的第 2 片起、prefill 与 decode 混批（`BatchMode::Paged`） |
| 额外参数 | — | `seq_lens`、`block_table`、`block_size`、`max_blocks`、`skip_single`（混批时跳过 q_len==1 行，交给 decode kernel） |

两者共用同一份 kernel 模板（`gqa.cu` 的 `gqa_fa_v5_kernel<PAGED, NBUF>`），计算部分完全相同：FA2 结构，block = 4 warp = (请求, q head, 64 行 query)，每 warp 16 行；Q 片、S（16×64）、O（16×128）全在寄存器，softmax 用 4-lane shuffle 归约，P 的 mma C 片直接 pack 成 PV 的 A 片；K/V 以 64 key 为一个 tile 经 cp.async 进 smem（行跨 136 个 bf16 避开 ldmatrix bank 冲突）。唯一差别就是 `load_tile` 里 K/V 的地址怎么算，以及 mask 的上界多了 hist。

"tc" 后缀 = tensor core（v3 起用 WMMA / mma.sync 做 QKᵀ 与 PV），区别于 v1/v2 的 CUDA core flash 版本。

### 5.2 对比数据（round 33 / 38，`cold_shape` 微基准，q head 16 / kv head 8 / D 128，warm）

| 形状 | v3（WMMA，smem 往返） | v4/v5（mma.sync，寄存器驻留） |
|---|---|---|
| prefill_tc，1 请求 q=1024 | 0.672 ms | **0.144 ms**（3090）/ 0.09 ms（4090） |
| paged_tc A：q=1024 hist=0（与上行同工作量） | 0.698 ms | 0.189 ms |
| paged_tc B：q=1024 hist=1024（历史 tile 翻倍） | 1.825 ms | 0.447 ms |
| paged_tc C：q=1024 + 7×decode(hist 1100)（c8-long 真实混批） | 0.867 ms | 0.305 ms；分流后 0.250 ms |
| T=1514 单请求 prefill（4090） | 0.66 ms | **0.14 ms** |

结论：同工作量下 paged 比 prefill 只慢 3–30%（间接寻址 + 多一次 `block_table` 读 + 零填充分支），**不是**早先 nsys 看到的 1.8×；那 1.8× 是形状差异（paged 承担了带历史、混批的更重工作），不是 kernel 质量差异。

### 5.3 为什么服务路径选 paged（而不是把 K/V 先拷成连续再用 prefill 版）

1. **历史 KV 本来就在 paged cache 里**。prefix cache 命中、chunked prefill 的后续片、decode 混批，这三种情况都要读历史 key；读 paged cache 是零拷贝，而"先 gather 成连续缓冲再算"要多一次 O(hist × Hkv × D) 的读写，对 c8-long 每层 ≈ 1100 × 8 × 128 × 2 B × 2（K、V）≈ 4.5 MB，28 层就是 126 MB/step，比 attention 本身的读量还大。
2. **调度器不需要分开三条路**。vLLM v1 的调度没有 prefill/decode 之分，只有 `num_computed_tokens` 追赶 `num_tokens`；dinfer 照搬这个设计。只要 step 里任何请求带历史，就统一走 paged；纯 prefill 步（所有请求 hist=0）才走 prefill 版省掉间接寻址。实测纯 prefill 步在压测里只占 3/32 个 prompt（其余都与别人的 decode 混批），所以 paged 是主路径，prefill 版是特例优化。
3. **代价已经很小**。A 形状 0.189 vs 0.144 ms 的差距（0.045 ms/层 × 28 = 1.3 ms / 1024-token step）相对 20 ms 的 prefill step 不到 7%；比起多一次 gather 拷贝划算。
4. **混批里 decode 行不该走 paged**（round 34）：q_len=1 的行若走 paged_tc，一个 block 只用 1/64 的行却要串行扫全部历史 tile（smem 占着 SM）。所以 `BatchMode::Paged` 下 `gqa_mixed_bf16` 先用 split-K decode kernel 算所有 q_len=1 行（按 `cu_seqlens_q` 定位行），再让 paged_tc `skip_single` 只算 prefill 行。

### 5.4 还能怎么快（分析，未做）

- **纯 prefill 步合并到 paged 版**：若把 qkv 输出的 K/V 直接写进 paged cache 后统一读 cache（现在 `qkv_post` 已经写 cache 了），prefill 版就可以删掉，少一份代码；代价是纯 prefill 步多 3–30% attention 时间。等 c8-long 稳定 >1.0× 后再评估。
- **双缓冲（v5 NBUF=2）在 sm86/sm89 不划算**：68 KB 动态 smem → 每 SM 1 block，丢掉的 block 间重叠比 tile 内 cp.async 重叠更多（3090 T=1900 0.38 → 0.44 ms）。保留 `GQA_PREFILL_NBUF=2` 开关，H100（228 KB smem）上可能反转。
- **真正的上限**：T=1514 单层 9.4 GFLOP，0.14 ms = 67 TFLOPS ≈ 4090 bf16 dense（fp32 累加 165 TFLOPS）的 40%。FA2 在 A100 上约 50–60%，再优化空间约 1.3×：128-key tile（减半 softmax 归约次数）、PV 用 fp16 累加（精度风险）、Q 片 ldmatrix 预取。按 prefill step 里 attention 只占 ~15%（v4 后，GEMM 占大头）来算，端到端收益 < 5%，暂不做。

## 6. 进一步优化空间分析（2026-10-17，仅分析；不受 vLLM 思路约束）

当前（bench16 / q3.sh 2026-10-17）：c1 1.12× / c8 1.07× / c32 1.02× / c8-long 1.00× vLLM。每一项按"理论下限 → 现状 → 差距在哪 → 要做什么 → 预期收益"写。

### 6.1 decode 步（TPOT 的全部）

- **下限**：B=1 一步要搬全部权重 1.2 GB（bf16，含 lm_head）+ KV；4090 1 TB/s → **≥1.2 ms**。现状 2.0 ms（c1 TPOT）。B=8 现状 2.5 ms，B=32 3.6 ms。
- **差距**：28 层 × 8 kernel + 采样 4 kernel ≈ 230 次 launch 在一张 CUDA graph 里；每个 kernel 1–3 µs 的尾部 + 依赖等待，launch-bound 占 ~0.6 ms。另 D2H 等结果 + CPU 调度 ≈ 0.1 ms/步串行。
- **做什么**（按收益排）：
  1. **逐层 megakernel**（persistent kernel，一个 block 组做完 norm → qkv GEMM → attention → o_proj → norm → MLP，grid 级同步用 cooperative groups 或 flag）：launch 230 → 1，B≤8 时 TPOT 2.0 → ~1.4 ms（**1.4×**）。这是 vLLM 结构上做不到的（它的 kernel 来自 PyTorch / Triton 独立编译）。工作量大（2–3 天），sm86/89 可做。
  2. 采样 4 kernel → 1（penalties / temperature / min_p / top-k-p 融合）：-3 launch，B=1 约 -10 µs，小但零风险，顺手做。
  3. **异步调度**：runner 不等本步结果就调度下一步（vLLM 的 async scheduling 也是这个），把 0.1 ms CPU 串行藏掉；dinfer 的 placeholder 机制已在 scheduler 里预留（`num_output_placeholders`），只差 executor 双缓冲。
  4. KV fp8（E4M3 + per-block scale）：decode attention 读量减半，c8-long 36 → ~20 µs/层，B=8 长序列 TPOT -0.4 ms（**~10%**）；精度需对拍。
- **不做**：GEMM 小 M 已在带宽下限（round 14 微基准）。

### 6.2 prefill 步（TTFT 与 c8-long 的 TPOT 抖动）

- **下限**：1024-token step ≈ 0.9 TFLOP（非 embedding 参数 0.44B × 2 × 1024）/ 165 TFLOPS ≈ **5.5 ms**；现状 15 ms（单发 TTFT，含 HTTP / tokenize ~1 ms）。
- **差距**：GEMM 占 ~9 ms（cuBLASLt 对 M=1024 选的 kernel 实测 ~125 TFLOPS，已 76% 峰值）；attention v4 ≈ 1.5 ms（28 × 0.05）；其他小 kernel（norm / rope / silu / cache write）≈ 1.5 ms；lm_head 只算最后一行可忽略。
- **做什么**：
  1. **GEMM 融合**：gate_up 与 silu·mul 融合（epilogue）、qkv 与 qkv_post 融合，省 2 次 M×N 的读写 ≈ 1 ms/step。cuBLASLt epilogue 只支持 bias/relu/gelu，要自己写 CUTLASS/裸 mma GEMM（有 v4 attention 的经验，可做）。
  2. **prefill 与 decode 跨流并行**（dinfer 独有机会）：decode 是带宽型、prefill 是算力型，同一张 GPU 上两条 stream 可以重叠。把 step 拆成"decode 批（graph）"与"prefill 批"两条流，prefill 的 20 ms 不再打断 decode（c8-long TPOT 3.9 → ~3.0，**+30%**；TTFT 不变）。vLLM 是单流单批，做不到。前提：KV 写与读的块不相交（prefill 写新块，decode 读旧块，天然成立）；激活 scratch 两套。这是 Q4 PD 分离的单卡版，值得先做。
  3. 预算 2048 下 chunked prefill 把 1024 prompt 切 2 片时的首 token 延迟：若 1 项做了，预算可回 4096 又不伤 TPOT。
- **不做**：attention 再快 1.3×（128-key tile）端到端 <3%。

### 6.3 启动（4090：3.0 s → 0.61–0.68 s，2026-10-17，loop `docs/loops/q3-startup.md` 10 轮）

- **现分段**（4090，页缓存热）：cuda_init 240–255（`CudaContext::new`：驾驭初始化 + primary context，**不可压**，除非常驻进程复用）+ model_load 305–393（open 110：mmap 后 8 线程并行填页 / layers 115：alloc + 设备拼接 / **h2d 180**：1.4 GB pageable `cuMemcpyHtoD` 7.8 GiB/s / kv 60：cuMemAlloc 17 GiB）+ engine 45（跨流副套 + 懒捕获不含图）。tokenizer 400 ms 在独立线程被完全掩盖。
- **做了的**：① qkv / gate_up 的 f32 host 拼接（~1 s）→ bf16 直传 + 设备拼接；② 权重不经 Vec，mmap 切片直接 H2D；③ tokenizer 并行；④ rope 表多线程（122 → 21）；⑤ mmap 并行填页（190 → 110）；⑥ decode 图按桶懒捕获（启动 -210，首次用到桶时 5–140 ms 一次性；`GRAPH_CAPTURE=eager` 恢复）；⑦ KV `alloc_uninit`；⑧ 去逐张量 synchronize。
- **试过无效（已回退，留记录）**：open 与 Device::new 并行（争内存带宽/页表锁，cuda_init 变 430，总时长不变）；自建 pinned 双缓冲 staging（mmap→pinned 的 host memcpy 成瓶颈，6.0 GiB/s 比驱动 pageable 还慢）。
- **再往下**：KV 28 层合一次 cuMemAlloc（-45）；多线程 memcpy 进 pinned + 多流 DMA（-60，复杂）；常驻进程 / 多模型共享 context（-250）；GDS 直读（权重冷读场景）。单进程冷启动的实际下限 ≈ 0.5 s。

### 6.4 吞吐上限（c32 及以上）

- B=32 一步 3.6 ms：GEMM M=32 仍是权重带宽主导（1.2 ms）+ attention 0.3 + 小 kernel；vLLM 3.3。差 0.3 ms 在 decode attention（splits 后仍 2.7× 慢于 B=1 的每请求成本）和 norm/silu 小 kernel 数。
- 继续涨 B（64 / 128）dinfer 的 GEMM 进入 tensorop 区间，per-token 成本下降更快；bench 没测 c64，建议加一组。

### 6.5 优先级建议

1. 跨流 prefill/decode 并行（6.2-2）：c8-long 直接超越，兼做 Q4 前置。
2. 启动 <0.8 s（6.3）：工程量小，用户可感。
3. 采样融合 + 异步调度（6.1-2/3）。
4. 逐层 megakernel（6.1-1）：最大单项，但工作量最大，放 Q3 之后独立阶段。

