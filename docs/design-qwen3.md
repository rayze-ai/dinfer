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
