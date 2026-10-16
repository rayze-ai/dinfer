# 设计：Qwen3-0.6B（`dinfer-qwen3`）

> 本文件只讲这个模型的结构事实、crate 模块与实现状态。整体架构、core / kernels / layers / server、阶段计划、性能分析见 `design.md`。

## 0. 状态

**Q2 通过（2026-10-16，4090 / CUDA 13.0，`scripts/verify/q2.sh`）**：per-request 采样全链路（penalties / temperature / min_p / top-k/p / Philox seed / logprobs）、stop 字符串、n>1、自动 num_blocks、prefix cache 默认开、`/metrics`、32 并发压测无 5xx。实现口径与已知简化见 `docs/loops/q2-serving.md`、`test/2026-10-16/q2-serving.md`。

**Q1 通过（2026-10-15）**：真权重（bf16 1.5 GB）在本机 3090 上 prefill logits / greedy 32 / 双请求混批 / 真模型 e2e 全部与 HF 一致（`scripts/verify/q1.sh` exit 0；`test/2026-10-15/q1-qwen3.md`）。
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
