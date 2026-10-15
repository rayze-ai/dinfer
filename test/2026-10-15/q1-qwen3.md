# Q1 Qwen3-0.6B 真权重（本机 3090）— 2026-10-15

## 结论

**第一次用真权重验证主流程，全部通过**（`scripts/verify/q1.sh` 全量见 `q1.log`）：

| 测试 | 结果 |
|---|---|
| `kernels gqa_*`（GQA prefill / cache_write / paged vs torch SDPA fp32） | 2/2 PASS，bf16 atol 1e-2 |
| `qwen3 chat_parity`（真 tokenizer.json + tokenizer_config 内嵌模板 vs HF） | 9/9 chat case 逐字符 + 逐 id 一致；encode / 增量 decode / specials / eos 一致 |
| `qwen3 real_parity::real_prefill_logits_match_hf` | 13 行 logits worst max abs 1.0；argmax 一致 11/13，2 行 near-tie（HF gap ≤ 1.0）。HF 自己 eager vs sdpa 就差 0.68、逐层 hidden 到第 27 层差 9.0 → 我们的偏差是参考噪声量级 |
| `real_greedy_matches_hf` | greedy 32 token **全部**与 HF 一致（Scheduler+Runner+Executor 路径） |
| `real_two_requests_concurrent` | 两条 prompt（第二条 `enable_thinking=false`）预算 16 混批 chunked prefill，32+32 token 全部一致 |
| `serve_e2e_qwen3`（真进程 HTTP） | `/v1/models`；`usage.prompt_tokens == HF`；thinking 模式 `reasoning_content` 468 字、`content == "2"`；流式拼接 == 非流式；`enable_thinking=false` 直接答 `1+1=2` |

性能快照（未优化参考 kernel，3090）：加载 1.5 GB bf16 17 s（优化 mmap→bf16 拷贝前 59 s）；13 token prefill 6.9 ms；32 步 decode 0.24 s（≈ 7.5 ms/步）。

## 本步新增 / 改动

- `crates/text`（dinfer-text）：从 glm53 抽出模型无关的 `HfTokenizer` 与 `ChatTemplate`（支持 `chat_template.jinja` 或 `tokenizer_config.json` 内嵌字段；`RenderParams.extra` 传模型特定变量）。glm53 改为薄封装，15/15 golden 仍过。
- `kernels/cuda/gqa.cu`：`gqa_cache_write / gqa_prefill / gqa_paged`，cache 布局 `[num_blocks, block_size, kv_heads, 2, D]`；`KvCacheSpec::FullAttention`。
- `crates/qwen3`：config / model（融合 qkv GEMM、per-head q/k RMSNorm、neox RoPE 全 head_dim、tied lm_head）/ executor / parser（`<think>` + JSON tool_call，7 单测）/ service。
- `WeightSource::load_bf16` BF16 路径改 u16 批量拷（加载 59 s → 17 s）。

## 日志节选
```
test real_greedy_matches_hf ... q1: loaded Qwen3-0.6B real weights in 16.8s
q1: engine 32 steps for 1 prompts × 32 tokens in 0.24s
q1: prompt: all 32 greedy tokens match HF
test real_prefill_logits_match_hf ... q1: loaded Qwen3-0.6B real weights in 17.1s
q1: prefill 13 tokens in 6.9 ms
q1: prefill logits vs HF: worst max abs 1.0000, argmax mismatches 0 (near-tie rows 2) over 13 rows
test real_two_requests_concurrent ... q1: loaded Qwen3-0.6B real weights in 17.2s
q1: engine 34 steps for 2 prompts × 32 tokens in 0.15s
q1: prompt (concurrent, chunked): all 32 greedy tokens match HF
q1: prompt2 (no_think, concurrent): all 32 greedy tokens match HF
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 64.29s
# e2e
e2e: finish="stop" reasoning[468]="Okay, so the question is 1 plus 1 equals what? And the user wants the answer jus" content="2"
e2e: no_think content=String("1+1=2") reasoning=Null
e2e: PASS
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 20.96s
```
