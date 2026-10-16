# loop 规格：Q2 可生产的 mix 服务（Qwen3-0.6B）

- 状态：进行中
- 创建：2026-10-16
- 验证脚本：`scripts/verify/q2.sh`
- 迭代记录：`test/2026-10-16/q2-serving.md`

## 目标

`dinfer serve --features qwen3` 作为一个**单实例 mix（prefill + decode 混批）服务**可以生产使用：OpenAI `/v1/chat/completions`
的采样与控制字段全部生效、与 vLLM 语义一致；资源按显存自动配置；prefix cache 开启；有 `/metrics`；并发 / 长跑 / 异常请求不崩。
PD 分离（Q4）在此之后。用户原话："只有 greedy，temperature / top_p / top_k 字段收了不生效，这些遗留问题都要解决"。

## 范围（全部要做，按依赖排序）

| 块 | 内容 | 对照 |
|---|---|---|
| a 采样 | per-request `SamplingParams` 下发到 executor；GPU 侧顺序与 vLLM `Sampler.forward` 一致：**penalties（repetition / frequency / presence，用 prompt+output 历史）→ temperature → min_p → top_k/top_p → Gumbel/Philox 采样**；temperature < 1e-5 走 greedy；`seed` 每请求独立流（Philox：seed + 该请求已采样步数作 offset）；logprobs 用**原始 logits**（惩罚 / 温度之前）的 log_softmax | `vllm/v1/sample/sampler.py`、`ops/penalties.py`、`ops/topk_topp_sampler.py` |
| b 请求功能 | `stop`（字符串，含 `include_stop_str_in_output`，vLLM `check_stop_strings` 语义：最早完成的 stop 优先、未完整匹配时缓冲 `max_len-1` 字符不下发）、`stop_token_ids`、`logprobs` + `top_logprobs`（OpenAI 形状：`choices[].logprobs.content[]{token, logprob, bytes, top_logprobs[]}`）、`n > 1`（同 prompt 多条请求，共享前缀靠 prefix cache）、`min_tokens`、`ignore_eos`、`skip_special_tokens`、`repetition_penalty / frequency_penalty / presence_penalty / min_p / seed` 字段 | `vllm/entrypoints/openai/`（protocol）、`v1/engine/detokenizer.py` |
| c 资源 | `num_blocks` 按显存自动算：`available = gpu_memory_utilization × total − 已用（模型 + 激活 scratch）`，再 `/ bytes_per_block`；`--num-blocks` 仍可覆盖；启动日志打印 KV 容量（token 数）| `gpu_worker.py:740`、`kv_cache_planning.py` |
| d prefix cache | 默认开启（`enable_prefix_caching=true`，`hash_block_size = block_size`）；多轮对话第二轮 `num_computed_tokens` ≥ 第一轮长度（对齐到块）；`/metrics` 暴露命中率 | P1 已实现，未在服务路径验证 |
| e 可观测 | `/metrics`（Prometheus 文本）：请求数 / 进行中 / 等待中、prompt & generation token 数、TTFT / TPOT / E2E 直方图、KV 利用率、prefix cache 命中率、每 step 耗时；指标名对齐 vLLM（`vllm:*` → `dinfer:*`） | `vllm/v1/metrics/loggers.py` 指标清单 |
| f 健壮性 | 客户端断开 → engine abort（释放 KV）；超 `max_model_len` 的 prompt → 400；非法参数（temperature < 0、top_p ∉ (0,1]、n > 8）→ 400；并发 32 请求 × 随机长度压测 1 分钟不崩、不泄漏（KV 用量回到 0）；`SIGTERM` 优雅退出 | — |

不在 Q2：性能优化（Q3）、PD（Q4）、structured output、多模态。

## 环境

```bash
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
cargo run -p dinfer --features qwen3 -- serve --model /data/ref/models/qwen3-0.6b --port 8012      # num_blocks 自动
# 采样 golden：/data/ref/venv-tf519/bin/python scripts/verify/gen_sampling_golden.py test/sampling_golden
```
依赖：本机 3090；`/data/ref/venv-tf519`（torch 2.6）；Python `openai` SDK（e2e 全字段用，装进同一 venv）。

## 观察方法

| 看什么 | 在哪里 | 命令 |
|---|---|---|
| 采样 kernel 顺序 / 数值 | `kernels/tests/ref_tests.rs` sampling 用例 | `scripts/verify/kernels.sh` |
| 采样器 vs torch（同 seed 同分布） | `core/src/sampler.rs` 单测 + `qwen3/tests/sampling_parity.rs` | 见 q2.sh |
| 请求字段 | `bins/dinfer/tests/serve_e2e_qwen3.rs` 扩展 + `scripts/verify/openai_sdk_check.py` | q2.sh |
| metrics | `curl localhost:8012/metrics` | q2.sh |
| 压测 | `scripts/dev/bench_concurrent.py` | q2.sh |

## 停止条件（机器判定，`scripts/verify/q2.sh` exit 0）

1. **采样 kernel**：penalties / temperature / min_p / top_k+top_p / log_softmax 各与 PyTorch 参考逐元素一致（fp32 atol 1e-5；mask 位置完全一致）。
2. **采样分布**：对同一组 logits，temperature=0.7 / top_p=0.9 / top_k=20 下采样 50k 次，频率与 PyTorch 同参数的 softmax(after mask) 的 L1 < 0.03；`seed` 固定时两次运行结果**逐 token 相同**；不同 seed 不同。
3. **greedy 不变**：temperature=0 的输出与 Q1 的 HF greedy 逐 token一致（回归）。
4. **stop 字符串**：vLLM `check_stop_strings` 的 6 个边界用例（跨 token、多 stop 最早完成优先、`include_stop_str_in_output` 两种、stop 恰在末尾、stop 永不出现）单测通过；e2e 用真模型 `stop=["。"]` 返回在第一个 "。" 处截断且 `finish_reason == "stop"`。
5. **logprobs**：`logprobs=true, top_logprobs=3` 时每个 token 带 `logprob` 与 3 个候选，`logprob` 等于原始 logits 的 log_softmax（与 HF 同 prompt 的 greedy 第一步 logprob 差 < 0.05）；`bytes` 字段正确。
6. **n**：`n=3, temperature=0.8` 返回 3 个 choices，index 0..2，至少两个不同；`n=3, temperature=0` 三个完全相同。
7. **penalties**：`repetition_penalty=1.5` 下重复 token 率低于不加时（统计检验，50 个样本）；`frequency/presence` 单测对照 torch 公式。
8. **num_blocks 自动**：不传 `--num-blocks` 启动，日志打印 `kv_cache: N blocks (M tokens), available X GiB`；N ≥ 按 3090 24 GB 估算的 80%。
9. **prefix cache**：同一 system+history 第二轮请求，`/metrics` 的 `prefix_cache_hits` 增长且 TTFT 比第一轮小 ≥ 30%。
10. **metrics**：`/metrics` 含 `dinfer:num_requests_running`、`dinfer:prompt_tokens_total`、`dinfer:time_to_first_token_seconds_bucket`、`dinfer:kv_cache_usage_perc`、`dinfer:prefix_cache_hits_total`。
11. **健壮性**：32 并发随机长度（prompt 10–300、max_tokens 1–200、随机采样参数）跑 60 s，无 5xx、无 panic，结束后 `num_requests_running == 0` 且 `kv_cache_usage_perc == 0`；客户端中途断开 10 次后同样回零；非法参数 400。
12. **q1.sh 回归**仍过（greedy 对拍、模板、e2e）。

## 上限

每块 6 轮；总计 2 个工作日。超出就停，把卡点写进本文件。

## 已知可能出错点

1. 采样顺序：vLLM 是 penalties → temperature → min_p → top_k/top_p；P2 kernel 的 `topk_topp_mask` 和 `sample` 内部是否已含温度要先看。
2. Philox 的 (seed, offset)：同 seed 两次运行逐 token 相同要求 offset 只依赖该请求的步数，不依赖 batch 位置。
3. stop 字符串与增量 detokenize：stop 可能跨 token，也可能和 `<think>` 标签解析器的保留逻辑叠加 → stop 判断放在解析器之后的**内容流**上，还是放在原始文本流？vLLM 放在原始 detokenize 文本上（不区分 reasoning / content）。先同 vLLM。
4. logprobs 要在 executor 里多算一次 log_softmax（原始 logits），`vocab=151936` 的 top-k 用 radix 还是全排序：参考版先全排序。
5. `n > 1` 的 prefix cache 命中要求 hash 块对齐，prompt 短于 1 块时无命中（正常）。
6. 自动 num_blocks：模型加载后 `cuMemGetInfo` 的 free 已扣掉权重与 scratch；KV 分配后还要留 cuBLAS workspace / 临时张量余量（vLLM 用 profile run；我们先按 `free × util − 1 GiB`）。
7. 客户端断开：axum 的 SSE stream drop → 要在 `stream_chunks` 的 future drop 时调 `EngineHandle::abort`（用 guard）。

## 记录

| 轮 | 现象 | 根因 | 处理 |
|---|---|---|---|
| 1 | a 采样 kernel | 新增 `penalties_f32 / temperature_f32 / min_p_mask_f32 / logprobs_f32`，`kernels.sh` 16/16 PASS（fp32 atol 1e-5） | 停止条件 1 达成；接线 executor 待做 |
