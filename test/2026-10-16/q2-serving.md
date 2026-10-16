# Q2 serving 迭代记录（2026-10-16）

| 轮 | 块 | 结果 | 备注 |
|---|---|---|---|
| 1 | a | kernels.sh exit 0，16 passed（含 4 个新采样 kernel） | 2026-10-16 03:20 |
| 2 | a | real_parity 3/3 仍过（65 s）；sampling_parity 4/5，分布测试因首 token 近确定（候选集 1）失败 → 改采样位置 | 2026-10-16 04:30 |
| 3 | a | sampling_parity 5/5（173 s）；real_parity 3/3；serve_e2e_qwen3 1/1；clippy --all-targets 干净 | 2026-10-16 05:10 |

## Q2-a 采样接线：结果摘要

- 实现：`core::sampler::SampleBatch`（随 `BatchPlan.sample` 下发，每请求一行：temperature/top_k/top_p/min_p/penalties/seed/offset/CSR 历史/logprobs k）；`ReqState.seed`（用户给或入队随机）；Philox offset = 已产出 token 数（与 batch 位置无关）。
- qwen3 executor：全 greedy 无惩罚无 logprobs → Q1 argmax 快路径；否则复制到工作区后 penalties → temperature → min_p → top-k/top-p → `sample_f32`（随机行传 1.0、greedy 行传 0），logprobs 对原始 logits 副本做 log_softmax。`ModelExecutor::take_logprobs` 默认空实现（glm53 不受影响）。
- 输出链：`ModelRunnerOutput.logprobs` → `StepResults.push_with_logprobs / logprobs_of` → `EngineEvent::Logprobs`（紧跟 Tokens）；server 先吞掉，渲染在 Q2-b。
- server：接 `min_p / repetition_penalty / frequency_penalty / presence_penalty / stop_token_ids / ignore_eos / min_tokens`，`SamplingParams::validate()` 失败 → 400。
- 测试（真权重 3090）：seed 确定性 + 混批位置无关；混批中 greedy 32 步 = HF；penalties 4 组 × 16 步 greedy 与 host 公式逐步一致（0 平局）；logprobs chosen/rank/top-3 与 host log_softmax 一致（1e-3）；分布：100 样本 L1 0.097（候选 3）。
- 教训：Qwen3 chat 首 token `<think>` logprob ≈ −2e-4，任何分布测试不能放在首 token；100×36 token 一次 prefill 需 budget 8192 + 512 块。
| 4 | b | StopChecker 6/6；server/text/core/qwen3 单测过；qwen3 + glm53 feature 均编译；e2e 待跑 | 2026-10-16 14:0x |
| 5 | b | serve_e2e_qwen3 2/2 on 4090（CUDA 13.0）：stop / include_stop / 流式 stop / logprobs / n=3 / 400 全过 | 2026-10-16 14:1x |
| 8 | c/d | auto_num_blocks + prefix_cache e2e PASS on 4090（TTFT 899→16 ms） | 2026-10-16 14:3x |
| 10 | e | serve_e2e_qwen3 5/5 on 4090（metrics / auto_num_blocks / prefix_cache / chat / request_features） | 2026-10-16 14:4x |
| 12 | f | robustness e2e PASS（20 s 试水：159 req, 0 5xx） | 2026-10-16 14:5x |
| 13 | 验收 | q2.sh exit 0 on 4090（60 s 压测 434 req / 40k token，0 5xx） | 2026-10-16 15:0x |

## Q2 验收结论（2026-10-16，4090 / CUDA 13.0 / sm_89）

`scripts/verify/q2.sh` exit 0。各步：

| 步 | 结果 |
|---|---|
| fmt / clippy（qwen3 全目标） | 干净 |
| CPU 单测 | core 33、qwen3 8、text 6（server 0） |
| kernel refs | 16/16（含 Q2-a 四个采样 kernel） |
| chat_parity / real_parity / sampling_parity | 2 / 3 / 5 全过（67 s / 192 s） |
| server e2e 6 项 | chat、request_features（stop / logprobs / n）、auto_num_blocks（9838 块）、prefix_cache（TTFT 899→16 ms）、metrics、robustness |
| robustness 压测 | 60 s × 32 并发随机：434 请求、40 405 completion token、0 个 5xx / 传输错误；结束后 running=waiting=0、kv_cache_usage=0；断开客户端后 abort 计数 +1、KV 归零 |

Q2 已知简化（进 Q3 / 后续）：logprobs 不含 `prompt_logprobs`；`n>1` 不做 KV fork（靠 prefix cache 共享）；`/metrics` 只有服务级（无 per-request `request_*` 直方图族、无 LoRA/多模态项）；glm53 后端未接 `metrics()` 与自动 num_blocks；采样工作区每 step 按 rows 复用但 logprobs 的 raw 副本仍是一次 D2D 拷贝（Q3 融合）；stop 字符串扣住 `max_len-1` 字节导致流式最多延迟这么多字符。
