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
