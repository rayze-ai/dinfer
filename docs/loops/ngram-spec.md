# Qwen3 n-gram 投机解码（A4）

> Qwen3-0.6B 无 MTP 层，EAGLE 头无现成权重；n-gram 不需要额外权重（vLLM `ngram_proposer.py`）。

## 目标

- `--speculative ngram --num-speculative-tokens K --ngram-min N --ngram-max M`（默认关）。
- 提案：每步对每个 decode 请求，用最后 n ∈ [N..M] 个 token 在 prompt + 已生成序列里找最近一次匹配，取其后 K 个 token 作 draft（vLLM 同算法，CPU）。
- 验证：复用现有 `drafts` 通道（`ModelRunnerOutput::drafts` → scheduler `update_draft_tokens` → 下一步 BatchPlan 多 token），executor 一步前向 K+1 行 logits，greedy / 随机采样按 vLLM `rejection_sampler` 口径逐位接受，首个拒绝处用本模型采样补一个 token。GLM 线 MTP 已走过这条路（`models/glm53` 的 verify 路径），接口对齐它。
- 指标：`spec_draft_total / spec_accepted_total`（`/metrics`），接受率。

## 环境

本机 3090，`QWEN3_DIR=/data/ref/models/qwen3-0.6b`。

## 观察方法

- 正确性：greedy 下开 / 关投机输出逐 token 一致（≥ 3 prompt，64 tok）。
- 收益：重复性强的 prompt（如"把下面文本原样重复三遍"）接受率 > 50%，TPOT 明显下降；普通 prompt 接受率低但不能比关闭时慢 > 10%。
- 旧 e2e 全部回归。

## 停止条件

`scripts/verify/ngram.sh` exit 0。

## 迭代上限

8 轮。

## 记录

| 轮 | 改动 | 结果 |
|---|---|---|
| 1 | `crates/core/src/spec/ngram.rs` 提案器（朴素倒序匹配，长 n 优先、最近位置、尾部不够截断）+ 6 单测；runner `with_ngram`：`finish` 里对 executor 无 draft、同步步、greedy 且无惩罚的请求用 host 侧 `token_ids` 提案（预算 = `num_prompt + max_tokens - len - 1`）；`BatchPlan.sample` 改为**每 logits 行一行**（带 draft 的请求 1+K 行）；Qwen3 executor 去掉 "不支持投机" 断言，eager 路径 `nrows ≥ n`，`verify_rows` host 侧逐行比对（接受前缀 + bonus，GLM MTP 同口径）并把逐行 logprobs 折回每请求；engine `publish_step` 计 `spec_draft_total / spec_accepted_total` 并把 `mro.drafts` 回 `update_draft_tokens`（此前 MTP 的 draft 在 engine 里被丢，只有 mtp_parity 测试接了）；service `--speculative ngram` 时 `num_spec_tokens=K`、`max_logits_rows = max_num_seqs×(1+K)`、`max_tokens += max_num_seqs×K`、强制 `ASYNC_SCHED=0`；CLI 4 个参数 | 单测 8/8 过；首跑 e2e 撞 `upload_plan` 把 "行数 = 请求数" 当前提（debug_assert (7,4)≠(4,4)）→ 拆开 token 维 / 请求维 / logits 行维三个 padding |
| 2 | `upload_plan` 返回 `(tt, nb, nrows)`，`forward_device` 显式 `rows` 参数；e2e 文本逐 token 相等 | 3 个普通 prompt：2 个逐 token一致，"poem about the sea" 第 51 token 分叉：基线 ` should` −0.633 vs ` need` −0.758，投机侧两者 −0.693 并列 → 翻转。手工对拍逐 token logprob：前 14 token 完全相同，之后 \|Δ\| ≈ 0.01–0.1（draft 验证行走 paged 多行 attention + 多行 GEMM，KV / logits 与单行 decode 有 bf16 级差异，只在并列处翻转）。重复 prompt：44 tok，draft 33 / accepted 32（**0.970**），88 ms vs 113 ms（−22%） |
| 3 | 正确性口径改为 vLLM `tests/v1/e2e/test_cpu_spec_decode.py` 的 `check_logprobs_close`：逐 token 比到首个不同 token，投机侧 token 必须 ∈ 基线 top-2 且基线 top-1/top-2 差 < 0.5（并列）；另要求 ≥ 1 个 prompt 完全一致；runner 单测 `ngram_drafts_flow_through_plan_and_rollback`（提案 → 1+K 行 plan → 拒绝回退 num_computed）；`ngram.sh` 补 `QWEN3_GOLDEN` 绝对路径（real_model_chat 需要） | **`scripts/verify/ngram.sh` exit 0**（2026-10-19，3090）：core 55/55；`e2e_qwen3_ngram`：5 普通 prompt 2 个逐 token 一致，3 个分叉全在并列处（gap 0.125 / 0.000 / 0.000，投机 token 都是基线 top-2）；普通 prompt 总耗时 884 ms vs 959 ms（**−7.8%**，接受 31/85）；重复 prompt 44 tok 52 ms vs 117 ms（**−55.8%**），接受率 **0.970**（32/33），整体 0.534；回归 `e2e_qwen3_request_features` / `e2e_qwen3_real_model_chat` 过；`cargo test --workspace --features qwen3` 全过；`--features glm53` 编译过（未跑 glm 测试） |

## 状态

- **A4 完成**：`scripts/verify/ngram.sh` exit 0（第 3 轮）。未 commit，交主会话。
- 已知取舍：
  - 只对 **greedy 且无惩罚** 的请求提案；随机采样请求退化为普通 decode（TODO：vLLM `rejection_sampler` 的 p/q 比接受 + 恢复采样；惩罚要 draft 行逐 token 历史）。
  - 开投机时**异步调度关闭**（同步 schedule → forward → finish → propose → 下一步 schedule 才能拿到真 draft；异步下 scheduler 用 0 占位、全被拒）。普通 prompt 代价：decode 步仍走 CUDA graph（无 draft 的请求 n=1 → `BatchMode::Decode`），损失的是 CPU/GPU 重叠；e2e 要求不慢 > 10%。
  - 带 draft 的步是 `Paged` 模式 eager（不进 graph）；TODO：统一形状（每请求 1+K 行、无 draft 的请求用 slot=-1 的 padding 行）后可进 graph，vLLM 也是这么做的。
  - 投机侧输出与关闭时在 logprob 并列处可能分叉（bf16 数值，vLLM 同样不要求逐位一致）。
