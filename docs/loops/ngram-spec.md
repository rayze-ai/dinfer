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

## 状态

- 规格写好，开工。
