# Qwen3 激活缓冲：每层一套 → 全模型共享一套（ActScratch）

结论先行：**改前在 `--max-num-batched-tokens 16384` 下根本起不来（CUDA OOM），改后 3/3 起来，KV 块数从
OOM 前的 3050 涨到 10808，激活缓冲从 ~13.7 GiB 降到 480 MiB。** 性能四组 A/B 无差异（≤0.6%）。

- 改动：`models/qwen3/src/scratch.rs`（新）、`models/qwen3/src/model.rs`、`models/qwen3/src/lib.rs`、
  `models/qwen3/tests/real_parity.rs`（新增一个 --ignored 回归）。
- 机器：本机 3090（GPU 0，24 GB），`CUDA_ARCHS=86` 本卡 arch 构建。**未上 B300**。
- 旧二进制 = 改动前 commit `33528b9~1`，在独立 `git worktree`（/tmp/q3base）编出，与新二进制同机同卡先后跑。

## 1. 启动对比（真权重 `/data/ref/models/qwen3-0.6b`，各 3 次）

命令：`dinfer serve --model /data/ref/models/qwen3-0.6b --tokenizer … --max-num-batched-tokens 16384`

| | 改前（33528b9~1） | 改后 |
|---|---|---|
| 每层激活缓冲 `zeros_mib` | **14013**（zeros_n=168 = 28 层 × 6 块） | 560（剩下的只有 qkv/gate_up **权重**拼接，28 × 20 MiB） |
| 共享 ActScratch | — | `act_scratch_rows=16400`、`act_scratch_bytes=503808000`（480 MiB） |
| `num_blocks`（KV） | 3050（`available_gib=8.11`） | **10808**（3 次完全一致，`available_gib=21.37`） |
| 起得来？ | **3/3 失败**：`Error: DriverError(CUDA_ERROR_OUT_OF_MEMORY)` | **3/3 成功**，`startup done total_ms=1227` |

改前失败的时机：KV 池（3050 块）分完之后，`xstream: alt stream + scratch ready` 那一步要给副流再开一套
**每层** 缓冲 → OOM。也就是说原实现在大预算下不是"浪费显存"，而是**直接起不来**。

`max_tokens` 的来源：`max_num_batched_tokens(=16384) + max_num_seqs(=16)`（decode 不占预算，service.rs:206）。

## 2. 性能 A/B（同一 bench 口径，无 vLLM 可对比）

本机没有 vLLM 镜像 / 未装 vllm，`scripts/verify/bench_compare.sh` 的四组门禁跑不了 → 做 **旧二进制 vs 新二进制**
（同 prompt/seed/并发，`bins/bench`；被测钉 0-7 核、压测端钉 8-15 核，互不重叠）。
`--max-num-batched-tokens 2048`（改前也能起来的预算，避免只比"能起来"）。

跑了两遍（每遍先旧后新，同 seed）。两遍的绝对数不同（机器噪声），但**每一遍内部两组差都 ≤1%**：

| 组 | 第 1 遍 旧 → 新 (out tok/s) | Δ | 第 2 遍 旧 → 新 (out tok/s) | Δ |
|---|---|---|---|---|
| c1 | 456.17 → 456.38 | +0.05% | 456.5 → 454.8 | −0.37% |
| c8 | 2966.04 → 2964.41 | −0.05% | 2978.1 → 2970.1 | −0.27% |
| c32 | 4772.72 → 4768.84 | −0.08% | 4768.0 → 4772.4 | +0.09% |
| c8-long | 1495.01 → 1493.85 | −0.08% | 1492.8 → 1493.4 | +0.04% |

TTFT p50 / TPOT p50 同步无差异（第 2 遍：c1 7.11→7.16 / 2.152→2.158；c32 467.2→466.5 / 3.135→3.131 ms）。

结论：共享缓冲不影响 kernel 形状（GEMM 的 `n` / attention 的 `q_stride` 仍是本层行宽，缓冲只是**更长**），
性能差在噪声内。`XSTREAM=0`（不走副套）同样四组在 ±1.5% 内（c8 TTFT 8.30 vs 9.19，是跳过跨流并行的预期）。

## 3. 正确性

- `cargo test -p dinfer-qwen3`：11 项全过（8 单测 + guided 1 + lmhead_colsplit 2）。
- `real_parity`（真权重，--ignored）：4 项全过 —— prefill logits 最差 max abs 0.6221 / argmax 0 不一致；
  greedy 32/32 对齐 HF；并发两请求 32/31 步对齐。
- **新增** `real_scratch_swap_is_bitwise_identical`：主套 → 换入副套 → 换回主套，logits **逐位相同**
  （`0/1975168` 元素不同，max abs 0.0），且先自检"同套连跑两次也逐位相同"保证判据有效。
- **反例记录**：HTTP 端到端 greedy 输出**不能**当判据。同一二进制 + 同一配置连跑两次，12 条里 6 条输出不同
  （批组成变化 + greedy 放大逐位分歧；prefill 是逐字确定的，但哪几条被混批不固定）。改前 / 改后二进制在
  不同轮次里既能对齐也能不对齐，属噪声，别拿它判回归。

## 4. 复现

```bash
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
CUDA_ARCHS=86 cargo build --release --features qwen3 -p dinfer
CUDA_VISIBLE_DEVICES=0 ./target/release/dinfer serve --model /data/ref/models/qwen3-0.6b \
  --tokenizer /data/ref/models/qwen3-0.6b --port 8137 --max-num-batched-tokens 16384
# 看三行：load timing: per-layer zeros / activation scratch (act_scratch_bytes=…) / kv_cache: auto num_blocks=…
CUDA_VISIBLE_DEVICES=0 QWEN3_DIR=/data/ref/models/qwen3-0.6b QWEN3_GOLDEN=$PWD/test/qwen3_golden \
  cargo test -p dinfer-qwen3 --features ref-tests --test real_parity -- --ignored --nocapture --test-threads=1
```
