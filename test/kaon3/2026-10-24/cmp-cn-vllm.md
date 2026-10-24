# 中文 prompt 对拍：我们的 Rust 前向 vs vLLM（compile 口径）

- 日期：2026-10-24
- 机器：5090，`/hf_data/yaochen`（**本文件之前该目录被清空，本轮重新推代码 + 编译**）
- 参照：`kaon3_oracle_server.sh` 起的 `kaon-oracle`（GPU 1，**:28100**，**compile 模式**，
  **未开 `enforce_eager`**，`VLLM_BATCH_INVARIANT=1`）
- 我们侧：`kaon3_probe --batch`，`RMSNORM_MODE=0`（compile 口径的 norm），
  `CUBLASLT_WORKSPACE_SIZE=1`
- 对比工具：`kaon3_oracle_diff <参照目录> <我们目录>`

## 参数对齐（逐项核过，见 `kaon3_oracle_server.sh` 头注释的对齐表）

| 参数 | oracle | 我们侧 |
|---|---|---|
| block size | 32 | 32（`kaon3_probe.rs:23 BLOCK_SIZE`） |
| max-num-batched-tokens | 4096 | 一次前向整串（最长用例 31 token） |
| KV dtype | fp8_e4m3 | 同 |
| max_model_len | 16384 | 单序列 |
| batch invariant | `VLLM_BATCH_INVARIANT=1` | `CUBLASLT_WORKSPACE_SIZE=1` 对齐 |
| 模式 | **compile**（`mode=VLLM_COMPILE`，日志确认） | `RMSNORM_MODE=0` |
| 模型 | `flowgpt_vllm` | 同快照 |

## 用例（中文 prompt，chat 模板分词）

| 用例 | prompt | token 数 |
|---|---|---|
| cn1 | 写一首关于秋天的五言绝句。 | 24 |
| cn2 | 请把下面这句话翻译成英文：今天的会议推迟到明天下午三点。 | 31 |

## 结果

### FLOOR（同一 oracle 抓两次，BI=1）

```
POOLED rows=51 decisive=49 flips=0 flip_pct=0.00 mean_common_d=0.0000 dnll=+0.0000
```

**FLOOR = 0** —— compile + BI=1 下 vLLM 逐位可复现。所以下面的任何差异都是真实差异。

### 我们 vs vLLM

| 口径 | flip | mean_common_d | dnll |
|---|---|---|---|
| **FLOOR** | **0.00%** | **0.0000** | +0.0000 |
| `MOE_COMBINE_F32=1`（当前默认） | **14.29%**（7/49） | **1.1599** | −0.3248 |
| `MOE_COMBINE_F32=0` | 16.33%（8/49） | 1.0823 | +0.0179 |

逐用例（`COMBINE=1`）：

```
CASE cn1  rows=24 decisive=23 flips=3  flip_pct=13.04  mean_common_d=1.0123  dnll=+0.3330
CASE cn2  rows=31 decisive=30 flips=4  flip_pct=13.33  mean_common_d=1.2214  dnll=-0.9829
```

## 结论

1. **中文 prompt 上同样是 ~14% 翻转、|Δlp| ~1.16**，与之前 30 用例英文/中文混合池的 14.08% / 1.25 一致。
   → 差异**与 prompt 语言无关**，是模型级的。
2. **FLOOR = 0**：vLLM compile + BI=1 完全可复现，所以这不是"参照噪声大"能解释的。
3. `MOE_COMBINE_F32` 两种口径都差 1.1 左右，**这个开关不是主因**（改它 flip 从 14.29 变 16.33，方向还反了）。
4. 之前把 combine 当成唯一差异源，是**基于 eager 参照**得出的；compile 口径下它解释不了。
