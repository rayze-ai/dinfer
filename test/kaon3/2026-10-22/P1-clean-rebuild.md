# P1 重建：干净参照（2026-10-22 深夜）

## 为什么全部推翻重来

用户 2026-10-22 指出：**共享基线 `kaon-vllm-pegaflow` 带了外部 KV（PegaKV 连接器）**。
外部 KV 会让 K/V 经过搬运与布局重排，数值路径与"纯 GPU 内 paged cache"不同 →
**它不是逐位置 logprob 对拍的合法参照**，且差异无法事后剥离。

因此以下全部作废并删除：

| 作废对象 | 原因 |
|---|---|
| `test/kaon3/oracle_p1/` `oracle_p1c/` | 从带 PegaKV 链路抓的 |
| `oracle_recheck*/` `oracle_dup/` `oracle_oldp/` `oracle_short/` `_one/` | 同上 + 端口抢占误抓 |
| 判据线 5.27% / 0.150 / 0.01 | 下限来自污染参照 |
| "修复前 28.87% → 修复后 15.26%" | 两端都在污染参照上测的，需重测 |

**保留**（已入库，作历史证据）：`oracle/`、`oracle_nobi/`、`oracle_layers/`。

## 环境清理（用户 2026-10-22 要求）

- 删除全部 `kaon-*` 容器（含 `kaon-mine`、`kaon-L12`、`kaon-L18`）
- `rm -rf /root/.cache/vllm`（torch.compile 缓存）+ `/hf_data/yaochen/{cmp,bins,runs}`
- 基线 `kaon-vllm-pegaflow` / `pegaflow-server-test` 已按用户要求**停掉**（不再作为参照）
- GPU 0–7 全部空闲（1 MiB）

## 新参照

`scripts/dev/kaon3_oracle_server.sh` 起两个**干净实例**，参数与探针假设逐项对齐，
**不带 `--kv-transfer-config`**：

| 容器 | GPU | 端口 | 角色 |
|---|---|---|---|
| `kaon-ref` | 1 | :28100 | REF（判定参照）|
| `kaon-floor` | 2 | :28101 | FLOOR（同参数第二进程 → 真实跨进程下限）|

共同参数：`--max-num-batched-tokens 4096 --block-size 32 --max-model-len 16384
--max-num-seqs 64 --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --dtype auto`，`VLLM_BATCH_INVARIANT=1`。

## 稳定结论（与参照无关，仍然成立）

**MoE 路由语义错误**：逐行读 vLLM `gemma4.py` 确认 Gemma4 用
`topk(原始 logits)` → **全专家 softmax** → top-k 上重新归一化 → 乘 `per_expert_scale`；
我们原先用 **sigmoid**（DeepSeekV3/GLM/K3 口径）。已新增 `moe_route_softmax_f32`
并在 `layer.rs` 启用。**这条不依赖任何参照数据**，是确定的修复。

## 干净参照下的实测（run `1022-144957-559547`，30/30 完整）

FLOOR = `oracle_clean` vs `oracle_clean_b`（两个干净实例，同参数、各单进程一次抓完）。

| 项 | FLOOR（vLLM vs vLLM）| OURS（我们 vs vLLM）| 线 | 判定 |
|---|---|---|---|---|
| 短 prompt 可判定行 top1 翻转率 | **0.83%** | **15.53%** | 4.25% | ❌ 超 3.7× |
| 共有 token \|Δlp\| 均值 | **0.0276** | **1.2578** | 0.041 | ❌ 超 30× |
| long_zh ΔNLL | −0.0817 | −0.0464 | 0.245 | ✅ |
| 短 prompt ΔNLL（参考）| −0.0025 | −0.0252 | — | — |
| 单 token（bos1，无 attention）| — | top1 一致 | — | ✅ |
| kernel 对拍（gqa_window / gemm_lowp）| — | 全过 | — | ✅ |

**通过 4 项 / 失败 2 项**（此前是 4 过 3 败，long_zh ΔNLL 由超限转为通过）。

## 污染参照 vs 干净参照：外部 KV 的代价

| 下限 | 污染参照（带 PegaKV 的基线链）| 干净参照（无外部 KV）| 倍数 |
|---|---|---|---|
| 跨进程翻转率 | 1.87% | **0.83%** | 2.3× |
| 共有 token Δ | 0.1227 | **0.0276** | 4.4× |
| long_zh ΔNLL | −0.0010 | −0.0817 | — |

**外部 KV 让参照自身的噪声放大 2–4 倍**，把判据线抬高了同样的倍数。这直接解释了为什么之前
"25.25% / 1.80 / +0.048" 看起来像"刚好压线"—— 那条线本身是被污染参照撑宽的。

## 现在的真实位置

- **可信的下限**：0.83% / 0.0276（两个同参数干净实例）
- **我们的实际差距**：15.53% / 1.2578 —— 是下限的 **19–46 倍**
- **长序列已通过**：ΔNLL −0.0464，在线内
- **单 token 已通过**：说明 embed / norm / proj / MLP / MoE / lm_head / softcap 在 T=1 下是对的

**结论**：MoE 路由 sigmoid→softmax 的修复是**确定的**（逐行读 vLLM 源码得出，不依赖参照），
但短 prompt 的 15.53% 说明**还有未找到的语义错误**，且量级远大于 vLLM 自身噪声。
分歧集中在多 token 路径（单 token 通过）。
