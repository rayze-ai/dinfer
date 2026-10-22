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
