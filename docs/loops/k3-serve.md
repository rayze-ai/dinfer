# Kimi-K3 里程碑 C-4b：executor + service（ChatBackend）+ `dinfer --features kimik3 serve`，前 N 层真权重冒烟

> 状态：**进行中（2026-10-20）**。crate 侧 `models/kimik3/src/{executor,service}.rs`，入口 `bins/dinfer/src/main.rs` 的 `kimik3` 分支。
> 前置：C-3（`k3-model.md`：`KimiK3Model::load(dir, n_layers, LayerBudget)` / `forward`）、C-4a（`k3-chat.md`：tokenizer / xtml / parser）。
> 下一步 C-5：EP / TP 切分（全模型专家 packed 1.45 TB，vLLM 基线 tp8 独占 8 卡）。

## 环境

- B300-50，权重 `/data/models/moonshotai/Kimi-K3`（物理路径，不拷贝）。**GPU 2–3**（`CUDA_DEVICE_ORDER=PCI_BUS_ID`；0/1 Qwen3.5 常驻、4–7 另一子代理）。
- 远端 `source /data/yaochen/infer/env.sh`；本地 HTTP 前 `unset http_proxy https_proxy`。生成物放 `/data/yaochen/infer/bootstrap/k3serve/`。
- 构建：`cargo build --release -p dinfer -p bench --features kimik3`。
- 启动（前 5 层 ≈ 75 GB，单卡最多 ~14 层）：
  ```bash
  CUDA_VISIBLE_DEVICES=2 RUST_LOG=info target/release/dinfer serve --model /data/models/moonshotai/Kimi-K3 \
    --layers 5 --port 8014 --max-num-seqs 16 --max-num-batched-tokens 2048 --max-model-len 4096
  ```
  `--layers N` = `KIMIK3_LAYERS=N`；日志必有 `TRUNCATED MODEL`。`NO_GRAPH=1` 关 decode 图、`ASYNC_SCHED=0` 关异步调度、`KDA_PREFILL=recurrent` 关 chunk prefill、`KIMIK3_TRACE=1` 每步打 plan。

## 设计要点（与 Qwen3.5 executor 的差别）

| 项 | Qwen3.5 | K3 |
|---|---|---|
| 公共件 | `PlanBuffers` / `StateSlotTable` / `DecodeGraphs` / `AsyncResultRing` | **同四件，直接用**（`executor.rs` 文件头） |
| 递归状态 | GDN，`num_slots` 个 + 1 废纸篓（GdnState 多分） | KDA，`LayerBudget.num_slots = max_num_seqs + 1`，最后一个是废纸篓（`model.num_slots() - 1`） |
| 全注意力 KV | `KvCacheSpec::FullAttention`，块数按剩余显存 | `KvCacheSpec::Mla`（head_size 576），块数 = `max_num_seqs × ceil(max_model_len / bs) + 1`（每 token 1.1 KB，不必压满；装载前就得定，MLA 层 load 时分配） |
| logits | `gather_rmsnorm` 只对采样行 | 新增 `forward_rows`（同法）；`forward` 保持每行 logits（对拍） |
| chunk prefill 工作区 | `GdnChunkWorkspace`，`t > n` 时 prepare | `KdaChunkWorkspace`（C-1 已有），同口径 |
| 块残差 AttnRes | 无 | `model.rs` 内按层维护，executor 一步一次 `forward_rows` 即可 |
| 输出流 | HF tokenizer `decode_with(skip_special)` | bytes 增量 detokenize，**特殊 token 不跳过**（XTML 标记就是特殊 token，parser 靠它切段），只滤 eos |
| 渲染 | minijinja 模板 | `xtml::build_chat_segments` → 逐段 `encode_into(allow_special)`；`--chat-dialect hf|serving` |
| eos | tokenizer eos_ids | `generation_config.json` → `<|end_of_msg|>` 163586（`[EOS]` 163585 不是终止符） |

## 观察方法 / 停止条件（机器可判定）

1. 启动日志：`TRUNCATED MODEL`、`kimik3 model loaded`（层数 / 显存 / 秒）、`kimik3 engine started`、`startup timing`；无 panic。
2. `curl :8014/v1/models` 有 `kimi-k3`；`curl :8014/metrics` 有 Prometheus 文本。
3. `/v1/chat/completions` 流式一条（`scripts/dev/chat.sh http://127.0.0.1:8014 "你好" --stream`）：有 delta、`finish_reason` 为 `length` 或 `stop`，**内容乱是预期**（5 层）。
4. `bench --url :8014 --concurrency 1 --num-requests 8 --prompt-len 128 --max-tokens 128`：8 条全部成功，记 TPOT / TTFT（只记录，不进趋势 CSV：5 层无可比性）。
5. 日志无 `CUDA error` / `panicked`；`graph_steps > 0`（decode 走了图）；c8 一轮看混批 / 槽归还不炸。
归档 `test/kimik3/2026-10-20/smoke5/`。迭代上限 8 轮。

## 轮次记录

| 轮 | 改动 | 结果 |
|---|---|---|
| 0 | 本机：kimik3 lib / bin 构建、clippy、`cargo test -p dinfer-kimik3` 24/24（含 chat_parity 5）、ref-tests `--no-run` 过 | 等 B300 |
