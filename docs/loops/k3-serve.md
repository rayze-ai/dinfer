# Kimi-K3 里程碑 C-4b：executor + service（ChatBackend）+ `dinfer --features kimik3 serve`，前 N 层真权重冒烟

> 状态：**冒烟通过（2026-10-20 第 2 轮，B300 GPU 2，`scripts/dev/k3_b300_smoke.sh` exit 0）**。crate 侧 `models/kimik3/src/{executor,service}.rs`，入口 `bins/dinfer/src/main.rs` 的 `kimik3` 分支。
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

## TODO（C-4b 之后）

- **decode CUDA graph 现在默认关**：要开必须去掉两处 host 同步——LatentMoe 的 `offsets` D2H（改 GPU 侧 tile 计划，`k3-moe.md` TODO）和 AttnRes 的 host softmax（融合 kernel，`k3-moe.md` C-2 TODO）。都做完后把 `K3_GRAPH` 默认翻回开并复测。
- `OutputStream` 没拿到本请求的 render 结果：`reasoning_effort=none` 的请求 parser 初态仍是 Think（靠 `fresh` 吞标记兜底）；server 侧若加"render → stream 钩子"再接。
- `tool_call_tags()` / `grammar_factory()` 返回 None：guided / 强制 tool_choice 未接（executor 也 bail `any_mask`）。
- 采样流水（`upload_sampling` / `sample_eager`）与 Qwen3 / Qwen3.5 三份同文：下一个公共件候选（`exec::Sampler`）。

## 轮次记录

| 轮 | 改动 | 结果 |
|---|---|---|
| 0 | 本机：kimik3 lib / bin 构建、clippy、`cargo test -p dinfer-kimik3` 24/24（含 chat_parity 5）、ref-tests `--no-run` 过 | 等 B300 |
| 1 | B300 GPU 2，`k3_b300_smoke.sh`（5 层） | 装载 5 层 + 顶层 **13.0 s**（layer 0 0.5 s、MoE 层各 2.3 s、MLA 层 3.0 s；health 可达 14.1 s），显存 82.2 GB；`/v1/models` 过；**首请求炸**：`graph capture: body: LatentMoe: pairs 0 ≠ t·K`——LatentMoe 每步 `dtoh_sync_copy(offsets)` 展平 tile 计划、AttnRes host softmax，体内 D2H 不能进 capture，引擎线程退出 → chat 无 chunk、/metrics 空、bench 失败 |
| 2 | executor：decode 图默认关（`K3_GRAPH=1` 强制开），其余不变 | **`k3_b300_smoke.sh` exit 0**（`test/kimik3/2026-10-20/smoke5/`）：装载 11.7 s（health 可达 14.1 s）、显存 85.5 GB（权重 ~75 + 激活 / KV / 采样缓冲；nvidia-smi 82.7 GB）；`/v1/models` = `kimi-k3`；流式 chat 66 chunk、`finish=length`、265 字 reasoning（乱文，预期）、content 0（5 层永远出不到 `<\|close\|>`）；`/metrics` 13 KB Prometheus（`dinfer:*`，脚本首版 grep 漏了冒号 → 0 行，已修）；**bench c1×8 8/8 ok：TPOT 12.5 ms、TTFT 386 ms**（p128：prefill 走 KDA chunk + MLA 展开 + MoE 解包）；**c8×16 16/16 ok：TPOT 42.7 ms、TTFT 2.2 s**（eager 单流、无图；仅记录，5 层数字不进趋势 CSV）；serve.log 无 ERROR / panic / CUDA error / slot 复用告警 |
| 回归 | `scripts/verify/k3_chat.sh`（本机）/ `K3_SKIP_GEN=1 K3_GPU=2 scripts/verify/k3_model.sh`（B300） | 均 exit 0：chat 24/24；model 隔离口径 5 层 cos ≥ 0.99998，链式 final 0.99882 / logits 0.999895 / decode-vs-prefill 0.999973 / top-1 一致（`forward` 改走 `forward_rows` + 单位行表后数字不变；日志 `bootstrap/k3serve/k3_model_rerun.log`） |
| 3 (C-6 首试) | B300 8 卡独占，`k3_c6_tp8.sh serve`（tp8 全 93 层，max-model-len 8192 / seqs 64 / budget 16384，graph 关） | 装载正常推进（rank 0–7 分片装载，专家 MXFP4 packed 每 rank 112 个 1.97 GB/层，切分正确），**但每层 15.7 GB/卡（设计 §2.4 估 2.3 GB/层/卡），10 层 148 GB/卡 → 推算第 18 层 OOM（288 GB）**，且每层 12–13 s（93 层 ≈ 20 min > 脚本 health 等待 900 s）。主动停。对照：单卡 smoke5 19.16 GB/层（packed 15.72 + 3.4）、tp2 10.4 GB/层（packed 7.86 + 2.5）、tp8 15.7（packed 1.97 + **13.7**）→ 多出的部分**随 world 变大而变大**，不是权重；嫌疑 per-layer 激活 / TP 归约 scratch 按 max_tokens（16384+）× world 或按全量专家分配。待查 `latent_moe.rs::load_tp` / `layer.rs` 的 `alloc_*` 清单 | ❌ 停，查分配 |
| 3b | 根因定位（读 `models/kimik3/src/{kda,mla,layer,latent_moe}.rs` 的 `Tensor::zeros` 清单）：**每层各持一套按 `max_tokens` 的激活缓冲**——KDA 13 块 `[tt, proj=12288]`、MLA 16 块 `[tm, …]`、LatentMoe `pairs = max_tokens × topk(8)` 四块 `[pairs, latent 3584 / 2·inter / inter / latent]` + 7 块 `[max_tokens, …]`、AttnRes 4 块。C-6 预算 16384（smoke5 是 2048）→ 全部 ×8：LatentMoe 一层 ≈ 131072 × (3584+6144+3072+3584) × 2 B ≈ **4.3 GB**，KDA 13 × 16384 × 12288 × 2 ≈ **5.2 GB**，再加 logits `[16384, 384]` f32 等 → ≈ 13 GB/层，与实测 15.7 − 1.97 = 13.7 吻合；tp 不切这些缓冲，所以"随 world 不变、随预算线性"。与 Qwen3.5 第 5H 轮同款坑（qwen35 已改 `ActScratch` 全模型共享一套，170 GB → 2.8 GB）。**修法**：kimik3 加 `ActScratch`（KDA / MLA / LatentMoe / AttnRes 四种层各取列数最大值，一套按 `&mut` 借用），93 层 ≈ 13 GB → 全模型 ≈ 0.2 GB；`kda.rs` 第 21 行注释早已写了这条待办。预计 world=8 每层回到 2.0–2.3 GB，93 层 ≈ 200 GB/卡，与 §2.4 一致 | 根因 |
