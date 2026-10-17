# Q3-j 跨流 prefill/decode 并行（loop 规格）

> 目标：同一 step 里 decode 批与 prefill 批放两条 CUDA stream 并行执行。decode 带宽型、prefill 算力型，单卡可重叠。
> 预期：c8-long TPOT 3.9 → ~3.0 ms（decode 不再被 20 ms 的 prefill step 打断），TTFT 不变；c1/c8/c32 不变（无混批时退化为单流）。
> 这是 vLLM 单流单批做不到的（design-qwen3.md §6.2-2），也是 Q4 PD 分离的单卡前置。

## 环境

- 4090 常驻 dinfer :8012（`--max-num-batched-tokens 2048`）vs vLLM :8013；测试 GPU 3。
- 本机 3090：`kernels.sh`、`real_parity`、`cold_shape`。

## 设计（最小改动）

1. **Runner 拆 plan**：`build_plan` 后若 `mode == Paged` 且同时含 q_len==1 行与 q_len>1 行，拆成 `plan_decode`（全部 q_len==1，mode=Decode）+ `plan_prefill`（其余，mode=Prefill 若全无历史 / Paged 否则）。`ModelExecutor` 新增 `forward_sample_pair(decode, prefill, ...)`，默认实现 = 顺序调用两次（保持其他 executor 不变）。
2. **Model scratch 双份**：`Qwen3Model` 的 per-step 缓冲（hidden / normed / attn_out / mlp_out / final_normed / logits_bf16 / decode_scratch + 每层 qkv_buf / attn / gate_up）抽成 `Scratch` 结构；model 持 `scratch[0]`（主流，容量 max_tokens）+ `scratch[1]`（副流，容量 max_tokens；显存 +~100 MB，Qwen3-0.6B 可接受）。`forward(..., scratch_idx)`。权重 / KV / rope 共享。
3. **Executor 双流**：主流 = 现有 `stream`（decode 图在此捕获）；副流 `stream2` + 第二套 `StepBuffers` / logits / work / raw + 第二个 cuBLASLt 句柄（workspace 不能共享）。`forward_sample_pair`：
   - 副流 `wait(event_main_prev)`（上一步主流结束，KV 一致）；
   - 主流：upload decode → graph launch → 采样；副流：upload prefill → forward(scratch 1) → 采样；
   - 主流 `wait(event_sub_done)`；两边 D2H 结果；合并 `generated` 顺序 = 原 plan 顺序。
4. **KV 一致性**：decode 读的块都是之前 step 写完的（prefix cache 只命中完整块；本 step prefill 写的新块 decode 不读）→ 无 RAW 冲突。prefill 行的 `qkv_post` 写 cache 与 decode 读 cache 的块不相交。唯一共享写：无（logits 行分开）。
5. 回退：`XSTREAM=0` 环境变量关掉，走原单流路径（对拍 / 排错用）。

## 观察方法

- `cold_shape`-级别不适用；直接看 4090 `bench_compare` c8-long 的 TPOT 与 TTFT，`/metrics` 的 `engine_step_execute_seconds` 直方图（50–100 ms 桶应消失或变少）。
- 正确性：`real_parity`（并发混批用例）+ `serve_e2e_qwen3` + 新增 runner 单测 `split_mixed_plan`。

## 停止条件（机器可判定）

1. 3090 `kernels.sh` 22/22、`real_parity` 3/3、`cargo test -p dinfer-core`（含新单测）全过。
2. 4090 `remote4090_baseline.sh` PASS。
3. 4090 bench：c8-long TPOT ≤ 3.5 ms（vLLM 3.7）且 out tok/s ≥ 1.05×；c1/c8/c32 不退（±2%）。
4. `q3.sh` exit 0。
迭代上限：6 轮 bench。不达则记录原因，回退默认关。

## 已知可能出错点

- cuBLASLt 句柄跨流：同一 handle 可在不同流上用，但 workspace 必须分开 → 第二个 `Gemm`。
- 副流上 `cudaFuncSetAttribute`（v5 动态 smem）是进程级，无需重复。
- 事件：cudarc `Stream::wait(&Event)`；记录点在每步末尾（主流）与副流 forward 末尾。
- CUDA graph 捕获期间副流不得活动（捕获在启动时完成，运行期无影响）。
- decode_scratch（split-K partial）两套；`gqa_mixed` 在副流上 prefill 行不再含 q_len==1（已被拆走）→ 副流用 `gqa_paged_bf16(skip_single=false)` 即可。

## 记录

| 轮 | 块 | 现象 | 根因 | 处理 |
|---|---|---|---|---|
| 0 | 规格 | 2026-10-17 写规格 | — | 开工：model Scratch 抽取 |
| 1 | 实现 | `Stream::record_event/wait_event`（kernels device.rs）；`Mlp::swap_bufs` / `Attention::swap_bufs` / `Qwen3Model::{alloc_scratch,swap_scratch}`（ModelScratch 整套激活缓冲）；executor `AltSet`（副流 + Gemm + StepBuffers + logits/work/raw + scratch）、`enqueue/collect/account` 拆分、`forward_sample_pair`（主流 decode 图 → 副流等 main_done 事件 → prefill eager → 主流等副流事件 → 各自 D2H）；runner `supports_pair` + Paged 混批拆 so → 两个 plan → 结果按原顺序散回；`XSTREAM=0` 关。release 编译过 | — | core 单测 + real_parity(两种 XSTREAM) → 4090 |
| 2 | 本机验收 | 3090：`real_parity` 3/3 在 XSTREAM=1 与 =0 下都过（并发混批用例走了 pair 路径，logits 与 HF 一致）；`cargo test -p dinfer-core` 33/33 | — | 4090 baseline + bench17 |
