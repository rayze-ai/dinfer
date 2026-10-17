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
| 3 | 4090 验收 | baseline PASS（refs 22 / parity 8 / 单测 8 / e2e 6）；副套分配 9 ms；KV 17.0 GiB（主套激活 +3.4 GB 被副套复制，KV 从 ~14.5 → 17.0 GiB 是因为 utilization 算法变化？待查）。bench17：c1 1.12× / c8 1.06× / c32 1.02× / c8-long 1911 = 1.00×，**TPOT 3.9 不变**；step breakdown：4000 步里 **pair=16**，eager=44 | 并行路径几乎没被走到：预算 2048 下一个 1024 prompt 要么独占 step（纯 Prefill，无 decode 可并行），要么切块后混批（才拆 pair）。"同一 step 凑齐两类行"的前提在当前调度下很少成立 | 下一步（调度侧）：scheduler 产出两个 SchedulerOutput（decode 组 + prefill 组）而不是一个混合批 —— 即 decode 批不再受 prefill 占预算影响、每步都并行；或更简单：prefill 请求单独一步时也把当时 running 的 decode 行拉进同一步（预算只约束 prefill token）。这是对 vLLM 调度模型的真正偏离，需单独一轮 |
| 4 | 可读性整理 | runner 拆 `try_execute_pair / is_decode_row / filter_output`；executor `on_alt / push_tokens / PairLogprobs`；删 v3 WMMA 死代码 302 行；clippy 只剩 1 条历史 warning（model forward 13 参数）。**回归**：core 单测 2 个失败 "executor returned 0/0 entries"——整理时把单 plan 的 `forward_sample` 回退分支丢了 | 提取 helper 时只保留了 pair 分支 | 补回 else 分支，重跑 core 单测 + parity |
| 5 | 回归修复 | 补回 `forward_sample` 单 plan 分支；core 33/33、real_parity 3/3 | — | 4090 baseline 重跑后进调度侧改动 |
| 6 | 调度侧：decode 免预算 | `SchedulerConfig.decode_exempt_from_budget`（默认开，`--no-decode-exempt` 关）：running 循环里 decode 行（1 token 且已有历史）不扣 `max_num_batched_tokens`，预算只约束 prefill token → 每步都能同时装下全部 decode 行 + 预算内 prefill，pair 路径才能每步命中。executor `max_tokens` = 预算 + max_num_seqs。新单测 `decode_exempt_from_budget_coschedules_decode_and_prefill`；core 34/34 | bench17 显示 pair 只命中 16/4000 步，根因是 vLLM 式单预算下 decode 与 prefill 很少同步 | 3090 parity → 4090 bench19 |
| 7 | 4090 bench19 | 免预算 + 跨流：c1 1.12× / c8 1.07× / c32 1.02× / c8-long 1913 = 1.00×，TPOT 3.9 不变；**pair 仍只 18/4000 步**；baseline 里 `serve_e2e_qwen3::e2e_qwen3_real_model_chat` FAIL（3090 复现，XSTREAM=0 也失败 → 与跨流无关，是免预算或可读性整理引入） | ① mixed step 本来就少：bench 闭环并发 8，每个 prompt 1024 token 在预算 2048 下 1 步算完，一次 bench 只有 32 个 prefill step，其余 ~4600 步全是纯 decode → 并行能省的上限 = 32 × ~20 ms ≈ 0.6 s / 32 请求 ≈ 对 TPOT 影响 ~0.15 ms，**与 3.9 → 3.7 的差距同量级但不够**；② e2e 失败待查 | runner 加 step 构成计数（mixed / pair / prefill_only 每 1000 步）；修 e2e；再评估是否继续 |
| 8 | e2e 排查 + 收益上限 | 3090 e2e FAIL 是 `QWEN3_GOLDEN` 相对路径找不到 golden（测试环境问题，绝对路径后 PASS）；4090 的 FAIL 原因另查。**收益上限估算**：bench c8-long 32 请求 × 1024 prompt，预算 2048 下每 prompt 1 个 prefill step（~20 ms），共 32 步；其余 ~4600 步纯 decode。跨流能藏的最多就是这 32 × 20 ms = 0.64 s，分摊到 32 × 128 个 decode token ≈ **0.15 ms/TPOT**（3.9 → 3.75）。pair 实际只命中 18 步，因为 bench 闭环下新请求到达时常常正好没有别的 decode 在跑（都在等它） | 跨流并行的价值取决于负载里 prefill 与 decode 的重叠程度；闭环固定并发的 bench 让两者天然错开。真实在线流量（泊松到达、长短混合）重叠会多得多 | 保留实现（默认开，对 c1/c8/c32 零影响）；c8-long 的 3.9 vs 3.7 另找原因（decode 步本身：B=8 时 GEMM gemv 2.5 ms 是带宽下限？vLLM 3.7 说明不是）→ 看 decode step 的 nsys 对比 |
| 9 | e2e 复核 | 4090 GPU 3 单独重跑 `serve_e2e_qwen3` 6/6 PASS。baseline 里那次 FAIL 原因未在日志里留下 panic 行（`run()` 只 grep 了 test/result/panicked），疑似 `e2e_qwen3_auto_num_blocks` 与前面测试进程在同卡显存上抢占（parity 刚结束未释放）；下次 baseline 若再现，把 e2e 单独跑并保留全量日志 | — | **Q3-j 阶段结论**：实现正确（parity / e2e / 单测过），对 c1/c8/c32 零影响，c8-long 在闭环 bench 下收益上限 ~0.15 ms TPOT、实测 0；默认保留开启，收益要在真实到达模式（泊松、长短混合）下评估 → 建一个 `bench --arrival poisson` 模式再测（后续）。c8-long 3.9 vs 3.7 的差距转去看 decode step 本身 |
| 10 | 开环 bench | `bench --request-rate R --prompt-len-mix 128:0.5,1024:0.35,2048:0.15`（泊松到达、不等完成、决定性 seed）；`scripts/verify/bench_poisson.sh`：dinfer XSTREAM=1 / =0 分别重启 + vLLM，rates 8/16/24 req/s × 96 请求 | 闭环 bench 看不到 prefill/decode 重叠（第 7–9 轮） | 4090 跑一轮，看 xs1 vs xs0 的 TPOT 差与 vLLM 对比 |
