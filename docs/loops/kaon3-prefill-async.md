# loop: kaon3-prefill-async —— 混批步里 decode 不等 prefill（prefill 异步收割）

> 来源：loop `kaon3-decode-opt` D219（2026-10-29）。五刀后 c16 混批步 65 ms（prefill 33 tok/ms），但 pair 步里 decode 子批（9 ms）
> 要等 prefill 副流（65 ms）完才 publish → 10 个在飞请求每 64 步被卡一次 65 ms ≈ 每步 +1.0 ms（c16）、c64 每步 +2.3 ms。
> 这是对 vLLM 剩余差额（c16 1.5×）里**调度结构**那一半；另一半是 prefill 步 GEMM 本身（MoE p64 57 TFLOPS 还有 1.5–2×）。

## D224 修正（2026-10-29 b56）
单流混批（`XSTREAM=0`）≥ pair 步（全档 0.4–1.3%）：pair 的"prefill 副流 ‖ decode 主流图"在 c16–c64 没拿到并行（prefill 65 ms 把 SM 占满，decode 图挤不进），还多付两流同步。
所以**基座改为单流混批**；异步收割要解决的不是同步点，是"prefill 跑着的时候 decode 图能不能真的并行推进"——需要：
(a) decode 流 `cudaStreamCreateWithPriority` 高优先级、prefill 流低优先级；(b) prefill 的 grid 不要一次把 SM 填满（p64 的 grid.y = R/64 + E 对 2148 tok 是 421 × 6 × 2 = 5052 CTA，占满 170 SM ×2 CTA 多轮；可能要分块 launch 让调度器交错）；(c) 收割语义同下。
先做一个**最小探针**：prefill 流低优先级 + decode 流高优先级，不改收割，只看 pair 步 exec 是否从 65 ms 降（decode 子批被优先调度 → decode 行先 publish 仍要等… 不会降）。所以探针无意义，直接做收割 + 优先级。

## D236 补充量化（2026-10-29）
混批步时间 ≈ prefill token 数 / 37 tok/ms，与 decode 行数无关（3800 tok：11 请求 95 ms、55 请求 101 ms）。vLLM 同形状 1500 tok 步 ~30 ms（50 tok/ms）。
→ 收割收益上限 = 混批步全部时间（c16 1.65 ms/步、c64 7.8 ms/步 摊到 TPOT）；prefill GEMM 再提速 1.35× 则另拿一半。两条路并行：本 loop 做收割，MoE p64 / attention prefill 另起。

## 目标
pair 步只等 decode 子批；prefill 子批在副流上跑完后，其首 token 在**之后某个 decode 步**的 publish 里并入（"晚到结果"）。
decode 步照常走 CUDA graph；prefill 期间 decode 不停。

## 环境
5090 容器 `kaon3-full` GPU0，`envab_c.sh` 口径（c8–c64、400 tok / 64 out / 8c 请求、两轮交错换 seed、HOST_TIME 分组、固定 prompt 输出）。
vLLM 基线 :28201（GPU2）。stepA5 的五刀二进制为 A 侧基线。

## 设计（core 改动，模型无关）
1. `runner::Pending` 加 `late: Option<LatePrefill>`：`try_execute_pair` 不再 `collect(rows_p)` 等 prefill；记录 prefill 子 plan 的 (req_ids, rows, event)。
2. `Runner::begin` 返回的 Pending 只含 decode 行的 `rows`；prefill 行留在 `self.late_queue`（按 event 排）。
3. `Runner::finish(pend)`：先按常规拿 decode 结果；再**非阻塞**检查 `late_queue` 队首 event（`cudaEventQuery`），完成的取 argmax、拼进 `ModelRunnerOutput`（req_ids / sampled 追加），未完成的留着。
4. 调度器：prefill 行在"已算完 / 未收割"期间 `num_computed_tokens` 已推进（begin 时就推），但不能再被调度出 token（标 `awaiting_result`），`update_from_output` 收到晚到结果时清标并追加 token。`finished_req_ids` / 抢占要排除 awaiting 行。
5. 多个 prefill 子批同时在飞：alt 流只有一套缓冲（`AltSet`），**同一时刻只能有一个 prefill 在飞**；下一个混批步若 prefill 还没完 → 这一步只调 decode（waiting 继续等）。等待期间 decode 步仍是纯 Decode 模式（走图）。
6. KV：prefill 写的 slot 与 decode 不重叠（已成立）；prefill 读的 KV 都是自己的。alt 流与主流只在 (a) prefill 开始前等主流上一步 (b) 主流在收割那步前等 prefill_done 事件——(b) 从"每步"变成"收割那一步"。

## 观察
- `HOST_TIME=1`：混批步 exec 应降到 ≈ 纯 decode 步（9–12 ms）；多出一类"收割步"（含 prefill 首 token）。
- bench：c16 TPOT 目标 10.8 → ≤ 9.8（−1.0）、c64 21.9 → ≤ 19.6（−2.3）；TTFT 不许升 > 5%（prefill 总时长不变，收割晚至多一个 decode 步 ~10 ms）。
- 输出：固定 prompt 两条逐字同；c≥2 并发输出与 A 侧比（批不变性口径）。

## 停止条件
① `envab_c` 两轮同向：c16 TPOT −8% 以上、c64 −8% 以上、TTFT 不升 > 5%；② c1/c8 不退化 > 2%；③ 输出一致；④ 核心单测（`cargo test -p dinfer-core`）全过。达不到：回滚到 `--xstream-late 0`（默认关）并记负结果进 CSV。

## 记录
E1 …（接 kaon3-decode-opt 的 D 编号体系，本 loop 用 E）
