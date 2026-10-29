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

## E1 2026-10-29 定稿设计（开工版，用户：模型相关配置放各自 crate，公共部分直接改 core）
**core（模型无关）**
- `runner::ModelExecutor` 新增两个可选方法（默认实现 = 不支持）：
  - `fn begin_prefill_async(&mut self, plan: &BatchPlan) -> Result<Option<Box<dyn Any>>>`：把 prefill 子批入队到执行器自己的副流，**不等**，返回句柄；`None` = 这次不能异步（副流忙 / 不支持），调用方退回同步。
  - `fn harvest_prefill(&mut self, h: Box<dyn Any>, gen: &mut Vec<Vec<u32>>, dr: &mut Vec<Vec<u32>>) -> Result<Option<Box<dyn Any>>>`：非阻塞查完成事件；完成 → 读回采样、返回 `None`；未完成 → 原句柄返还。
- `Runner`：
  - `try_execute_pair` 在 `PREFILL_ASYNC=1`（env，后续改 CLI）且 `late.is_none()` 且 prefill 子批全是 `end_of_seq`（最后一个 chunk；中间 chunk 没有 token 可收割，照旧同步）时：decode 子批 `forward_sample`（图），prefill 子批 `begin_prefill_async`；成功则把 prefill 行**从本步 Pending 里剔除**（`req_ids`/`rows` 只留 decode 行），prefill 的 (rid, n, num_draft) 存 `self.late`。
  - `finish()` 末尾：若 `self.late` 有 → `harvest_prefill` 非阻塞；完成则把这些请求的 token 追加进本步 `ModelRunnerOutput`（`req_ids`/`sampled`/`drafts`/`logprobs` 各追加一项）并更新 `states[rid].token_ids` / `num_computed`。
- `Scheduler`：不加状态位。"已调度未收割"的请求天然 `num_computed_tokens == num_tokens`（prefill 整段已记成算完），`schedule()` 的 running 循环算出 `num_new == 0` → `continue`（已有逻辑），不会再被调度；`update_from_output` 按 mro 的 `req_ids` 驱动，晚到的请求照常 `append_output` + 停止判定。只加一条防御：抢占受害者选择跳过 `num_new == 0` 的 running 请求（它的 KV 还在被副流写）。
**kaon3 crate（模型相关）**
- `KaonExecutor::begin_prefill_async`：`alt` 存在且 `self.late_busy == false` → `swap_alt`；`enqueue(prefill, false)`；`record_event` → 句柄 `{ rows, done: Event }`；`swap_alt` 回；`late_busy = true`。不再让主流 `wait_event(prefill_done)`，也不让副流等 `main_done`（两流只共享只读权重与不同 slot 的 KV，无需同步）。
- `harvest_prefill`：`done.is_done()?` 假 → 返还；真 → `swap_alt`；`collect(rows)`；`swap_alt`；`late_busy = false`；按 `logits_row_start` 填 gen/dr。
- 流优先级：第一版不做（cudarc 0.18 没暴露 `cuStreamCreateWithPriority`，CudaStream 字段私有）；若 A/B 显示 decode 图被 prefill 挤（decode 步 exec 从 8 ms 涨到 ≫ 8），再在 kernels crate 加 `Stream::with_priority`（走 sys）。
**门禁 / 观察**：`envab_c` ABAB `PREFILL_ASYNC=0/1`，c8–c64 两轮；HOST_TIME 看混批步 exec 应降到 ≈ 纯 decode 步 + 收割步出现；TTFT 不升 > 5%；固定 prompt 输出同；核心单测 + `cargo test -p dinfer-kaon3`。

## E1 落地（2026-10-29，commit 见下）
- core：`ModelExecutor::{supports_prefill_async, begin_prefill_async, harvest_prefill}`（默认不支持）；`Runner.late: Option<LatePrefill>`；`try_execute_pair` 在 `PREFILL_ASYNC=1` 且副流空闲且 prefill 行全是末 chunk 时：prefill 子批 `begin_prefill_async`，decode 子批照常走图，本步 Pending 只含 decode 行；`finish()` 末尾（含空步）`harvest_late` 非阻塞收割 → 追加进 `ModelRunnerOutput`。scheduler 只加抢占受害者守卫（跳过"已下发未收割"的请求）。`runner step mix` 日志加 `late_begin / late_harvest`。
- kaon3：`late_busy` + `LateHandle{rows, done: Event, starts}`；`begin_prefill_async` = swap_alt → enqueue(eager) → record_event → swap 回（**不**等 main_done，主流**不**等 prefill_done）；`harvest_prefill` = `done.is_done()` → collect → 填 gen。
- 流优先级未做（cudarc 0.18 不暴露）。第一轮 A/B 看 decode 步 exec 是否被 prefill 挤。
- 链 b62：A = `XSTREAM=0`（现状）/ B = `PREFILL_ASYNC=1 XSTREAM=1`。

## E2 预登记判据（b62 出结果前）
- 合入默认开的条件：两轮同向 c16 TPOT −8% 以上、c64 −8% 以上；TTFT 不升 > 5%；c1/c8 不退化 > 2%；固定 prompt 输出同；B 侧 `runner step mix` 的 `late_begin` ≈ 混批步数、`late_harvest == late_begin`。
- 收益显著小于上限（c16 1.7 / c64 8.9 ms）的解释顺序：① decode 图被 prefill 副流挤（HOST_TIME decode 步 exec 在 prefill 期间涨多少）→ 流优先级 / 限 grid；② `can_late` 条件没命中（中间 chunk / 副流忙）→ late_begin ≪ mixed；③ 收割晚一步的 TTFT 代价。
- 任一退化 → 默认留关，记负结果进 CSV。

## E2 2026-10-29 b62 第 1 轮：A 侧正常（c16 9.50 / c64 19.62）；**B 侧（PREFILL_ASYNC=1 XSTREAM=1）serve 起来后 bench 全空**（c8–c64 四档 `prefill: none || decode: none`，固定 prompt 无输出）→ 请求进了引擎没出来。链已停，先定位（serve 日志 / 是否 panic / 是否 late 永不收割 → 请求卡在 awaiting）。
- **E2 根因**：`engine step failed: req 12: in-flight token without last_row`。late 路径把 prefill 行从本步 Pending 剔除后，runner 的 `states[rid].num_computed` **没推进**（原 pair 路径由 `finish` 的 `st.num_computed += base` 推进），而 scheduler 在 schedule 时已记成算完 → 下一步 scheduler 给它排 1 个 decode token，runner 这边 `start + 1 == len + 1` 被当成异步在飞行、要 `last_row` → ensure 失败 → 整步失败、后续请求全挂。**修**：`begin_prefill_async` 成功后立刻 `st.num_computed += n − num_draft`（与 scheduler 同步）；收割只 `+= 接受的 draft 数` 并 append token。
- 单测：`late_prefill_advances_state_and_harvests`（Echo 式 mock 执行器：`supports_prefill_async` 真、`begin` 存句柄、第二次 `harvest` 才完成）。
- **单测抓到第二个 bug**：scheduler 对"已下发未收割"的请求没有任何标记，`num_computed == num_tokens` 且无 placeholder 时 `num_new = num_tokens − computed = 0`… 实测它仍排了 1 个 token（`num_tokens_with_spec + placeholders − computed` 在首 token 未 append 前确实是 0——但 running 循环里 prefill 整段算完的请求下一步 `num_new` 应为 0；测试显示排了 (2,1)，说明 computed 口径与我以为的不同：scheduler 的 `num_computed_tokens` 是 "本步结束后"值 = num_tokens，而 `num_tokens` 在 scheduler 侧… ）。不再推断：**显式集合** `Scheduler.awaiting: IdHashSet`，runner `take_late_started()` → engine `mark_awaiting` → `schedule()` 跳过（不排 token、不当受害者）→ `update_from_output` 收到 token 时移除。单测 `late_prefill_advances_state_and_harvests` 覆盖 begin / 跳过 / 收割 / 恢复 decode 四步。

## ★★★ E3 2026-10-29 b62 定稿（A = XSTREAM=0 现状 / B = PREFILL_ASYNC=1 XSTREAM=1；二进制 eff4408b；两轮）
| c | A tpot / ttft / tps | B tpot / ttft / tps | Δ |
|---|---|---|---|
| 8 | 7.08 / 94 / 949 | 7.28 / 87 / 934 | TPOT **+2.8%**，TTFT −8%，tps −1.6% |
| 16 | 9.46 / 116 / 1424 | 9.58 / 108 / 1410 | TPOT +1.3%，TTFT −7%，tps −1% |
| 32 | 12.28 / 193 / 2115 | 12.77 / 149 / 2089 | TPOT **+4%**，TTFT **−23%**，tps −1.2% |
| 64 | 19.73 / 206 / 2814 | 19.52 / 220 / 2776 | TPOT −1%，TTFT +7%，tps −1.4% |
- 机制正确：`late_begin=50/51 == late_harvest`（77 混批步里 2/3 走了 late，其余是中间 chunk → pair）；输出逐字同；无 ERROR。
- **收益没出来，判负（默认留关）**。解释（E2 预登记 ①）：prefill 副流（2148 tok 的 p64 GEMM，grid 5052 CTA）把 SM 占满，decode 图在它旁边跑不动——decode 行不再"等 prefill 完"，但"和 prefill 抢 SM"结果一样慢，还多付两流同步 + 收割晚一步。TTFT c8–c32 降是因为 prefill 不再排在 decode 图后面；c64 TTFT 升是 late 只允许一个在飞、第二波 prefill 要等。
- **下一步只有一条路**：让 prefill 让 SM —— (a) prefill 流低优先级 + decode 流高优先级（硬件抢占粒度是 CTA 边界，p64 一个 CTA 几十 us，decode 图 ~800 个小 kernel 应能插进去）；(b) 不够再限 prefill grid（分块 launch）。cudarc 0.18 没暴露 `cuStreamCreateWithPriority`，在 kernels crate 用 sys 直接建流（`Stream::with_priority`）。
- CSV：b62 记负结果（c16 +1.3% / c32 +4%）。

## E4 2026-10-29 流优先级（E3 的唯一出路）
- kernels crate：`Device::new_stream_with_priority(prio)`——cudarc 0.18 没暴露 `cuStreamCreateWithPriority`，`CudaStream` 字段 `pub(crate)`；用 `sys::cuStreamCreateWithPriority(NON_BLOCKING, prio)` 建流，再按 `CudaStream { cu_stream, ctx }` 的内存布局 `transmute` 成 `Arc<CudaStream>`（`#[repr(Rust)]` 两字段，与 cudarc 同版本锁定 0.18.2，加 `size_of` 断言；Drop 走 cudarc 自己的 destroy，正确）。`Device::stream_priority_range()` 给 (least, greatest)。
- kaon3：`MAIN_STREAM_PRIO=1` 时主流（decode 图）用 greatest（数值最小 = 最高）优先级建，alt（prefill）用默认（最低）。只在 kaon3 executor 构造处生效，其它模型不变。
- 原理：CUDA 流优先级在 CTA 边界抢占——高优先级流的 pending CTA 先于低优先级流的 pending CTA 被调度。p64 一个 CTA 几十 us、5052 个 CTA 排队 → decode 图的 ~800 小 kernel 应能插进去。
- 链 b63：A = `PREFILL_ASYNC=1 XSTREAM=1`（E1 无优先级，= b62 B）；B = `PREFILL_ASYNC=1 XSTREAM=1 MAIN_STREAM_PRIO=1`。看 B 的纯 decode 步 exec 在 prefill 期间是否接近无 prefill 时（HOST_TIME 的 decode 步 p90 从 12.4 降），TPOT 是否真降。
- **E4 第一版 SIGSEGV**（b63 r1 B serve 起到 "main stream prio = greatest least=0 greatest=-5" 后 Segmentation fault；本机 3090 `stream_prio_smoke` 复现）：按 cudarc 源码字段顺序 `transmute` 构造 `CudaStream`——`repr(Rust)` 字段顺序不保证，ctx 指针被当成流句柄。改法：正常 `new_stream()` 拿合法 `Arc<CudaStream>`，sys 建优先级流，在两个指针宽的字里**按值**找旧句柄、销毁旧流、原地替换（字段顺序无关）。`crates/kernels/tests/stream_prio_smoke.rs` 作门禁。
- E4 v2 二进制 8eb03496，b63 重跑中。判据除 E2 预登记外，加一条机理判据：B 侧 HOST_TIME 纯 decode 步 **p90**（A 侧 12.4 ms，无 prefill 干扰时 ~8.5）必须明显回落，否则优先级没起作用（CUDA 流优先级只影响 pending CTA 的调度顺序，若 prefill CTA 已驻留满 SM，decode CTA 仍要等它们退出）。

## E5 2026-10-29 b63 第 1 轮 B 侧（MAIN_STREAM_PRIO=1，二进制 8eb03496）中途读数
| c | A（E1 无优先级） tpot / ttft | B（+优先级） tpot / ttft | 混批步 exec A→B |
|---|---|---|---|
| 8 | 7.30 / 86 | 7.28 / 106 | 44 → 18 ms |
| 16 | 9.21 / 108 | **8.96 / 110** | 58 → **44 ms** |
| 32 | 12.43 / 152 | **11.87 / 228** | 70 → 71 ms |
- **优先级起作用了**：c16 TPOT −2.7%、c32 −4.5%（b62 的 E1 是 +1.3% / +4%，现在反过来）；c8 混批步 exec 44 → 18 ms 说明 decode 图真的插进了 prefill 之间（混批步的 host exec 现在只量 decode 子批）。
- **代价**：TTFT 升（c8 86 → 106、c32 152 → 228）——prefill 被压到低优先级，首 token 晚。这是 TPOT↔TTFT 的直接 tradeoff；我们 TTFT 本来反超 vLLM 很多（c32 192 vs 231），有空间让。
- 等 r1 c64 + r2 定稿；若两轮同向，合入时要同时看 TTFT 不超 vLLM（c32 228 vs vLLM 231 刚好贴边）。
