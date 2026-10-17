# Q3-o 插队 prefill（异步调度的 TTFT 代价优化，loop 规格）

> 异步调度后新请求要等 "在飞 decode 步 n 结束 → 调度 n+1" 才进 batch，TTFT 多等一步（c8-long 39 → 54 ms）。
> 做法（vLLM 没有）：步 n 在飞时若 waiting 非空，立刻 `schedule()` 一次——此时 running 的 decode 行都有 placeholder、不会被再次调度，
> 产出的 step 只含新请求的 prefill；同步跑完（主流上排在 n 之后，GPU 无空隙），结果进 `sync_done`；再 finish n。
> 新请求的首 token 提前一步出来；下一步 n+1 它已是 running decode 行。

## 风险与对策
1. 插队步的采样会把 `chosen → prev_chosen`，覆盖步 n 的结果 → n+1 的 gather 读错。对策：`BatchPlan.skip_prev_chosen`，插队步不写 prev_chosen。
2. 插队步的 `upload_plan` 调 `PinnedUploader::reset_step` 回卷 host 区，而步 n 的异步 H2D 可能还在读 → 对策：`PINNED_UPLOAD=1` 时 reset 前等上次 flush 的事件（默认关闭路径是同步 pageable，无此问题）。
3. 插队步若与别人混批（不会：running 行全被 placeholder 挡住）。
4. 开关 `CUT_IN=0`。

## 停止条件
1. core 单测、3090 real_parity、serve_e2e（ASYNC=1）全过；batch=1 串行等价性不变。
2. 4090 bench27：c8-long TTFT ≤ 42 ms（bench26 54）、c1/c8/c32 不退；开环泊松 TTFT 对比 poisson1。
## 记录
| 轮 | 块 | 现象 | 根因 | 处理 |
|---|---|---|---|---|
| 1 | 实现 | 2026-10-17 | — | engine pending 分支 + runner `begin_cut_in` + executor skip flag |
| 2 | 本机验收 | core 34/34、real_parity 3/3、serve_e2e 6/6（ASYNC=1 + CUT_IN=1）。实现细节：插队步结果占用 `sync_done` 槽后，本轮不再预排 n+1（否则要两个 sync_done）；即插队那一步之后 decode 流水有一次"空档"（n finish → publish 插队结果 → 再 schedule n+1）。代价 ≈ 一次 host 往返（~0.3 ms）只在有新请求到达的步发生，换新请求 TTFT -1 步（3–4 ms） | — | 4090 bench27 + 等价性 |
| 3 | 4090 翻车 | baseline PASS、c8-long 单测 TTFT p50 33.6（bench26 54）说明方向对；但 bench_compare 中 engine 报 `cut-in step contains an in-flight decode row` 退出 | `schedule()` 只对 "token 在飞" 的 decode 行跳过（placeholder 条件），**刚完成 prefill、首个 decode 还没调度过**的 running 行仍会被排进来，与新请求混批 → begin_cut_in 的 any_gather 断言不成立的另一种情况：它有 gather（上一步是它的 prefill，首 token 在飞） | scheduler 加 `schedule_waiting_only()`（跳过 running 循环，只接 waiting，预算全给新请求）；engine 的 sync_done 改队列（插队步 + 预排的 n+1 都能放下，不留空档）|
| 4 | 4090 bench27 | 无引擎错误。c1 510 = 1.14× / c8 3104 = 1.07× / c32 8795 = 1.06× / c8-long 1912 = 1.00×。**c8-long TTFT 54 → 43 ms（目标 ≤42 擦边）**，但 c8-long **TPOT 3.7 → 3.9**、c32 TTFT 23 → 36、c32 TPOT 3.4 持平。`serve_e2e_qwen3::e2e_qwen3_request_features` 在 3090 与 4090 都 FAIL（同一处）| ① TPOT 变差：插队步是同步 prefill，在飞 decode 步 n 结束后要等它跑完才 finish n（GPU 上两步串行、host 上多一次等待）——比不插队时 decode 被打断得更频繁；c32 TTFT 变差同理（每个到达的新请求都触发一次独占 prefill 步，取代了原来 "等一步、与别人一起 prefill" 的批处理）；② e2e 回归待看 panic 行 | 看 e2e；评估是否加门槛：waiting 累积到 ≥ k 个 / 距上次插队 ≥ t ms 才插队，或只在 running 少（低负载）时插队 |
| 5 | e2e 根因 | 断言 `", 2, 3, …"` vs `"1, 2, 3, …"`：**首 token 丢了**。插队步的结果进 `sync_done` 队列，而紧接着就 `schedule()` 了 n+1——scheduler 还没 `update_from_output(so_c)`，不知道该请求已有 1 个输出 token，把它当 "prefill 刚完" 再排一步；等插队结果最后才 update，append 的顺序错位 | 结果发布顺序与调度顺序脱节 | 插队步 begin → finish → **立刻 update_from_output + 推送**（不进队列），再 schedule n+1。另加负载门槛：在飞步行数 ≤ 8 才插队（bench27：大 batch 下插队把 prefill 打散反而伤 c32 TTFT / c8-long TPOT）|
