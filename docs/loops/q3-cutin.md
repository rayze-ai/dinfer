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
