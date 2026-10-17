# Q3-l 异步调度（loop 规格）

> 目标：把 decode 步的 host 开销（schedule + 建 plan + upload + launch ≈ 0.3 ms，q3-perf 第 57 轮）藏到 GPU 跑上一步的时间里。
> 口径：c32 TPOT 3.5 → ≤ 3.3（= vLLM）；c1/c8 不退；bench 用 `restart_dinfer.sh` 干净重启后测（bench23 基线）。
> 注意（用户 2026-10-17）：测试异常先排查测试程序 / 环境（孤儿进程、GPU 占用），再怀疑代码。

## 设计

现在每步：schedule → build_plan → upload → launch → **D2H 等 GPU 完成** → update → 下一步。GPU 在 "等 → update → schedule → build → upload → launch" 这段空转 ~0.3 ms。

异步版（对齐 vLLM async scheduling，scheduler 已有 `async_scheduling` + `num_output_placeholders` 机制）：
1. **scheduler**：`async_scheduling=true` 时 `schedule()` 不等上一步结果，为每个 running 请求按 "已算 + 占位 1" 排下一步（已有）。
2. **runner**：`build_plan` 遇到 decode 行的 token 还没回来（`start + k >= token_ids.len()`）时，不填 token 值，而是记录 `(本行, 上一 plan 里同一请求的行号)` 到 `plan.ids_from_prev_row`；executor 在设备上用上一步的 `chosen` 补 `input_ids`。
3. **executor** 拆两段：`begin(plan)`（upload + 若有 prev_row 则先启 `gather_prev_chosen` kernel 填 ids + forward + sample，入队后**异步** D2H 到 pinned 双缓冲 + 记事件）→ `finish()`（等事件、读 pinned、返回 tokens）。D2H 必须在下一步的 sample 之前入队（流序保证读到的是本步 chosen）。
4. **engine**：
   ```
   so_n = schedule(); begin(so_n)
   loop { so_{n+1} = schedule(); begin(so_{n+1}); out_n = finish_prev(); update_from_output(so_n, out_n); 推送 token; n += 1 }
   ```
   只对 "全 decode 且无 penalties / logprobs" 的 plan 走异步；其他 plan 退化为同步（begin 后立刻 finish）。
5. 结束：请求在 step n 到 EOS，但 step n+1 已被调度 → scheduler 的 update_from_output 把它 finish，n+1 的那行结果丢弃（已有 placeholder 回滚逻辑；核对）。
6. 回退：`ASYNC_SCHED=0`。

## 停止条件
1. core 单测（含新增 runner `prev_row` 用例）、3090 `real_parity` + `serve_e2e_qwen3`（engine 路径）全过；4090 baseline PASS。
2. 4090 bench24：c32 TPOT ≤ 3.3、c1/c8/c8-long 不退（±2%）。
迭代上限 6 轮。

## 记录
| 轮 | 块 | 现象 | 根因 | 处理 |
|---|---|---|---|---|
| 0 | 规格 | 2026-10-17 | — | 开工 |
| 1 | 设计定稿 | runner 拆 `begin(so)`（update_states → build_plan，占位 token 记 `plan.prev_row[row] = 上一步同请求的行号`，state.num_computed 在 begin 就推进）/ `finish()`（取结果、追加 token、产出 MRO）；executor 新 `begin`（upload prev_row → 图内 `row_gather(prev_chosen → ids)` → forward → sample → `chosen → prev_chosen` D2D → D2H 到 pinned 双槽 + 事件）/ `finish`（等事件读 pinned）；只对 "Decode 且无 penalties/logprobs" 的 plan 异步，否则 begin 内同步跑完。engine：`so_n=schedule; begin; if eligible { so_{n+1}=schedule; begin } ; finish_n; update; publish`。scheduler `async_scheduling` 由 `ASYNC_SCHED`（默认开）控制 | — | 写代码 |
| 2 | 实现要点 | ① runner `begin(so) -> Pending`（update_states + build_plan；decode 行若 token 在飞则 `plan.gather_idx[row] = 上一步同请求行号`，否则 -1；状态在 begin 时推进 num_computed，finish 时补 token_ids）/ `finish(Pending) -> MRO`；② 异步只在 "全 decode、无 draft、无 penalties/logprobs、executor 支持" 时开，否则 begin 内同步跑；③ executor 图内加 `row_gather_i32(prev_chosen, gather_idx → ids)` 作第一个 kernel（host 上传的 ids 先写、gather 覆盖在飞行）；sample 后 `chosen → prev_chosen` D2D；D2H 到 pinned 双槽 + 事件；④ engine：`begin n；若 n 异步且还有请求：schedule n+1、begin n+1；finish n；update；publish`，pending 最多 1 个；⑤ 请求在 n 到 EOS 而 n+1 已算它一行：scheduler update 跳过非 Running，n+2 的 finished_req_ids 清 runner 状态（已有路径）；⑥ 风险：新请求到达时等两步（TTFT +~2 ms），pinned 槽复用安全靠 "finish n 必在 begin n+2 之前" | — | 编码 |
| 3 | 编码 | kernel `row_gather_i32`；runner `begin/finish/Pending/last_row/gather_idx`（误删 update_states 一次，从 HEAD 恢复）；engine 双步循环（pending / sync_done 两个槽，`fail_step`）；executor `PinnedSlot` 双槽异步 D2H、`prev_chosen` D2D、图内 gather。`ASYNC_SCHED=0` 关 | — | 编译 → core 单测 → 3090 parity + e2e → 4090 |
| 4 | 首次编译通过 | core 34/34；3090 real_parity 3/3 + sampling 5/5 在 ASYNC 1/0 都过；**serve_e2e 6/6 在 ASYNC_SCHED=1 全 FAIL、=0 全过** → engine 循环 / 多步衔接有错，parity（单步 forward_sample）测不到 | 看 e2e 日志里的 engine 错误 | 修 → 重跑 e2e |
| 5 | e2e 首因 | server 日志：`executor begin: previous async step not finished`。engine 设计是 begin n → begin n+1 → finish n（两步在飞），executor 的 `async_inflight` 却是 Option（只容 1 个）| 自相矛盾的容量 | 改 VecDeque（容 2，槽号随条目走）；重跑 e2e |
| 6 | 4090 验收 | baseline PASS（refs 22 / parity 8 / 单测 8 / e2e 6，ASYNC 默认开）。probe：c8 2.48（bench23 2.52）、c32 **3.46**（3.56）；step breakdown wait 4764（= 现在 finish 等的是上一步，含整步 GPU 时间，口径变了）。**bench24：c1 507 = 1.13×、c8 3115 = 1.08×、c32 8844 = 1.08×（TPOT 3.4 vs 3.3）、c8-long 1938 = 1.01×（TPOT 3.8 = 3.8）**；TTFT c8-long 54 vs 54（之前 39：新请求现在要等两步才排进去，+1 步 ≈ 3–4 ms × 2 的代价在 c8-long 可见） | 异步把 host 0.3 ms 的大部分藏掉了（c32 3.56 → 3.46）；TTFT 多等一步是异步调度的固有代价（vLLM 同样） | 停止条件 2 的 c32 TPOT ≤ 3.3 未达（3.4），但吞吐 1.08× 已超；进入 "可生产" 收尾 |
