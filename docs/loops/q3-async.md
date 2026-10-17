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
