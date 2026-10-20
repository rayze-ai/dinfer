# 多子代理并行作业（2026-10-19/20 两天 ~15 个子代理的教训）

## brief 模板（每条都要有）
1. 目标一句话 + 起点数字（bench / nsys 行号）+ 停止条件。
2. 必读：loop 规格文件、CLAUDE.md 硬规则（原文摘要：调优数值 / 分文件 / 前缀 / 中文 / commit 格式）。
3. 可复用件的**文件路径 + 行号**（kernel、Rust 包装、调用方、对拍测试、参考实现），不要让它从头找。
4. 交付清单按序 + 每步 commit + 验证命令（parity / golden / 四 feature 构建 / `cargo test --no-run --features ref-tests`）。
5. **边界**：不碰哪台机器、不碰哪些文件（列出其它并行子代理在改的文件集）、不 push、只 `git add` 自己的文件。
6. 环境一行（CUDA_HOME / rustc_info 坑 / torch 路径）+ 输出纪律（截断、tee /tmp）。
7. **后台任务给 timeout 并主动查产物时间戳，不要只等通知**（一个子代理因通知丢失空等 4.5 h）。
8. 回报格式：数字表、commit 列表、未完成项、踩坑。

## 并行分工
- 按**文件集**切：kernels/gemm* ↔ kernels/gdn* ↔ models/kimik3/* 三线可并行；共享文件（`arch.rs`、`kernels.h`、`ffi.rs`、`ops.rs`）谁加谁 commit，**声明与定义同一 commit**。
- 共用远端目录（B300 `/data/yaochen/infer/dinfer`）：`remote_b300.sh push --delete` 会删掉别人远端生成、未 pull 的文件。规矩：生成物放 `$R/bootstrap/`（push 不覆盖）或先 pull 再 push；有人在 B300 跑长任务时主会话冻结 push。
- GPU 分配写进 brief（B300：0/1 Qwen3.5 常驻、2–3 K3、4–7 nsys / 临时）。

## 主会话的职责
- 子代理不碰 bench 机器；主会话统一 push / bench / 进 CSV / 更新交接区。
- 子代理回报要**审**：第 9 轮 brief 里的"疑点 1 bug"是主会话的误读，子代理证伪了——brief 的假设要标注为假设。
- 子代理报"中间状态等后台"超过 30 min 没动 → 查它等的产物时间戳，叫醒。
- **"无响应"不等于卡住**（2026-10-20 误停一次）：子代理在写代码时转录可能十几分钟不更新。判卡住的条件是：它**自己说在等后台任务** + 该任务产物已完成 + 超过 20 min；否则先 SendMessage 问一句，等 10 min 回复再决定。停错一个正在做关键归因的子代理，损失比多等 10 分钟大得多。

## 上下文纪律
- 主会话每 ~10 次调用写一次 loop / progress 并 commit；超 60 次建议 `/clear`，靠交接区恢复（本次会话 1500+ 次调用全靠这个）。

## 远端同步只由主会话 push；子代理 put 单文件、产物落 /tmp（2026-10-20 两次事故）

- **判断**：`remote_b300.sh push` 带 `--delete`，谁的本机工作区不全谁 push 就会删别人的远端目录。多子代理并行时只有主会话工作区含全部 commit，所以只有主会话能 push；子代理要同步代码就发消息。
- **为什么**：10-20 上午 K3 refs 被另一会话 push 删掉、下午 r8 的 engine_r8 产物被 r9 push 删掉，各损失 20–40 min GPU 时间。
- **怎么用**：子代理 brief 固定一句"不要 push；要推代码发消息；远端产物写 /tmp 或 bootstrap/，主会话 pull"。主会话 push 前 `git status` 确认子代理的 commit 都已在本地 dev。
- **证据**：`docs/pitfalls.md` 2026-10-20 两条。

## 子代理"收口"后仍可能被用户直接追问而重新上远端（2026-10-20 r9）

- **判断**：子代理回报"不再碰 B300"不等于它不会再上——用户直接对子代理提问（"为什么还比 vLLM 慢"）会让它为取证重新起 GPU 进程，且不知道主会话此刻在跑什么。主会话是唯一知道全局 GPU 占用的人，**派子代理时要在 brief 里写死"任何 GPU 动作前先发消息问主会话"**，并在自己起长测前再发一次"X 卡 Y 分钟内禁用"的广播。
- **为什么**：C-6 全 93 层 tp8 占 8 卡装载期间，r9 在 GPU 6 起了 `fla-prof` 容器跑 torch.profiler（2.3 GB），两次消息令停都晚于它的下一轮动作，最后主会话直接 `nerdctl rm -f` 才停住；若晚 5 分钟进 bench 计时，c32 / c8-long 数字作废。
- **怎么用**：brief 模板加一句；主会话起独占测试前 `SendMessage` 广播所有活着的子代理；发现越界先 `nvidia-smi --query-compute-apps` 定位容器 / PID，直接停，不等确认。
- **证据**：`docs/progress.md` 2026-10-20 C-6 第三轮条目。
