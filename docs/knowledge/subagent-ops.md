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

## 远端长链的超时（2026-10-20 K3 C-6 教训）
- **判断**：`timeout N bash chain.sh`（serve → smoke → 多组 bench → stop）的 N 要按"装载 + 全部组"的实测上界给，或者**不包 serve**（serve 单独起、bench 每组各自 `timeout`）。C-6 第三轮 1800 s 把 25 min 装载 + 四组 bench 的链截在第四组（c8-long，一组 10 min+），serve 被连带杀掉，只能重装 25 min 单独补。
- **怎么用**：每组 bench 落盘后立刻可用（json 独立），链的超时只保护单组；装载慢的模型（K3 93 层 tp8 ≈ 1500 s）把 serve 放 tmux 常驻、链只管 bench。
- **证据**：`docs/loops/k3-serve.md` 3j（"c8-long 链上 1800 s 超时未落，单独补跑中"）；B300 `bootstrap/k3_c6_tp8_2/serve.log` 止于正常 step mix、无错误。

## 远端 `test/` 下有本机没有的目录（尤其容器 root 写的）会让 push 报错退出（2026-10-20 b17 链）

- **判断**：`remote_b300.sh push` 带 `--delete`，远端 `test/` 下任何本机没有的目录都会被尝试删除；容器以 root 写的（参考 .npy）删不掉 → rsync 报 Permission denied 并以非 0 退出，`push && 后续` 的链就断在这里，但代码其实已传到。
- **为什么**：K3 24 层参考由容器生成到远端 `test/kimik3/2026-10-20/model24/`（root 属主），本机没有 → 下一次 push 想删它失败 → bench_b300_17 链没编译就退了，白等一轮装载。
- **怎么用**：参考 / 产物生成到远端后**立刻 pull 回本机**（决定是否进仓库另说，至少本机要有同名目录），push 才幂等；链式脚本里 push 后不要用 `&&` 串关键步骤，或 `push || true` 再检查代码是否到位。
- **证据**：`docs/progress.md` 2026-10-20 b17 两次失败条目。

## 可能失败的本地步骤（merge / rebase / 编译）单独一条命令，看到结果再推远端起长链（2026-10-20 用户："你为啥干这种事，耽误时间"）

- **判断**：`git merge --ff-only`、rebase、本地编译这类有失败分支的步骤，不能和 `remote push` + 起 30 分钟远端链放在同一个并行 batch 里；失败了远端链照样跑，推上去的是旧代码，白等一轮才发现。
- **为什么**：20:4x 合体测试第一次发：ff 失败（dev 有后加 commit）→ 我没看结果 → B300 / 4090 两条链用合并前代码起跑 → 10 分钟后才发现作废重发。同一轮 B300 链还被自己的 `pkill -f "dinfer serve"` 误杀（pattern 匹配包装 bash）。
- **怎么用**：顺序严格为：① 本地 merge/rebase（单独发，看结果）→ ② 本地验证（build / 单测，可后台）→ ③ 远端 push 后 `grep` 一个只有新代码才有的符号确认到位 → ④ 起长链。链里杀进程用 `pkill -x <进程名>`。
- **证据**：`docs/progress.md` 2026-10-20 合并条目；`docs/pitfalls.md` 同日 pkill 条。

## 远端排队等前一个作业（r10，2026-10-20）
- `ssh run 'until ! pgrep -f "gdn_parity"; do sleep 5; done; …'` **会匹配到自己**：`pgrep -f` 看整条命令行，而外层 `bash -c` 的命令行里就含 "gdn_parity"，于是永远不退出——两个后台作业各空转 20 min，GPU 7 全程空闲，表面看像"还在跑"。`pkill -f` 同理会把自己杀掉（整条链 exit 1、日志文件都没建）。
- 正确做法：按**二进制路径**等（`pgrep -f "^/data/.../release/deps/gdn_parity-"`，锚定行首，bash -c 的命令行不会以它开头），或干脆不排队：一个后台作业把 build / parity / bench / ncu 串在同一条链里顺序跑。
- 排队作业启动后 1–2 min 要回头看一次产物文件是否已建（`ls -la bootstrap/<log>`）；没建 = 还在等 = 多半自锁。

## 远端探针队列的卫生（2026-10-28，一晚踩四次）
- 探针脚本 `echo DONE` 放 `trap … EXIT`，否则 python 异常 / serve 503 后 watcher 永远等；watcher 一律带超时（`for i in $(seq 1 N)`）。
- 宿主机路径 `/hf_data/yaochen/...` ≠ 容器路径 `/w/...`：在宿主机补写 DONE 标记要用宿主路径。
- 容器 `kaon3-full` 只挂 GPU 0；`CUDA_VISIBLE_DEVICES=1` → `CUDA_ERROR_NO_DEVICE` 秒退，脚本继续打到旧端口 → 503，看起来像服务 bug。
- 上一条链的 serve 没退干净时起下一条 → OOM → 503；链首 `pkill -x dinfer; sleep 2`。
- bN 的 A 侧二进制：脚本开头 `cp 当前 → prev`；若链里先编译再调 bN，prev 已是新二进制——用 `[ -f prev ] || cp` 变体保留真旧版。
- **链之间只能靠 DONE 标记串行；链内绝不 `pkill -x dinfer`**（只 kill 自己起的 `$SP`）：2026-10-28 b29.sh 开头的 `pkill -x dinfer` 在 watcher 提前放行时把 b28 bN 的 serve 杀了（bN 只剩 A 组、无 DONE，下游全卡）。
- 用 `sed s/b28/b29/g` 复制链脚本时，**先 `grep -n "^for i"` 核 watcher 等的是哪个文件再启动**；我把 "等 pair2 DONE" 改成了 "等 b29 自己的 DONE"，watcher 立刻放行。
