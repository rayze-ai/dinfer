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

## 上下文纪律
- 主会话每 ~10 次调用写一次 loop / progress 并 commit；超 60 次建议 `/clear`，靠交接区恢复（本次会话 1500+ 次调用全靠这个）。
