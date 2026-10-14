# 踩坑记录

> 一条一个坑，标题写**现象**（搜索时按现象搜），正文讲根因与解法。新条目加在最上面。
>
> 条目格式（标题用 `## YYYY-MM-DD 现象一句话`，正文五项：现象 / 根因 / 解法 / 如何避免 / 相关）：
>
>     ## 2026-01-01 启动时报 8080 端口被占用
>     - 现象：
>     - 根因：
>     - 解法：
>     - 如何避免：
>     - 相关：`文件/命令/链接`

## 2026-10-14 criterion `iter_batched` 把 setup 成本算进了被测函数

- 现象：调度器 bench 报 197 / 310 µs，但手工 `Instant` 分段计时只有 62 / 92 µs；两条 bench 之间 110 µs 的差距与代码路径差异不符。
- 根因：`iter_batched(setup, routine, SmallInput)` 每次迭代都跑一遍 setup（重建 1024 个请求、prefill 一步）。criterion 文档说 setup 不计时，但 `SmallInput` 模式下 setup 产生的内存分配 / cache 污染会让 routine 本身变慢，且 setup 里 `Scheduler::new` + 1024 次 `add_request` 的 drop 成本落在 routine 之后的同一测量窗口内。
- 解法：对"有持久状态、每步修改状态"的被测对象，用 `b.iter(|| ...)` 持续跑同一个实例；把 `max_tokens` / `max_model_len` / 块数设到测量期间不会触发完成或抢占。
- 如何避免：写 bench 前先用 `examples/probe_sched.rs` 这类手工计时对一遍量级；两者差 > 30% 先怀疑 bench 写法。
- 相关：`crates/core/benches/scheduler.rs`、`crates/core/examples/probe_sched.rs`、`docs/loops/p1-scheduler-perf.md`

## 2026-10-14 Claude Code 的 Bash 工具被平台安全分类器间歇拦截

- 现象：`Bash` 调用返回 "claude-opus-5[1m] is temporarily unavailable, so auto mode cannot determine the safety of Bash"，连续数十次；`Read/Write/Edit` 多数时候可用。
- 根因：Claude Code auto mode 依赖一个独立分类器判定命令安全性，该服务不可用时所有非只读工具被拒。与本项目无关。
- 解法：拦截期间只做写文件；把要跑的命令列给用户在终端执行并贴回输出；或用户关闭 auto mode（本次用户说"拦截已经放开"后恢复一段时间，随后又复发）。
- 如何避免：长命令写进 `scripts/`，拦截时用户可直接跑；每次会话开头先用一条无害命令试探工具是否可用。
- 相关：`docs/progress.md` 2026-10-13/14 日志

## 2026-10-14 cc crate 的 `cuda(true)` 给 nvcc 塞 gcc 参数导致编译失败

- 现象：`cargo build` 报 `nvcc fatal : Unknown option '-ffunction-sections'`，另有 `incompatible redefinition for option 'optimize'`。
- 根因：`cc::Build::cuda(true)` 仍把 debug profile 的 gcc 参数（`-ffunction-sections -fdata-sections -O0 -gdwarf-4`）原样传给 nvcc；nvcc 不认。
- 解法：`build.rs` 直接 `Command::new(nvcc)` 编 `.cu` 为 `.o`，再用 `cc::Build::object()` 只做归档。
- 如何避免：所有 CUDA 编译不经 `cc` 的 compiler 路径；需要 host 侧参数用 `-Xcompiler` 显式传。
- 相关：`crates/kernels/build.rs`、`test/2026-10-14/build-glm53.log`

## 2026-10-14 k8s Pod 申请了 `nvidia.com/gpu` 但容器里没有 nvidia-smi

- 现象：`nvidia-smi: not found`，`/dev/nvidia*` 不存在，但 `/proc/driver/nvidia/gpus/*/information` 能看到 4 张 RTX 4090。
- 根因：集群用 containerd + nvidia RuntimeClass，默认 runtime 不注入 GPU 设备与驱动库；device plugin 只负责调度计数。
- 解法：Pod spec 加 `runtimeClassName: nvidia`。
- 如何避免：所有 GPU Pod 模板都带该字段；`scripts/dev/k8s/*.yaml` 已加。
- 相关：`kubectl get runtimeclass` → `nvidia`

## 2026-10-13 本机 3090 跑不了两模型的任何快路径 kernel

- 现象：vLLM 中 Kimi-K3 / GLM-5.3 的 MLA decode（FlashMLA）、FP8 块 GEMM（DeepGEMM）、MXFP4 MoE、KDA fused decode 全部要求 sm90 / sm100。
- 根因：RTX 3090 是 sm86，无 FP8 / FP4 张量核。
- 解法：每类算子先写 sm80+ 可跑的参考版（bf16 / dequant 回退）做正确性验证；快路径在目标机器做（design.md 决策表）。
- 如何避免：所有 kernel 的 FFI 保持同一签名，参考版与快路径可互换；验证脚本按 `nvidia-smi` 的 compute capability 自动选路径。
- 相关：`docs/design.md` §6 §7；`vllm/models/kimi_k3/nvidia/kda.py:161-357`（各 backend 的架构门限）

## 2026-10-13 HF 上 Kimi-K3 / GLM-5.3 只有主仓库可匿名访问

- 现象：`moonshotai/Kimi-K3-Instruct|Thinking|Flash`、`zai-org/GLM-5.3-Air`、`THUDM/GLM-5.3` 返回 401；`moonshotai/Kimi-K3`、`zai-org/GLM-5.3`、`zai-org/GLM-5.3-Flash` 返回 200。
- 根因：前者不存在或为 gated 仓库。
- 解法：目标固定为 `moonshotai/Kimi-K3` 与 `zai-org/GLM-5.3`；`GLM-5.3-Flash`（`Glm5NextForCausalLM`，结构不同）不在范围。
- 如何避免：拉文件前先 `curl -s -o /dev/null -w '%{http_code}' https://huggingface.co/api/models/<repo>`。
- 相关：`/data/ref/models/*/meta.json`
