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

## 2026-10-14 用 scratch 缓冲的 `shape[0]` 当 token 数，把 max_tokens 当成了本步 token 数

- 现象：MLA forward 内 `rope_n` ensure 失败：`q.len 12288 < 64*4*16`，positions 只有 14。
- 根因：所有层的 scratch 按 `max_tokens` 预分配；`forward` 里 `let t = x.shape[0]` 取到的是预分配的 64，不是本步的 14。
- 解法：`forward(stream, gemm, t, x, ...)` 显式传 `t`；所有 `_n` 后缀的 kernel 包装按"容量 ≥ 所需"校验；GEMM 的激活/输出缓冲也改 `>=`，权重仍精确相等。
- 如何避免：张量形状只用于校验容量，不用于推导"本步有多少 token"；token 数永远从调度器/输入显式传。
- 相关：`crates/layers/src/{mla,nn}.rs`、`crates/kernels/src/{ops,gemm}.rs`

## 2026-10-14 Rust 包装层参数顺序与 C ABI 不一致（copy_cols）

- 现象：`copy_cols: range` ensure 失败，但调用处参数看起来合理。
- 根因：`ops.rs` 里 `copy_cols_bf16` 的参数顺序写成 `(src_cols, col0, width, dcol0, dst_cols)`，头文件是 `(src_cols, col0, dst_cols, dcol0, width)`；调用方按头文件传。
- 解法：包装层参数顺序改成与 `dinfer_kernels.h` 完全一致。
- 如何避免：写包装时直接从头文件复制参数列表；考虑在 ops.rs 里给多 usize 参数的函数用结构体参数（P4 起）。
- 相关：`crates/kernels/src/ops.rs::copy_cols_bf16`

## 2026-10-14 greedy 对拍在 bf16 平局步必然分叉，不是 bug

- 现象：prefill logits 全对、32 步 decode logits 最大偏差 0.007，但 greedy 第 7 步 token 不同。
- 根因：HF 该步 top1=0.508、top2=0.504，差 0.0039 = bf16 在 0.5 附近的 1 ULP；两边各自的累加顺序决定谁略大，无法也不应要求一致。
- 解法：对拍口径改为三层：① 全词表 logits max abs < 0.05；② argmax 只在 HF top1-top2 gap > 0.02 时要求一致；③ greedy 允许在平局步分叉，且分叉 token 必须是 HF top-2 之一。
- 如何避免：所有 greedy 对拍都带 `expect_step_logits`（HF 全序列前向逐步取最后一行），用 gap 判定平局，不用"token 序列完全一致"做唯一标准。
- 相关：`crates/glm53/tests/tiny_parity.rs`、`scripts/verify/gen_tiny_glm.py`、`docs/loops/p3-mla-dense.md`

## 2026-10-14 top-p 掩码与 vLLM 口径不一致：top-p 必须在 top-k 之后的分布上算

- 现象：`topk_topp_mask` 对拍，`k=5, p=0.5` 这一行 kernel 保留 5 个、参考保留 2 个。
- 根因：第一版 kernel 把 top-k 阈值和 top-p 阈值各自在 **原始分布** 上算再取 max；vLLM `apply_top_k_top_p_pytorch` 是先 top-k 置 -inf，再对剩余元素 softmax 后做 top-p（`probs_sum <= 1 - p` 置 -inf，至少留 1 个）。两者在 k 小、p 小时结果不同。
- 解法：kernel 里 top-p 的 softmax 分母与质量累计只统计 `v >= top_k_threshold` 的元素；Python 参考改成升序 + `cumsum <= 1-p` 与 vLLM 完全同形。
- 如何避免：采样链的任何算子先抄 vLLM 的 PyTorch 参考实现到 `gen_kernel_refs.py`，再写 kernel；容差按"边界 token 可能差 ≤ max(2, 0.1%)"设。
- 相关：`crates/kernels/cuda/sampling.cu::topk_topp_mask_kernel`、`scripts/verify/gen_kernel_refs.py::mask_ref`、`vllm/v1/sample/ops/topk_topp_sampler.py:473`

## 2026-10-14 cargo 集成测试的 cwd 是 crate 目录，不是工作区根

- 现象：`tests/ref_tests.rs` 用相对路径 `test/kernel_refs` 读 `.npy`，10 个测试全部 "No such file"。
- 根因：`cargo test` 跑集成测试时 cwd = `crates/kernels/`。
- 解法：相对路径按 `env!("CARGO_MANIFEST_DIR")/../..` 解析；脚本里用 `realpath -m` 传绝对路径。
- 如何避免：测试读外部文件一律走环境变量 + `CARGO_MANIFEST_DIR` 兜底。
- 相关：`crates/kernels/tests/ref_tests.rs::refs()`

## 2026-10-14 clippy 偶发 "failed to run rustc to learn about target-specific information"

- 现象：`cargo clippy` 报 rustc 解析 `<anon>` 失败，错误内容是一段 shell 脚本（`awk '-F: '`、`SN='Not Specified'`）。
- 根因：cargo 用 `rustc -` 从 stdin 探测目标信息，本机某个后台进程（疑似监控 agent）往同一终端 stdin 写了东西。与代码无关，重跑即过。
- 解法：重跑。若频繁出现，给 verify 脚本的 cargo 命令加 `< /dev/null`。
- 相关：`test/2026-10-14/p2.log`（第一次运行）

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
