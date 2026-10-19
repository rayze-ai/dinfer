# B 类：NVLink 机器上的 TP 通信优化 / DCP / TP 真实性能（B300 ×8）

> 用户 2026-10-19："沿用 /data/yaochen/infer，实现 B"。机器：host-172-16-3-225，8 × B300 SXM6 275 GB，全 NV18 NVLink，CUDA 13.1，sm_103。

## 目标（design §13.6 T3 / T4 + 重测）

1. **B0 环境**：`scripts/dev/remote_b300.sh` 常驻连接；`bootstrap_remote_b300.sh`（rust 工具链 / NCCL / env.sh，`CUDA_ARCHS=103`）；`probe_ipc` 确认 P2P；Qwen3-0.6B 权重 rsync；vLLM 基线容器（`scripts/dev/vllm_docker.sh`，看 B300 上能否起 0.31.0）。
2. **B1 TP 真实性能**：`bench_multi.sh` tp2 / tp4 / tp8 vs vLLM 单卡与 vLLM tp2，进 CSV（`kind=tp`）。这是 4090 上 c32 0.64× 那条结论的重测。
3. **B2 T3 通信优化**（`crates/kernels/src/comm.rs` 的 TODO(T3)）：
   - 小消息 one-shot all-reduce（vLLM `csrc/custom_all_reduce.cu`：world ≤ 8、消息 ≤ 几百 KB 时各 rank 直接读对端 IPC 缓冲求和，比 NCCL ring 省一半延迟）；需要 cuda-ipc 跨卡（NVLink 下可用）。
   - `fused_allreduce_rmsnorm`：all-reduce + residual add + RMSNorm 一个 kernel（vLLM flashinfer `trtllm_allreduce_fusion` 思路），省一次 hidden 读写。
   - lm_head 列切（vocab ÷ world，各 rank 算自己段 logits → 采样只在 rank 0 需要 all-gather logits 或各 rank 算局部 top-k 后归并）。
   - 每项单独 bench 进 CSV，不达标回退开关。
4. **B3 T4 DCP**（design §13.5）：长上下文 KV 按 token 段切到多卡，decode 复用 split-K 的 lse 归并（`gqa_decode` 已有 scratch 布局 `[B][H][splits][D+2]`）；wire 布局已 head-major。目标：单请求 context 超单卡 KV 仍可跑。

## B2 设计（2026-10-19，B1 跑的同时定稿；对齐 vLLM `csrc/custom_all_reduce.cuh`，§11 台账记"采用"）

- **何时走自定义**：world ∈ {2,4,8}、全 P2P、消息 ≤ 512 KB（world ≤ 4）/ 256 KB（world 8）走 one-shot；更大消息先保留 NCCL（two-stage reduce-scatter + all-gather 留 TODO）。decode `t×hidden×2B`：Qwen3-0.6B t=32 时 64 KB，全部命中 one-shot；prefill 2048 token = 4 MB 走 NCCL。开关 `CUSTOM_AR=0` 回退。
- **缓冲**：每 rank 一块 cuda-ipc 设备区 = `Signal`（start/end flag `[kMaxBlocks=36][8]`，128 B 对齐）+ 输入镜像区（max_size）。rank 建区 `cuIpcGetMemHandle`，经 `TpExecutor::dispatch/collect` 在进程内交换 handle（不走文件 / NCCL），各 rank `cuIpcOpenMemHandle` 打开其余 world-1 份 → `RankData{ptrs[8]}` 上传设备。复用 `shared.rs` 的 `RegionKind::CudaIpc` 代码路径（抽一个 `IpcBuf::{create, open}`）。
- **kernel `ar_oneshot_bf16<world>`**（`crates/kernels/cuda/allreduce.cu`）：① 本 rank 输入 D2D 进自己镜像区（或让 o_proj / down 直接写镜像区省一次拷贝，第二步再做）；② `barrier_at_start`（每 block 对每个 peer 写 start flag、自旋等齐，`st.volatile` / `ld.volatile`）；③ 每线程按**固定 rank 顺序** 0..world 累加 8 个 bf16 的 packed 读（所有 rank 相同顺序 → 逐位一致，TP 采样一致性依赖这点）；④ 写 result；⑤ `barrier_at_end`（release / acquire）。grid ≤ 36 block × 512 线程。
- **融合 `ar_oneshot_add_rmsnorm_bf16`**：③ 的和不落回 attn_out / mlp_out，直接 `hidden += sum` 再对该行 RMSNorm 写 normed（一行 = 一个 block 处理 hidden=1024 → 512 线程 × 2 元素，归约在 block 内）。替换 forward 里 "all-reduce ① + fused_add_rmsnorm(post_norm)" 与 "all-reduce ② + 下一层 fused_add_rmsnorm(input_norm)" 两处（第二处跨层，`pending_mlp` 逻辑要改成"上一层的 mlp 部分和还在镜像区"）。省一次 t×h 读写 + 一次 kernel launch。
- **CUDA graph**：flag 用单调递增的 `_flag[block]` 计数器（vLLM 做法），图内重放无需重置；镜像区地址固定。
- **lm_head 列切**：vocab 151936 ÷ world 各 rank 算一段 logits（GEMM N 维切），采样需要全 vocab → rank 0 `all_gather` logits（t × 151936 × 4 B，decode t=32 时 19 MB，比省下的 GEMM 不一定划算）。**先 bench 再决定**：只做开关 + all_gather 版本，若 TPOT 不降就不默认开。
- **验证**：① 单测 `ar_oneshot` vs NCCL 结果逐位一致（随机 bf16，world 2/4/8，size 1 KB–512 KB）；② `q5_tp.sh` 一致性不变；③ `bench_tp_ab.sh` 每项改动一行进 CSV（变体名 `tp2_car` / `tp2_car_fused` / `tp2_lmcol`），停止条件 TPOT 相对上一轮不退化、至少一项 ≥ 3%。
- **分工**：kernel + IpcBuf + 单测一个子代理；model.rs 接线 + 开关 + bench 另一个；lm_head 列切第三个（独立）。brief 固定加"`crates/` 不得出现模型特定内容"。

## 环境

- 远端 `/data/yaochen/infer/{dinfer,models,cargo,rustup,pip,lib}`，`source /data/yaochen/infer/env.sh`。
- JumpServer token 一次性：`remote_b300.sh up` 建常驻连接后不要断；断了向用户要新 token。
- 选卡：先 `nvidia-smi` 看占用；两个 NUMA 域（GPU 0–3 / 4–7），tp2 / tp4 取同域。

## 观察方法

- 正确性：`scripts/verify/q5_tp.sh N <BASE_GPU>`（单卡 vs tpN greedy 一致）。
- 性能：`scripts/dev/bench_multi.sh test/qwen3/<日期>/benchN "<改动>" <BASE_GPU> --tp N`，每轮进 CSV。
- 通信：`NCCL_DEBUG=INFO` 看 transport 是 NVLink（`NVL`）；nsys 看 all-reduce kernel 时长。

## 停止条件

B1：tp2 c32 ≥ 单卡 dinfer 的 1.0×（NVLink 下 TP 不应低于单卡）且输出一致。B2：每项改动 TPOT 相对上一轮不退化、至少一项 ≥ 3% 提升进 CSV。B3：`max_model_len` 超单卡 KV 预算的请求在 dcp2 下跑通且输出与单卡（能跑的长度内）一致。

## 迭代上限

B0–B1 4 轮；B2 8 轮；B3 8 轮。

## 记录

| 轮 | 改动 | 结果 |
|---|---|---|
| 0 | 用户给三个 JumpServer token（12:07 / 12:11 / 12:14 过期）。第一个首次登录成功后再登被拒，第二个直接被拒，**第三个拿到立刻 `remote_b300.sh up` 建常驻 ControlMaster 成功**。机器：登录用户 `jumpserver`，`sudo NOPASSWD ALL`；`/data/yaochen` 原归 root，`sudo chown` 后建 `/data/yaochen/infer`；系统自带 `/usr/lib/x86_64-linux-gnu/libnccl.so.2.30.3` 与 `/usr/local/cuda-13.1`；**无 cargo、无 docker、无 torch、无 vllm、外网全不通**（只有内网 apt 镜像 100.66.200.250:8080）；glibc 2.35 与本机同 → 工具链 / registry / 权重全部从本机 rsync，cargo 走 `CARGO_NET_OFFLINE=true`。8 卡空闲（各 1.6 GB 常驻）。 | B0：rsync 工具链（rust 1.89 + registry 869 MB）/ 权重 / 仓库完成；`env.sh`（`CUDA_ARCHS=103`、`CARGO_NET_OFFLINE=true`、系统 CUDA 13.1 + NCCL 2.30.3）。**`probe_ipc`：8 卡两两 `canAccessPeer=1`，IPC 64 MB 拷贝 0.09 ms ≈ 740 GB/s，跨 NUMA（0→4）同速** → PD cuda-ipc 跨卡、one-shot all-reduce 都可用。**sm_103 release 编译通过**（`cargo build --release --offline --features qwen3`，nvcc 13.1 `-gencode arch=compute_103,code=sm_103` 全部 .cu 直接过，无需改 kernel）。B0 完成 |
| 1 | B1：B300 GPU 0–7 `q5_tp.sh 2 0` / `q5_tp.sh 4 0` 一致性 + `bench_tp_ab.sh` single vs tp2 / tp4 / tp8（`test/qwen3/2026-10-19/ab_tp_b300`，进 CSV tp 页签） | 首跑：**`q5_tp.sh 4 0` PASS（2/3 逐字一致 + 12 词前缀一致，与 4090 tp2 口径同）**；tp2 那次失败是权重 rsync 还没完成（`no *.safetensors`），bench 失败是 `bench_tp_ab.sh` 写在 push 之后（exit 127）——都是顺序问题不是代码问题。重推后重跑 tp2 + bench |

## 状态

- B0 完成（连接 / 工具链 / 权重 / sm_103 编译 / P2P 确认）。B1 运行中。**vLLM 基线在这台机器上暂无法起**（无 docker / torch / 外网），B1 先用 dinfer 单卡 vs tpN 自比，vLLM 对比列留空。
