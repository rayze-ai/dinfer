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
| 0 | 用户给三个 JumpServer token（12:07 / 12:11 / 12:14 过期）。第一个首次登录成功后再登被拒，第二个直接被拒，**第三个拿到立刻 `remote_b300.sh up` 建常驻 ControlMaster 成功**。机器：登录用户 `jumpserver`，`sudo NOPASSWD ALL`；`/data/yaochen` 原归 root，`sudo chown` 后建 `/data/yaochen/infer`；系统自带 `/usr/lib/x86_64-linux-gnu/libnccl.so.2.30.3` 与 `/usr/local/cuda-13.1`；**无 cargo、无 docker、无 torch、无 vllm、外网全不通**（只有内网 apt 镜像 100.66.200.250:8080）；glibc 2.35 与本机同 → 工具链 / registry / 权重全部从本机 rsync，cargo 走 `CARGO_NET_OFFLINE=true`。8 卡空闲（各 1.6 GB 常驻）。 | B0 进行中：rsync 工具链 + 权重 + 仓库 |

## 状态

- B0 进行中：常驻连接已建；工具链 / registry / 权重 rsync 中；下一步远端 `cargo build --release --features qwen3`（sm_103 首次编译）、`probe_ipc`、`q5_tp.sh`。**vLLM 基线在这台机器上暂无法起**（无 docker / torch / 外网），B1 先用 dinfer 单卡 vs tpN 自比，vLLM 对比列留空。
