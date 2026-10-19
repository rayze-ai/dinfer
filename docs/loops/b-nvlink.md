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

## vLLM 基线（B300，用户 2026-10-19："需要 vllm 基线，机器上有 containerd 可以用，用完记得清理"）

- 无 docker / 外网；containerd `k8s.io` 命名空间已缓存 `docker.io/vllm/vllm-openai:nightly`（**vllm 0.29.1rc1.dev573**，torch 2.13 cu130，B300 sm_103 可跑）与 `:kimi-k3`、`ghcr.io/novitalabs/vllm:glm53-*`。镜像站 image.paigpu.com / ppinfra 对 `v0.31.0` manifest 不响应，所以基线版本是 **nightly 0.29.1**，不是 4090 上的 0.31.0 —— 跨卡比值不可直接对照，CSV `baseline` 列写 `vllm-0.29.1rc1`。
- `nerdctl` 默认桥接网络缺 CNI bridge 插件 → `--net host`；GPU `--gpus device=N`。脚本 `scripts/dev/vllm_nerdctl.sh run|wait|smoke|logs|stop|ps`。**每次 bench 完 `stop`**（删容器，不删机器原有镜像）。
- 占卡约定：vLLM 基线 GPU 7（`:8013`）；dinfer single / tpN 从 GPU 0 起；lm_head 列切 bench 用 GPU 4–6。
- `bench_compare.sh` 在 B300 两次 exit 141：真因是 8 卡下 `nvidia-smi --query-gpu=name | head -1` 触发 SIGPIPE（pipefail），已改 `-i 0`（pitfalls）；`docker inspect` 那行顺带加 `/version` 回退。`bench_b300_1`（dinfer GPU 6 vs vLLM GPU 7）第三次跑。

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
| 1 | B1：B300 GPU 0–7 `q5_tp.sh 2 0` / `q5_tp.sh 4 0` 一致性 + `bench_tp_ab.sh` single vs tp2 / tp4 / tp8（`test/qwen3/2026-10-19/ab_tp_b300`，进 CSV tp 页签） | 首跑：**`q5_tp.sh 4 0` PASS（2/3 逐字一致 + 12 词前缀一致，与 4090 tp2 口径同）**；tp2 那次失败是权重 rsync 还没完成（`no *.safetensors`），bench 失败是 `bench_tp_ab.sh` 写在 push 之后（exit 127）——都是顺序问题不是代码问题。重推后重跑：**`q5_tp.sh 2 0` PASS（3/3 逐字一致）**。bench（GPU 0–7，`--num-blocks 4000`，CUDA graph 3954/4000 步）：
| 2 | B2 lm_head 列切（开关 `LMHEAD_COLSPLIT=1`，默认关；本机 3090 子代理，不上 B300）：`Linear::load_shard(dim 0)` 按 vocab 切 `lm_head.weight`（world>1 且文件有独立 lm_head.weight 才生效，纯 tied 不切并打 info；Qwen3-0.6B 文件带 lm_head.weight，可生效）；`Qwen3Model::lm_head_logits` 统一 forward / `logits_for_rows` 末尾：GEMM `[rows, vocab/world]` bf16 → `comm::all_gather_bf16`（新包装，NCCL，可入图）→ 新 kernel `gather_blocks_bf16_to_f32`（`[world][rows][vl]` 块布局一趟拼成 `[rows][vocab]` f32，省"转置 + 升精度"两趟）。传 bf16 让 NVLink 字节减半。单测 `models/qwen3/tests/lmhead_colsplit.rs`（切片拼回 == 全量逐位；块布局修正 == CPU 参考逐位，3090 PASS）；`cargo test --workspace --features qwen3` 全过，glm53 编译过。**预估**：Qwen3-0.6B lm_head 311 MB bf16，tp2 每卡 GEMM 权重读 155 MB（省 ~20 µs @8 TB/s），多 1 次 all_gather（decode t=32：9.7 MB bf16 ≈ 5 µs NVLink + 同步）；收益小且依赖 t，**bench 说话**。方案 B（局部 top-k 归并免传全 vocab）留 TODO 在 `lm_head_logits` 注释。 | **待 B300**（GPU 4–7，避开 0–3）：① `CUDA_VISIBLE_DEVICES=4,5,6,7 LMHEAD_COLSPLIT=1 scripts/verify/q5_tp.sh 2 0` 一致性；② `LMHEAD_COLSPLIT=1 scripts/dev/bench_tp_ab.sh test/qwen3/2026-10-19/ab_tp_b300_lmcol "B2: lm_head 列切" 4 "2 4"` 变体名 `tp2_lmcol` / `tp4_lmcol`，对照第 1 轮 tp2 / tp4。启动日志应有 `lm_head colsplit enabled`；若 TPOT 不降就保持默认关。 |

| 变体 | c1 tok/s / TPOT | c8 tok/s / TPOT | c32 tok/s / TPOT | c8-long tok/s / TPOT |
|---|---|---|---|---|
| single | 646 / 1.54 | 4643 / 1.65 | 14392 / 2.11 | 3432 / 2.23 |
| tp2 | 496 / 2.00 | 3435 / 2.24 | 11522 / 2.68 | 2761 / 2.76 |
| tp4 | 446 / 2.23 | 3345 / 2.31 | 12079 / 2.54 | 2712 / 2.83 |
| tp8 | 361 / 2.75 | 2706 / 2.83 | 9105 / 3.30 | 2288 / 3.37 |

B300 单卡 TPOT 1.54 ms（4090 3.5 ms 的 2.3×）。**NVLink 下 tp2 仍比单卡慢 30%（TPOT +0.46 ms）**：0.6B 模型每层 GEMM 太小，56 次 NCCL all-reduce × ~8 µs ≈ 0.45 ms 正好是差值，通信延迟主导而非带宽——这就是 B2 one-shot all-reduce 要吃掉的部分（vLLM custom_all_reduce 在 2 卡上单次 ~3 µs）。tp4 比 tp2 在 c32 略好（GEMM 再切小仍有余量），tp8 全面更差（8 rank 同步成本）。CSV `ab_tp_b300`（kind=tp，gpu=B300）。**B1 停止条件（tp2 ≥ 单卡）未达成，原因明确，转 B2** |

## 状态

- B0 完成。B1 完成：tp2 / tp4 一致性 PASS；性能 tp2 慢单卡 30%，差值 = 56 × NCCL all-reduce 延迟，B2 的目标就是这 0.45 ms。趋势图已加卡型维度（GPU 文件 + gpu 列 + 顶层卡型按钮）。**vLLM 基线在这台机器上暂无法起**（无 docker / torch / 外网），B1 先用 dinfer 单卡 vs tpN 自比，vLLM 对比列留空。
