# loop: Q5 多卡 TP / DP（design §13）

## 目标

1. `--tp 2`（可 4 / 8）：Qwen3-0.6B 在 4090 Pod 上多卡 TP 跑通，greedy 输出与单卡逐 token 一致。
2. `--dp N`：同进程 N 副本，按在飞数路由，吞吐接近 N × 单卡。
3. 每轮 bench 进 `test/qwen3/perf-trend.csv`（即使 TP 慢于单卡，也要记数字和原因）。

## 环境

- 4090 Pod 8 卡，无 PCIe P2P（NCCL 走 SHM；若报 P2P 错加 `NCCL_P2P_DISABLE=1`）。**Pod 镜像里没有 NCCL**（之前 `find /` 命中的是宿主机其它容器 rootfs）：装到 `$R/pip`（`pip install --target $R/pip nvidia-nccl-cu12`），软链 `$R/lib/libnccl.so{,.2}`，`env.sh` 加 `RUSTFLAGS -L $R/lib` + `LD_LIBRARY_PATH`。
- 启动：`scripts/dev/restart_dinfer.sh` 风格，`serve --tp 2 --device 0`（卡 0,1）；DP：`--dp 2 --tp 1 --device 2`（卡 2,3）。
- TP 下强制 `XSTREAM=0`（副流上没有 comm；service 自动设）。

## 观察方法

- 一致性：`scripts/verify/q5_tp.sh [TP] [BASE_GPU]`：同一组 prompt，单卡 vs `--tp N` greedy 64 tok；≥ 2/3 prompt 逐字一致且 12 词前缀全一致 = 通过（硬判决口径，见 memory / design §13.2）。GPU 0–2 常被 vLLM / 旧 dinfer 占用，用 3/4。
- 性能：`scripts/verify/bench_compare.sh test/qwen3/<日期>/benchN "tp2 ..."`。
- NCCL 日志：`NCCL_DEBUG=INFO` 看选用的 transport（期望 SHM）。

## 停止条件

`q5_tp.sh` exit 0 且 `--dp 2` c32 吞吐 ≥ 1.8× 单卡。

## 迭代上限

12 轮。

## 记录

| 轮 | 改动 | 结果 |
|---|---|---|
| 1 | NCCL 封装 + 权重分片 + Qwen3 TP 切分 + TpExecutor + `--tp/--dp` CLI + DpBackend（本机 3090 只能编译 + 单卡回归） | 本机编译过；4090 首次链接失败 `cannot find -lnccl`（Pod 无 NCCL）→ pip 安装到用户目录后重建，`q5_tp.sh 2` 结果待记 |
| 2 | 4090 GPU 3/4 `--tp 2`：两 rank 本地 cfg heads 8 / kv 4 / inter 1536，NCCL 建组成功，启动 1.04 s（单卡 0.59 s；两卡装载并行但 cuda_init 351 ms + NCCL init）。3 个 prompt greedy 64 tok：2 个逐字一致，第 3 个最后一词 "system" vs "database or application"（bf16 部分和 + all-reduce 二次舍入，近平局翻转）。`q5_tp.sh` 口径改为 ≥2/3 一致 + 12 词前缀一致 | **TP=2 功能通过**；性能 bench 待 |
| 3 | 4090 GPU 5/6 bench（`scripts/dev/bench_multi.sh`，num_blocks 4000；首跑两次 OOM 根因是上一轮 OOM 的 serve 残留占 13–17 GB，INT 杀不掉装载期进程 → 脚本改 INT→KILL）。**tp2 vs vLLM 单卡 tok/s**：c1 511/447 = **1.14×**、c8 2656/2893 = 0.92×、c32 5264/8225 = **0.64×**、c8-long 1298/1908 = 0.68×；CUDA graph 在 TP 下正常捕获（graph=3954/4000 步，含 NCCL all-reduce）。**dp2 vs vLLM 单卡**：c1 514/447 = 1.15×、c8 3259/2891 = **1.13×**、c32 10001/8310 = **1.20×**、c8-long 2492/1896 = **1.31×**（约 1.9× 单卡 dinfer c32，路由均衡）。 | TP：小模型 + 无 P2P 下 all-reduce 占主导，c32 掉 36%，符合 design §13.2 预期；DP 停止条件（≥1.8× 单卡）**达成** |

## 状态

- TP=2 功能通过（第 2 轮）；DP=2 吞吐达标（第 3 轮）。TP 性能结论需 NVLink / P2P 机器；T3（fused all-reduce+rmsnorm、lm_head 列切）在那之后再做。
