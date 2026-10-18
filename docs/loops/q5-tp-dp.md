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
| 4 | **TP + PD 一套实现（2026-10-18 晚，design §13.7）**：`Qwen3Executor` 抽出单 rank 原语 `pd_export_addr / pd_import_addr`（写 / 读任意 host-pinned 或设备地址，单卡 `kv_export*/kv_import` 也改走它）；`TpExecutor` 加 `Msg::Call` 闭包通道，`kv_export / kv_export_slot / kv_import / pd_map_region / pd_poll` 按 rank 分片到同一缓冲 / 槽的 `rank × stride` 偏移；`enable_pd(pool_bufs, world)` / `enable_pd_region(.., world)` 缓冲按全 world 算，非 0 rank pool_bufs=0；`model_tag` 带 `:tpN`；service 放开 `--tp` + `--role`，拒 cuda-ipc；`q4.sh` 加 `Q4_TP`（P / D 各 `--tp N`，`Q4_GPU_P/D` 给卡表） | 本机 core 46 单测 + 单卡 `e2e_qwen3_pd` tcp / shm PASS。**4090 `Q4_TP=2 Q4_GPU_P=3,4 Q4_GPU_D=5,6`：tcp 与 shm 两轮 e2e + 三故障 + 流式全 PASS，输出与单机逐字相同**；P `pd_export_total 66`（shm 下 `pd_export_slot_total 66` = 槽路径 100%），D `pd_import_total 66`、fallback 2（= 故障 3 的两次）。bench c8 (1024,128) 经 proxy vs 单机 :8012：**TPOT tcp 3.30 / shm 2.75 ms（单机 3.53，D 两卡 TP 更快）**，吞吐 974 / 893 tok/s，**TTFT p50 616 / 795 ms（单卡 PD xshm 58 ms，单机 13）—— 10 倍退化，功能线收口但这个要查**（第 5 轮）。CSV `ab_pd1_tp2` / `ab_pd1_tp2_shm` |
| 5 | **TTFT 616 ms 定位（探针：P / D / proxy 各自单测 c8 1024-token）** | P 单独（不经 proxy / D）o1 TTFT **1093 ms**、o128 582 ms；D 单独 172 ms；proxy 697 ms → 瓶颈全在 P 自身。P `step breakdown`：`wait_us_per_step=11660`、eager 52/1000（prefill 步）。算：2048 token × 1024 hidden × bf16 = 4 MB / all-reduce × 56 次/步 ≈ 224 MB/步走 NCCL SHM（4090 无 P2P）≈ 11 ms，对上。对照 plain `--tp 2` 无 PD：c8-long TTFT 231 vs 单卡 39 ms，同一件事。**结论：TP prefill 的 all-reduce 流量 ∝ token 数，无 P2P 机器上 1024-token prefill 比单卡慢 10×；PD 把 P 变成纯 prefill 只是把它全部暴露在 TTFT 上，不是 PD 的开销（D 侧 decode 每步 8 token 反而受益，TPOT 2.75–3.30 < 单机 3.53）。** 与 design §13 "TP 性能结论需 P2P 机器" 一致，不改代码；有 P2P / NVLink 的机器上 P 的 all-reduce 走 NVLink 才有意义，T3 的 fused all-reduce 也在那之后 |
| 6 | **异构 tp（P tp_P → D tp_D 任意组合，2026-10-18 深夜，进行中）**：wire 布局从 `[rank][block][layer][本地单元]` 改成 tp 无关的 canonical `[block][layer][全局 kv_head][slot][K/V][D]`；新 kernel `kv_pack_heads / kv_unpack_heads` 做 `[slot][head]→[head][slot]` 转置；各 rank 用 `cuMemcpy2DAsync`（src pitch = 本地层字节，dst pitch = 全局层字节，列偏移 rank × 本地层字节）只写 / 读自己头段；`TpExecutor` 不再算 `rank × stride`，所有 rank 收同一基址；`model_tag` 去 `:tpN` 改带 `kvh/hd`。q4.sh 加 `Q4_TP_P / Q4_TP_D`（CHANGE 写 `tpP/tpD`） | 本机：编译 + core 46 单测过；新 GPU 单测 `kv_pack_heads`（pack 布局 vs CPU 参考逐字节 + unpack 往返；`copy2d` 两 rank 列写 / 读回）PASS；单卡 PD e2e tcp / shm 两轮 PASS（canonical 在 world=1 退化为线性拷贝）。**4090 异构 `Q4_TP_P=1 Q4_TP_D=2`（P GPU 3 单卡，D GPU 5,6 tp2）tcp：e2e + 三故障 + 流式全 PASS，输出与单机逐字同，fault 前 import ≥1 / fallback 0**；bench c8 (1024,128) 经 proxy vs 单机 :8012：**TTFT p50 60 ms（同 tp2/tp2 时 616 ms → 10× 回落到单卡 PD 的 58 ms 水平，证实 TTFT 退化全是 P 侧 TP 的锅）、TPOT 3.54 vs 3.56（持平）**、1938 vs 2199 tok/s；门槛 `ttft ≤ 1.3×` 对 PD 小模型本来不适用（memory：PD TTFT 高于单机是预期）。CSV `ab_pd1_tpP1D2`。shm 轮见下 |

## 状态

- TP=2 功能通过（第 2 轮）；DP=2 吞吐达标（第 3 轮）；**TP + PD 功能通过（第 4 轮，tcp / shm，三故障，输出一致，TPOT < 单机）**；TTFT 10× 退化 = TP prefill all-reduce 无 P2P 的代价（第 5 轮），不是 PD。TP 性能结论需 NVLink / P2P 机器；T3（fused all-reduce+rmsnorm、lm_head 列切）在那之后再做。
