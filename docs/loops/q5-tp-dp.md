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

- 一致性：`scripts/verify/q5_tp.sh`：同一组 prompt，单卡 vs `--tp 2` greedy `max_tokens 64`，逐 token diff；exit 0 = 全一致。
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
