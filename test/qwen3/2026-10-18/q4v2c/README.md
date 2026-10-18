# PD v2 (c) 4090 验证（2026-10-18，commit a12d627）

结论：e2e PASS（含流式断言）、三故障 PASS（fault 3 流式回退输出与单机相同）；bench c8 (1024,128) 经 proxy vs 单机 :8012：
TTFT p50 97 → 67 ms（p99 423 → 140；单机 13 / 19）、TPOT 4.72（v2a 4.43，单机 3.53，噪声内）、吞吐 1464 tok/s（v2a 1480，单机 2222）、0 失败。
门槛（TTFT ≤ 1.3× 单机、TPOT ≤ 单机 +0.05）仍不达 → `Q4_BENCH_GATE=1` FAIL，预期内（同卡 0.6B 结构性代价，见 loop 第 11 轮）。

跑法：`QWEN3_DIR=/data/yaochen/infer/models/qwen3-0.6b Q4_REMOTE=1 bash scripts/verify/q4.sh test/qwen3/2026-10-18/q4v2c`
结果：`../ab_pd1/`（v2c）、`../ab_pd1_v2a/`、`../ab_pd1_v1/`，三行都在 `test/qwen3/perf-trend.csv`。日志：本目录 `e2e.log`、`p.log`、`d.log`、`x.log`、`p_badpeer.log`。
分析与下一步：`docs/loops/q4-pd.md` 第 14–15 轮。
