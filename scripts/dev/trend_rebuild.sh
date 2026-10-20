#!/usr/bin/env bash
# 本机重生成所有模型的趋势表 + 总览 index（pull 回 bench 目录后跑；远端跑的 bench_trend 只看到远端那台卡的目录，
# 本机才有全部卡型，所以权威的 perf-trend.* / index.html 以本机生成为准。2026-10-20：index 曾只有 qwen3/4090）。
#   bash scripts/dev/trend_rebuild.sh            # 全部模型
set -euo pipefail
cd "$(dirname "$0")/../.."
for m in test/*/; do
  [ -f "$m/perf-trend.json" ] || ls "$m"/2026-*/bench* >/dev/null 2>&1 || continue
  python3 -I scripts/dev/bench_trend.py --root "${m%/}" | tail -n 1
done
python3 -I scripts/dev/bench_index.py
