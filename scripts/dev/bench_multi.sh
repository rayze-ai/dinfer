#!/usr/bin/env bash
# 4090 上多卡 bench 一键脚本（docs/loops/q5-tp-dp.md）：杀残留 → 起 serve（--tp/--dp）→ bench_compare 进 CSV → 停。
#   bash scripts/dev/bench_multi.sh <OUT_DIR> "<改动一句话>" <BASE_GPU> [extra serve args, e.g. --tp 2 | --dp 2]
# 固定 --num-blocks 4000（每 rank 3.4 GiB；TP 下每 rank 存一半 kv 头）；端口 8331。
set -uo pipefail
OUT="$1"; CHANGE="$2"; BASE="$3"; shift 3
R="${R:-/data/yaochen/infer}"; PORT=8331; LOG=/tmp/bench_multi.log
# 残留：先 INT，3 s 后 KILL（装载期 INT 可能被忽略 → 上一轮 OOM 的根因）
pkill -INT -f "dinfer serve .*--port $PORT" 2>/dev/null; sleep 3; pkill -KILL -f "dinfer serve .*--port $PORT" 2>/dev/null; sleep 1
RUST_LOG=info nohup ./target/release/dinfer serve --model "$R/models/qwen3-0.6b" --host 127.0.0.1 --port $PORT \
  --max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.85 --num-blocks 4000 \
  --device "$BASE" "$@" > $LOG 2>&1 &
for _ in $(seq 1 240); do curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 0.5; done
curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null || { echo "serve DOWN:"; sed 's/\x1b\[[0-9;]*m//g' $LOG | grep -iE "error|panick" | tail -5; exit 1; }
DINFER_URL="http://127.0.0.1:$PORT" bash scripts/verify/bench_compare.sh "$OUT" "$CHANGE" 2>&1 | tail -6 | cut -c1-200
pkill -INT -f "dinfer serve .*--port $PORT"; sleep 3; pkill -KILL -f "dinfer serve .*--port $PORT" 2>/dev/null
echo "--- serve log tail:"; sed 's/\x1b\[[0-9;]*m//g' $LOG | grep -iE "error|panick|step breakdown|cuda graphs|serving" | cut -c1-200 | tail -5
