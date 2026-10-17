#!/usr/bin/env bash
# host tier A/B（4090，GPU 3）：tier 关 / 开 两个隔离实例，同一真挤出负载，结果落 JSON 到 test/qwen3/<日期>/ab_hosttier/
# 文件名 <variant>__<group>.json（bench_trend.py 的 ab_* 规则：字典序第一个 variant 作基线 → tier00 vs tier16）。
#   bash scripts/verify/ab_hosttier.sh [OUT_DIR] ["改动一句话"]
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-test/qwen3/$(date +%F)/ab_hosttier}"; mkdir -p "$OUT"
[ -n "${2:-}" ] && printf '%s\n' "$2" > "$OUT/CHANGE"
GPU="${GPU:-3}"; PORT=8096; SEED="${SEED:-$(date +%s)}"
GROUP="c16_p1024_rep50_o128"
for gib in 0 16; do
  var=$(printf 'tier%02d' "$gib")
  bash scripts/dev/ab_serve.sh start $PORT $GPU --max-model-len 2048 --max-num-seqs 32 --max-num-batched-tokens 2048 --num-blocks 512 --kv-host-gib $gib | cut -c1-140
  timeout 900 ./target/release/bench --url http://127.0.0.1:$PORT --label "$var" --concurrency 16 --num-requests 96 --prompt-len 1024 --prefix-repeat 0.5 --max-tokens 128 --seed "$SEED" --warmup 2 --out "$OUT/${var}__${GROUP}.json" > "$OUT/${var}__${GROUP}.log" 2>&1 || { tail -3 "$OUT/${var}__${GROUP}.log"; }
  grep -E "requests=|TTFT|TPOT|throughput" "$OUT/${var}__${GROUP}.log" | cut -c1-110
  curl -s http://127.0.0.1:$PORT/metrics | grep -E "^dinfer:(host_tier_|prefix_cache_hits|num_preempt)" > "$OUT/${var}__metrics.txt"; sed "s/dinfer://" "$OUT/${var}__metrics.txt" | tr "\n" " "; echo
  sed "s/\x1b\[[0-9;]*m//g" /tmp/ab_serve_$PORT.log | grep -E "finishing with Length|empty step" > "$OUT/${var}__warns.txt" || true; echo "warns: $(wc -l < "$OUT/${var}__warns.txt")"
  bash scripts/dev/ab_serve.sh stop $PORT | tail -1
done
python3 -I scripts/dev/bench_trend.py --root test/qwen3 2>&1 | tail -1
