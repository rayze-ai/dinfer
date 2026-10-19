#!/usr/bin/env bash
# 开环泊松到达压测：dinfer（XSTREAM 开 / 关两种常驻配置，分别重启）vs vLLM，同 seed、同到达序列、长短混合 prompt。
#   bash scripts/verify/bench_poisson.sh [OUT_DIR] ["改动一句话"]     # 在 4090 上跑
# 环境：RATES（默认 "8 16 24" req/s）、REQS（每组请求数，默认 96）、MIX（默认 "128:0.5,1024:0.35,2048:0.15"）、OUT_TOK（默认 128）。
# 结论看：同 rate 下 out tok/s、TTFT p50/p99、TPOT p50；XSTREAM 开 vs 关的 TPOT 差即跨流并行收益。
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-test/qwen3/$(date +%F)/poisson}"; mkdir -p "$OUT"
[ -n "${2:-}" ] && printf '%s\n' "$2" > "$OUT/CHANGE"
RATES="${RATES:-8 16 24}"; REQS="${REQS:-96}"; MIX="${MIX:-128:0.5,1024:0.35,2048:0.15}"; OUT_TOK="${OUT_TOK:-128}"
SEED="${SEED:-$(date +%s)}"
BENCH=./target/release/bench; [ -x "$BENCH" ] || cargo build --release -p dinfer-bench
R=/data/yaochen/infer
V="${VLLM_URL:-http://127.0.0.1:8013}"
restart_dinfer() { # $1 = XSTREAM value
  XSTREAM=$1 bash scripts/dev/restart_dinfer.sh >/dev/null
}
run() { # side url rate
  local side=$1 url=$2 rate=$3
  local f="$OUT/${side}_r${rate}.json"
  "$BENCH" --url "$url" --label "$side" --request-rate "$rate" --num-requests "$REQS" --prompt-len-mix "$MIX" --max-tokens "$OUT_TOK" --seed "$SEED" --warmup 4 --out "$f" > "$OUT/${side}_r${rate}.log" 2>&1 \
    || { echo "bench $side rate=$rate failed:"; tail -3 "$OUT/${side}_r${rate}.log"; exit 1; }
}
MD="$OUT/summary.md"
{
  echo "# 开环泊松到达：dinfer(xstream on/off) vs vLLM（$(date '+%F %T')）"; echo
  echo "seed=$SEED  reqs/组=$REQS  mix=$MIX  out=$OUT_TOK  rates=$RATES"; echo
  echo "| rate req/s | 端 | 完成 | out tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | latency p50 (ms) |"
  echo "|---|---|---|---|---|---|---|"
} > "$MD"
row() { python3 - "$1" "$2" "$3" >> "$MD" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); rate, side = sys.argv[2], sys.argv[3]
f = lambda x: f"{x:.1f}"
print(f"| {rate} | {side} | {d['completed']}/{d['num_requests']} | {f(d['output_token_throughput_tps'])} | "
      f"{f(d['ttft_ms']['p50'])} / {f(d['ttft_ms']['p99'])} | {f(d['tpot_ms']['p50'])} / {f(d['tpot_ms']['p99'])} | {f(d['latency_ms']['p50'])} |")
PY
}
for x in 1 0; do
  restart_dinfer "$x"; side="our_xs$x"
  for r in $RATES; do run "$side" http://127.0.0.1:8012 "$r"; row "$OUT/${side}_r${r}.json" "$r" "$side"; done
done
for r in $RATES; do run vllm "$V" "$r"; row "$OUT/vllm_r${r}.json" "$r" vllm; done
{ head -6 "$MD"; tail -n +7 "$MD" | sort -t'|' -k2,2n -k3,3; } > "$MD.tmp" && mv "$MD.tmp" "$MD"
cat "$MD"
restart_dinfer 1   # 恢复默认（XSTREAM 开）
