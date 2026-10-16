#!/usr/bin/env bash
# dinfer vs vLLM 对比压测（在 4090 上跑）。同一 bench 二进制、同 prompt 集（seed）、同 max_tokens、ignore_eos、思考关。
#   bash scripts/verify/bench_compare.sh [OUT_DIR]      # 默认 test/$(date +%F)/bench
# 矩阵：并发 1 / 8 / 32 × (prompt 128, out 128)，再加一组长 prompt (1024, out 128) 并发 8。每组先 bench --self-check 一次。
# 环境变量：DINFER_URL / VLLM_URL（默认 :8012 / :8013）、REQS（每组请求数，默认 = 4×并发，最少 16）。
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-test/$(date +%F)/bench}"; mkdir -p "$OUT"
BENCH=./target/release/bench
[ -x "$BENCH" ] || cargo build --release -p dinfer-bench
A="${DINFER_URL:-http://127.0.0.1:8012}"
B="${VLLM_URL:-http://127.0.0.1:8013}"
"$BENCH" --self-check > "$OUT/self_check.log" 2>&1 || { tail -3 "$OUT/self_check.log"; echo "bench self-check FAILED"; exit 1; }
echo "self-check PASS"
MD="$OUT/summary.md"
{
  echo "# dinfer vs vLLM（$(date '+%F %T')，$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)）"
  echo
  echo "dinfer: $A（$(curl -s $A/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')）；vLLM: $B（$(docker inspect vllm-qwen3 --format '{{.Config.Image}}' 2>/dev/null || echo ?)）。"
  echo "口径：流式；TTFT = 首个带内容 chunk；TPOT = (latency − TTFT)/(out−1)；ignore_eos=true；temperature 0；enable_thinking=false；每组前 2 个预热请求不计。"
  echo
  echo "| 并发 | prompt | out | 端 | 请求 | req/s | out tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | ITL p50 (ms) | latency p50 (ms) |"
  echo "|---|---|---|---|---|---|---|---|---|---|---|"
} > "$MD"
run_one() { # conc prompt out
  local c=$1 p=$2 o=$3
  local reqs=${REQS:-$(( c * 4 > 16 ? c * 4 : 16 ))}
  for side in dinfer vllm; do
    local url; [ $side = dinfer ] && url=$A || url=$B
    local f="$OUT/${side}_c${c}_p${p}_o${o}.json"
    "$BENCH" --url "$url" --label "$side" --concurrency "$c" --num-requests "$reqs" --prompt-len "$p" --max-tokens "$o" --out "$f" > "$OUT/${side}_c${c}_p${p}_o${o}.log" 2>&1 \
      || { echo "bench $side c=$c failed:"; tail -3 "$OUT/${side}_c${c}_p${p}_o${o}.log"; exit 1; }
    python3 - "$f" "$c" "$p" "$o" "$side" >> "$MD" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); c, p, o, side = sys.argv[2:6]
f = lambda x: f"{x:.1f}"
print(f"| {c} | {p} | {o} | {side} | {d['completed']}/{d['num_requests']} | {d['request_throughput_rps']:.2f} | {f(d['output_token_throughput_tps'])} | "
      f"{f(d['ttft_ms']['p50'])} / {f(d['ttft_ms']['p99'])} | {f(d['tpot_ms']['p50'])} / {f(d['tpot_ms']['p99'])} | {f(d['itl_ms']['p50'])} | {f(d['latency_ms']['p50'])} |")
PY
    grep -E 'throughput|TTFT|TPOT' "$OUT/${side}_c${c}_p${p}_o${o}.log" | sed "s/^/  [$side c=$c p=$p o=$o] /"
  done
}
run_one 1 128 128
run_one 8 128 128
run_one 32 128 128
run_one 8 1024 128
echo; cat "$MD"
