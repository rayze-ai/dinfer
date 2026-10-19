#!/usr/bin/env bash
# dinfer vs vLLM 对比压测（在 4090 上跑）。同一 bench 二进制、同 prompt 集（seed）、同 max_tokens、ignore_eos、思考关。
#   bash scripts/verify/bench_compare.sh [OUT_DIR] ["本轮改动一句话"]   # 默认 test/qwen3/$(date +%F)/bench；改动写进 OUT/CHANGE
# 矩阵：并发 1 / 8 / 32 × (prompt 128, out 128)，再加一组长 prompt (1024, out 128) 并发 8。每组先 bench --self-check 一次。
# 环境变量：DINFER_URL / VLLM_URL（默认 :8012 / :8013）、REQS（每组请求数，默认 = 4×并发，最少 16）。
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-test/qwen3/$(date +%F)/bench}"; mkdir -p "$OUT"
# 本轮改了什么（进趋势表）：第二个参数或 $CHANGE
[ -n "${2:-${CHANGE:-}}" ] && printf '%s\n' "${2:-$CHANGE}" > "$OUT/CHANGE"
# 卡型（进趋势表的 gpu 维度；bench_trend.py 没有该文件按 4090）：含 4090 → 4090，含 B300 → B300，其余取 name 去空格
gpu_tag() { local n; n=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1); case "$n" in *4090*) echo 4090;; *B300*) echo B300;; *) echo "${n// /}";; esac; }
[ -n "$(gpu_tag)" ] && gpu_tag > "$OUT/GPU"
BENCH=./target/release/bench
# 每次运行换 seed（默认取当前秒）：prompt 不与历史轮次重复，两边都只命中聊天模板前缀的 prefix cache。固定 seed 会让常驻 vLLM 全命中、重启的 dinfer 冷启动，口径不公（pitfalls 2026-10-16）。
SEED="${SEED:-$(date +%s)}"
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
  echo "seed：$SEED（每次运行不同，prompt 不与历史轮次重复；两边 prefix cache 只命中模板前缀）"
  echo
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
    "$BENCH" --url "$url" --label "$side" --concurrency "$c" --num-requests "$reqs" --prompt-len "$p" --max-tokens "$o" --seed "$SEED" --out "$f" > "$OUT/${side}_c${c}_p${p}_o${o}.log" 2>&1 \
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
# 趋势：所有轮次汇总成表 + 曲线（用户要求：每轮改动在每个指标上的提升）
python3 -I scripts/dev/bench_trend.py --root "$(dirname "$(dirname "$OUT")")" | tail -n +2 | tail -3
