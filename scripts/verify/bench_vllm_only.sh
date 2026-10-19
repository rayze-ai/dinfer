#!/usr/bin/env bash
# 只跑 vLLM 基线（dinfer 侧尚无该模型时用，如 Qwen3.5-27B 里程碑 4 之前）。口径与 bench_compare.sh 完全一致（四组、
# 每组前 2 个预热不计、ignore_eos、temperature 0、每次换 seed），文件名 vllm_c<c>_p<p>_o<o>.{json,log}，之后 dinfer 跑同目录
# bench_compare.sh 即可补齐对比。用户 2026-10-19："qwen3.5 可以在 B300-50 上做基线测试了"。
#   bash scripts/verify/bench_vllm_only.sh <OUT_DIR> <served-model-name> ["改动一句话"]   # VLLM_URL 默认 :8013
set -uo pipefail
OUT="$1"; MODEL_NAME="$2"; mkdir -p "$OUT"
[ -n "${3:-}" ] && printf '%s\n' "$3" > "$OUT/CHANGE"
gpu_tag() { local n; n=$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader 2>/dev/null); case "$n" in *4090*) echo 4090;; *B300*) echo B300;; *) echo "$n" | tr -d " ";; esac; }
[ -n "$(gpu_tag)" ] && gpu_tag > "$OUT/GPU"
printf '%s\n' "$MODEL_NAME" > "$OUT/MODEL"
B="${VLLM_URL:-http://127.0.0.1:8013}"
BENCH=./target/release/bench; SEED=${SEED:-$(date +%s)}
MD="$OUT/summary.md"
{
  echo "# vLLM 基线（$(date '+%F %T')，$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader)）"
  echo; echo "vLLM: $B（$MODEL_NAME，$(curl -s -m 3 $B/version 2>/dev/null | tr -d '\n' | cut -c1-80)）；seed $SEED"
  echo; echo "口径：流式；TTFT = 首个带内容 chunk；TPOT = (latency − TTFT)/(out−1)；ignore_eos=true；temperature 0；enable_thinking=false；每组前 2 个预热请求不计。"
  echo; echo "| 并发 | prompt | out | 端 | 请求 | req/s | out tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | ITL p50 (ms) | latency p50 (ms) |"
  echo "|---|---|---|---|---|---|---|---|---|---|---|"
} > "$MD"
row() { python3 -I - "$1" "$2" "$3" "$4" "$5" <<'PY'
import json,sys
c,p,o,side,f=sys.argv[1:6]; j=json.load(open(f))
g=lambda k,d=float('nan'): j.get(k,d)
print(f"| {c} | {p} | {o} | {side} | {g('completed','?')}/{g('num_requests','?')} | {g('req_per_s',0):.2f} | {g('output_tok_per_s',0):.1f} | {g('ttft_p50_ms',0):.1f} / {g('ttft_p99_ms',0):.1f} | {g('tpot_p50_ms',0):.2f} / {g('tpot_p99_ms',0):.2f} | {g('itl_p50_ms',0):.2f} | {g('latency_p50_ms',0):.1f} |")
PY
}
for grp in "1 128 128" "8 128 128" "32 128 128" "8 1024 128"; do
  set -- $grp; c=$1; p=$2; o=$3
  reqs=${REQS:-$(( c * 4 > 16 ? c * 4 : 16 ))}
  f="$OUT/vllm_c${c}_p${p}_o${o}.json"
  "$BENCH" --url "$B" --model "$MODEL_NAME" --label vllm --concurrency "$c" --num-requests "$reqs" --prompt-len "$p" --max-tokens "$o" --temperature 0 --seed "$SEED" --warmup 2 --out "$f" > "$OUT/vllm_c${c}_p${p}_o${o}.log" 2>&1 \
    && row "$c" "$p" "$o" vllm "$f" >> "$MD" || { echo "bench vllm c=$c failed:"; tail -3 "$OUT/vllm_c${c}_p${p}_o${o}.log"; }
done
grep -E "^\| (1|8|32) " "$MD"
