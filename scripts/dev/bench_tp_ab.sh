#!/usr/bin/env bash
# 同机 single vs --tp N 的 A/B（无 vLLM 基线的机器，如 B300；docs/loops/b-nvlink.md B1）。产出 bench_trend.py 的 ab_ 目录布局：
#   <OUT>/{single,tp2,tp4,...}__c<c>_p<p>_o<o>.json + CHANGE；bench_trend 以字典序 / "single" 为基线，HTML 进 tp 页签。
#   bash scripts/dev/bench_tp_ab.sh <OUT_DIR> "<改动一句话>" <BASE_GPU> <TP 列表，如 "2 4 8"> [extra serve args]
# 每个变体：起 serve（--device BASE [--tp N]）→ 四组 bench（c1/c8/c32 p128 o128 + c8 p1024 o128）→ 停。固定 --num-blocks 4000。
set -uo pipefail
OUT="$1"; CHANGE="$2"; BASE="$3"; TPS="$4"; shift 4
R="${R:-/data/yaochen/infer}"; PORT="${PORT:-8341}"; BENCH=./target/release/bench; BIN=./target/release/dinfer
mkdir -p "$OUT"; echo "$CHANGE" > "$OUT/CHANGE"
gpu_tag() { local n; n=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1); case "$n" in *4090*) echo 4090;; *B300*) echo B300;; *) echo "${n// /}";; esac; }
[ -n "$(gpu_tag)" ] && gpu_tag > "$OUT/GPU"   # 卡型，bench_trend.py 的 gpu 维度
[ -x "$BENCH" ] || cargo build --release -p dinfer-bench
SEED=$(( $(date +%s) % 100000 ))
# ab_serve.sh 用 CUDA_VISIBLE_DEVICES=<gpus> 钉卡并自带 --model；tpN 给 BASE..BASE+N-1 的逗号卡表，serve 内 --device 0 起
gpus() { local n=$1 s=""; for i in $(seq 0 $((n-1))); do s+="${s:+,}$((BASE+i))"; done; echo "$s"; }
one() { # variant ngpu [serve args]
  local var=$1 n=$2; shift 2
  bash scripts/dev/ab_serve.sh start "$PORT" "$(gpus "$n")" --max-model-len 8192 --max-num-seqs 64 \
    --max-num-batched-tokens 2048 --gpu-memory-utilization 0.85 --num-blocks 4000 "$@" | tail -2
  curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null || { echo "$var serve DOWN"; bash scripts/dev/ab_serve.sh stop "$PORT"; return 1; }
  for g in "1 128 128" "8 128 128" "32 128 128" "8 1024 128"; do
    set -- $g; local c=$1 p=$2 o=$3 reqs=$(( $1 * 4 > 16 ? $1 * 4 : 16 ))
    "$BENCH" --url "http://127.0.0.1:$PORT" --label "$var" --concurrency "$c" --num-requests "$reqs" --prompt-len "$p" --max-tokens "$o" --seed "$SEED" \
      --out "$OUT/${var}__c${c}_p${p}_o${o}.json" > "$OUT/${var}__c${c}_p${p}_o${o}.log" 2>&1 || { echo "bench $var c=$c failed"; tail -3 "$OUT/${var}__c${c}_p${p}_o${o}.log"; }
    python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(f'  [{sys.argv[2]} c={sys.argv[3]} p={sys.argv[4]}] {d[\"output_token_throughput_tps\"]:.0f} tok/s  TTFT {d[\"ttft_ms\"][\"p50\"]:.1f}  TPOT {d[\"tpot_ms\"][\"p50\"]:.2f}')" "$OUT/${var}__c${c}_p${p}_o${o}.json" "$var" "$c" "$p" 2>/dev/null
  done
  bash scripts/dev/ab_serve.sh stop "$PORT" >/dev/null
}
one single 1 "$@"
for tp in $TPS; do one "tp$tp" "$tp" --tp "$tp" "$@"; done
python3 -I scripts/dev/bench_trend.py --root "$(dirname "$(dirname "$OUT")")" | tail -n +2 | tail -3 || true
exit 0
