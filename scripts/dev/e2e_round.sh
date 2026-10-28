#!/usr/bin/env bash
# Step C 端到端结果入库一行（loop kaon3-decode-opt，用户 2026-10-28：每轮含回滚都进 CSV + dashboard）。
#   scripts/dev/e2e_round.sh <model> <round> "<change 一句话>" <group> <d_tps> <v_tps> <d_ttft> <v_ttft> <d_tpot> <v_tpot> [gpu]
# 追加 3 行（out_tok_per_s / ttft_p50_ms / tpot_p50_ms）到 test/<model>/e2e-rounds.csv，再重生成 perf-trend.*。
set -euo pipefail
cd "$(dirname "$0")/../.."
m=$1 r=$2 chg=$3 g=$4 dt=$5 vt=$6 dtt=$7 vtt=$8 dtp=$9 vtp=${10} gpu=${11:-5090}
f=test/$m/e2e-rounds.csv
[ -f "$f" ] || head -1 test/$m/perf-trend.csv > "$f"
date=$(date +%F)
chg=${chg//\"/\'}
row() { python3 -I -c "import sys;a,b=float(sys.argv[1]),float(sys.argv[2]);print(f'{a:.3f},{b:.3f},{(a/b if b else 0):.3f}')" "$2" "$3" | { read v; echo "$r,$date,\"$chg\",$g,$1,$v,,$4,main,$gpu,vllm,dinfer,$m"; }; }
row out_tok_per_s "$dt" "$vt" higher >> "$f"
row ttft_p50_ms "$dtt" "$vtt" lower >> "$f"
row tpot_p50_ms "$dtp" "$vtp" lower >> "$f"
python3 -I scripts/dev/bench_trend.py --root test/$m | tail -1
tail -3 "$f" | cut -c1-160
