#!/usr/bin/env bash
# Q4 PD 分离验收（design §10.3 / docs/loops/q4-pd.md）。exit 0 = 通过。
# 本机 3090：只跑 e2e（同卡 single + P + D + proxy 四进程）。
# 4090（Q4_REMOTE=1）：GPU 3 起 P(:8101, kv 9101) + D(:8102, kv 9102) + proxy(:8100)，杀 P / 杀 D / 断 kv-port 三故障，
#   再 bench c8 (1024,128) 打 proxy vs 单机 :8012，写 test/qwen3/<日期>/pd1 进 CSV。
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-test/qwen3/$(date +%F)/q4}"; mkdir -p "$OUT"
step() { rm -f target/.rustc_info.json; printf '\n==> %s\n' "$*"; }
MODEL="${QWEN3_DIR:-/data/ref/models/qwen3-0.6b}"
step "e2e_qwen3_pd (single + prefill + decode + proxy, same GPU)"
QWEN3_DIR="$MODEL" QWEN3_GOLDEN="$(realpath -m test/qwen3_golden)" cargo test -p dinfer --features qwen3,e2e --test serve_e2e_qwen3 -- --ignored --test-threads=1 e2e_qwen3_pd </dev/null 2>&1 | tee "$OUT/e2e.log" | grep -E '^test |test result|panicked'
grep -qE 'test result: ok\. 1 passed' "$OUT/e2e.log"
[ "${Q4_REMOTE:-0}" = 1 ] || { echo "==> Q4 e2e PASS (set Q4_REMOTE=1 on the 4090 box for fault + bench)"; exit 0; }

# ---------------- 4090 部分 ----------------
BIN=./target/release/dinfer
GPU="${Q4_GPU:-3}"
COMMON=(--model "$MODEL" --host 0.0.0.0 --max-model-len 4096 --max-num-seqs 32 --max-num-batched-tokens 2048 --device 0)
start_d() { CUDA_VISIBLE_DEVICES=$GPU RUST_LOG=info nohup $BIN serve "${COMMON[@]}" --port 8102 --role decode --kv-port 9102 > "$OUT/d.log" 2>&1 & echo $! > "$OUT/d.pid"; }
start_p() { CUDA_VISIBLE_DEVICES=$GPU RUST_LOG=info nohup $BIN serve "${COMMON[@]}" --port 8101 --role prefill --peer 127.0.0.1:9102 > "$OUT/p.log" 2>&1 & echo $! > "$OUT/p.pid"; }
start_x() { RUST_LOG=info nohup $BIN proxy --host 0.0.0.0 --port 8100 --prefill http://127.0.0.1:8101 --decode http://127.0.0.1:8102 > "$OUT/proxy.log" 2>&1 & echo $! > "$OUT/x.pid"; }
wait_up() { for _ in $(seq 1 240); do curl -s -m 2 "http://127.0.0.1:$1/health" >/dev/null && return 0; sleep 0.5; done; echo "port $1 did not come up"; return 1; }
stop_all() { for f in x p d; do [ -f "$OUT/$f.pid" ] && kill "$(cat "$OUT/$f.pid")" 2>/dev/null || true; done; sleep 2; }
trap stop_all EXIT
ask() { curl -s -m 60 "http://127.0.0.1:$1/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"List three primary colors, comma separated, then stop."}],"max_tokens":24,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 -c 'import sys,json; v=json.load(sys.stdin); print(v["choices"][0]["message"]["content"])'; }
metric() { curl -s "http://127.0.0.1:$1/metrics" | awk -v n="dinfer:$2" '$1==n{print $2}'; }

step "start D, P, proxy on GPU $GPU"
stop_all; start_d; wait_up 8102; start_p; wait_up 8101; start_x; wait_up 8100
WANT="$(ask 8012)"; echo "single : $WANT"
GOT="$(ask 8100)";  echo "pd     : $GOT"; [ "$GOT" = "$WANT" ]
[ "$(metric 8101 pd_export_total)" -ge 1 ] && [ "$(metric 8102 pd_import_total)" -ge 1 ] && [ "$(metric 8102 pd_fallback_total)" = 0 ]

step "fault 1: kill P → proxy falls back to D-only"
kill "$(cat "$OUT/p.pid")"; sleep 1; GOT="$(ask 8100)"; [ "$GOT" = "$WANT" ]; start_p; wait_up 8101
step "fault 2: kill D → proxy returns 5xx, no hang; restart D"
kill "$(cat "$OUT/d.pid")"; sleep 1
code=$(curl -s -o /dev/null -w '%{http_code}' -m 30 "http://127.0.0.1:8100/v1/chat/completions" -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}')
echo "with D dead: http $code"; [ "$code" -ge 500 ]
start_d; wait_up 8102
step "fault 3: P cannot reach kv-port → D times out, serves local prefill, output still equal"
kill "$(cat "$OUT/p.pid")"; sleep 1
CUDA_VISIBLE_DEVICES=$GPU RUST_LOG=info nohup $BIN serve "${COMMON[@]}" --port 8101 --role prefill --peer 127.0.0.1:1 > "$OUT/p_badpeer.log" 2>&1 & echo $! > "$OUT/p.pid"; wait_up 8101
GOT="$(ask 8100)"; [ "$GOT" = "$WANT" ]; [ "$(metric 8102 pd_fallback_total)" -ge 1 ]
kill "$(cat "$OUT/p.pid")"; sleep 1; start_p; wait_up 8101

step "bench c8 (1024,128): proxy :8100 vs single :8012 → $OUT/../ab_pd1 (CSV: variant single 基线 vs pd)"
AB="$(dirname "$OUT")/ab_pd1"; mkdir -p "$AB"
B=./target/release/bench
$B --url http://127.0.0.1:8012 --concurrency 8 --num-requests 64 --prompt-len 1024 --max-tokens 128 --out "$AB/single__c8_p1024_o128.json" > "$AB/single.log" 2>&1 || { tail -3 "$AB/single.log"; exit 1; }
$B --url http://127.0.0.1:8100 --concurrency 8 --num-requests 64 --prompt-len 1024 --max-tokens 128 --out "$AB/pd__c8_p1024_o128.json" > "$AB/pd.log" 2>&1 || { tail -3 "$AB/pd.log"; exit 1; }
echo "PD v1 TCP：pd = proxy(:8100 → P GPU$GPU + D GPU$GPU) vs single = 单机 :8012，c8 (1024,128)" > "$AB/CHANGE"
python3 - "$AB" <<'PY'
import json, sys, os
d = sys.argv[1]
a = json.load(open(os.path.join(d, "pd__c8_p1024_o128.json"))); b = json.load(open(os.path.join(d, "single__c8_p1024_o128.json")))
rows = [("ttft_p50_ms", a["ttft_ms"]["p50"], b["ttft_ms"]["p50"]), ("tpot_p50_ms", a["tpot_ms"]["p50"], b["tpot_ms"]["p50"]),
        ("out_tok/s", a["output_token_throughput_tps"], b["output_token_throughput_tps"]), ("failed", a.get("failed", 0), b.get("failed", 0))]
for k, x, y in rows:
    print(f"{k:12s} pd {x:9.2f}  single {y:9.2f}")
ttft_ok = a["ttft_ms"]["p50"] <= 1.3 * b["ttft_ms"]["p50"]
tpot_ok = a["tpot_ms"]["p50"] <= b["tpot_ms"]["p50"] + 0.05
print("ttft<=1.3x:", ttft_ok, " tpot<=1.0x(+0.05ms):", tpot_ok, " failed:", a.get("failed", 0))
sys.exit(0 if (ttft_ok and tpot_ok and a.get("failed", 0) == 0) else 1)
PY
python3 scripts/dev/bench_trend.py >/dev/null 2>&1 || true
echo "==> Q4 PASS"
