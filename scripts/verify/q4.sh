#!/usr/bin/env bash
# Q4 PD 分离验收（design §10.3 / docs/loops/q4-pd.md）。exit 0 = 通过。
# 本机 3090：只跑 e2e（同卡 single + P + D + proxy 四进程）。Q4_BENCH_GATE=0：bench 只记录进 CSV 不判门槛（v1 TCP 同卡不达标，见 loop 第 11 轮）。
# 4090（Q4_REMOTE=1）：GPU 3 起 P(:8101, kv 9101) + D(:8102, kv 9102) + proxy(:8100)，杀 P / 杀 D / 断 kv-port 三故障，
#   再 bench c8 (1024,128) 打 proxy vs 单机 :8012，写 test/qwen3/<日期>/pd1 进 CSV。
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-test/qwen3/$(date +%F)/q4}"; mkdir -p "$OUT"
step() { rm -f target/.rustc_info.json; printf '\n==> %s\n' "$*"; }
MODEL="${QWEN3_DIR:-/data/ref/models/qwen3-0.6b}"
[ -d "$MODEL" ] || { echo "model dir not found: $MODEL (set QWEN3_DIR; 4090: /data/yaochen/infer/models/qwen3-0.6b)"; exit 1; }
# 共享机：GPU 0/1 常驻服务，测试一律钉在 Q4_GPU（默认 3）；本机单卡设 Q4_GPU=0
GPU="${Q4_GPU:-3}"
export CUDA_VISIBLE_DEVICES="$GPU"
step "e2e_qwen3_pd (single + prefill + decode + proxy, same GPU)"
QWEN3_DIR="$MODEL" QWEN3_GOLDEN="$(realpath -m test/qwen3_golden)" cargo test -p dinfer --features qwen3,e2e --test serve_e2e_qwen3 -- --ignored --test-threads=1 e2e_qwen3_pd </dev/null 2>&1 | tee "$OUT/e2e.log" | grep -E '^test |test result|panicked'
grep -qE 'test result: ok\. 1 passed' "$OUT/e2e.log"
[ "${Q4_REMOTE:-0}" = 1 ] || { echo "==> Q4 e2e PASS (set Q4_REMOTE=1 on the 4090 box for fault + bench)"; exit 0; }

# ---------------- 4090 部分 ----------------
BIN=./target/release/dinfer
# 同一张卡跑 P + D：显存预算要分开给（默认 0.9 会让第二个进程 OOM）。P 只持在飞请求的 KV，D 要大池。
COMMON=(--model "$MODEL" --host 0.0.0.0 --max-model-len 4096 --max-num-seqs 32 --max-num-batched-tokens 2048 --device 0) # device 0 = CUDA_VISIBLE_DEVICES 里的第一张
# 用固定块数而不是 utilization：自动预算 = min(free, total×u − used)，同卡第二个进程时 used 已含第一个进程，
# u=0.3 → cap−used ≈ 0 → "not enough GPU memory"。P 只持在飞请求的 KV（512 块 = 8k token），D 大池（2048 块 = 32k token）。
P_BLOCKS="${Q4_P_BLOCKS:-512}"; D_BLOCKS="${Q4_D_BLOCKS:-2048}"
start_d() { CUDA_VISIBLE_DEVICES=$GPU RUST_LOG=info nohup $BIN serve "${COMMON[@]}" --port 8102 --role decode --kv-port 9102 --num-blocks $D_BLOCKS > "$OUT/d.log" 2>&1 & echo $! > "$OUT/d.pid"; }
start_p() { CUDA_VISIBLE_DEVICES=$GPU RUST_LOG=info nohup $BIN serve "${COMMON[@]}" --port 8101 --role prefill --peer 127.0.0.1:9102 --num-blocks $P_BLOCKS > "$OUT/p.log" 2>&1 & echo $! > "$OUT/p.pid"; }
start_x() { RUST_LOG=info nohup $BIN proxy --host 0.0.0.0 --port 8100 --prefill http://127.0.0.1:8101 --decode http://127.0.0.1:8102 > "$OUT/proxy.log" 2>&1 & echo $! > "$OUT/x.pid"; }
wait_up() { for _ in $(seq 1 240); do curl -s -m 2 "http://127.0.0.1:$1/health" >/dev/null && return 0; sleep 0.5; done; echo "port $1 did not come up"; for f in d p proxy p_badpeer; do [ -f "$OUT/$f.log" ] && { echo "--- $f.log tail"; grep -vE " INFO " "$OUT/$f.log" | tail -n 4; }; done; return 1; }
stop_all() { for f in x p d; do [ -f "$OUT/$f.pid" ] && kill "$(cat "$OUT/$f.pid")" 2>/dev/null || true; done; sleep 2; }
trap stop_all EXIT
ask() { curl -s -m 60 "http://127.0.0.1:$1/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"List three primary colors, comma separated, then stop."}],"max_tokens":24,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 -c 'import sys,json; v=json.load(sys.stdin); print(v["choices"][0]["message"]["content"])'; }
# 流式同问题，拼接 delta.content（v2 (c)：proxy 先发 P 的首 chunk，再接 D；回退时 D 从 prompt+首 token 续写）
ask_stream() { curl -s -N -m 60 "http://127.0.0.1:$1/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"List three primary colors, comma separated, then stop."}],"max_tokens":24,"temperature":0,"stream":true,"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 -c '
import sys,json
out=""
for line in sys.stdin:
    if not line.startswith("data: "): continue
    d=line[6:].strip()
    if d=="[DONE]": break
    c=json.loads(d)["choices"][0]["delta"].get("content")
    if c: out+=c
print(out)'; }
metric() { curl -s "http://127.0.0.1:$1/metrics" | awk -v n="dinfer:$2" '$1==n{print $2}'; }

step "start D, P, proxy on GPU $GPU"
pgrep -af "dinfer (serve|proxy)" | grep -E "810[0-2]" && { echo "stale PD processes on 810x; kill them first"; exit 1; }
stop_all; start_d; wait_up 8102; start_p; wait_up 8101; start_x; wait_up 8100
WANT="$(ask 8012)"; echo "single : $WANT"
GOT="$(ask 8100)";  echo "pd     : $GOT"; [ "$GOT" = "$WANT" ]
GOT="$(ask_stream 8100)"; [ "$GOT" = "$WANT" ] || { echo "pd stream mismatch: [$GOT] vs [$WANT]"; exit 1; }
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
CUDA_VISIBLE_DEVICES=$GPU RUST_LOG=info nohup $BIN serve "${COMMON[@]}" --port 8101 --role prefill --peer 127.0.0.1:1 --num-blocks $P_BLOCKS > "$OUT/p_badpeer.log" 2>&1 & echo $! > "$OUT/p.pid"; wait_up 8101
GOT="$(ask 8100)"; [ "$GOT" = "$WANT" ]; [ "$(metric 8102 pd_fallback_total)" -ge 1 ]
GOT="$(ask_stream 8100)"; [ "$GOT" = "$WANT" ] || { echo "stream fallback mismatch: [$GOT] vs [$WANT]"; exit 1; }
kill "$(cat "$OUT/p.pid")"; sleep 1; start_p; wait_up 8101

step "bench c8 (1024,128): proxy :8100 vs single :8012 → $OUT/../ab_pd1 (CSV: variant single 基线 vs pd)"
AB="$(dirname "$OUT")/ab_pd1"; mkdir -p "$AB"
B=./target/release/bench
$B --url http://127.0.0.1:8012 --concurrency 8 --num-requests 64 --prompt-len 1024 --max-tokens 128 --out "$AB/single__c8_p1024_o128.json" > "$AB/single.log" 2>&1 || { tail -3 "$AB/single.log"; exit 1; }
$B --url http://127.0.0.1:8100 --concurrency 8 --num-requests 64 --prompt-len 1024 --max-tokens 128 --out "$AB/pd__c8_p1024_o128.json" > "$AB/pd.log" 2>&1 || { tail -3 "$AB/pd.log"; exit 1; }
echo "PD ${Q4_VARIANT:-v2 (c) 首 token 先发} TCP：pd = proxy(:8100 → P GPU$GPU + D GPU$GPU) vs single = 单机 :8012，c8 (1024,128)" > "$AB/CHANGE"
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
gate = os.environ.get("Q4_BENCH_GATE", "1") == "1"
ok = ttft_ok and tpot_ok and a.get("failed", 0) == 0
print("bench gate:", "enforced" if gate else "record-only", "→", "PASS" if ok else "FAIL")
sys.exit(0 if (ok or not gate) else 1)
PY
python3 scripts/dev/bench_trend.py >/dev/null 2>&1 || true
echo "==> Q4 PASS"
