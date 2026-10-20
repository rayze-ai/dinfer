#!/usr/bin/env bash
# B300：Kimi-K3 前 N 层 serve 冒烟（docs/loops/k3-serve.md）。启动 dinfer（前 N 层）→ /v1/models、/metrics、流式 chat 一条 → bench c1 ×8 →
# 收集日志。exit 0 = 全部步骤通过（内容乱是预期，只看流程）。在远端直接跑：
#   K3_LAYERS=5 K3_GPU=2 PORT=8014 bash scripts/dev/k3_b300_smoke.sh
# 产物：$OUT（默认 /data/yaochen/infer/bootstrap/k3serve/smoke${K3_LAYERS}）：serve.log / models.json / metrics.txt / chat_stream.txt /
#       bench_c1.json / bench_c8.json / summary.txt。拷回仓库 test/kimik3/<日期>/smoke<N>/。
set -uo pipefail
source /data/yaochen/infer/env.sh
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
cd /data/yaochen/infer/dinfer
LAYERS="${K3_LAYERS:-5}"
GPU="${K3_GPU:-2}"
PORT="${PORT:-8014}"
MODEL="${K3_MODEL:-/data/models/moonshotai/Kimi-K3}"
OUT="${OUT:-/data/yaochen/infer/bootstrap/k3serve/smoke${LAYERS}}"
MAX_SEQS="${MAX_SEQS:-16}"
BUDGET="${BUDGET:-2048}"
URL="http://127.0.0.1:$PORT"
mkdir -p "$OUT"
rc=0
step() { printf '\n==> %s\n' "$*" | tee -a "$OUT/summary.txt"; }
fail() { printf 'FAIL: %s\n' "$*" | tee -a "$OUT/summary.txt"; rc=1; }
: > "$OUT/summary.txt"

# 残留进程
pkill -f "dinfer serve .*--port $PORT" 2>/dev/null && sleep 2

step "启动 dinfer（前 $LAYERS 层，GPU $GPU，:$PORT）"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" RUST_LOG="${RUST_LOG:-info}"
T0=$(date +%s.%N)
nohup target/release/dinfer serve --model "$MODEL" --layers "$LAYERS" --port "$PORT" \
  --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$BUDGET" --max-model-len 4096 "$@" > "$OUT/serve.log" 2>&1 < /dev/null &
SPID=$!
for i in $(seq 1 240); do
  curl -sf -m 2 "$URL/health" > /dev/null 2>&1 && break
  kill -0 "$SPID" 2>/dev/null || { fail "dinfer 进程退出"; tail -20 "$OUT/serve.log" | cut -c1-200; exit 1; }
  sleep 2
done
T1=$(date +%s.%N)
curl -sf -m 2 "$URL/health" > /dev/null 2>&1 || { fail "health 超时（480 s）"; tail -20 "$OUT/serve.log" | cut -c1-200; kill "$SPID"; exit 1; }
printf 'startup_wall_s=%.1f\n' "$(echo "$T1 - $T0" | bc)" | tee -a "$OUT/summary.txt"
grep -E 'TRUNCATED|model loaded|engine started|startup timing|kimik3 eos|single-stream' "$OUT/serve.log" | cut -c1-260 | tee -a "$OUT/summary.txt"
grep -q 'TRUNCATED' "$OUT/serve.log" || fail "日志没有 TRUNCATED MODEL 标记"

step "/v1/models"
curl -sf -m 5 "$URL/v1/models" > "$OUT/models.json" || fail "/v1/models"
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print([m["id"] for m in d["data"]])' "$OUT/models.json" | tee -a "$OUT/summary.txt"

step "流式 chat 一条（内容乱是预期）"
MODEL_ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["data"][0]["id"])' "$OUT/models.json")
curl -sN -m 300 "$URL/v1/chat/completions" -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"用一句话介绍你自己\"}],\"max_tokens\":64,\"stream\":true}" \
  > "$OUT/chat_stream.txt" || fail "chat stream curl"
python3 - "$OUT/chat_stream.txt" <<'PY' | tee -a "$OUT/summary.txt"
import json, sys
n = 0; fin = None; reasoning = ""; content = ""; err = None
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    line = line.strip()
    if not line.startswith("data: "):
        if line.startswith("{"): err = line[:200]
        continue
    d = line[6:]
    if d == "[DONE]": break
    ch = json.loads(d)["choices"][0]; n += 1
    delta = ch.get("delta", {})
    reasoning += delta.get("reasoning_content") or ""
    content += delta.get("content") or ""
    if ch.get("finish_reason"): fin = ch["finish_reason"]
print(f"chunks={n} finish={fin} reasoning_chars={len(reasoning)} content_chars={len(content)} err={err}")
print("reasoning[:120]=", repr(reasoning[:120]))
print("content[:120]=", repr(content[:120]))
sys.exit(0 if (n > 0 and fin in ("stop", "length")) else 1)
PY
[ "${PIPESTATUS[0]}" = 0 ] || fail "流式 chat 没有 chunk / 没有 finish_reason"

step "/metrics"
curl -sf -m 5 "$URL/metrics" > "$OUT/metrics.txt" || fail "/metrics"
grep -cE '^[a-z_]+(\{|\s)' "$OUT/metrics.txt" | sed 's/^/metrics_lines=/' | tee -a "$OUT/summary.txt"

step "bench c1 ×8（p128 o128，只记录）"
target/release/bench --url "$URL" --label dinfer --concurrency 1 --num-requests 8 --prompt-len 128 --max-tokens 128 \
  --seed 7 --out "$OUT/bench_c1.json" > "$OUT/bench_c1.log" 2>&1 || fail "bench c1"
grep -iE 'tpot|ttft|tok/s|success|fail|error' "$OUT/bench_c1.log" | head -8 | cut -c1-200 | tee -a "$OUT/summary.txt"

step "bench c8 ×16（混批 / 槽归还）"
target/release/bench --url "$URL" --label dinfer --concurrency 8 --num-requests 16 --prompt-len 128 --max-tokens 128 \
  --seed 8 --out "$OUT/bench_c8.json" > "$OUT/bench_c8.log" 2>&1 || fail "bench c8"
grep -iE 'tpot|ttft|tok/s|success|fail|error' "$OUT/bench_c8.log" | head -8 | cut -c1-200 | tee -a "$OUT/summary.txt"

step "日志检查"
if grep -qiE 'panicked|CUDA error|CUDA_ERROR|illegal' "$OUT/serve.log"; then fail "serve.log 有 panic / CUDA error"; grep -iE 'panicked|CUDA error|CUDA_ERROR|illegal' "$OUT/serve.log" | head -5 | cut -c1-200; fi
grep -E 'decode step via cuda graph|slot 被复用|warn' "$OUT/serve.log" | sort | uniq -c | sort -rn | head -6 | cut -c1-200 | tee -a "$OUT/summary.txt"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$GPU" | tee -a "$OUT/summary.txt"

kill "$SPID" 2>/dev/null; sleep 2; kill -9 "$SPID" 2>/dev/null
printf '\nsmoke_rc=%s\n' "$rc" | tee -a "$OUT/summary.txt"
exit $rc
