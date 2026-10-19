#!/usr/bin/env bash
# openai Python SDK 冒烟（docs/loops/api-complete.md）：起一个 Qwen3-0.6B serve（本机 3090，--num-blocks 256 显存很小），
# 等 /health 200 后跑 scripts/verify/sdk_smoke.py（python3 -I），无论成败 kill serve。exit 0 = 通过。
# 可调：QWEN3_DIR（权重目录）、BIN（dinfer 可执行文件；默认取 target/{release,debug}/dinfer 中较新的那个）、GPU_WAIT_MAX（秒）。
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
QWEN3="${QWEN3_DIR:-/data/ref/models/qwen3-0.6b}"
GPU_WAIT_MAX="${GPU_WAIT_MAX:-600}"

# 可执行文件：优先 release，但 debug 更新时用 debug（release 可能是旧构建、缺新端点）
if [[ -z "${BIN:-}" ]]; then
  BIN=""
  for b in target/release/dinfer target/debug/dinfer; do
    [[ -x "$b" ]] || continue
    if [[ -z "$BIN" || "$b" -nt "$BIN" ]]; then BIN="$b"; fi
  done
  [[ -n "$BIN" ]] || { echo "no dinfer binary; run: cargo build --release -p dinfer --features qwen3" >&2; exit 2; }
fi
echo "==> bin: $BIN"

# 等别的 dinfer 进程让出 GPU（和其他 e2e 共存一般没问题，但别同时起两个自己的）
waited=0
while pgrep -x dinfer >/dev/null 2>&1; do
  if (( waited >= GPU_WAIT_MAX )); then echo "other dinfer still running after ${GPU_WAIT_MAX}s" >&2; exit 3; fi
  echo "==> other dinfer running (pids: $(pgrep -x dinfer | tr '\n' ' ')); wait 30s"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null | tail -n +2 | head -5 || true
  sleep 30; waited=$((waited + 30))
done

PORT=$(python3 -I -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
LOG=/tmp/sdk_smoke_serve.log
RUST_LOG="${RUST_LOG:-info}" "$BIN" serve --model "$QWEN3" --port "$PORT" \
  --max-model-len 2048 --max-num-batched-tokens 512 --num-blocks 256 >"$LOG" 2>&1 &
SERVE_PID=$!
cleanup() { kill "$SERVE_PID" 2>/dev/null || true; wait "$SERVE_PID" 2>/dev/null || true; }
trap cleanup EXIT
echo "==> serve pid=$SERVE_PID port=$PORT log=$LOG"

for _ in $(seq 1 180); do
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then echo "serve exited early:" >&2; tail -20 "$LOG" >&2; exit 4; fi
  if [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" || true)" == "200" ]]; then break; fi
  sleep 1
done
[[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" || true)" == "200" ]] || { echo "health timeout" >&2; tail -20 "$LOG" >&2; exit 5; }

echo "==> sdk_smoke.py"
rc=0
python3 -I scripts/verify/sdk_smoke.py "http://127.0.0.1:$PORT" "$@" || rc=$?
if (( rc != 0 )); then echo "--- serve log tail ---"; tail -30 "$LOG" | cut -c1-200; fi
exit "$rc"
