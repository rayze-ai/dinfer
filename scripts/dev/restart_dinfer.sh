#!/usr/bin/env bash
# 4090 常驻 dinfer 的可靠重启：按端口模式 kill 旧进程（tmux kill-session 杀不到 tee 管道里的子进程，会留下孤儿继续占端口——
# 2026-10-17 发现 :8012 跑了 20 小时的旧进程，bench16–22 的 GPU 0 数字部分打在它上面），等显存释放，再起新的。
#   bash scripts/dev/restart_dinfer.sh [extra serve args...]      # 在 4090 上跑；默认参数见下
set -euo pipefail
R=/data/yaochen/infer
PORT="${PORT:-8012}"; DEV="${DEV:-0}"; UTIL="${UTIL:-0.9}"; BUDGET="${BUDGET:-2048}"
cd "$(dirname "$0")/../.."
tmux kill-session -t dinfer 2>/dev/null || true
pkill -INT -f "dinfer serve .*--port $PORT" 2>/dev/null || true
for _ in $(seq 1 40); do pgrep -f "dinfer serve .*--port $PORT" >/dev/null || break; sleep 0.5; done
pgrep -f "dinfer serve .*--port $PORT" >/dev/null && { pkill -KILL -f "dinfer serve .*--port $PORT"; sleep 2; }
# 等该卡显存回到基线（旧进程释放 ~20 GiB 需要几秒）
for _ in $(seq 1 40); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$DEV"); [ "$used" -lt 2000 ] && break; sleep 0.5; done
tmux new-session -d -s dinfer -c "$PWD" "source $R/env.sh; RUST_LOG=info ./target/release/dinfer serve --model $R/models/qwen3-0.6b --host 0.0.0.0 --port $PORT --max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens $BUDGET --gpu-memory-utilization $UTIL --device $DEV $* 2>&1 | tee $R/bootstrap/serve.log"
for _ in $(seq 1 120); do sleep 0.5; curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break; done
curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "dinfer did not come up"; tail -5 $R/bootstrap/serve.log; exit 1; }
echo "dinfer :$PORT ready (pid $(pgrep -f "dinfer serve .*--port $PORT" | head -1)); $(pgrep -fc "dinfer serve .*--port $PORT") serve process(es)"
sed "s/\x1b\[[0-9;]*m//g" $R/bootstrap/serve.log | grep -E "startup timing|num_blocks=" | sed -E "s/^.*INFO [a-z_0-9:]+ //" | cut -c1-160
