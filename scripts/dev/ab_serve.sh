#!/usr/bin/env bash
# 起一个隔离的 dinfer 实例做 A/B（4090 上）：先按端口杀残留、等端口空、起新、等 ready，打印 pid 与 num_blocks。
#   bash scripts/dev/ab_serve.sh start <port> <gpu> [serve args...]
#   bash scripts/dev/ab_serve.sh stop  <port>
# 教训（2026-10-17 q3-hosttier 第 10 轮）：`pkill -INT -f "port $P"` 后不等退出就起新实例，bench 会打在旧进程上。
set -euo pipefail
R=/data/yaochen/infer
cd "$(dirname "$0")/../.."
cmd=$1; port=$2
kill_port() {
  pkill -INT -f "dinfer serve .*--port $port" 2>/dev/null || true
  for _ in $(seq 1 40); do pgrep -f "dinfer serve .*--port $port" >/dev/null || break; sleep 0.5; done
  pgrep -f "dinfer serve .*--port $port" >/dev/null && { pkill -KILL -f "dinfer serve .*--port $port"; sleep 2; }
  for _ in $(seq 1 20); do curl -s -m 1 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 || break; sleep 0.5; done
}
wait_gpu_free() { # $1 = gpu index 或逗号卡表（TP）；等每张卡显存回落（刚退出的实例要几秒才释放）
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | sort -n | tail -1); used=${used:-0}
    [ "$used" -lt 2000 ] && return 0; sleep 0.5
  done
  echo "WARNING: GPU $1 still has ${used} MiB used"
}
case "$cmd" in
  stop) kill_port; echo "port $port free"; exit 0 ;;
  start)
    gpu=$3; shift 3
    kill_port
    wait_gpu_free "$gpu"
    log=/tmp/ab_serve_$port.log
    ( RUST_LOG=info CUDA_VISIBLE_DEVICES=$gpu ./target/release/dinfer serve --model $R/models/qwen3-0.6b --port "$port" "$@" > "$log" 2>&1 & )
    for _ in $(seq 1 120); do sleep 0.5; curl -s -m 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && break; done
    curl -s -m 2 "http://127.0.0.1:$port/v1/models" >/dev/null || { echo "instance on :$port did not come up"; tail -3 "$log"; exit 1; }
    n=$(ps -eo args | grep -cE "^\./target/release/dinfer serve .*--port $port")
    echo "instance :$port pid=$(pgrep -f "dinfer serve .*--port $port" | head -1) real_procs=$n $(sed "s/\x1b\[[0-9;]*m//g" "$log" | grep -oE "num_blocks=[0-9]+|kv host tier enabled num_slots=[0-9]+" | tr "\n" " ")"
    [ "$n" = 1 ] || { echo "WARNING: $n serve processes on :$port"; exit 1; }
    ;;
  *) echo "usage: ab_serve.sh start <port> <gpu> [args] | stop <port>"; exit 2 ;;
esac
