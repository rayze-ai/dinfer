#!/usr/bin/env bash
# 抓 qwen35 闭环 decode（c32 p128 / c8 p1024）的 kernel 分解（r11 decode 侧，engine-gap 9e）。
# 收尾三步同 q35_nsys_prefill.sh（pitfalls 2026-10-20 nsys 三坑）：发够请求 → sleep 20 → SIGINT 给 nsys → 等 nsys 退出。
# 用法（B300-50 远端）：bash scripts/dev/q35_nsys_decode.sh <tag> [gpu=3] [port=8031] [conc=32] [plen=128] [out=128] [reqs=128]
set -uo pipefail
tag=${1:?tag}; gpu=${2:-3}; port=${3:-8031}; conc=${4:-32}; plen=${5:-128}; out=${6:-128}; reqs=${7:-128}
M=${MODEL_DIR:-/data/models/Qwen/Qwen3.8-27B}
cd "$(dirname "$0")/../.." && source /data/yaochen/infer/env.sh 2>/dev/null
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

pkill -f "dinfer serve .*--port $port\b" 2>/dev/null; sleep 2
rm -f /tmp/q35_$tag.nsys-rep /tmp/q35_$tag.sqlite
tmux kill-session -t nsdec 2>/dev/null
tmux new-session -d -s nsdec -c "$PWD" "source /data/yaochen/infer/env.sh; CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu \
  nsys profile -t cuda --cuda-graph-trace=node --cuda-flush-interval=1000 -o /tmp/q35_$tag --force-overwrite true \
  taskset -c 32-47 ./target/release/dinfer serve --model $M --served-model-name Qwen3.8-27B --host 127.0.0.1 --port $port \
  --max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens 16384 > /tmp/q35_${tag}_serve.log 2>&1"
for i in $(seq 1 240); do sleep 1; curl -s -m 2 127.0.0.1:$port/v1/models >/dev/null 2>&1 && break; done
curl -s -m 3 127.0.0.1:$port/v1/models | grep -oE '"id":"[^"]+"' | head -1 || { echo "serve 未起"; tail -5 /tmp/q35_${tag}_serve.log; exit 1; }

# 与 bench_compare 同口径（ignore_eos、temperature 0、每次换 seed），钉核 96-111
taskset -c 96-111 ./target/release/bench --url http://127.0.0.1:$port --model Qwen3.8-27B --label dinfer \
  --concurrency $conc --num-requests $reqs --prompt-len $plen --max-tokens $out --seed $(date +%s) --warmup 2 \
  --out /tmp/q35_${tag}_bench.json 2>&1 | grep -E "throughput|TTFT  |TPOT" | head -3 | cut -c1-140
sleep 20
np=$(pgrep -x nsys | head -1); echo "nsys pid=$np → SIGINT"; [ -n "$np" ] && kill -INT "$np"
for i in $(seq 1 300); do sleep 1; pgrep -x nsys >/dev/null || break; done; sleep 2
ls -la /tmp/q35_$tag.nsys-rep 2>/dev/null | cut -c25-
pkill -f "dinfer serve .*--port $port\b" 2>/dev/null; tmux kill-session -t nsdec 2>/dev/null
