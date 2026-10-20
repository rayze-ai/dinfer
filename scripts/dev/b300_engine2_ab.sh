#!/usr/bin/env bash
# B300 GPU 5：engine2 负载（rate 8 × 48 req，prompt 512/1024/1536，out 256）一次前后对比的单边运行（engine-gap 第 7 轮）。
#   bash b300_engine2_ab.sh <tag> [serve 二进制]      # 在 /data/yaochen/infer/dinfer 下运行；产物 /tmp/r7_<tag>.*
# dinfer 钉 32-63、bench 钉 96-127；nsys 全程跟踪（含加载期，分析用 nsys_steps_sqlite.py 的稳态窗口）。
set -u
tag=$1; BIN=${2:-./target/release/dinfer}; GPU=${GPU:-5}; PORT=${PORT:-8412}
M=/data/models/Qwen/Qwen3.8-27B
source /data/yaochen/infer/env.sh; unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
for p in $(pgrep -x dinfer); do grep -qz -- "--port $PORT" /proc/$p/cmdline 2>/dev/null && kill -KILL $p; done; sleep 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU RUST_LOG=info
taskset -c 32-63 nsys profile -t cuda --cuda-graph-trace=node -o /tmp/r7_$tag --force-overwrite true \
  $BIN serve --model $M --served-model-name Qwen3.8-27B --host 127.0.0.1 --port $PORT \
  --max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens 16384 ${SERVE_ARGS:-} >/tmp/r7_${tag}_serve.log 2>&1 &
for i in $(seq 1 1200); do curl -sf 127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break; sleep 1; done
curl -sf 127.0.0.1:$PORT/v1/models >/dev/null || { echo "serve not ready"; exit 2; }
sleep 3
curl -s 127.0.0.1:$PORT/metrics >/tmp/r7_${tag}.metrics.before
taskset -c 96-127 ./target/release/bench --url http://127.0.0.1:$PORT --model Qwen3.8-27B --request-rate 8 --num-requests 48 \
  --prompt-len-mix "512:0.25,1024:0.5,1536:0.25" --max-tokens 256 --seed 7 ${BENCH_ARGS:-} --out /tmp/r7_${tag}_bench.json >/tmp/r7_${tag}_bench.log 2>&1
echo BENCH_EXIT=$?
curl -s 127.0.0.1:$PORT/metrics >/tmp/r7_${tag}.metrics.after
# 只给 dinfer 进程发 SIGINT（`pkill -f` 会连 nsys 一起杀 → 没有 .nsys-rep）；nsys 等被跟踪进程退出后落盘
for p in $(pgrep -x dinfer); do grep -qz -- "--port $PORT" /proc/$p/cmdline 2>/dev/null && kill -INT $p; done
for i in $(seq 1 180); do pgrep -f "nsys profile.*r7_$tag" >/dev/null || break; sleep 1; done
sleep 2
nsys export --type sqlite --force-overwrite true -o /tmp/r7_$tag.sqlite /tmp/r7_$tag.nsys-rep >/dev/null 2>&1 && echo SQLITE_OK
python3 /data/yaochen/infer/bootstrap/nsys_steps_sqlite.py /tmp/r7_$tag.sqlite 6 >/tmp/r7_${tag}_steps.txt 2>&1
python3 scripts/dev/metrics_delta.py /tmp/r7_${tag}.metrics.before /tmp/r7_${tag}.metrics.after >/tmp/r7_${tag}_delta.txt 2>&1
echo DONE_$tag
