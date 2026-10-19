#!/usr/bin/env bash
# B300 上用 nsys 抓 dinfer 单卡 decode 的 kernel 耗时分布（docs/loops/b4-sm100.md）。在 B300 上跑：
#   bash scripts/dev/b300_nsys_decode.sh <tag> <并发> <prompt_len> [GPU=6]   # 结果 /tmp/b4_<tag>.nsys-rep + kernel 汇总
set -uo pipefail
tag=$1; conc=$2; plen=$3; gpu=${4:-6}
cd "$(dirname "$0")/../.." && source ../env.sh
export CUDA_VISIBLE_DEVICES=$gpu
M=${MODEL_DIR:-/data/yaochen/infer/models/qwen3-0.6b}   # B300-50 无该目录：MODEL_DIR=/data/models/Qwen/Qwen3-0.6B（物理路径，不软链）; PORT=8412
pkill -KILL -f "dinfer serve .*--port $PORT" 2>/dev/null; sleep 1
nsys profile -t cuda --cuda-graph-trace=node -o /tmp/b4_$tag --force-overwrite true \
  ./target/release/dinfer serve --model $M --tokenizer $M --port $PORT --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.5 ${SERVE_ARGS:-} >/tmp/b4_${tag}_serve.log 2>&1 &
for i in $(seq 1 120); do curl -sf 127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break; sleep 0.5; done
n=$((conc*4)); [ $n -lt 16 ] && n=16
./target/release/bench --url http://127.0.0.1:$PORT --model qwen3-0.6b --concurrency $conc --num-requests $n \
  --prompt-len $plen --max-tokens 128 --temperature 0 --seed 7 --warmup 2 --out /tmp/b4_${tag}_bench.json 2>&1 \
  | grep -iE "tpot|ttft|tok/s" | head -3
pkill -INT -f "dinfer serve .*--port $PORT"
for i in $(seq 1 60); do pgrep -f "dinfer serve .*--port $PORT" >/dev/null || break; sleep 0.5; done
nsys stats --report cuda_gpu_kern_sum --format csv -q /tmp/b4_$tag.nsys-rep 2>/dev/null \
  | cut -d, -f1,2,3,4,9 | head -20 | cut -c1-160
