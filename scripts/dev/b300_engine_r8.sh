#!/usr/bin/env bash
# r8（design §15.7 GDN 状态按块缓存）B300 验收驱动：GPU 5、端口 8412、dinfer 钉 32-63 核、压测钉 96-127 核。
#   bash scripts/dev/b300_engine_r8.sh build            # 远端 release 构建（带 feature；-j 可用 JOBS 限）
#   bash scripts/dev/b300_engine_r8.sh serve [extra]    # 起 dinfer :8412（GPU 5），等 /v1/models 就绪；extra 追加 serve 参数
#   bash scripts/dev/b300_engine_r8.sh engine OUT "改动一句话"   # bench_engine.sh 单端（VLLM_URL=），rate 4/8，REQS 64
#   bash scripts/dev/b300_engine_r8.sh compare OUT "改动一句话"  # bench_compare.sh 四组闭环单端
#   bash scripts/dev/b300_engine_r8.sh stop             # 停 :8412
# 远端运行（在 /data/yaochen/infer/dinfer 下，先 source ../env.sh）。GPU 0/1 常驻不碰。
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU="${GPU:-5}"; PORT="${PORT:-8412}"; M="${M:-/data/models/Qwen/Qwen3.8-27B}"; BIN=./target/release/dinfer
URL="http://127.0.0.1:$PORT"
case "${1:-}" in
  build)
    cargo build --release --features qwen35 -p dinfer -p dinfer-bench ${JOBS:+-j $JOBS} 2>&1 | grep -E "^error|warning: unused|Finished" | head -20 ;;
  serve)
    shift
    pgrep -f -- "dinfer serve .*--port $PORT" >/dev/null && { echo "already serving on :$PORT"; exit 0; }
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU RUST_LOG=info
    nohup taskset -c 32-63 $BIN serve --model $M --served-model-name Qwen3.8-27B --host 127.0.0.1 --port $PORT \
      --max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens 16384 "$@" >/tmp/r8_serve_$PORT.log 2>&1 &
    for _ in $(seq 1 240); do curl -s -m 2 "$URL/v1/models" >/dev/null && break; sleep 1; done
    curl -s -m 2 "$URL/v1/models" >/dev/null || { echo "dinfer :$PORT did not come up"; tail -20 /tmp/r8_serve_$PORT.log; exit 1; }
    grep -E "snapshot pool|engine started|kv / gdn" /tmp/r8_serve_$PORT.log | cut -c1-220 ;;
  engine)
    OUT="${2:?OUT}"; CH="${3:-r8 GDN 状态按块缓存}"
    RATES="${RATES:-4 8}" REQS="${REQS:-64}" BENCH_CPUS="${BENCH_CPUS:-96-127}" PREFIX_REPEAT="${PREFIX_REPEAT:-0.5}" ALLOW_EOS=1 \
      SLO_TTFT=500 SLO_TPOT=20 VLLM_URL= SERVE_URL="$URL" bash scripts/verify/bench_engine.sh "$OUT" "$CH" ;;
  compare)
    OUT="${2:?OUT}"; CH="${3:-r8 GDN 状态按块缓存}"
    VLLM_URL= SERVE_URL="$URL" bash scripts/verify/bench_compare.sh "$OUT" "$CH" ;;
  stop)
    pkill -INT -f -- "dinfer serve .*--port $PORT" || true
    for _ in $(seq 1 40); do pgrep -f -- "dinfer serve .*--port $PORT" >/dev/null || break; sleep 0.5; done
    pgrep -f -- "dinfer serve .*--port $PORT" >/dev/null && pkill -KILL -f -- "dinfer serve .*--port $PORT" || true
    echo stopped ;;
  *) sed -n 2,8p "$0"; exit 2 ;;
esac
