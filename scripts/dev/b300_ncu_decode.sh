#!/usr/bin/env bash
# B300 上用 ncu 看某个 decode kernel 的真实瓶颈（docs/loops/b4-sm100.md）。在 B300 上跑：
#   bash scripts/dev/b300_ncu_decode.sh <kernel正则> <并发> <prompt_len> [GPU=6]   # 结果 /tmp/b4_ncu_<kernel>.csv
set -uo pipefail
kre=$1; conc=$2; plen=$3; gpu=${4:-6}
cd "$(dirname "$0")/../.." && source ../env.sh
export CUDA_VISIBLE_DEVICES=$gpu
M=/data/yaochen/infer/models/qwen3-0.6b; PORT=8413; OUT=/tmp/b4_ncu_$kre
pkill -KILL -f "dinfer serve .*--port $PORT" 2>/dev/null; sleep 1
ncu --target-processes all --kernel-name "regex:$kre" --launch-skip ${SKIP:-500} --launch-count 3 --log-file $OUT.csv \
  --section SpeedOfLight --section Occupancy --section WarpStateStats --section LaunchStats --section SchedulerStats \
  --csv ./target/release/dinfer serve --model $M --tokenizer $M --port $PORT --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.5 --no-cuda-graph >$OUT.serve.log 2>&1 &
for i in $(seq 1 240); do curl -sf 127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break; sleep 0.5; done
./target/release/bench --url http://127.0.0.1:$PORT --model qwen3-0.6b --concurrency $conc --num-requests $((conc*2)) \
  --prompt-len $plen --max-tokens 64 --temperature 0 --seed 7 --warmup 1 --out $OUT.bench.json >/dev/null 2>&1
pkill -INT -f "dinfer serve .*--port $PORT"
for i in $(seq 1 120); do pgrep -f "dinfer serve .*--port $PORT" >/dev/null || break; sleep 0.5; done
python3 - "$OUT.csv" <<'PY'
import csv, sys
rows = [r for r in csv.reader(open(sys.argv[1])) if r and not r[0].startswith("==")]
h = rows[0]; mi = h.index("Metric Name"); vi = h.index("Metric Value"); ui = h.index("Metric Unit")
seen = set()
for r in rows[1:]:
    if len(r) <= vi or (r[mi], r[vi]) in seen: continue
    seen.add((r[mi], r[vi])); print(r[mi][:50].ljust(50), r[vi][:14].ljust(14), r[ui][:12])
PY
grep -c "PROF" $OUT.serve.log
