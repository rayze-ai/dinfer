#!/usr/bin/env bash
# r11a（docs/loops/qwen35-attn-r11a.md）：B300 GPU 4 上 build → gqa_fa_bench（默认 + 变体）→ ncu 单 kernel。
# 在远端跑：bash /data/yaochen/infer/bootstrap/r11a/r11a_b300.sh <轮号> [变体环境变量串，如 "GQA_PREFILL_NBUF=2"]...
# 产物：/data/yaochen/infer/bootstrap/r11a/r<N>_{default,<变体>}.log、r<N>_ncu.txt、r<N>_build.log
set -u
ROUND="${1:?round}"; shift
R=/data/yaochen/infer
source "$R/env.sh"
export CARGO_TARGET_DIR=$R/target_r11a CUDA_VISIBLE_DEVICES=4
OUT=$R/bootstrap/r11a
cd "$R/dinfer"
echo "== build $(date +%T)"
taskset -c 64-79 cargo test -p dinfer-kernels --release --test gqa_fa_bench --test ops_parity --no-run > "$OUT/r${ROUND}_build.log" 2>&1 \
  || { echo BUILD_FAIL; grep -E '^error|error\[' "$OUT/r${ROUND}_build.log" | head -20; exit 1; }
BIN=$(ls -t "$CARGO_TARGET_DIR"/release/deps/gqa_fa_bench-* | grep -v '\.d$' | head -1)
echo "== bench $(date +%T) $BIN"
taskset -c 48-55 "$BIN" --ignored --nocapture 2>&1 | grep -E '^\[attn' | tee "$OUT/r${ROUND}_default.log"
for VAR in "$@"; do
  TAG=$(echo "$VAR" | tr '=, ' '___' | tr -c 'A-Za-z0-9_\n' '_')
  echo "== variant $VAR"
  env $VAR taskset -c 48-55 "$BIN" --ignored --nocapture 2>&1 | grep -E '^\[attn' | tee "$OUT/r${ROUND}_${TAG}.log"
done
# ncu：GQA_BENCH_ITERS=1 → 每形状 2 次 launch（预热 + 1），6019 是第 5 次（skip 4）；NCU_VAR 可指定变体环境
if [ "${NCU:-1}" = 1 ]; then
  echo "== ncu $(date +%T)"
  sudo -n env GQA_BENCH_ITERS=1 ${NCU_VAR:-} CUDA_VISIBLE_DEVICES=4 LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    taskset -c 48-55 /usr/local/cuda-13.1/bin/ncu --target-processes all -k regex:gqa_fa --launch-skip 4 --launch-count 1 \
    --set full "$BIN" --ignored --nocapture > "$OUT/r${ROUND}_ncu.txt" 2>&1
  grep -E 'Duration|Registers Per Thread|Achieved Occupancy|Theoretical Occupancy|Block Limit|Eligible Warps|Issued Warp|No Eligible|DRAM Throughput|L2 Cache Throughput|Compute \(SM\) Throughput|Memory Throughput|Shared Memory Configuration Size|Dynamic Shared|Waves Per SM|Grid Size|Block Size|Executed Ipc Active|Stall' "$OUT/r${ROUND}_ncu.txt" | sed 's/  */ /g' | head -60
fi
echo "== done $(date +%T)"
