#!/usr/bin/env bash
# 真权重 MXFP4 解包核对（docs/loops/k3-moe.md）。exit 0 = 通过。
#   1) k3_mxfp4_realcheck.py：numpy 从 safetensors 解包作参考（纯 numpy，`python3 -I`）
#   2) mxfp4_realcheck.rs：Rust 侧走 mxfp4.cu 解包，逐位比。
# 环境：MODEL_DIR 模型目录（必填，含 model.safetensors.index.json）；OUT 输出目录；
#       LAYER / EXPERT / W 选张量；ROWS_LIMIT 只取前 N 行（0 = 全量）；PY python 解释器。
#       无 numpy 的机器可以只跑第 2 步：REFS_DIR 指向已生成的目录 + SKIP_GEN=1。
set -uo pipefail
cd "$(dirname "$0")/../.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
PY="${PY:-python3}"
OUT="${OUT:-/tmp/mxfp4_real}"
LAYER="${LAYER:-1}"; EXPERT="${EXPERT:-0}"; W="${W:-1,2,3}"; ROWS_LIMIT="${ROWS_LIMIT:-0}"
LOG=/tmp/k3_mxfp4_realcheck.log

if [ "${SKIP_GEN:-0}" = "1" ]; then
  OUT="${REFS_DIR:-$OUT}"
  printf '==> 跳过参考生成（SKIP_GEN=1），用 %s\n' "$OUT"
  [ -f "$OUT/manifest.txt" ] || { echo "缺少 $OUT/manifest.txt"; exit 1; }
else
  : "${MODEL_DIR:?需要 MODEL_DIR（模型目录）}"
  printf '==> 参考 %s ← %s layer %s expert %s w %s rows %s\n' "$OUT" "$MODEL_DIR" "$LAYER" "$EXPERT" "$W" "$ROWS_LIMIT"
  "$PY" -I scripts/verify/k3_mxfp4_realcheck.py --model "$MODEL_DIR" --out "$OUT" \
      --layer "$LAYER" --expert "$EXPERT" --w "$W" --rows-limit "$ROWS_LIMIT" || exit 1
fi

printf '\n==> 逐位对拍\n'
MXFP4_REALCHECK_DIR="$OUT" cargo test -p dinfer-kernels --features ref-tests --test mxfp4_realcheck \
  -- --ignored --nocapture 2>&1 | tee "$LOG" | grep -E "^  |^test |test result|panicked|^error"
rc=${PIPESTATUS[0]}
printf '\n日志：%s\n' "$LOG"
exit $rc
