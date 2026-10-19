#!/usr/bin/env bash
# Kimi-K3 MXFP4 对拍（docs/loops/k3-moe.md）。exit 0 = 通过。
#   1) gen_mxfp4_refs.py：解包 3 case + grouped GEMM 3 case
#   2) mxfp4_parity.rs：GPU 侧解包逐位比、grouped GEMM 相对 1e-2 / cos 0.999
# 环境：MXFP4_REFS_DIR 改参考目录；MXFP4_SKIP_GEN=1 跳过生成（远端无 torch 时用仓库里的参考）。
set -uo pipefail
cd "$(dirname "$0")/../.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
PY="${PY:-/data/ref/venv-tf519/bin/python}"
DIR="${MXFP4_REFS_DIR:-test/kimik3/2026-10-19/mxfp4_refs}"
LOG=/tmp/dinfer_mxfp4_parity.log

rm -f target/.rustc_info.json
if [ "${MXFP4_SKIP_GEN:-0}" = "1" ]; then
  printf '==> 跳过参考生成（MXFP4_SKIP_GEN=1），用 %s\n' "$DIR"
  [ -d "$DIR" ] || { echo "缺少参考目录 $DIR"; exit 1; }
else
  printf '==> 生成参考 %s\n' "$DIR"
  "$PY" scripts/verify/gen_mxfp4_refs.py "$DIR" || exit 1
fi
printf '\n==> 对拍\n'
MXFP4_REFS_DIR="$DIR" cargo test -p dinfer-kernels --features ref-tests --test mxfp4_parity -- --ignored --test-threads=1 --nocapture 2>&1 | tee "$LOG" | grep -E "^  [A-Za-z_0-9]+ |^test |test result|panicked|error\[|^error"
rc=${PIPESTATUS[0]}
printf '\n日志：%s\n' "$LOG"
exit $rc
