#!/usr/bin/env bash
# Kimi-K3 KDA 算子对拍（docs/loops/k3-kda.md）。exit 0 = 通过。
#   1) gen_kda_refs.py：torch 参考生成 .npy（纯 torch，不依赖 vllm / fla）
#   2) kda_parity.rs：Rust 侧跑 kernel + 断言
# 环境：KDA_REFS_DIR 改参考目录（默认 test/kimik3/2026-10-19/kda_refs）；PY 指定 python。
set -uo pipefail
cd "$(dirname "$0")/../.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
PY="${PY:-/data/ref/venv-tf519/bin/python}"
DIR="${KDA_REFS_DIR:-test/kimik3/2026-10-19/kda_refs}"
LOG=/tmp/dinfer_kda_parity.log

rm -f target/.rustc_info.json
printf '==> 生成参考 %s\n' "$DIR"
"$PY" scripts/verify/gen_kda_refs.py "$DIR" || exit 1
printf '\n==> 对拍\n'
KDA_REFS_DIR="$DIR" cargo test -p dinfer-kernels --features ref-tests --test kda_parity -- --ignored --test-threads=1 2>&1 | tee "$LOG" | grep -E "^  [a-z_]+/|^test |test result|panicked|error\[|^error" 
rc=${PIPESTATUS[0]}
printf '\n日志：%s\n' "$LOG"
exit $rc
