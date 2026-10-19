#!/usr/bin/env bash
# Qwen3.5 GDN 算子对拍（docs/loops/qwen35-gdn.md）。exit 0 = 通过。
#   1) gen_gdn_refs.py：torch 参考生成 .npy（纯 torch，不依赖 vllm / fla）
#   2) gdn_parity.rs：Rust 侧跑 kernel + 断言
#   3) 回归：公共 trait 未被破坏（cargo build --workspace --features qwen3）
# 环境：GDN_REFS_DIR 改参考目录（默认 test/qwen35/2026-10-19/gdn_refs）；PY 指定 python；
#       CUDA_ARCHS 交给 kernels 的 build.rs（3090 = 86、4090 = 89、B300 = 103）。
set -uo pipefail
cd "$(dirname "$0")/../.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
PY="${PY:-python3}"
DIR="${GDN_REFS_DIR:-test/qwen35/2026-10-19/gdn_refs}"
LOG=/tmp/dinfer_gdn_parity.log

rm -f target/.rustc_info.json
# 参考是模型无关的 .npy（已进仓库），无 torch 的机器（4090 / B300 容器外）直接复用：
#   SKIP_REF_GEN=1 跳过生成；PY 指定带 torch 的解释器。
if [ "${SKIP_REF_GEN:-0}" = "1" ] || { [ -d "$DIR" ] && [ -z "${FORCE_REF_GEN:-}" ]; }; then
  printf '==> 1/3 复用已有参考 %s\n' "$DIR"
else
  printf '==> 1/3 生成参考 %s\n' "$DIR"
  "$PY" -I scripts/verify/gen_gdn_refs.py "$DIR" || exit 1
fi

printf '\n==> 2/3 GDN 对拍（CUDA_ARCHS=%s）\n' "${CUDA_ARCHS:-默认}"
GDN_REFS_DIR="$DIR" cargo test -p dinfer-kernels --features ref-tests --test gdn_parity -- \
  --ignored --test-threads=1 2>&1 | tee "$LOG" | grep -E '^  [a-z_]+/|max_err|^test |test result|panicked' | cut -c1-200
rc=${PIPESTATUS[0]}

printf '\n==> 3/3 回归：cargo build --workspace --features qwen3\n'
cargo build --workspace --features qwen3 >/tmp/dinfer_qwen3_build.log 2>&1
reg=$?
[ "$reg" -ne 0 ] && tail -20 /tmp/dinfer_qwen3_build.log

printf '\n日志：%s\n' "$LOG"
[ "$rc" -eq 0 ] || { echo "GDN 对拍失败（exit $rc）"; exit "$rc"; }
[ "$reg" -eq 0 ] || { echo "qwen3 回归构建失败（exit $reg）"; exit "$reg"; }
echo "GDN 对拍通过"
exit 0
