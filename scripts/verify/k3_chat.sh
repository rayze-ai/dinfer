#!/usr/bin/env bash
# Kimi-K3 里程碑 C-4a：tokenizer（tiktoken）+ XTML 渲染 + 输出解析，纯 CPU，exit 0 = 通过。
#   1) python 生成 golden（只用模型目录的 tokenization_kimi.py / encoding_k3.py，不需要 torch）→ test/kimik3/golden/chat.json
#   2) cargo test -p dinfer-kimik3（单测 + chat_parity 对拍）
# 注意：crate 依赖 dinfer-kernels（nvcc 构建），所以本机要有 CUDA_HOME；测试本身不碰 GPU。
# 环境：KIMIK3_TOK_DIR（默认 /data/ref/models/kimik3）、PY_TF519（默认 /data/ref/venv-tf519/bin/python）、
#       K3_SKIP_GEN=1 跳过生成直接用仓库里的 golden。
set -uo pipefail
cd "$(dirname "$0")/../.."
TOK="${KIMIK3_TOK_DIR:-/data/ref/models/kimik3}"
PY="${PY_TF519:-/data/ref/venv-tf519/bin/python}"
GOLDEN="test/kimik3/golden/chat.json"
LOG=/tmp/k3_chat.log
rm -f target/.rustc_info.json
step() { printf '\n==> %s\n' "$*"; }

if [ "${K3_SKIP_GEN:-0}" != "1" ]; then
  step "生成 golden → $GOLDEN"
  mkdir -p "$(dirname "$GOLDEN")"
  "$PY" -I scripts/verify/gen_k3_chat_golden.py "$GOLDEN" --model "$TOK" 2>&1 </dev/null | grep -v -i warning | tail -3 || exit 1
  [ -s "$GOLDEN" ] || { echo "golden 未生成"; exit 1; }
  sz=$(stat -c %s "$GOLDEN"); [ "$sz" -lt 1048576 ] || { echo "golden $sz B ≥ 1 MB"; exit 1; }
fi

step "cargo test -p dinfer-kimik3（CPU：单测 + chat_parity）"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" PATH="${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"
KIMIK3_TOK_DIR="$TOK" cargo test -p dinfer-kimik3 2>&1 </dev/null | tee "$LOG" \
  | grep -E '^test |test result|panicked|cases checked|differ|failed' | cut -c1-200 | tail -40
rc=${PIPESTATUS[0]}
printf '\n日志：%s\n' "$LOG"
exit $rc
