#!/usr/bin/env bash
# 验证脚本：kaon3-fp4dense
# 约定：全部通过 exit 0；任一失败 exit 1，并把失败线索打印到 stdout。
#
# 停止条件对应 docs/loops/kaon3-fp4dense.md：
#   ① dense FP4 GEMM 与参照逐位相同（ndiff == 0）
#   ② 端到端 flip ≤ 4.25%（1.5×FLOOR(0.83%) + 3pp）
#   ③ 端到端共有 token |Δlp| ≤ 0.0414（1.5×FLOOR(0.0276)）
#
# 本脚本只在 5090 上跑（sm_120），远端路径由 remote_5090.sh 提供。
# 用法：bash scripts/verify/kaon3-fp4dense.sh [--dense-only]
set -uo pipefail
cd "$(dirname "$0")/../.."

R="${R:-scripts/dev/remote_5090.sh}"
FP4D="${FP4D:-/w/tmp/fp4d}"
PASS=0
FAIL=0
DENSE_ONLY=0
[ "${1:-}" = "--dense-only" ] && DENSE_ONLY=1

check() {
  local desc="$1"; shift
  if "$@" >/tmp/verify_fp4dense.$$ 2>&1; then
    echo "✅ $desc"
    PASS=$((PASS+1))
  else
    echo "❌ $desc"
    sed 's/^/     /' /tmp/verify_fp4dense.$$ | head -30
    FAIL=$((FAIL+1))
  fi
}

if ! "$R" status 2>/dev/null | grep -q alive; then
  echo "❌ ControlMaster 不在（token 过期？向用户要新的 → ~/.ssh/jms_5090_{user,pass}，"
  echo "   然后第一条命令必须是 scripts/dev/remote_5090.sh up）"
  exit 1
fi

# ---- ① dense FP4 GEMM 逐位 ----
dense_pass() {
  GPU=0 "$R" ctr "cd /w/dinfer && FP4_DIR=$FP4D CUDA_ARCHS=120 \
    cargo test --release -p dinfer-kernels --test fp4_dense_parity -- --ignored --nocapture" 2>&1 \
    | tee /tmp/fp4dense_out.$$ | grep -q 'FP4_DENSE.*ndiff=0\b' \
    || { echo "--- 期望 'ndiff=0'，实际输出："; tail -40 /tmp/fp4dense_out.$$; return 1; }
}
check "dense FP4 GEMM 与参照逐位相同（ndiff=0）" dense_pass

if [ $DENSE_ONLY -eq 0 ]; then
  # ---- ②③ 端到端：复用 kaon3-parity.sh 的判据（flip / |Δlp| / ΔNLL），它已按 FLOOR 基线判定 ----
  e2e_pass() {
    bash scripts/verify/kaon3-parity.sh >/tmp/fp4dense_e2e.$$ 2>&1
    local rc=$?
    tail -40 /tmp/fp4dense_e2e.$$
    return $rc
  }
  check "端到端 flip / |Δlp| / ΔNLL 全部落在 FLOOR 线内（kaon3-parity.sh）" e2e_pass
fi

echo
rm -f /tmp/verify_fp4dense.$$ /tmp/fp4dense_out.$$ /tmp/fp4dense_e2e.$$ 2>/dev/null
if [ "$FAIL" -eq 0 ]; then
  echo "全部通过（$PASS 项）"
  exit 0
else
  echo "失败 $FAIL 项 / 通过 $PASS 项"
  exit 1
fi
