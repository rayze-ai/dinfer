#!/usr/bin/env bash
# P4 验收：p3 全部 + tiny MoE GLM（第 0 层 dense、第 1 层 MoE 4 专家 top-2 + shared）与 HF 对拍。exit 0 = 通过。
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
TINY_MOE="$(realpath -m "${DINFER_TINY_GLM_MOE:-test/tiny_glm_moe}")"

step() { printf '\n==> %s\n' "$*"; }

scripts/verify/p3.sh

step "generate tiny MoE GLM reference → $TINY_MOE"
python3 scripts/verify/gen_tiny_glm.py "$TINY_MOE" --first-k-dense 1

step "tiny MoE GLM parity (prefill logits + greedy 32 + step logits)"
DINFER_TINY_GLM="$TINY_MOE" cargo test -p dinfer-glm53 --features ref-tests --test tiny_parity -- --ignored --test-threads=1 2>&1 \
  | tee /tmp/dinfer_p4_moe.log | grep -E '^test |test result|panicked|diverges|max abs|argmax'
grep -qE "test result: ok\. 3 passed" /tmp/dinfer_p4_moe.log

printf '\n==> P4 PASS\n'
