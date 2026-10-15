#!/usr/bin/env bash
# P3 验收：p2 全部 + tiny GLM（MLA + dense MLP）与 HF 对拍。exit 0 = 通过。
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
TINY="$(realpath -m "${TINY_GLM_DIR:-test/tiny_glm}")"

step() { rm -f target/.rustc_info.json; printf '\n==> %s\n' "$*"; }  # 每步前清掉可能被 cron trace 污染的 rustc 探测缓存

scripts/verify/p2.sh

step "generate tiny GLM reference → $TINY"
python3 scripts/verify/gen_tiny_glm.py "$TINY"

step "tiny GLM parity (prefill logits + greedy 32)"
TINY_GLM_DIR="$TINY" cargo test -p dinfer-glm53 --features ref-tests --test tiny_parity -- --ignored --test-threads=1 </dev/null 2>&1 \
  | tee /tmp/dinfer_p3.log | grep -E '^test |test result|panicked|diverges|max abs|argmax'
grep -qE "test result: ok\. 3 passed" /tmp/dinfer_p3.log

printf '\n==> P3 PASS\n'
