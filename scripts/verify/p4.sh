#!/usr/bin/env bash
# P4 验收：p3 全部 + tiny MoE GLM（第 0 层 dense、第 1 层 MoE 4 专家 top-2 + shared）与 HF 对拍。exit 0 = 通过。
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
TINY_MOE="$(realpath -m "${DINFER_TINY_GLM_MOE:-test/tiny_glm_moe}")"
TINY_FP8="$(realpath -m "${DINFER_TINY_GLM_FP8:-test/tiny_glm_fp8}")"
# cargo 探测 rustc 的结果缓存若被污染（见 docs/pitfalls.md），所有 cargo 命令假失败：先清掉
rm -f target/.rustc_info.json

step() { printf '\n==> %s\n' "$*"; }

scripts/verify/p3.sh

step "generate tiny MoE GLM reference → $TINY_MOE"
python3 scripts/verify/gen_tiny_glm.py "$TINY_MOE" --first-k-dense 1

step "tiny MoE GLM parity (prefill logits + greedy 32 + step logits)"
DINFER_TINY_GLM="$TINY_MOE" cargo test -p dinfer-glm53 --features ref-tests --test tiny_parity -- --ignored --test-threads=1 2>&1 \
  | tee /tmp/dinfer_p4_moe.log | grep -E '^test |test result|panicked|diverges|max abs|argmax'
grep -qE "test result: ok\. 3 passed" /tmp/dinfer_p4_moe.log

step "core runner unit tests (mock executor)"
cargo test -p dinfer-core runner:: 2>&1 | tee /tmp/dinfer_p4_runner_unit.log | grep -E '^test |test result|panicked'
grep -qE "test result: ok\. 2 passed" /tmp/dinfer_p4_runner_unit.log

for d in "$(realpath -m "${DINFER_TINY_GLM:-test/tiny_glm}")" "$TINY_MOE"; do
  step "scheduler + runner + Glm53Executor: 2 concurrent requests vs HF, plain + chunked/mixed ($d)"
  DINFER_TINY_GLM="$d" cargo test -p dinfer-glm53 --features ref-tests --test runner_parity -- --ignored --nocapture --test-threads=1 2>&1 \
    | tee /tmp/dinfer_p4_runner.log | grep -E '^test |test result|panicked|steps =|gap 0\.0[01]'
  grep -qE "test result: ok\. 2 passed" /tmp/dinfer_p4_runner.log
done

step "generate tiny fp8 (block-quantized, MoE) GLM reference → $TINY_FP8"
python3 scripts/verify/gen_tiny_glm.py "$TINY_FP8" --first-k-dense 1 --fp8

step "tiny fp8 GLM parity: load-time 128x128 block dequant → same 3 + 2 tests"
DINFER_TINY_GLM="$TINY_FP8" cargo test -p dinfer-glm53 --features ref-tests --test tiny_parity --test runner_parity -- --ignored --test-threads=1 2>&1 \
  | tee /tmp/dinfer_p4_fp8.log | grep -E '^test |test result|panicked|diverges|max abs|steps ='
test "$(grep -cE 'test result: ok\. (3|2) passed' /tmp/dinfer_p4_fp8.log)" -eq 2

printf '\n==> P4 PASS\n'
