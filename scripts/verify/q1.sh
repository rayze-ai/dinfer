#!/usr/bin/env bash
# Q1 验收：Qwen3-0.6B 真权重（本机 3090）。exit 0 = 通过。不含 p 系列（GLM 线另跑 p6.sh）。
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
QWEN3="${QWEN3_DIR:-/data/ref/models/qwen3-0.6b}"
GOLDEN="$(realpath -m "${QWEN3_GOLDEN:-test/qwen3_golden}")"
PY519="${PY_TF519:-/data/ref/venv-tf519/bin/python}"
step() { rm -f target/.rustc_info.json; printf '\n==> %s\n' "$*"; }

step "fmt / clippy (qwen3 feature)"
cargo fmt --all -- --check </dev/null
cargo clippy --workspace --features qwen3 -- -D warnings </dev/null

step "Q1: config structural asserts"
cargo test -p dinfer-qwen3 </dev/null 2>&1 | grep -E '^test |test result|panicked'

step "Q1: GQA kernels vs torch"
KERNEL_REFS_DIR="$(realpath -m test/kernel_refs)" "$PY519" -I scripts/verify/gen_kernel_refs.py test/kernel_refs --gqa 2>&1 </dev/null | grep -v -i warning | tail -2
KERNEL_REFS_DIR="$(realpath -m test/kernel_refs)" cargo test -p dinfer-kernels --features ref-tests --test ref_tests gqa -- --ignored --test-threads=1 </dev/null 2>&1 | tee /tmp/q1_gqa.log | grep -E '^test |test result|panicked'
grep -qE "test result: ok\. [1-9]" /tmp/q1_gqa.log

step "Q1: HF golden (real weights, same-GPU bf16) → $GOLDEN"
"$PY519" -I scripts/verify/gen_qwen3_golden.py "$GOLDEN" --model "$QWEN3" 2>&1 </dev/null | grep -v -i warning | tail -3

step "Q1: real-weight parity (prefill logits + greedy 32) + chat template golden"
QWEN3_DIR="$QWEN3" QWEN3_GOLDEN="$GOLDEN" cargo test -p dinfer-qwen3 --features ref-tests --test real_parity --test chat_parity -- --ignored --nocapture --test-threads=1 </dev/null 2>&1 \
  | tee /tmp/q1_parity.log | grep -E '^test |test result|panicked|q1:'
grep -cE "test result: ok\." /tmp/q1_parity.log | grep -qE '^2$'

step "Q1: server e2e with real model"
QWEN3_DIR="$QWEN3" QWEN3_GOLDEN="$GOLDEN" cargo test -p dinfer --features qwen3,e2e --test serve_e2e_qwen3 -- --ignored --nocapture --test-threads=1 </dev/null 2>&1 \
  | tee /tmp/q1_e2e.log | grep -E '^test |test result|panicked|e2e:'
grep -qE "test result: ok\." /tmp/q1_e2e.log

printf '\n==> Q1 PASS\n'
