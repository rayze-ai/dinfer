#!/usr/bin/env bash
# Q2 验收：Qwen3-0.6B 真权重，mix 服务可生产（docs/loops/q2-serving.md 停止条件 1–12）。exit 0 = 通过。
# 在 4090 测试机跑：source /data/yaochen/infer/env.sh && bash scripts/verify/q2.sh
# 需要：$QWEN3_DIR 权重；test/kernel_refs、test/qwen3_golden 已由有 torch 的机器生成并同步（PY_TF519 可用时会重新生成）。
# STRESS_SECS：压测秒数（默认 60）。
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
# 共享机器：GPU 0 / 1 常驻 dinfer / vLLM 服务，测试默认用 GPU 2（可用 CUDA_VISIBLE_DEVICES 覆盖）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
QWEN3="${QWEN3_DIR:-/data/ref/models/qwen3-0.6b}"
GOLDEN="$(realpath -m "${QWEN3_GOLDEN:-test/qwen3_golden}")"
REFS="$(realpath -m "${KERNEL_REFS_DIR:-test/kernel_refs}")"
PY519="${PY_TF519:-/data/ref/venv-tf519/bin/python}"
export STRESS_SECS="${STRESS_SECS:-60}"
step() { rm -f target/.rustc_info.json; printf '\n==> %s\n' "$*"; }
ok_count() { grep -cE "test result: ok\." "$1" | grep -qE "^$2\$" || { echo "expected $2 passing suites in $1"; exit 1; }; }

step "fmt / clippy (qwen3, all targets)"
cargo fmt --all -- --check </dev/null
cargo clippy --workspace --features qwen3 --all-targets -- -D warnings </dev/null

step "Q2: CPU unit tests (core sampler/metrics/sched, text StopChecker, qwen3 config)"
cargo test -p dinfer-core -p dinfer-text -p dinfer-qwen3 -p dinfer-server --lib </dev/null 2>&1 | tee /tmp/q2_unit.log | grep -E 'test result|panicked'
! grep -qE 'FAILED|panicked' /tmp/q2_unit.log

if [ -x "$PY519" ]; then
  step "Q2: regenerate kernel refs + HF golden (torch available)"
  "$PY519" -I scripts/verify/gen_kernel_refs.py "$REFS" 2>&1 </dev/null | grep -v -i warning | tail -1
  "$PY519" -I scripts/verify/gen_qwen3_golden.py "$GOLDEN" --model "$QWEN3" 2>&1 </dev/null | grep -v -i warning | tail -1
else
  step "Q2: no torch here; using synced $REFS and $GOLDEN"
  [ -d "$REFS" ] && [ -d "$GOLDEN" ] || { echo "missing refs/golden"; exit 1; }
fi

step "Q2: kernel refs (incl. penalties / temperature / min_p / logprobs / topk_topp / sample)"
KERNEL_REFS_DIR="$REFS" cargo test -p dinfer-kernels --features ref-tests --test ref_tests -- --ignored --test-threads=1 </dev/null 2>&1 \
  | tee /tmp/q2_kernels.log | grep -E 'test result|panicked'
ok_count /tmp/q2_kernels.log 1

step "Q2: real-weight parity (Q1 回归) + sampling parity (seed / greedy-in-mixed / mask+distribution / penalties / logprobs)"
QWEN3_DIR="$QWEN3" QWEN3_GOLDEN="$GOLDEN" cargo test -p dinfer-qwen3 --features ref-tests --test real_parity --test sampling_parity --test chat_parity -- --ignored --test-threads=1 </dev/null 2>&1 \
  | tee /tmp/q2_parity.log | grep -E '^test |test result|panicked|q2:'
ok_count /tmp/q2_parity.log 3

step "Q2: server e2e (chat / request features / auto num_blocks / prefix cache / metrics / robustness ${STRESS_SECS}s)"
QWEN3_DIR="$QWEN3" QWEN3_GOLDEN="$GOLDEN" cargo test -p dinfer --features qwen3,e2e --test serve_e2e_qwen3 -- --ignored --nocapture --test-threads=1 </dev/null 2>&1 \
  | sed 's/\x1b\[[0-9;]*m//g' | tee /tmp/q2_e2e.log | grep -E '^test |test result|panicked|^e2e: (stress|after|metrics|prefix cache|num_blocks|.*PASS)'
ok_count /tmp/q2_e2e.log 1

printf '\n==> Q2 PASS\n'
