#!/usr/bin/env bash
# 4090 测试机迁移基线：在远端跑 kernels.sh（含 Q2-a 采样 kernel）+ qwen3 real_parity + sampling_parity。exit 0 = 全过。
# 在远端执行：source /data/yaochen/infer/env.sh && bash scripts/verify/remote4090_baseline.sh
# 本机触发：scripts/dev/remote4090.sh run 'source /data/yaochen/infer/env.sh && cd /data/yaochen/infer/dinfer && bash scripts/verify/remote4090_baseline.sh'
set -uo pipefail
ROOT=/data/yaochen/infer
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source $ROOT/env.sh   # PATH / LD_LIBRARY_PATH（libcudart.so.13）/ CUDA_HOME / CUDA_ARCHS
export QWEN3_DIR=$ROOT/models/qwen3-0.6b QWEN3_GOLDEN="$PWD/test/qwen3_golden" KERNEL_REFS_DIR="$PWD/test/kernel_refs"
LOG=$ROOT/bootstrap/baseline.log; : > "$LOG"
fail=0
step() { printf '\n==> %s\n' "$*" | tee -a "$LOG"; }
run() { local step_log; step_log=$(mktemp); "$@" > "$step_log" 2>&1; local rc=$?; cat "$step_log" >> "$LOG"; grep -E '^test |test result|panicked' "$step_log" | tail -n 12; rm -f "$step_log"; [ $rc -eq 0 ] || { echo "FAILED (rc=$rc): $*"; fail=1; }; }

nvidia-smi -L | head -1 | tee -a "$LOG"
step "kernels (ref tests, needs test/kernel_refs synced from a torch machine)"
run env CUDA_ARCHS=89 cargo test -p dinfer-kernels --features ref-tests --test ref_tests -- --ignored --test-threads=1
step "qwen3 real_parity (prefill logits / greedy 32 / concurrent)"
run cargo test -p dinfer-qwen3 --features ref-tests --test real_parity -- --ignored --test-threads=1
step "qwen3 sampling_parity (Q2-a)"
run cargo test -p dinfer-qwen3 --features ref-tests --test sampling_parity -- --ignored --test-threads=1
step "qwen3 unit tests"
run cargo test -p dinfer-qwen3 -p dinfer-core
step "qwen3 server e2e (real model)"
run cargo test -p dinfer --features qwen3,e2e --test serve_e2e_qwen3 -- --ignored --test-threads=1
[ $fail -eq 0 ] && step "remote4090 baseline PASS" || step "remote4090 baseline FAIL (see $LOG)"
exit $fail
