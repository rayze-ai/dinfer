#!/usr/bin/env bash
# P2 算子对拍：PyTorch 生成期望 → Rust GPU 测试读取比对。exit 0 = 全过。
set -euo pipefail
cd "$(dirname "$0")/../.."

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
REF_DIR="$(realpath -m "${KERNEL_REFS_DIR:-test/kernel_refs}")"

step() { rm -f target/.rustc_info.json; printf '\n==> %s\n' "$*"; }  # 每步前清掉可能被 cron trace 污染的 rustc 探测缓存

step "generate references with torch → $REF_DIR"
python3 scripts/verify/gen_kernel_refs.py "$REF_DIR"

step "GPU reference tests"
# 不过滤输出：失败时需要看到 panic 消息。
KERNEL_REFS_DIR="$REF_DIR" cargo test -p dinfer-kernels --features ref-tests --test ref_tests -- --ignored --test-threads=1 </dev/null 2>&1 \
  | tee /tmp/dinfer_kernels.log | grep -E '^test |test result|panicked|mismatch|got .* want'
grep -qE 'test result: ok\. [1-9]' /tmp/dinfer_kernels.log
! grep -qE 'FAILED|panicked' /tmp/dinfer_kernels.log

step "kernels PASS"
