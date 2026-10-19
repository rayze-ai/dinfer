#!/usr/bin/env bash
# n-gram 投机解码验收（docs/loops/ngram-spec.md）：Qwen3-0.6B 真权重（本机 3090）。
# 1) core 单测（spec::ngram / runner / sched）；2) e2e_qwen3_ngram：greedy 开 / 关投机逐 token 一致、重复性 prompt
# 接受率 > 0.5、普通 prompt 不比关闭慢 > 10%；3) 回归 e2e_qwen3_request_features + e2e_qwen3_real_model_chat。exit 0 = 通过。
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
QWEN3="${QWEN3_DIR:-/data/ref/models/qwen3-0.6b}"
export QWEN3_GOLDEN="${QWEN3_GOLDEN:-$(realpath test/qwen3_golden)}"   # real_model_chat 读 chat_golden.json；cargo test 的 cwd 是 crate 目录
step() { rm -f target/.rustc_info.json; printf '\n==> %s\n' "$*"; }

# GPU 被别的 dinfer 进程占着就等（最多 10 分钟）
wait_gpu() {
  for _ in $(seq 1 20); do
    if ! nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -q dinfer; then return 0; fi
    echo "gpu busy (dinfer running), waiting 30s..."; sleep 30
  done
  echo "gpu still busy after 10 min" >&2; return 1
}

step "core 单测（CPU）"
cargo test -p dinfer-core </dev/null 2>&1 | grep -E 'spec::|test result|panicked' || true
cargo test -p dinfer-core </dev/null 2>&1 | grep -qE 'test result: FAILED' && { echo "core tests FAILED"; exit 1; }

run_e2e() {
  local name="$1"
  step "e2e: $name（真权重，3090）"
  wait_gpu
  QWEN3_DIR="$QWEN3" cargo test -p dinfer --features qwen3,e2e --test serve_e2e_qwen3 -- --ignored --test-threads=1 --nocapture --exact "$name" </dev/null 2>&1 \
    | tee "/tmp/ngram_$name.log" | grep -E '^test |test result|panicked|e2e:'
  grep -qE "test result: ok\. 1 passed" "/tmp/ngram_$name.log"
}

run_e2e e2e_qwen3_ngram
run_e2e e2e_qwen3_request_features
run_e2e e2e_qwen3_real_model_chat
echo "ngram: PASS"
