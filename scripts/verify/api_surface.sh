#!/usr/bin/env bash
# API 补全验收（docs/loops/api-complete.md）：Qwen3-0.6B 真权重（本机 3090），起 serve 后打 OpenAI 服务面：
# /v1/completions（流式 / 非流式 / echo / logprobs / 多 prompt×n / stop / 错例）、/tokenize、/detokenize、/version、/ping、
# /v1/models[/{id}]、chat 的 model 404、pooling 等 501 列表。exit 0 = 通过。
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
QWEN3="${QWEN3_DIR:-/data/ref/models/qwen3-0.6b}"
step() { rm -f target/.rustc_info.json; printf '\n==> %s\n' "$*"; }

step "server 单测（CPU）"
cargo test -p dinfer-server </dev/null 2>&1 | grep -E '^test |test result|panicked' || true

step "e2e: OpenAI surface（真权重，3090）"
QWEN3_DIR="$QWEN3" cargo test -p dinfer --features qwen3,e2e --test serve_e2e_qwen3 -- --ignored --test-threads=1 e2e_qwen3_openai_surface </dev/null 2>&1 \
  | tee /tmp/api_surface.log | grep -E '^test |test result|panicked|e2e:'
grep -qE "test result: ok\. 1 passed" /tmp/api_surface.log
echo "api_surface: PASS"
