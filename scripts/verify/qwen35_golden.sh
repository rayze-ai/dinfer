#!/usr/bin/env bash
# Qwen35 里程碑 4 验收：tiny golden 对拍（prefill logits / greedy / chunked 教师强制 / 双请求混批）。
# 前置：GPU 可用（本机 3090 或 4090 GPU 7）；golden 已进仓库（test/qwen35/golden）。
# 重生成 golden（要 torch）：/data/ref/venv-tf519/bin/python -I scripts/dev/make_qwen35_tiny.py test/qwen35/golden --gen 16
# 用法：scripts/verify/qwen35_golden.sh [GPU=0]
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$HOME/.cargo/bin:$PATH"   # 远端（4090）cargo 在 ~/.cargo/bin，非交互 ssh 不读 profile
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU:-0}}"   # 调用方已指定（如 4090 的 GPU 7）就不要覆盖
export QWEN35_GOLDEN="$(pwd)/test/qwen35/golden"
echo "== qwen35 golden parity (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) =="
cargo test -p dinfer-qwen35 --features ref-tests --test golden_parity -- --ignored --test-threads=1
echo "== qwen35 unit tests =="
cargo test -p dinfer-qwen35
echo "== ok =="
