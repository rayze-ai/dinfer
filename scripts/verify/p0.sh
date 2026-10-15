#!/usr/bin/env bash
# P0 验收：工作区能 build / test / clippy；有 GPU 时 add kernel 跑通。exit 0 = 通过。
set -euo pipefail
cd "$(dirname "$0")/../.."

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

step() { rm -f target/.rustc_info.json; printf '\n==> %s\n' "$*"; }  # 每步前清掉可能被 cron trace 污染的 rustc 探测缓存

if command -v nvcc >/dev/null 2>&1; then
  FEAT=()
  step "nvcc: $(nvcc --version | tail -1)"
else
  FEAT=(--features no-nvcc)
  step "nvcc not found: building with --features no-nvcc"
fi

step "cargo fmt --check"
cargo fmt --all -- --check </dev/null

step "cargo build (workspace, glm53)"
cargo build --workspace --features glm53 "${FEAT[@]}" </dev/null

step "cargo build (bin, kimik3)"
cargo build -p dinfer --features kimik3 "${FEAT[@]}" </dev/null

step "cargo clippy"
cargo clippy --workspace --features glm53 "${FEAT[@]}" -- -D warnings </dev/null

step "cargo test (CPU)"
cargo test --workspace --features glm53 "${FEAT[@]}" </dev/null

if [ ${#FEAT[@]} -eq 0 ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  step "GPU present: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)"
  step "cargo test (GPU, ignored tests)"
  cargo test -p dinfer-kernels -- --ignored </dev/null
  step "dinfer device-info"
  cargo run -q -p dinfer --features glm53 -- device-info </dev/null
else
  step "no GPU: skipping GPU tests"
fi

step "P0 PASS"
