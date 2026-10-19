#!/usr/bin/env bash
set -uo pipefail
source /data/yaochen/infer/env.sh
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
cd /data/yaochen/infer/dinfer
export CUDA_VISIBLE_DEVICES=0
rm -f target/.rustc_info.json
cargo build --release -p dinfer-kernels --features ref-tests --tests 2>&1 | tail -2
KH=$(ls -t target/release/deps/kda_parity-* | grep -v '\.d$' | head -1)
MH=$(ls -t target/release/deps/mxfp4_parity-* | grep -v '\.d$' | head -1)
echo "=== KDA bench ==="
KDA_REFS_DIR=test/kimik3/2026-10-19/kda_refs "$KH" kda_chunk_bench --ignored --nocapture 2>&1 | grep -E "^  |test result|panicked"
echo "=== MXFP4 GEMM bench ==="
MXFP4_REFS_DIR=test/kimik3/2026-10-19/mxfp4_refs "$MH" mxfp4_gemm_bench --ignored --nocapture 2>&1 | grep -E "^  |test result|panicked"
