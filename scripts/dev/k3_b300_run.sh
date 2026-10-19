#!/usr/bin/env bash
set -uo pipefail
source /data/yaochen/infer/env.sh
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
cd /data/yaochen/infer/dinfer
export CUDA_VISIBLE_DEVICES=0
KH=$(ls -t target/release/deps/kda_parity-* | grep -v '\.d$' | head -1)
MH=$(ls -t target/release/deps/mxfp4_parity-* | grep -v '\.d$' | head -1)
echo "=== KDA parity ($KH) ==="
KDA_REFS_DIR=test/kimik3/2026-10-19/kda_refs "$KH" --ignored --test-threads=1 --nocapture 2>&1 | tee /tmp/k3_kda_b300.log | grep -E "^  |^test |test result|panicked"
echo "kda_rc=${PIPESTATUS[0]}"
echo "=== MXFP4 parity ($MH) ==="
MXFP4_REFS_DIR=test/kimik3/2026-10-19/mxfp4_refs "$MH" --ignored --test-threads=1 --nocapture 2>&1 | tee /tmp/k3_moe_b300.log | grep -E "^  |^test |test result|panicked"
echo "moe_rc=${PIPESTATUS[0]}"
