#!/usr/bin/env bash
set -uo pipefail
source /data/yaochen/infer/env.sh
cd /data/yaochen/infer/dinfer
rm -f target/.rustc_info.json
export CUDA_VISIBLE_DEVICES=0
echo "== archs=$CUDA_ARCHS nvcc=$(nvcc --version | tail -1)"
time cargo build --release -p dinfer-kernels --features ref-tests --tests 2>&1 | tail -40
echo "build_rc=${PIPESTATUS[0]}"
