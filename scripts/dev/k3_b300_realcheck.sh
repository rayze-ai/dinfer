#!/usr/bin/env bash
# B300：真权重 MXFP4 解包核对（只读 /data/models/... ，不拷贝权重）。
set -uo pipefail
source /data/yaochen/infer/env.sh
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
cd /data/yaochen/infer/dinfer
export CUDA_VISIBLE_DEVICES=0
export PY=/data/yaochen/infer/pyenv-numpy/bin/python
export MODEL_DIR=/data/models/moonshotai/Kimi-K3
export OUT=/data/yaochen/infer/dinfer/test/kimik3/2026-10-19/b300_50/realcheck
mkdir -p "$OUT"
bash scripts/verify/k3_realcheck.sh 2>&1 | tail -25
echo "realcheck_rc=${PIPESTATUS[0]}"
