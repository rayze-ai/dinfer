#!/usr/bin/env bash
# P2 验收：p1 全部 + kernels 对拍。exit 0 = 通过。
set -euo pipefail
cd "$(dirname "$0")/../.."
scripts/verify/p1.sh
scripts/verify/kernels.sh
printf '\n==> P2 PASS\n'
