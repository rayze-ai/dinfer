#!/usr/bin/env bash
# 验证脚本：kaon3-parity —— Kaon-V3 Rust 前向（5090，GPU ${GPU:-0}）与 vLLM oracle 逐位置 logprob 比对。
# 约定：全部通过 exit 0；任一失败 exit 1，并打印失败线索。规格：docs/loops/kaon3-parity.md
# 用法：bash scripts/verify/kaon3-parity.sh [case...]      # 缺省 = test/kaon3/oracle/*.ids 全部
# 前提：remote_5090 ControlMaster 活着；oracle 由 scripts/dev/kaon3_oracle.sh 抓好（<case>.ids + <case>.json）。
# 判据写在 models/kaon3/src/bin/kaon3_probe.rs 顶部常量（MARGIN / MEAN_TOL / TOP1_TOL），探针打印 PARITY PASS|FAIL。
set -uo pipefail
cd "$(dirname "$0")/../.."
R=scripts/dev/remote_5090.sh
export GPU=${GPU:-0}
cases=("$@")
if [ ${#cases[@]} -eq 0 ]; then
  for f in test/kaon3/oracle/*.ids; do cases+=("$(basename "$f" .ids)"); done
fi
PASS=0; FAIL=0
timeout 300 $R sync >/dev/null || { echo "❌ sync 到远端失败"; exit 1; }
# 每轮都重编：上轮踩过"链接失败后静默跑旧二进制"
build=$(timeout 1200 $R ctr 'cargo build --release -p dinfer-kaon3 --bin kaon3_probe 2>&1 | grep -E "^error|^warning: unused var|Finished" | tail -5')
echo "$build" | grep -q Finished || { echo "❌ 编译失败"; echo "$build" | cut -c1-200; exit 1; }
for c in "${cases[@]}"; do
  out=$(timeout 900 $R ctr "M=\$(ls -d /hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/*/ | head -1); \
    ./target/release/kaon3_probe \$M --forward --oracle test/kaon3/oracle/$c.json \$(cat test/kaon3/oracle/$c.ids) 2>&1 \
    | grep -E 'PARITY|^row|^worst|一次前向|rror|panicked' | head -24")
  echo "$out" | sed "s/^/   [$c] /" | cut -c1-230
  if echo "$out" | grep -q 'PARITY PASS'; then echo "✅ $c"; PASS=$((PASS+1)); else echo "❌ $c"; FAIL=$((FAIL+1)); fi
done
echo
echo "通过 $PASS 项，失败 $FAIL 项"
[ "$FAIL" -eq 0 ] && [ "$PASS" -gt 0 ]
