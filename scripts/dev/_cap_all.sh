#!/usr/bin/env bash
# 单进程一次抓完整套 oracle：_cap_all.sh <out_dir> <port> [案例过滤]
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=$1; PORT=$2; FILTER=${3:-}
mkdir -p "$OUT"
fail=0; n=0
for f in test/kaon3/oracle/*.ids; do
  b=$(basename "$f" .ids)
  [ -n "$FILTER" ] && [ "$b" != "$FILTER" ] && continue
  ORACLE_OUT=$OUT ORACLE_PORT=$PORT scripts/dev/kaon3_oracle.sh "$b" ids $(cat "$f") >/dev/null 2>&1 || { echo "FAIL $b"; fail=$((fail+1)); }
  n=$((n+1))
done
echo "CAP_DONE n=$n fail=$fail out=$OUT"
