#!/usr/bin/env bash
# P1 验收：p0 全部 + 调度器/KV 单测 + criterion 门禁（1024 decode step < 200 µs）。exit 0 = 通过。
set -euo pipefail
cd "$(dirname "$0")/../.."

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
THRESHOLD_US="${P1_SCHED_US:-200}"

step() { rm -f target/.rustc_info.json; printf '\n==> %s\n' "$*"; }  # 每步前清掉可能被 cron trace 污染的 rustc 探测缓存

step "p0"
scripts/verify/p0.sh

step "core unit tests (verbose)"
cargo test -p dinfer-core -- --nocapture </dev/null 2>&1 | grep -E '^test |test result' | tail -40

step "criterion bench (scheduler)"
# criterion 输出形如：time:   [123.45 µs 125.00 µs 126.80 µs]
cargo bench -p dinfer-core --bench scheduler -- --warm-up-time 1 --measurement-time 3 </dev/null 2>&1 | tee /tmp/bench_bin.log | grep -E 'schedule\+update|time:'

step "threshold check (< ${THRESHOLD_US} µs median)"
python3 - "$THRESHOLD_US" <<'EOF'
import re, sys
thr = float(sys.argv[1])
log = open('/tmp/bench_bin.log').read()
names = re.findall(r'(schedule\+update[^\n]*)\n\s+time:\s+\[([\d.]+) (\S+) ([\d.]+) (\S+) ([\d.]+) (\S+)\]', log)
if not names:
    print("no bench results parsed"); sys.exit(1)
unit = {'ns': 1e-3, 'µs': 1.0, 'us': 1.0, 'ms': 1e3, 's': 1e6}
ok = True
for name, _, _, med, mu, _, _ in names:
    us = float(med) * unit[mu]
    flag = 'OK ' if us < thr else 'FAIL'
    print(f"{flag} {name.strip()}: median {us:.1f} µs")
    ok &= us < thr
sys.exit(0 if ok else 1)
EOF

step "P1 PASS"
