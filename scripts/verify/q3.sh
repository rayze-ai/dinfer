#!/usr/bin/env bash
# Q3 验收：单卡性能不低于 vLLM（docs/loops/q3-perf.md 停止条件 5）。exit 0 = 通过。在 4090 跑：
#   source /data/yaochen/infer/env.sh && bash scripts/verify/q3.sh
# 前提：dinfer :8012 与 vLLM :8013 常驻（scripts/dev/README-remote4090.md）。Q3_RATIO=0.8 可放宽阈值（迭代期）。
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
RATIO="${Q3_RATIO:-1.0}"
OUT="${1:-test/qwen3/$(date +%F)/q3}"; mkdir -p "$OUT"
step() { rm -f target/.rustc_info.json; printf '\n==> %s\n' "$*"; }

step "kernel refs (GQA flash prefill / paged / split-K decode + sampling)"
KERNEL_REFS_DIR="$(realpath -m test/kernel_refs)" cargo test -p dinfer-kernels --features ref-tests --test ref_tests -- --ignored --test-threads=1 </dev/null 2>&1 | tee "$OUT/kernels.log" | grep -E 'test result|panicked'
grep -qE 'test result: ok\. [1-9]' "$OUT/kernels.log"

step "parity with CUDA graphs on (real_parity + sampling_parity)"
QWEN3_DIR="${QWEN3_DIR:-/data/yaochen/infer/models/qwen3-0.6b}" QWEN3_GOLDEN="$(realpath -m test/qwen3_golden)" cargo test -p dinfer-qwen3 --features ref-tests --test real_parity --test sampling_parity -- --ignored --test-threads=1 </dev/null 2>&1 | tee "$OUT/parity.log" | grep -E 'test result|panicked'
[ "$(grep -cE 'test result: ok\.' "$OUT/parity.log")" = 2 ]

step "bench_compare (dinfer :8012 vs vLLM :8013)"
unset CUDA_VISIBLE_DEVICES
bash scripts/verify/bench_compare.sh "$OUT/bench" > "$OUT/bench.log" 2>&1 || { tail -5 "$OUT/bench.log"; exit 1; }
grep -E '^\|' "$OUT/bench.log"

step "ratio check: dinfer out tok/s ≥ ${RATIO}× vLLM on every group"
python3 - "$OUT/bench" "$RATIO" <<'PY'
import json, sys, glob, os
d, ratio = sys.argv[1], float(sys.argv[2])
bad = 0
for f in sorted(glob.glob(os.path.join(d, "dinfer_*.json"))):
    g = os.path.basename(f)[len("dinfer_"):-5]
    a = json.load(open(f)); b = json.load(open(os.path.join(d, f"vllm_{g}.json")))
    r = a["output_token_throughput_tps"] / b["output_token_throughput_tps"]
    ok = r >= ratio
    bad += not ok
    print(f"  {g:<16} {a['output_token_throughput_tps']:7.0f} / {b['output_token_throughput_tps']:7.0f} tok/s = {r:.2f}x  TTFT {a['ttft_ms']['p50']:.0f}/{b['ttft_ms']['p50']:.0f} ms  TPOT {a['tpot_ms']['p50']:.1f}/{b['tpot_ms']['p50']:.1f} ms  {'OK' if ok else 'BELOW'}")
sys.exit(1 if bad else 0)
PY
printf '\n==> Q3 PASS (ratio ≥ %s)\n' "$RATIO"
