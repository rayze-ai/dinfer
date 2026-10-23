#!/usr/bin/env bash
# kaon3 gelu 口径验证（exit 0 = 通过）。规格 docs/loops/kaon3-gelu.md。
#   条件 1：kernel 级 —— 我们的 gelu_tanh·mul vs torch.compile 参考逐位相同（0/67584）
#   条件 2：截层级 —— kaon3_probe --layers 1 vs compile oracle，实际 token |Δlp| 均值 ≤ 0.30
set -uo pipefail
cd "$(dirname "$0")/../.."
R=scripts/dev/remote_5090.sh
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
GPU=${GPU:-0}
fail=0
note() { printf '%-24s %s\n' "$1" "$2"; }

# --- 条件 1：kernel 级 ---
if [ ! -f /tmp/kg_out.txt ]; then
  GPU=$GPU timeout 1200 $R ctr "cd /w/dinfer && CUDA_ARCHS=120 cargo test --release -p dinfer-kernels --test gelu_mode -- --ignored --nocapture 2>&1 | grep -E 'GELU|error\['" > /tmp/kg_test.txt 2>&1
  GPU=4 timeout 600 $R ctr "cd /w/tmp && python3 gelu_cmp.py 2>&1 | grep -E 'GELU_CMP'" > /tmp/kg_out.txt 2>&1
fi
line=$(grep 'GELU_CMP vs compile' /tmp/kg_out.txt 2>/dev/null | tail -1)
if grep -q 'GELU_CMP vs compile ndiff=0/' /tmp/kg_out.txt 2>/dev/null; then note "kernel 级" "PASS  $line"
else note "kernel 级" "FAIL  ${line:-无输出（见 /tmp/kg_test.txt）}"; fail=1; fi

# --- 条件 2：截层级 ---
out=$(GPU=$GPU timeout 1200 $R ctr "cd /w/dinfer && ./target/release/kaon3_probe $M --forward --layers 1 --oracle /w/tmp/lay/L1.json \$(cat test/kaon3/oracle_clean/p02.ids) 2>&1 | tail -1" 2>&1)
act=$(printf '%s' "$out" | sed -n 's/.*实际 token |Δlp| 均值 \([0-9.]*\).*/\1/p')
if [ -z "$act" ]; then note "截层级" "FAIL  探针无输出：$out"; fail=1
else
  ok=$(awk -v a="$act" 'BEGIN{print (a<=0.30)?"1":"0"}')
  if [ "$ok" = 1 ]; then note "截层级" "PASS  实际 token |Δlp|=$act (≤0.30)"
  else note "截层级" "FAIL  实际 token |Δlp|=$act (>0.30)"; fail=1; fi
fi
exit $fail
