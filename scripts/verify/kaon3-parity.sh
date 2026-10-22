#!/usr/bin/env bash
# 验证脚本：kaon3-parity —— Kaon-V3 Rust 前向（5090，GPU ${GPU:-0}）与 vLLM 的一致性，判据以 vLLM 自身噪声为基准。
# 约定：全部通过 exit 0；任一失败 exit 1。规格（含判据由来）：docs/loops/kaon3-parity.md
# 用法：bash scripts/verify/kaon3-parity.sh            PROBE_ENV="GQA_PREFILL_CFG=-1" 可给探针加环境变量做 A/B
# 前提：test/kaon3/oracle（vLLM 开 batch-invariant，= 基线口径）与 test/kaon3/oracle_nobi（关 BI）已由
#       scripts/dev/kaon3_oracle_set.sh / kaon3_oracle.sh 抓好；两者之差 = vLLM 对自身数值细节的敏感度（噪声下限）。
set -uo pipefail
cd "$(dirname "$0")/../.."
R=scripts/dev/remote_5090.sh
export GPU=${GPU:-0}
PASS=0; FAIL=0
ok() { echo "✅ $1"; PASS=$((PASS+1)); }
ng() { echo "❌ $1"; FAIL=$((FAIL+1)); }
timeout 300 $R sync >/dev/null || { echo "❌ sync 到远端失败"; exit 1; }
out=$(timeout 3000 $R ctr "cargo build --release -p dinfer-kaon3 --bins 2>&1 | grep -E '^error' | head -5
  cargo test --release -p dinfer-kernels --test gqa_window_parity --test gemm_lowp_parity -- --ignored --test-threads 1 2>&1 | grep -E '^test result' | sed 's/^/KTEST /'
  M=\$(ls -d /hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/*/ | head -1); mkdir -p test/kaon3/ours
  for f in test/kaon3/oracle/*.ids; do c=\$(basename \$f .ids)
    ${PROBE_ENV:-} ./target/release/kaon3_probe \$M --forward --dump-oracle test/kaon3/ours/\$c.json \$(cat \$f) > /tmp/probe_\$c.txt 2>&1 || echo \"PROBEFAIL \$c \$(tail -1 /tmp/probe_\$c.txt)\"
  done
  ./target/release/kaon3_oracle_diff test/kaon3/oracle test/kaon3/oracle_nobi | sed 's/^/FLOOR /'
  ./target/release/kaon3_oracle_diff test/kaon3/oracle test/kaon3/ours test/kaon3/oracle_nobi | sed 's/^/OURS /'
  ./target/release/kaon3_oracle_diff test/kaon3/oracle/bos1.json test/kaon3/ours/bos1.json | sed 's/^/BOS1 /'")
echo "$out" | grep -E '^error|PROBEFAIL' | head -5
kt=$(echo "$out" | grep -c '^KTEST test result: ok')
[ "$kt" -ge 2 ] && ok "kernel 对拍（gqa_window_parity / gemm_lowp_parity）" || ng "kernel 对拍：$(echo "$out" | grep KTEST | tr '\n' ' ')"
echo "$out" | grep -q PROBEFAIL && ng "探针运行失败" || true
F=$(echo "$out" | grep '^FLOOR POOLED'); O=$(echo "$out" | grep '^OURS POOLED')
FL=$(echo "$out" | grep '^FLOOR LONG long_zh'); OL=$(echo "$out" | grep '^OURS LONG long_zh')
echo "   噪声下限 $F" | cut -c1-220; echo "   我们     $O" | cut -c1-220
echo "   长序列   $FL" | cut -c1-200; echo "   长序列   $OL" | cut -c1-200
kv() { echo "$1" | tr ' ' '\n' | grep "^$2=" | cut -d= -f2; }
if [ -z "$F" ] || [ -z "$O" ]; then ng "汇总缺失（oracle 未抓或探针失败）"; else
  # 判据（规格里写定，改要先改规格）：翻转率 ≤ 1.5×下限+3pp；|ΔNLL| ≤ max(3×|下限ΔNLL|, 0.01)；共有 token Δ ≤ 1.5×下限
  awk -v of="$(kv "$O" flip_pct)" -v ff="$(kv "$F" flip_pct)" 'BEGIN{exit !(of <= 1.5*ff + 3)}' \
    && ok "可判定行 top1 翻转率 $(kv "$O" flip_pct)% ≤ 1.5×$(kv "$F" flip_pct)%+3" || ng "翻转率 $(kv "$O" flip_pct)% > 1.5×$(kv "$F" flip_pct)%+3"
  awk -v od="$(kv "$O" dnll)" -v fd="$(kv "$F" dnll)" 'function a(x){return x<0?-x:x} BEGIN{t=3*a(fd); if(t<0.01)t=0.01; exit !(a(od) <= t)}' \
    && ok "ΔNLL $(kv "$O" dnll) 在 max(3×|$(kv "$F" dnll)|, 0.01) 内" || ng "ΔNLL $(kv "$O" dnll) 超出 max(3×|$(kv "$F" dnll)|, 0.01)"
  awk -v od="$(kv "$O" mean_common_d)" -v fd="$(kv "$F" mean_common_d)" 'BEGIN{exit !(od <= 1.5*fd)}' \
    && ok "共有 token |Δlp| $(kv "$O" mean_common_d) ≤ 1.5×$(kv "$F" mean_common_d)" || ng "共有 token |Δlp| $(kv "$O" mean_common_d) > 1.5×$(kv "$F" mean_common_d)"
fi
if [ -n "$FL" ] && [ -n "$OL" ]; then
  awk -v od="$(kv "$OL" dnll)" -v fd="$(kv "$FL" dnll)" 'function a(x){return x<0?-x:x} BEGIN{t=3*a(fd); if(t<0.01)t=0.01; exit !(a(od) <= t)}' \
    && ok "长序列（>1024，滑窗）ΔNLL $(kv "$OL" dnll)" || ng "长序列 ΔNLL $(kv "$OL" dnll) 超出 max(3×|$(kv "$FL" dnll)|, 0.01)"
fi
B=$(echo "$out" | grep '^BOS1'); [ "$(kv "$B" flips)" = 0 ] && ok "单 token（无 attention）top1 一致" || ng "单 token top1：$B"
echo
echo "通过 $PASS 项，失败 $FAIL 项"
[ "$FAIL" -eq 0 ] && [ "$PASS" -gt 0 ]
