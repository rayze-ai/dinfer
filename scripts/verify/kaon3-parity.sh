#!/usr/bin/env bash
# 验证脚本：kaon3-parity —— Kaon-V3 Rust 前向（5090）与 vLLM 的一致性，判据以 vLLM 自身噪声为基准。
# 约定：全部通过 exit 0；任一失败 exit 1。规格（含判据由来）：docs/loops/kaon3-parity.md
# 用法：bash scripts/verify/kaon3-parity.sh
#   GPUS="0 2 3 4 5 6"（默认；1 = 私有 oracle、7 = 基线，禁用）  PROBE_ENV="GQA_PREFILL_CFG=-1"（给探针加环境变量做 A/B）
# 隔离（用户 2026-10-22：并行时编译不要互相覆盖）：编译只在本脚本开头串行做一次，产物立即拷到
#   /w/bins/<run>/ 快照、输出写 /w/runs/<run>/，并行分片只用快照；别的编译改 target/ 影响不到正在跑的测试。
# 前提：test/kaon3/oracle（vLLM 开 BI = 基线口径）与 test/kaon3/oracle_nobi（关 BI）已抓好；两者之差 = 噪声下限。
set -uo pipefail
cd "$(dirname "$0")/../.."
R=scripts/dev/remote_5090.sh
read -r -a G <<< "${GPUS:-0 2 3 4 5 6}"
for g in "${G[@]}"; do case $g in 1|7) echo "❌ GPU $g 禁用（1=私有 oracle，7=基线）"; exit 1;; esac; done
RUN=$(date +%m%d-%H%M%S)-$$
B=/w/bins/$RUN; O=/w/runs/$RUN
PASS=0; FAIL=0
ok() { echo "✅ $1"; PASS=$((PASS+1)); }
ng() { echo "❌ $1"; FAIL=$((FAIL+1)); }
timeout 300 $R sync >/dev/null || { echo "❌ sync 到远端失败"; exit 1; }
# 1) 串行：编译 + kernel 对拍 + 产物快照
pre=$(GPU=${G[0]} timeout 3000 $R ctr "cargo build --release -p dinfer-kaon3 --bins 2>&1 | grep -E '^error' -A4 | head -8
  cargo test --release -p dinfer-kernels --test gqa_window_parity --test gemm_lowp_parity -- --ignored --test-threads 1 2>&1 | grep -E '^test result' | sed 's/^/KTEST /'
  mkdir -p $B $O/ours && cp target/release/kaon3_probe target/release/kaon3_oracle_diff $B/ && echo SNAP_OK")
echo "$pre" | grep -E '^error' | head -5
echo "$pre" | grep -q SNAP_OK || { echo "❌ 编译 / 快照失败"; exit 1; }
[ "$(echo "$pre" | grep -c '^KTEST test result: ok')" -ge 2 ] && ok "kernel 对拍（gqa_window_parity / gemm_lowp_parity）" \
  || ng "kernel 对拍：$(echo "$pre" | grep KTEST | tr '\n' ' ')"
# 2) 并行：按卡分片跑批量前向（每卡只加载一次权重）
n=${#G[@]}
for i in "${!G[@]}"; do
  GPU=${G[$i]} timeout 2400 $R ctr "M=\$(ls -d /hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/*/ | head -1)
    ${PROBE_ENV:-} $B/kaon3_probe \$M --batch test/kaon3/oracle --out $O/ours --shard $i/$n 2>&1 | grep -E 'BATCH|rror|panicked' | tail -3" \
    > /tmp/kaon3_shard_${RUN}_$i.txt 2>&1 &
done
wait
grep -hE 'rror|panicked' /tmp/kaon3_shard_${RUN}_*.txt | head -5
cnt=$(timeout 300 $R run "ls /hf_data/yaochen/runs/$RUN/ours/*.json 2>/dev/null | wc -l")   # 宿主机路径（容器里是 /w）
tot=$(ls test/kaon3/oracle/*.ids | wc -l)
[ "$cnt" = "$tot" ] && ok "分片前向完成 $cnt/$tot（${n} 卡并行，run $RUN）" || ng "分片前向只完成 $cnt/$tot（看 /tmp/kaon3_shard_${RUN}_*.txt）"
# 3) 汇总判定
out=$(GPU=${G[0]} timeout 600 $R ctr "$B/kaon3_oracle_diff test/kaon3/oracle test/kaon3/oracle_nobi | sed 's/^/FLOOR /'
  $B/kaon3_oracle_diff test/kaon3/oracle $O/ours test/kaon3/oracle_nobi | sed 's/^/OURS /'
  $B/kaon3_oracle_diff test/kaon3/oracle/bos1.json $O/ours/bos1.json | sed 's/^/BOS1 /'")
F=$(echo "$out" | grep '^FLOOR POOLED'); O2=$(echo "$out" | grep '^OURS POOLED')
FL=$(echo "$out" | grep '^FLOOR LONG long_zh'); OL=$(echo "$out" | grep '^OURS LONG long_zh')
echo "   噪声下限 $F" | cut -c1-220; echo "   我们     $O2" | cut -c1-220
echo "   长序列   $FL" | cut -c1-200; echo "   长序列   $OL" | cut -c1-200
kv() { echo "$1" | tr ' ' '\n' | grep "^$2=" | cut -d= -f2; }
if [ -z "$F" ] || [ -z "$O2" ]; then ng "汇总缺失（oracle 未抓或探针失败）"; else
  # 判据（规格里写定，改要先改规格）：翻转率 ≤ 1.5×下限+3pp；共有 token Δ ≤ 1.5×下限；自然长文本 |ΔNLL| ≤ max(3×|下限|, 0.01)
  awk -v of="$(kv "$O2" flip_pct)" -v ff="$(kv "$F" flip_pct)" 'BEGIN{exit !(of <= 1.5*ff + 3)}' \
    && ok "可判定行 top1 翻转率 $(kv "$O2" flip_pct)% ≤ 1.5×$(kv "$F" flip_pct)%+3" || ng "翻转率 $(kv "$O2" flip_pct)% > 1.5×$(kv "$F" flip_pct)%+3"
  awk -v od="$(kv "$O2" mean_common_d)" -v fd="$(kv "$F" mean_common_d)" 'BEGIN{exit !(od <= 1.5*fd)}' \
    && ok "共有 token |Δlp| $(kv "$O2" mean_common_d) ≤ 1.5×$(kv "$F" mean_common_d)" || ng "共有 token |Δlp| $(kv "$O2" mean_common_d) > 1.5×$(kv "$F" mean_common_d)"
  echo "   （参考，不判定）短 prompt ΔNLL：我们 $(kv "$O2" dnll) / 下限 $(kv "$F" dnll)"
fi
if [ -n "$FL" ] && [ -n "$OL" ]; then
  awk -v od="$(kv "$OL" dnll)" -v fd="$(kv "$FL" dnll)" 'function a(x){return x<0?-x:x} BEGIN{t=3*a(fd); if(t<0.01)t=0.01; exit !(a(od) <= t)}' \
    && ok "自然长文本（>1024，滑窗）ΔNLL $(kv "$OL" dnll)" || ng "长文本 ΔNLL $(kv "$OL" dnll) 超出 max(3×|$(kv "$FL" dnll)|, 0.01)"
else ng "长文本用例缺失"; fi
Bo=$(echo "$out" | grep '^BOS1'); [ "$(kv "$Bo" flips)" = 0 ] && ok "单 token（无 attention）top1 一致" || ng "单 token top1：$Bo"
echo
echo "通过 $PASS 项，失败 $FAIL 项（run $RUN，产物 $B，输出 $O）"
[ "$FAIL" -eq 0 ] && [ "$PASS" -gt 0 ]
