#!/usr/bin/env bash
# 验证脚本：kaon3-layer1 —— compile 口径下的逐层（截层）对拍。
#
# 设计原则（上一轮的教训）：**先证明判据可信，再比我们与参照**。
#   ① FLOOR 必须为 0（同一 oracle 抓两次，逐位一致）；
#   ② 参数逐项核对（compile / 不开 enforce_eager / BI=1 / 截层数两侧一致）；
#   ③ 只有 ①② 都过，才拿差异下结论。
#
# 用法：
#   bash scripts/verify/kaon3-layer1.sh              # 默认截到 1 层
#   N=6 bash scripts/verify/kaon3-layer1.sh          # 截到 6 层
set -uo pipefail
cd "$(dirname "$0")/../.."

N="${N:-1}"
R=scripts/dev/remote_5090.sh
OG="${ORACLE_GPU:-1}"
OP="${ORACLE_PORT:-28101}"
ON="${ORACLE_NAME:-kaon-oracle-$N}"
SNAP_E="/models/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478"
PASS=0; FAIL=0
ok() { echo "✅ $1"; PASS=$((PASS+1)); }
ng() { echo "❌ $1"; FAIL=$((FAIL+1)); }
die() { echo "❌ $1"; echo "整轮作废，不要用它的数字下任何结论。"; exit 1; }

[ "$OG" = 7 ] && die "GPU 7 是基线，禁止占用"
# 显存预检：2026-10-24 踩过（另一个 oracle 占着 27 GB，gpu_memory_utilization=0.85
# 需要 26.65 GB → ValueError，但容器名不同所以没察觉）。
used=$($R run "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $OG" 2>/dev/null | tail -1 | tr -d ' ')
[ -n "${used:-}" ] && [ "$used" -gt 2000 ] && die "GPU $OG 已被占用 ${used} MiB（需要空卡；先 docker rm -f 旧 oracle）"
$R status 2>/dev/null | grep -q alive || die "ControlMaster 不在（token 过期？向用户要新的）"

# ---------- 参数对齐：能机器判定的逐项核 ----------
echo "== 参数对齐 =="
# ① oracle 必须是 compile（cmd 里不能有 enforce_eager / enforce-eager）
cmd=$($R run "docker inspect -f '{{json .Config.Cmd}}' $ON 2>/dev/null" 2>/dev/null | tail -1)
if [ -z "$cmd" ] || [ "$cmd" = "null" ]; then
  ng "容器 $ON 不存在或未运行（先起 oracle，见 loop 规格）"
else
  echo "$cmd" | grep -qiE 'enforce[-_]eager' && die "oracle 带了 enforce_eager → 不是 compile 口径"
  echo "$cmd" | grep -q -- "--block-size 32" || echo "$cmd" | grep -q -- "--block-size" || ng "block size 未显式设为 32"
  echo "$cmd" | grep -q -- "--max-num-batched-tokens 4096" || ng "max-num-batched-tokens 不是 4096"
  [ "$(echo "$cmd" | grep -oE '"num_hidden_layers":[0-9]+' | grep -oE '[0-9]+$')" = "$N" ] \
    && ok "截层数 = $N（与探针侧一致）" || ng "截层数不是 $N：$cmd"
  # ② BI=1
  envv=$($R run "docker inspect -f '{{json .Config.Env}}' $ON 2>/dev/null" 2>/dev/null | tail -1)
  echo "$envv" | grep -q 'VLLM_BATCH_INVARIANT=1' && ok "BI=1" || die "oracle 不是 BI=1"
  # ③ 实际是 compile：日志里应有 torch_compile / VLLM_COMPILE 痕迹
  $R run "docker logs $ON 2>&1 | grep -qE 'torch_compile_cache|VLLM_COMPILE|Dynamo bytecode'" \
    && ok "日志确认走 torch.compile" || die "日志里没有 compile 痕迹 → 可能退化成 eager"
fi

# ---------- FLOOR：同一 oracle 抓两次，必须逐位一致 ----------
echo "== FLOOR（判据可信度） =="
FB=/w/runs/floor_$(date +%Y%m%d-%H%M)_r$(printf %02d "$N")
$R ctr "cd /w/dinfer && mkdir -p $FB/a $FB/b" >/dev/null 2>&1
for side in a b; do
  ORACLE_PORT=$OP ORACLE_OUT="test/kaon3/_floor_$side" \
    scripts/dev/kaon3_oracle.sh cn1 chat '写一首关于秋天的五言绝句。' >/dev/null 2>&1
done
$R sync >/dev/null 2>&1
fl=$($R ctr "cd /w/dinfer && ./target/release/kaon3_oracle_diff test/kaon3/_floor_a test/kaon3/_floor_b 2>&1 | grep POOLED" 2>/dev/null | tail -1)
echo "  $fl"
echo "$fl" | grep -q 'flip_pct=0.00' && ok "FLOOR = 0（参照逐位可复现）" || die "FLOOR ≠ 0：$fl"

# ---------- 我们 vs 参照 ----------
echo "== 截到 $N 层：我们 vs vLLM（compile） =="
ORACLE_PORT=$OP ORACLE_OUT="test/kaon3/_ref_L$N" \
  scripts/dev/kaon3_oracle.sh cn1 chat '写一首关于秋天的五言绝句。' >/dev/null 2>&1
$R sync >/dev/null 2>&1
out=$($R ctr "cd /w/dinfer && M=\$(ls -d /hf_data/hub/models--kaon*/snapshots/*/ | head -1) && \
  CUBLASLT_WORKSPACE_SIZE=1 RMSNORM_MODE=0 ./target/release/kaon3_probe \$M \
  --batch test/kaon3/_ref_L$N --layers $N --out /w/_ours_L$N 2>&1 | grep -cE 'BATCH'" 2>/dev/null | tail -1)
[ "${out:-0}" -ge 1 ] && ok "我们的前向跑完（$out 个用例）" || die "我们的前向没跑成功"

res=$($R ctr "cd /w/dinfer && ./target/release/kaon3_oracle_diff test/kaon3/_ref_L$N /w/_ours_L$N 2>&1 | grep POOLED" 2>/dev/null | tail -1)
echo "  $res"
flip=$(echo "$res" | grep -oE 'flip_pct=[0-9.]+' | cut -d= -f2)
d=$(echo "$res" | grep -oE 'mean_common_d=[0-9.]+' | cut -d= -f2)
echo
echo "第 $N 层：flip=${flip}%  mean_common_d=${d}"
awk -v f="${flip:-999}" 'BEGIN{exit !(f+0 == 0)}' \
  && ok "第 $N 层逐位一致（flip=0）→ 继续往上加层" \
  || echo "⚠️  第 $N 层已有差异（flip=${flip}%）→ 差异在第 $N 层之前/之内"

echo
if [ "$FAIL" -eq 0 ]; then echo "判据可信（FLOOR=0、参数对齐），可以下结论"; exit 0; fi
echo "失败 $FAIL 项 / 通过 $PASS 项"; exit 1
