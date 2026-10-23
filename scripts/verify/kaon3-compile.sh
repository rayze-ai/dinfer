#!/usr/bin/env bash
# 验证脚本：kaon3-compile —— Kaon-V3 Rust 前向在 **compile 口径**下与 vLLM compile 服务一致。
# 规格：docs/loops/kaon3-compile.md
# 铁律（用户 2026-10-23）：全程 compile；oracle 必须干净启动（不许残留截层实例）；判据只用 compile 参照。
# 用法：bash scripts/verify/kaon3-compile.sh
#   环境变量：REF / FLR（参照集 / 下限集）、RT_CASES、GPUS 透传给 kaon3-parity.sh
#   SKIP_ORACLE=1 跳过 oracle 重启（自己确认过干净时用）
set -uo pipefail
cd "$(dirname "$0")/../.."
R=scripts/dev/remote_5090.sh
PASS=0; FAIL=0
ok() { echo "✅ $1"; PASS=$((PASS+1)); }
ng() { echo "❌ $1"; FAIL=$((FAIL+1)); }

# ---------- 1) 口径干净：oracle 是 compile、全 30 层、无残留 ----------
if [ -z "${SKIP_ORACLE:-}" ]; then
  timeout 120 $R run 'docker rm -f kaon-oracle >/dev/null 2>&1' >/dev/null 2>&1
  timeout 300 bash scripts/dev/kaon3_oracle_server.sh > /tmp/kc_oracle.txt 2>&1
  tail -2 /tmp/kc_oracle.txt | sed 's/^/     /'
fi
insp=$(timeout 90 $R run "docker inspect kaon-oracle --format '{{.State.Running}}|{{range .Config.Cmd}}{{.}} {{end}}|{{range .Config.Env}}{{.}} {{end}}'" 2>&1)
run_state=$(printf '%s' "$insp" | cut -d'|' -f1 | tr -d '[:space:]')
cmdline=$(printf '%s' "$insp" | cut -d'|' -f2)
envs=$(printf '%s' "$insp" | cut -d'|' -f3)

[ "$run_state" = "true" ] && ok "oracle 容器在跑" || ng "oracle 容器未运行（inspect: $(printf '%s' "$insp" | cut -c1-160)）"
printf '%s' "$cmdline" | grep -q 'num_hidden_layers' \
  && ng "oracle 仍是截层实例（cmd 含 num_hidden_layers）：$(printf '%s' "$cmdline" | cut -c1-160)" \
  || ok "oracle 是全 30 层（cmd 无 num_hidden_layers）"
printf '%s' "$envs" | grep -qi 'eager' \
  && ng "oracle 环境里出现 eager 开关：$(printf '%s' "$envs" | tr ' ' '\n' | grep -i eager | head -2 | tr '\n' ' ')" \
  || ok "oracle 未开 eager（compile 口径）"
printf '%s' "$envs" | grep -q 'VLLM_BATCH_INVARIANT=1' \
  && ok "oracle BI=1（与参照集一致）" || ng "oracle 的 VLLM_BATCH_INVARIANT 不是 1"
code=$(timeout 60 $R run "curl -s -m 8 -o /dev/null -w '%{http_code}' http://127.0.0.1:28100/v1/models" 2>/dev/null | tr -d '[:space:]')
[ "$code" = "200" ] && ok "oracle :28100 /v1/models 200" || ng "oracle :28100 返回 $code"

# 我们自己不许有残留进程占卡（会静默拖垮下一轮）
apps=$(timeout 60 $R run "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader" 2>/dev/null | grep -c . )
[ "${apps:-0}" -le 1 ] && ok "GPU 无我们的残留进程（共 ${apps:-0} 个占用者）" \
  || ng "GPU 上有 $apps 个计算进程，可能含残留（先清理再测）"

# ---------- 2) 端到端（compile 口径：显式 RMSNORM_MODE=0） ----------
echo
echo "     整跑（这步要几分钟）..."
PROBE_ENV="CUBLASLT_WORKSPACE_SIZE=1 RMSNORM_MODE=0${PROBE_EXTRA:+ $PROBE_EXTRA}" \
  timeout 3600 bash scripts/verify/kaon3-parity.sh > /tmp/kc_parity.txt 2>&1
rc=$?
grep -E 'POOLED|噪声下限|我们     |长序列|❌|通过 .* 项' /tmp/kc_parity.txt | cut -c1-220 | sed 's/^/     /'
[ $rc -eq 0 ] && ok "kaon3-parity.sh 全项通过" || ng "kaon3-parity.sh 有失败项（见 /tmp/kc_parity.txt；摘要见下）"

# 取出放大的判定数（parity 脚本已按 FLOOR 现算判据；这里只做可读回显与二次核对）
F=$(grep -o '噪声下限 POOLED[^|]*' /tmp/kc_parity.txt | head -1)
O=$(grep -o '我们     POOLED[^|]*' /tmp/kc_parity.txt | head -1)
[ -n "$F" ] && [ -n "$O" ] && echo "     FLOOR: $(printf '%s' "$F" | cut -c1-200)" && echo "     OURS : $(printf '%s' "$O" | cut -c1-200)"
grep -q '❌ 翻转率' /tmp/kc_parity.txt && hn=1 || hn=0
grep -q '❌ 共有 token' /tmp/kc_parity.txt && hd=1 || hd=0
[ "$hn" = 0 ] && [ "$hd" = 0 ] && ok "翻转率与 |Δlp| 均在 FLOOR 线内" || ng "翻转率/|Δlp| 超线（翻转超线=$hn，Δ 超线=$hd）"
grep -q '❌ 长文本 ΔNLL' /tmp/kc_parity.txt && ng "长文本 ΔNLL 超线" || ok "长文本 ΔNLL 在 max(3×FLOOR, 0.01) 内"
grep -q '❌ 单 token' /tmp/kc_parity.txt && ng "单 token top1 不一致" || ok "单 token（无 attention）top1 一致"

echo
echo "通过 $PASS 项，失败 $FAIL 项"
[ "$FAIL" -eq 0 ] && [ "$PASS" -gt 0 ]
