#!/usr/bin/env bash
# 造一个符合 docs/data-layout.md 规范的数据目录，并把 README 里的口径字段填好。
#
# 用法（容器内、cwd=/w/dinfer）：
#   RUN=$(bash scripts/dev/mkrun.sh moe-seg r07 "参照 MoE 分段，compile 口径")
#   echo $RUN            # /w/runs/moe-seg_20261024-1430_r07
#   ./tool --out "$RUN"
#
# 参数：
#   $1 用途（moe-seg / hidden / e2e / fp4 / oracle）
#   $2 轮次（rNN）
#   $3 一句话说明
# 环境变量：
#   RUNNAME  覆盖自动生成的目录名（重跑同一轮时用 xxx_b）
set -euo pipefail

PURPOSE="${1:?用法: mkrun.sh <用途> <rNN> <说明>}"
ROUND="${2:?缺少轮次，例如 r07}"
NOTE="${3:-}"

case "$PURPOSE" in
  moe-seg|hidden|e2e|fp4|oracle) ;;
  *) echo "❌ 未知用途 '$PURPOSE'（允许：moe-seg hidden e2e fp4 oracle）" >&2; exit 1 ;;
esac
case "$ROUND" in
  r[0-9][0-9]) ;;
  *) echo "❌ 轮次格式应为 rNN（拿到 '$ROUND'）" >&2; exit 1 ;;
esac

ROOT="${RUNS_ROOT:-/w/runs}"
NAME="${RUNNAME:-${PURPOSE}_$(date +%Y%m%d-%H%M)_${ROUND}}"
DIR="$ROOT/$NAME"

if [ -e "$DIR" ]; then
  echo "❌ $DIR 已存在。同一轮重跑请设 RUNNAME=${NAME}_b 之类，不要覆盖。" >&2
  exit 1
fi
mkdir -p "$DIR"

cat > "$DIR/README.txt" <<EOF
用途      : $PURPOSE
轮次      : $ROUND
创建      : $(date '+%Y-%m-%d %H:%M:%S %z')
说明      : $NOTE
主机      : $(hostname)
目录      : $DIR

## 口径（填全，缺一项就不许拿去当判据）
参照口径  : <compile / eager>            # 规格口径是 compile；eager 只能做诊断
RMSNORM_MODE      : ${RMSNORM_MODE:-<未设>}
MOE_COMBINE_F32   : ${MOE_COMBINE_F32:-<未设>}
CUBLASLT_WORKSPACE_SIZE : ${CUBLASLT_WORKSPACE_SIZE:-<未设>}
其他环境变量       : <补>

## 来源
脚本 / 命令行 :

## 非零校验（抓完立刻填，勿凭印象）
文件  行数  nz比例

EOF

echo "$DIR"
