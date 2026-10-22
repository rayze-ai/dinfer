#!/usr/bin/env bash
# 批量抓 oracle：test/kaon3/prompts.tsv（name<TAB>用户消息）逐条套 chat 模板抓取。
# 用法：ORACLE_OUT=<目录> scripts/dev/kaon3_oracle_set.sh       # 当前私有 oracle 的配置（BI 开 / 关）决定结果
# 同一批 prompt 在 BI 开 / 关两套 oracle 下各抓一份 → kaon3_oracle_diff 汇总得 vLLM 自身噪声下限。
set -euo pipefail
cd "$(dirname "$0")/../.."
while IFS=$'\t' read -r name msg; do
  [ -z "$name" ] && continue
  timeout 300 scripts/dev/kaon3_oracle.sh "$name" chat "$msg" | tail -1
done < test/kaon3/prompts.tsv
