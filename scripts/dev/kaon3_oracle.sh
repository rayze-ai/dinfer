#!/usr/bin/env bash
# 从 vLLM 取对拍 oracle（默认私有副本 :28100；ORACLE_PORT=28067 = 共享基线，只许短 prompt）：
# 每个 prompt 位置的 top-5 logprob（prompt_logprobs）+ prompt 之后下一个 token 的 top-5。只发 HTTP，不占卡。
# 用法：
#   scripts/dev/kaon3_oracle.sh <name> ids <id...>         # 直接给 token id
#   scripts/dev/kaon3_oracle.sh <name> chat '<用户消息>'    # 基线 /tokenize 套 chat 模板（add_generation_prompt）
#   scripts/dev/kaon3_oracle.sh <name> file <路径> <字符数>  # 取本地文件前 N 字符原样分词（造 >1024 token 的滑窗用例）
# 产物：test/kaon3/oracle/<name>.ids（空格分隔）与 <name>.json（/v1/completions 原始响应）。
set -euo pipefail
cd "$(dirname "$0")/../.."
name=$1; mode=$2; shift 2
R=scripts/dev/remote_5090.sh
PORT=${ORACLE_PORT:-28100}   # 私有副本 kaon-oracle（GPU 1）；共享基线 28067 只许短 prompt（2026-10-22 长请求打崩过它）
URL=http://127.0.0.1:$PORT
MODEL=flowgpt_vllm
OUT=test/kaon3/oracle
mkdir -p "$OUT"
post() { # post <path>：stdin 是 JSON body，经 ssh 交给远端 curl；跳板偶发 "handshake failed" 且 rc=0，按响应是否为 JSON 判定并重试
  local body out i
  body=$(cat)
  for i in 1 2 3; do
    out=$(printf '%s' "$body" | $R runin "curl -sf $URL$1 -H 'Content-Type: application/json' --data-binary @-" 2>/dev/null || true)
    if printf '%s' "$out" | jq -e . >/dev/null 2>&1; then printf '%s' "$out"; return 0; fi
    sleep 2
  done
  echo "post $1 三次失败" >&2; return 1
}
case "$mode" in
  ids)  ids="$*" ;;
  chat) ids=$(jq -n --arg m "$MODEL" --arg c "$1" \
          '{model:$m, messages:[{role:"user", content:$c}], add_generation_prompt:true}' | post /tokenize | jq -r '.tokens | join(" ")') ;;
  file) ids=$(head -c "$2" "$1" | jq -Rs --arg m "$MODEL" '{model:$m, prompt:., add_special_tokens:true}' \
          | post /tokenize | jq -r '.tokens | join(" ")') ;;
  *) echo "mode 只能是 ids / chat / file"; exit 2 ;;
esac
[ -n "$ids" ] || { echo "分词失败"; exit 1; }
if [ "$PORT" = 28067 ] && [ "$(echo "$ids" | wc -w)" -gt 64 ]; then echo "共享基线只许 ≤64 token 的请求，改用 ORACLE_PORT=28100"; exit 2; fi
echo "$ids" > "$OUT/$name.ids"
echo "$ids" | jq -R --arg m "$MODEL" \
  '{model:$m, prompt:(split(" ") | map(tonumber)), max_tokens:1, temperature:0, logprobs:5, prompt_logprobs:5,
    return_tokens_as_token_ids:true}' | post /v1/completions > "$OUT/$name.json"
n=$(wc -w < "$OUT/$name.ids")
echo "$name: $n tokens → $OUT/$name.{ids,json}（$(wc -c < "$OUT/$name.json") B）"
