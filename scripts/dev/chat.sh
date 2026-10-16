#!/usr/bin/env bash
# 向 dinfer / vLLM 的 OpenAI 兼容接口发一条 chat 请求。
#   scripts/dev/chat.sh [BASE_URL] "问题" [--stream] [--think]
#   BASE_URL 默认 $CHAT_URL 或 http://127.0.0.1:8012 ；--think 开 Qwen3 思考模式（默认关，输出短）
# 例：scripts/dev/chat.sh http://10.121.192.98:8012 "用一句话介绍你自己"
set -euo pipefail
URL="${CHAT_URL:-http://127.0.0.1:8012}"
[[ "${1:-}" == http* ]] && { URL="$1"; shift; }
MSG="${1:-用一句话介绍你自己}"; shift || true
STREAM=false; THINK=false
for a in "$@"; do case "$a" in --stream) STREAM=true;; --think) THINK=true;; esac; done
MODEL=$(curl -s -m 10 "$URL/v1/models" | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')
BODY=$(python3 - "$MODEL" "$MSG" "$STREAM" "$THINK" <<'PY'
import json, sys
model, msg, stream, think = sys.argv[1], sys.argv[2], sys.argv[3] == "true", sys.argv[4] == "true"
print(json.dumps({"model": model, "messages": [{"role": "user", "content": msg}], "max_tokens": 256,
                  "temperature": 0.7, "top_p": 0.9, "stream": stream,
                  "chat_template_kwargs": {"enable_thinking": think}}, ensure_ascii=False))
PY
)
if $STREAM; then
  curl -sN -m 300 "$URL/v1/chat/completions" -H 'content-type: application/json' -d "$BODY" \
    | python3 -c '
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data: "): continue
    d = line[6:]
    if d == "[DONE]": print(); break
    ch = json.loads(d)["choices"][0]
    delta = ch.get("delta", {})
    for k in ("reasoning_content", "content"):
        if delta.get(k): sys.stdout.write(delta[k]); sys.stdout.flush()
    fr = ch.get("finish_reason")
    if fr: print("\n[finish: " + fr + "]")
'
else
  curl -s -m 300 "$URL/v1/chat/completions" -H 'content-type: application/json' -d "$BODY" \
    | python3 -c '
import sys, json
d = json.load(sys.stdin)
if "error" in d: print("ERROR:", d["error"]); sys.exit(1)
m = d["choices"][0]["message"]
if m.get("reasoning_content"): print("[reasoning]", m["reasoning_content"])
print(m.get("content"))
print("--", d["usage"], "finish:", d["choices"][0]["finish_reason"])
'
fi
