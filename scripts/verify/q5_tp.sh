#!/usr/bin/env bash
# Q5 TP 一致性（docs/loops/q5-tp-dp.md）：单卡 vs --tp N greedy 逐 token 一致。在 4090 Pod 上跑。
# 用法：scripts/verify/q5_tp.sh [TP=2] [BASE_GPU=0]   需 ./target/release/dinfer + $R/models/qwen3-0.6b
set -uo pipefail
TP="${1:-2}"; BASE="${2:-0}"
R="${R:-/data/yaochen/infer}"; MODEL="${MODEL:-$R/models/qwen3-0.6b}"; BIN="${BIN:-./target/release/dinfer}"
OUT="${OUT:-/tmp/q5_tp}"; mkdir -p "$OUT"
COMMON=(--model "$MODEL" --host 127.0.0.1 --max-model-len 2048 --max-num-seqs 32 --gpu-memory-utilization 0.5 --num-blocks 2000)  # 固定块数：避免各 rank 自动预算不一致 / 上一进程显存未释放
up() { for _ in $(seq 1 240); do curl -s -m 2 "http://127.0.0.1:$1/health" >/dev/null && return 0; sleep 0.5; done; echo "port $1 down"; tail -20 "$OUT/$2.log"; return 1; }
ask() { curl -s -m 120 "http://127.0.0.1:$1/v1/chat/completions" -H 'content-type: application/json' \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":$(printf '%s' "$2" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')}],\"max_tokens\":64,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
  | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["choices"][0]["message"]["content"])'; }
pkill -INT -f "dinfer serve .*--port 83[01]1" 2>/dev/null; sleep 1
RUST_LOG=info nohup "$BIN" serve "${COMMON[@]}" --port 8301 --device "$BASE" >"$OUT/single.log" 2>&1 &
up 8301 single || exit 1
PROMPTS=("Explain what a KV cache is in two sentences." "Write a haiku about Rust." "List three prime numbers greater than 100 and explain why.")
: > "$OUT/single.txt"; for p in "${PROMPTS[@]}"; do ask 8301 "$p" >> "$OUT/single.txt"; echo "---" >> "$OUT/single.txt"; done
pkill -INT -f "dinfer serve .*--port 8301"; for _ in $(seq 1 40); do pgrep -f "dinfer serve .*--port 8301" >/dev/null || break; sleep 0.5; done; sleep 1
NCCL_DEBUG=${NCCL_DEBUG:-WARN} RUST_LOG=info nohup "$BIN" serve "${COMMON[@]}" --port 8311 --device "$BASE" --tp "$TP" >"$OUT/tp.log" 2>&1 &
up 8311 tp || exit 1
: > "$OUT/tp.txt"; for p in "${PROMPTS[@]}"; do ask 8311 "$p" >> "$OUT/tp.txt"; echo "---" >> "$OUT/tp.txt"; done
pkill -INT -f "dinfer serve .*--port 8311"
grep -E 'tp: local shard|NCCL INFO.*(SHM|P2P|NET)' "$OUT/tp.log" | head -5
if diff -q "$OUT/single.txt" "$OUT/tp.txt" >/dev/null; then echo "Q5 TP=$TP PASS (outputs identical, $(wc -l < "$OUT/tp.txt") lines)"; exit 0
else echo "Q5 TP=$TP FAIL"; diff "$OUT/single.txt" "$OUT/tp.txt" | head -20; exit 1; fi
