#!/usr/bin/env bash
# 受限解码 + chat 字段补全验收（docs/loops/guided-decoding.md）：Qwen3-0.6B 真权重（本机 3090）。
# 1) core / guided / server 单测；2) e2e_qwen3_guided（json_object / json_schema / tool_choice / logit_bias /
# prompt_logprobs / truncate_prompt_tokens / continue_final_message / tokenize / completions echo+logprobs）；
# 3) 回归：e2e_qwen3_request_features / e2e_qwen3_real_model_chat / e2e_qwen3_openai_surface。exit 0 = 通过。
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
QWEN3="${QWEN3_DIR:-/data/ref/models/qwen3-0.6b}"
export QWEN3_GOLDEN="${QWEN3_GOLDEN:-$(realpath test/qwen3_golden)}"   # real_model_chat 读 chat_golden.json；cargo test 的 cwd 是 crate 目录
step() { rm -f target/.rustc_info.json; printf '\n==> %s\n' "$*"; }
wait_gpu() { # 一次只跑一个 e2e：等别的 dinfer 进程退出（最多 10 分钟）
  for _ in $(seq 1 20); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && return 0; sleep 30; done
  echo "gpu busy" >&2; return 1
}

step "单测（CPU + 真 tokenizer 文法走查）"
cargo test -p dinfer-core -p dinfer-guided -p dinfer-server </dev/null 2>&1 | grep -E 'test result|panicked|^error' || true
cargo test -p dinfer-core -p dinfer-guided -p dinfer-server </dev/null 2>&1 | grep -qE 'test result: FAILED' && { echo "unit FAILED"; exit 1; }

for t in e2e_qwen3_guided e2e_qwen3_request_features e2e_qwen3_real_model_chat e2e_qwen3_openai_surface; do
  step "e2e: $t（真权重，3090）"
  wait_gpu
  QWEN3_DIR="$QWEN3" cargo test -p dinfer --features qwen3,e2e --test serve_e2e_qwen3 -- --ignored --test-threads=1 "$t" </dev/null 2>&1 \
    | tee "/tmp/guided_$t.log" | grep -E '^test |test result|panicked|e2e:'
  grep -qE "test result: ok\. 1 passed" "/tmp/guided_$t.log" || { echo "$t FAILED"; exit 1; }
done
echo "guided: PASS"
