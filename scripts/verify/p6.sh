#!/usr/bin/env bash
# P6 验收：p5 全部 + GLM-5.3 真 tokenizer / chat 模板对拍 + 解析器单测 + server e2e + MTP 对拍。exit 0 = 通过。
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
TINY_DSA="$(realpath -m "${TINY_GLM_DSA_DIR:-test/tiny_glm_dsa}")"
TINY_FULL="$(realpath -m "${TINY_GLM_DSA_FULLVOCAB_DIR:-test/tiny_glm_dsa_fullvocab}")"   # 词表 154880：真 tokenizer 的 id 能喂进模型
TOK_DIR="${GLM_TOKENIZER_DIR:-/data/ref/models/glm53}"
PY519="${PY_TF519:-/data/ref/venv-tf519/bin/python}"
step() { rm -f target/.rustc_info.json; printf '\n==> %s\n' "$*"; }

scripts/verify/p5.sh

step "P6-a: HF golden (chat template / encode / incremental decode) → $TINY_DSA/chat_golden.json"
"$PY519" -I scripts/verify/gen_chat_golden.py "$TINY_DSA/chat_golden.json" --tokenizer "$TOK_DIR" 2>&1 </dev/null | grep -v -i warning | tail -3

step "P6-a: tokenizer + chat template parity vs HF"
TINY_GLM_DIR="$TINY_DSA" GLM_TOKENIZER_DIR="$TOK_DIR" cargo test -p dinfer-glm53 --features ref-tests --test chat_parity -- --nocapture </dev/null 2>&1 \
  | tee /tmp/p6_chat.log | grep -E '^test |test result|panicked'
grep -qE "test result: ok\." /tmp/p6_chat.log

step "P6-b: output parser unit tests"
cargo test -p dinfer-glm53 parser:: </dev/null 2>&1 | tee /tmp/p6_parser.log | grep -E '^test |test result|panicked'
grep -qE "test result: ok\. [1-9]" /tmp/p6_parser.log

step "P6-c: generate full-vocab tiny DSA GLM (vocab 154880) → $TINY_FULL"
"$PY519" -I scripts/verify/gen_tiny_glm.py "$TINY_FULL" --layers 3 --first-k-dense 1 --dsa --vocab 154880 --long 64 --gen 16 2>&1 </dev/null | grep -v -i warning | tail -1
cp "$TINY_DSA/chat_golden.json" "$TINY_FULL/chat_golden.json"

step "P6-c: server e2e (spawn serve on full-vocab tiny DSA + real tokenizer; stream == non-stream; usage; /v1/models)"
TINY_GLM_DIR="$TINY_FULL" GLM_TOKENIZER_DIR="$TOK_DIR" cargo test -p dinfer --features glm53,e2e --test serve_e2e -- --ignored --nocapture --test-threads=1 </dev/null 2>&1 \
  | tee /tmp/p6_e2e.log | grep -E '^test |test result|panicked|e2e:'
grep -qE "test result: ok\." /tmp/p6_e2e.log

TINY_MTP="$(realpath -m "${TINY_GLM_DSA_MTP_DIR:-test/tiny_glm_dsa_mtp}")"
step "P6-d: generate tiny DSA+MTP GLM (layers.3 = MTP) → $TINY_MTP"
"$PY519" -I scripts/verify/gen_tiny_glm.py "$TINY_MTP" --layers 3 --first-k-dense 1 --dsa --mtp --long 64 --gen 16 2>&1 </dev/null | grep -v -i warning | tail -2

step "P6-d: MTP parity (draft logits vs HF MTP layer; greedy with MTP == without)"
TINY_GLM_DIR="$TINY_MTP" cargo test -p dinfer-glm53 --features ref-tests --test mtp_parity -- --ignored --nocapture --test-threads=1 </dev/null 2>&1 \
  | tee /tmp/p6_mtp.log | grep -E '^test |test result|panicked|mtp:'
grep -qE "test result: ok\." /tmp/p6_mtp.log

printf '\n==> P6 PASS\n'
