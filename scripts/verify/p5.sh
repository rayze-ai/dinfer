#!/usr/bin/env bash
# P5 验收：p4 全部 + tiny DSA GLM（3 层 full/shared/full，索引器 top-k 8）与 transformers 5.19 GlmMoeDsa 对拍。exit 0 = 通过。
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
TINY_DSA="$(realpath -m "${DINFER_TINY_GLM_DSA:-test/tiny_glm_dsa}")"
PY519="${DINFER_PY_TF519:-/data/ref/venv-tf519/bin/python}"
rm -f target/.rustc_info.json

step() { printf '\n==> %s\n' "$*"; }

scripts/verify/p4.sh

step "transformers 5.19 venv ($PY519)"
"$PY519" -I -c "import transformers; from transformers import GlmMoeDsaForCausalLM; print('transformers', transformers.__version__)"

step "generate tiny DSA GLM reference → $TINY_DSA"
"$PY519" scripts/verify/gen_tiny_glm.py "$TINY_DSA" --layers 3 --first-k-dense 1 --dsa 2>&1 | grep -v -i warning

# 注：tiny_parity / runner_parity 的 "每步 logits < 0.05" 口径不适用于 DSA fixture —— 索引器 top-k 与 MoE 路由两类硬判决
# 在 bf16 下会在两边翻转（见 docs/pitfalls.md 2026-10-15），翻转后的行/步不可比。DSA 的验收全部放在 dsa_parity（可判定行上断言）。
step "tiny DSA GLM: indexer top-k sets vs HF; sparse attention with HF top-k injected; self-run (tie-aware)"
DINFER_TINY_GLM="$TINY_DSA" cargo test -p dinfer-glm53 --features ref-tests --test dsa_parity -- --ignored --test-threads=1 --nocapture 2>&1 \
  | tee /tmp/dinfer_p5_dsa.log | grep -E '^test |test result|panicked|topk rows|worst|decidable'
grep -qE "test result: ok\. 3 passed" /tmp/dinfer_p5_dsa.log

printf '\n==> P5 PASS\n'
