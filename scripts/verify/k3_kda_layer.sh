#!/usr/bin/env bash
# Kimi-K3 里程碑 C-1：真权重单层 KDA 对拍（docs/loops/k3-kda.md）。exit 0 = 通过。
#   1) （可选）容器里 gen_k3_kda_layer_refs.py 生成 .npy 参考（K3_SKIP_GEN=1 跳过，直接用 REFS）
#   2) cargo test -p dinfer-kimik3 --features ref-tests --test kda_layer_parity
# 环境：K3_MODEL（默认 /data/models/moonshotai/Kimi-K3）、REFS（默认 test/kimik3/2026-10-19/kda_layer）、
#      K3_GPU（跑 Rust 测试的卡，默认 2；用户约束：GPU 0/1 被占，只用 2–7）、K3_REF_GPU（容器参考用的卡，默认 3）。
set -uo pipefail
cd "$(dirname "$0")/../.."
MODEL="${K3_MODEL:-/data/models/moonshotai/Kimi-K3}"
REFS="${REFS:-test/kimik3/2026-10-19/kda_layer}"
GPU="${K3_GPU:-2}"
REF_GPU="${K3_REF_GPU:-3}"
LOG=/tmp/k3_kda_layer.log
rm -f target/.rustc_info.json

if [ "${K3_SKIP_GEN:-0}" != "1" ]; then
  printf '==> 生成参考（容器，GPU %s）→ %s\n' "$REF_GPU" "$REFS"
  mkdir -p "$REFS"
  ABS_REFS="$(cd "$REFS" && pwd)"
  sudo -n nerdctl -n k8s.io run --rm --net host --gpus "device=$REF_GPU" \
    -v /data/models:/data/models:ro -v "$(pwd)/scripts/verify:/work:ro" -v "$ABS_REFS:/out" \
    -e HF_HUB_OFFLINE=1 --entrypoint python3 docker.io/vllm/vllm-openai:nightly \
    -I /work/gen_k3_kda_layer_refs.py "$MODEL" /out "$@" 2>&1 | grep -vE '^INFO|^WARNING' | tail -20 || exit 1
  [ -f "$REFS/y_fla.npy" ] || { echo "参考未生成"; exit 1; }
fi

printf '\n==> 对拍（GPU %s）\n' "$GPU"
CUDA_VISIBLE_DEVICES="$GPU" KIMIK3_DIR="$MODEL" KIMIK3_REFS="$REFS" \
  cargo test -p dinfer-kimik3 --features ref-tests --test kda_layer_parity -- --ignored --nocapture 2>&1 \
  | tee "$LOG" | grep -E '^\s+(第|装载|o_norm|out|chunk|recurrent|decode|fla)|^==|test result|panicked|error' | cut -c1-200
rc=${PIPESTATUS[0]}
printf '\n日志：%s\n' "$LOG"
exit $rc
