#!/usr/bin/env bash
# Kimi-K3 里程碑 C-3：真权重**前 N 层主干**对拍（embed → N 层（KDA/MLA × Dense/MoE + AttnRes）→ output_attn_res → norm → lm_head；
# docs/loops/k3-model.md）。exit 0 = 通过。
#   1) （可选）容器里 gen_k3_model_refs.py 生成 .npy 参考（K3_SKIP_GEN=1 跳过，直接用 REFS）
#   2) cargo test -p dinfer-kimik3 --features ref-tests --test model_parity
# 环境：K3_MODEL（默认 /data/models/moonshotai/Kimi-K3）、REFS（默认 test/kimik3/2026-10-20/model）、K3_LAYERS（默认 5：0 稠密、1/2 KDA+MoE、3 MLA+MoE、4 KDA+MoE）、
#      K3_GPU（跑 Rust 测试的卡，默认 2；tp 时逗号分隔如 2,3）、K3_REF_GPU（容器参考用的卡，默认 3）。用户约束：GPU 0/1 被占，只用 2–3。
#      K3_TP（默认 1；=2 → KIMIK3_TP=2 两 rank lockstep，docs/loops/k3-tp.md）、K3_DUMP（目录；rank 0 写各层 raw f32，供 k3_tp_compare.py）。
# 注意：remote_b300.sh push 带 --delete，参考生成后先 pull 回本地再 push（docs/loops/k3-kda.md 踩坑 0）。
set -uo pipefail
cd "$(dirname "$0")/../.."
MODEL="${K3_MODEL:-/data/models/moonshotai/Kimi-K3}"
REFS="${REFS:-test/kimik3/2026-10-20/model}"
LAYERS="${K3_LAYERS:-5}"
GPU="${K3_GPU:-2}"
REF_GPU="${K3_REF_GPU:-3}"
TP="${K3_TP:-1}"
DUMP="${K3_DUMP:-}"
LOG=/tmp/k3_model.log; [ "$TP" != 1 ] && LOG="/tmp/k3_model_tp$TP.log"
rm -f target/.rustc_info.json

if [ "${K3_SKIP_GEN:-0}" != "1" ]; then
  printf '==> 生成参考（容器，GPU %s，前 %s 层）→ %s\n' "$REF_GPU" "$LAYERS" "$REFS"
  mkdir -p "$REFS"
  ABS_REFS="$(cd "$REFS" && pwd)"
  sudo -n nerdctl -n k8s.io run --rm --net host --gpus "device=$REF_GPU" \
    -v /data/models:/data/models:ro -v "$(pwd)/scripts/verify:/work:ro" -v "$ABS_REFS:/out" \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 --entrypoint python3 docker.io/vllm/vllm-openai:nightly \
    -I /work/gen_k3_model_refs.py "$MODEL" /out --layers "$LAYERS" "$@" 2>&1 | grep -vE '^INFO|^WARNING' | tail -30 || exit 1
  [ -f "$REFS/final.npy" ] || { echo "参考未生成"; exit 1; }
fi

printf '\n==> 对拍（GPU %s，前 %s 层，tp %s）\n' "$GPU" "$LAYERS" "$TP"
# 用 env 显式传：之前 `${DUMP:+KIMIK3_DUMP="$DUMP"} RUST_LOG=info cargo …` 在 DUMP 为空时展开成空词，
# bash 把 `RUST_LOG=info` 当命令名 → "command not found"，cargo test 根本没跑（2026-10-20 C-6 乱文取证时发现）。
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" KIMIK3_DIR="$MODEL" KIMIK3_REFS="$REFS" KIMIK3_LAYERS="$LAYERS" KIMIK3_TP="$TP" ${DUMP:+KIMIK3_DUMP="$DUMP"} RUST_LOG=info \
  cargo test --release -p dinfer-kimik3 --features ref-tests --test model_parity -- --ignored --nocapture 2>&1 \
  | tee "$LOG" | grep -E '^\s+(第|装载|layer|embed|attn|res_out|final|logits|top|decode|前|显存|耗时|prefill|TP)|kimik3 layer loaded|kimik3: TP|^==|test result|panicked|error' | cut -c1-240
rc=${PIPESTATUS[0]}
grep -qE "^test result: ok" "$LOG" || { echo "FAIL: cargo test 未产出 test result（看 $LOG）"; rc=1; }
printf '\n日志：%s\n' "$LOG"
exit $rc
