#!/usr/bin/env bash
# B300 上的 vLLM 对照服务（docs/loops/b-nvlink.md）：机器无 docker、无外网，但 containerd 的 k8s.io 命名空间里已缓存
# 模型用 /data/models 下的物理路径（VLLM_MODEL=/data/models/Qwen/Qwen3.8-27B），容器挂 /data/models:ro，不用软链接（用户 2026-10-19）。
# `docker.io/vllm/vllm-openai:nightly`（vllm 0.29.1rc1.dev573 / torch 2.13 cu130，sm_103 可跑）。用 nerdctl 起，`--net host`
# （无 CNI bridge 插件，桥接网络起不来），GPU 用 `--gpus device=N`。**用完必须 `stop`**（用户 2026-10-19：用完记得清理）；
# `stop` 删容器；镜像是机器原有的，不删。
#   bash scripts/dev/vllm_nerdctl.sh run [GPU=7] [PORT=8013] [TP=1]   # 后台起 vllm-qwen3-b300，HF 离线，模型只读挂 /models
#   bash scripts/dev/vllm_nerdctl.sh wait                             # 等 /health 200（最多 10 min）
#   bash scripts/dev/vllm_nerdctl.sh smoke | logs | stop | ps
set -euo pipefail
ROOT=/data/yaochen/infer
NS=k8s.io
IMG="${VLLM_IMG:-docker.io/vllm/vllm-openai:nightly}"
NAME="${VLLM_NAME:-vllm-qwen3-b300}"
MODEL="${VLLM_MODEL:-/data/models/Qwen/Qwen3-0.6B}"   # 物理路径（用户 2026-10-19：不要软链接）；served-model-name 取 basename
MAX_LEN="${VLLM_MAX_LEN:-8192}"
N() { sudo -n nerdctl -n $NS "$@" 2> >(grep -vE "DEPRECATION|level=warning" >&2); }
case "${1:-}" in
  run)
    GPU="${2:-7}"; PORT="${3:-8013}"; TP="${4:-1}"
    N rm -f "$NAME" >/dev/null 2>&1 || true
    mkdir -p $ROOT/vllm/hf-cache
    N run -d --name "$NAME" --net host --gpus "device=$GPU" --ipc host --shm-size 16g \
      -v /data/models:/data/models:ro -v $ROOT/vllm/hf-cache:/root/.cache/huggingface \
      -e HF_HUB_OFFLINE=1 -e VLLM_LOGGING_LEVEL=INFO \
      "$IMG" --model "$MODEL" --served-model-name "$(basename "$MODEL")" --tensor-parallel-size "$TP" \
      --max-model-len "$MAX_LEN" --gpu-memory-utilization 0.9 --max-num-seqs 64 \
      --enable-prefix-caching --host 127.0.0.1 --port "$PORT" >/dev/null
    echo "started $NAME on GPU $GPU, port $PORT, tp $TP（记得 stop）" ;;
  wait)
    PORT="${2:-8013}"
    for _ in $(seq 1 600); do curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null && { echo "vllm up on :$PORT"; exit 0; }; sleep 1; done
    echo "vllm did not come up"; N logs --tail 30 "$NAME" 2>&1 | tail -30; exit 1 ;;
  logs)  N logs -f --tail 50 "$NAME" ;;
  ps)    N ps -a | grep -E "NAMES|$NAME" || true ;;
  smoke)
    PORT="${2:-8013}"
    curl -s -m 120 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"用一句话介绍你自己\"}],\"max_tokens\":48,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" | head -c 400; echo ;;
  stop)  N rm -f "$NAME" && echo "removed $NAME" ;;
  *) sed -n 2,10p "$0"; exit 2 ;;
esac
