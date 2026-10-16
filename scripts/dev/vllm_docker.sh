#!/usr/bin/env bash
# vLLM 对照服务（Q3 性能基线）：Docker 跑官方镜像，与 dinfer 同模型、同 max_model_len，占 GPU 1、端口 8013。
#   bash vllm_docker.sh pull    # 经 daocloud 镜像站拉 vllm/vllm-openai:$TAG（docker hub 直连不通，镜像站直连可达）
#   bash vllm_docker.sh run     # 起容器（后台），模型目录只读挂到 /models，HF 离线
#   bash vllm_docker.sh logs    # 跟日志
#   bash vllm_docker.sh smoke   # 一条 chat 请求
#   bash vllm_docker.sh stop
# 注意：镜像层落在 docker 的 data-root（/var/lib/docker），不在 /data/yaochen/infer（docker 不支持按镜像指定目录）；
# 本目录只放脚本 / 日志 / HF 缓存。
set -euo pipefail
ROOT=/data/yaochen/infer
TAG="${VLLM_TAG:-v0.31.0}"
IMG="docker.m.daocloud.io/vllm/vllm-openai:$TAG"
NAME=vllm-qwen3
GPU="${VLLM_GPU:-1}"
PORT="${VLLM_PORT:-8013}"
MODEL="${VLLM_MODEL:-qwen3-0.6b}"
MAX_LEN="${VLLM_MAX_LEN:-8192}"
mkdir -p $ROOT/vllm/hf-cache

case "${1:-}" in
  pull) docker pull "$IMG" ;;
  run)
    docker rm -f $NAME >/dev/null 2>&1 || true
    docker run -d --name $NAME --runtime nvidia --gpus "\"device=$GPU\"" --ipc=host \
      -p "$PORT:8000" \
      -v $ROOT/models:/models:ro -v $ROOT/vllm/hf-cache:/root/.cache/huggingface \
      -e HF_HUB_OFFLINE=1 -e VLLM_LOGGING_LEVEL=INFO \
      "$IMG" --model "/models/$MODEL" --served-model-name "$MODEL" \
      --max-model-len "$MAX_LEN" --gpu-memory-utilization 0.9 --max-num-seqs 64 \
      --enable-prefix-caching --host 0.0.0.0 --port 8000
    echo "started $NAME on GPU $GPU, port $PORT; logs: bash $0 logs" ;;
  logs)  docker logs -f --tail 50 $NAME ;;
  stop)  docker rm -f $NAME ;;
  smoke)
    curl -s -m 120 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"用一句话介绍你自己\"}],\"max_tokens\":64,\"temperature\":0.7,\"chat_template_kwargs\":{\"enable_thinking\":false}}" ;;
  *) sed -n 2,9p "$0"; exit 2 ;;
esac
