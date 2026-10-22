#!/usr/bin/env bash
# 重启私有 vLLM oracle（kaon-oracle，GPU 1，:28100，与基线同镜像同参数、去掉 PegaKV 连接器）。
# 用法：scripts/dev/kaon3_oracle_server.sh          # 全 30 层
#       scripts/dev/kaon3_oracle_server.sh <N>      # 截到前 N 层（--hf-overrides），配合探针 --layers N 逐层二分
# 等到 /v1/models 返回 200 才退出；失败打印容器日志尾部。绝不碰基线容器 kaon-vllm-pegaflow（GPU 7）。
set -euo pipefail
cd "$(dirname "$0")/../.."
R=scripts/dev/remote_5090.sh
N=${1:-}
OV=""
[ -n "$N" ] && OV="--hf-overrides '{\"text_config\":{\"num_hidden_layers\":$N}}'"
SNAP=/models/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
timeout 120 $R run "docker rm -f kaon-oracle >/dev/null 2>&1; docker run -d --name kaon-oracle --restart no --gpus device=1 \
  --network host --ipc=host -v /hf_data/hub:/models:ro -e VLLM_BATCH_INVARIANT=1 \
  image.paigpu.com/library/vllm-openai:v0.28.0-pegaflow-pr455-51261c6-cu130 $SNAP --port 28100 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 --dtype auto --served-model-name flowgpt_vllm --max-num-seqs 64 --max-model-len 16384 \
  --block-size 32 $OV >/dev/null"
for _ in $(seq 1 60); do
  s=$(timeout 30 $R run "curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:28100/v1/models; docker inspect -f ' {{.State.Status}}' kaon-oracle" 2>/dev/null || true)
  case "$s" in
    200*) echo "kaon-oracle 就绪（layers=${N:-30}）"; exit 0 ;;
    *exited*|*dead*) break ;;
  esac
  sleep 10
done
echo "kaon-oracle 启动失败："; timeout 60 $R run "docker logs --tail 15 kaon-oracle 2>&1 | cut -c1-200"; exit 1
