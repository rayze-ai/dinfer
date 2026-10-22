#!/usr/bin/env bash
# 重启私有 vLLM oracle（kaon-oracle，GPU 1，:28100，与基线同镜像同参数、去掉 PegaKV 连接器）。
# 用法：scripts/dev/kaon3_oracle_server.sh          # 全 30 层
#       scripts/dev/kaon3_oracle_server.sh <N>      # 截到前 N 层（--hf-overrides），配合探针 --layers N 逐层二分
#       BI=0 scripts/dev/kaon3_oracle_server.sh     # 关 VLLM_BATCH_INVARIANT（换 kernel）：标定 vLLM 自身的数值敏感度
# 等到 /v1/models 返回 200 才退出；失败打印容器日志尾部。绝不碰基线容器 kaon-vllm-pegaflow（GPU 7）。
set -euo pipefail
cd "$(dirname "$0")/../.."
R=scripts/dev/remote_5090.sh
N=${1:-}
OV=""
[ -n "$N" ] && OV="--hf-overrides '{\"text_config\":{\"num_hidden_layers\":$N}}'"
SNAP=/models/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
OG=${ORACLE_GPU:-1}; OP=${ORACLE_PORT:-28100}; ON=${ORACLE_NAME:-kaon-oracle}   # 多实例并行：卡 / 端口 / 容器名各不相同
case $OG in 7) echo "GPU 7 是基线，禁止"; exit 2;; esac
timeout 120 $R run "docker rm -f $ON >/dev/null 2>&1; docker run -d --name $ON --restart no --gpus device=$OG \
  --network host --ipc=host -v /hf_data/hub:/models:ro -e VLLM_BATCH_INVARIANT=${BI:-1} \
  image.paigpu.com/library/vllm-openai:v0.28.0-pegaflow-pr455-51261c6-cu130 $SNAP --port $OP --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 --dtype auto --served-model-name flowgpt_vllm --max-num-seqs 64 --max-model-len 16384 \
  --block-size 32 $OV >/dev/null"
for _ in $(seq 1 60); do
  s=$(timeout 30 $R run "curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$OP/v1/models; docker inspect -f ' {{.State.Status}}' $ON" 2>/dev/null || true)
  case "$s" in
    200*) echo "$ON 就绪（GPU $OG :$OP layers=${N:-30}）"; exit 0 ;;
    *exited*|*dead*) break ;;
  esac
  sleep 10
done
echo "$ON 启动失败："; timeout 60 $R run "docker logs --tail 15 $ON 2>&1 | cut -c1-200"; exit 1
