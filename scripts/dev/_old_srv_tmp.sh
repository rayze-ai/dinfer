#!/usr/bin/env bash
# 重启私有 vLLM oracle（kaon-oracle，GPU 1，:28100，与基线同镜像同参数、去掉 PegaKV 连接器）。
# 用法：scripts/dev/kaon3_oracle_server.sh          # 全 30 层
#       scripts/dev/kaon3_oracle_server.sh <N>      # 截到前 N 层（--hf-overrides），配合探针 --layers N 逐层二分
#       BI=0 scripts/dev/kaon3_oracle_server.sh     # 关 VLLM_BATCH_INVARIANT（换 kernel）：标定 vLLM 自身的数值敏感度
#       BLOCK=16 scripts/dev/kaon3_oracle_server.sh # 换 block size，标定 paged attention 布局差异的影响
#
# **对拍参数对齐（用户 2026-10-22）**：生产命令没写的参数不等于不能配；对拍时必须把 oracle 的参数
# 与探针侧的实际假设逐项对齐，否则看到的差异里混着配置差异。当前对齐关系：
#   | 参数            | 探针侧假设                       | oracle 取值                      |
#   | block size      | 32（kaon3_probe.rs:23 BLOCK_SIZE）| 32（默认；BLOCK= 可覆盖做噪声标定）|
#   | prefill 分块    | 一次 prefill 整串，不分块         | --max-num-batched-tokens 4096（> 最长用例 1252，等价不分块）|
#   | KV dtype        | FP8 e4m3（config kv_cache_quant_algo）| FP8（checkpoint 自带）      |
#   | max_model_len   | 无（单序列）                      | 16384                            |
#   | batch invariance| 无此概念                          | BI=1 作判据参照 / BI=0 作噪声下限 |
# 生产（`vllm serve <model> --port 8081 --max_num_batched_tokens 4096`）用 block 16 / max_num_seqs 256，
# 与上表不同；生产口径只用于阶段 4 之后的引擎层面比较，不用于逐位置 logprob 对拍。
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
  --block-size ${BLOCK:-32} $OV >/dev/null"
for _ in $(seq 1 60); do
  s=$(timeout 30 $R run "curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$OP/v1/models; docker inspect -f ' {{.State.Status}}' $ON" 2>/dev/null || true)
  case "$s" in
    200*) echo "$ON 就绪（GPU $OG :$OP layers=${N:-30} block=${BLOCK:-32} BI=${BI:-1}）"; exit 0 ;;
    *exited*|*dead*) break ;;
  esac
  sleep 10
done
echo "$ON 启动失败："; timeout 60 $R run "docker logs --tail 15 $ON 2>&1 | cut -c1-200"; exit 1
