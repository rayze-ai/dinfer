#!/usr/bin/env bash
# Q3-d 逐段耗时对比：在 GPU 3 上分别起独立的 dinfer 与 vLLM 实例，nsys 抓同一 bench 负载，按阶段并排。
#   bash scripts/verify/stage_profile.sh [OUT_DIR]   # 默认 test/qwen3/$(date +%F)/stage
# 负载：bench --concurrency 8 --num-requests 24 --prompt-len 256 --max-tokens 64（decode 为主，含 8 次 prefill）。
# vLLM：宿主 `nsys profile docker run …` 抓不到（容器由 daemon 起，不是 nsys 的子进程，实测 0 kernel）；改为把宿主
# CUDA 13 toolkit 目录只读挂进容器，用其中的 nsys 作 entrypoint 包住 vLLM（实测可抓到 kernel）。
set -euo pipefail
cd "$(dirname "$0")/../.."
R=/data/yaochen/infer
OUT="${1:-test/qwen3/$(date +%F)/stage}"; mkdir -p "$OUT"
GPU="${STAGE_GPU:-3}"
LOAD=(--concurrency 8 --num-requests 24 --prompt-len 256 --max-tokens 64 --warmup 2)
BENCH=./target/release/bench; [ -x $BENCH ] || cargo build --release -p dinfer-bench
source $R/env.sh

echo "== dinfer under nsys (GPU $GPU, :8033)"
nsys profile -o "$OUT/dinfer" --force-overwrite true --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --duration=90 ./target/release/dinfer serve --model $R/models/qwen3-0.6b --port 8033 --max-model-len 8192 \
  --max-num-seqs 64 --max-num-batched-tokens 4096 --num-blocks 2048 --device "$GPU" > "$OUT/dinfer_serve.log" 2>&1 &
for i in $(seq 1 90); do sleep 1; grep -q listening "$OUT/dinfer_serve.log" 2>/dev/null && break; done
$BENCH --url http://127.0.0.1:8033 "${LOAD[@]}" --label dinfer --out "$OUT/dinfer_bench.json" | tee "$OUT/dinfer_bench.log" | grep -E 'throughput|TTFT|TPOT'
pkill -INT -f "port 8033" || true
for i in $(seq 1 60); do sleep 1; pgrep -f "nsys profile.*$OUT/dinfer" >/dev/null || break; done
nsys stats --report cuda_gpu_kern_sum --format csv --force-export true "$OUT/dinfer.nsys-rep" > "$OUT/dinfer_kern.csv" 2>/dev/null
D_STEPS=$(python3 -c "import json;d=json.load(open('$OUT/dinfer_bench.json'));print(int(d['output_tokens_mean']*d['completed']/8))")

echo "== vLLM under nsys inside the container (GPU $GPU, :8034); host CUDA 13 toolkit bind-mounted for nsys"
docker rm -f vllm-prof >/dev/null 2>&1 || true
ABS_OUT="$(realpath "$OUT")"
docker run -d --name vllm-prof --runtime nvidia --gpus "\"device=$GPU\"" --ipc=host --cap-add=SYS_ADMIN -p 8034:8000 \
  -v $R/models:/models:ro -v $R/vllm/hf-cache:/root/.cache/huggingface -v $R/cuda-13.0:/opt/cuda13:ro -v "$ABS_OUT":/out \
  -e HF_HUB_OFFLINE=1 --entrypoint /opt/cuda13/bin/nsys docker.m.daocloud.io/vllm/vllm-openai:v0.31.0 \
  profile -o /out/vllm --force-overwrite true --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --duration=300 \
  python3 -m vllm.entrypoints.openai.api_server --model /models/qwen3-0.6b --served-model-name qwen3-0.6b \
    --max-model-len 8192 --gpu-memory-utilization 0.9 --max-num-seqs 64 --enable-prefix-caching --host 0.0.0.0 --port 8000 \
  > "$OUT/vllm_serve.log" 2>&1
for i in $(seq 1 240); do sleep 1; curl -s -m 2 http://127.0.0.1:8034/v1/models >/dev/null 2>&1 && break; done
$BENCH --url http://127.0.0.1:8034 "${LOAD[@]}" --label vllm --out "$OUT/vllm_bench.json" | tee "$OUT/vllm_bench.log" | grep -E 'throughput|TTFT|TPOT'
# 让 nsys 正常收尾：给容器内进程 SIGINT（nsys 转发给 python），等报告落盘
docker kill -s INT vllm-prof >/dev/null 2>&1 || true
for i in $(seq 1 120); do sleep 1; [ -s "$OUT/vllm.nsys-rep" ] && ! docker ps -q -f name=vllm-prof | grep -q . && break; done
docker rm -f vllm-prof >/dev/null 2>&1 || true
nsys stats --report cuda_gpu_kern_sum --format csv --force-export true "$OUT/vllm.nsys-rep" > "$OUT/vllm_kern.csv" 2>/dev/null || true
V_STEPS=$(python3 -c "import json;d=json.load(open('$OUT/vllm_bench.json'));print(int(d['output_tokens_mean']*d['completed']/8))")

{
  echo "# 逐段 GPU 耗时：dinfer vs vLLM（$(date '+%F %T')，GPU $GPU，负载 ${LOAD[*]}）"
  echo
  echo "注：nsys 抓的是整个进程生命周期（含启动 / 预热 / 图捕获）；按 step 归一化用 bench 完成 token 数 / 并发 ≈ decode step 数（dinfer $D_STEPS，vLLM $V_STEPS）。vLLM 的 kernel 若为空，说明宿主 nsys 没注入到容器（见脚本头注释）。"
  echo
  python3 -I scripts/dev/nsys_stage_compare.py "$OUT/dinfer_kern.csv" "$OUT/vllm_kern.csv" --steps "$D_STEPS" "$V_STEPS"
} > "$OUT/stage-compare.md"
cat "$OUT/stage-compare.md"
