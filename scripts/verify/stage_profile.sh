#!/usr/bin/env bash
# Q3-d 逐段耗时对比：在 GPU 3 上分别起独立的 dinfer 与 vLLM 实例，nsys 抓同一 bench 负载，按阶段并排。
#   bash scripts/verify/stage_profile.sh [OUT_DIR]   # 默认 test/qwen3/$(date +%F)/stage
# 负载：bench --concurrency 8 --num-requests 24 --prompt-len 256 --max-tokens 64（decode 为主，含 8 次 prefill）。
# vLLM：宿主 nsys 启动 docker run（nsys 注入 CUDA 层到容器内进程需要 --pid=host 不行；用 `nsys profile` 包 docker run 的方式
# 在 CUDA 13 + nsys 2025 下可用；不可用时回退：在容器里 pip 装 nsys 不现实 → 用 torch.profiler 导出 kernel 表（见 fallback）。
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

echo "== vLLM under nsys (GPU $GPU, :8034)"
docker rm -f vllm-prof >/dev/null 2>&1 || true
nsys profile -o "$OUT/vllm" --force-overwrite true --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --duration=240 \
  docker run --rm --name vllm-prof --runtime nvidia --gpus "\"device=$GPU\"" --ipc=host -p 8034:8000 \
    -v $R/models:/models:ro -v $R/vllm/hf-cache:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
    docker.m.daocloud.io/vllm/vllm-openai:v0.31.0 --model /models/qwen3-0.6b --served-model-name qwen3-0.6b \
    --max-model-len 8192 --gpu-memory-utilization 0.9 --max-num-seqs 64 --enable-prefix-caching --host 0.0.0.0 --port 8000 \
  > "$OUT/vllm_serve.log" 2>&1 &
for i in $(seq 1 200); do sleep 1; curl -s -m 2 http://127.0.0.1:8034/v1/models >/dev/null 2>&1 && break; done
$BENCH --url http://127.0.0.1:8034 "${LOAD[@]}" --label vllm --out "$OUT/vllm_bench.json" | tee "$OUT/vllm_bench.log" | grep -E 'throughput|TTFT|TPOT'
docker stop -t 5 vllm-prof >/dev/null 2>&1 || true
for i in $(seq 1 90); do sleep 1; pgrep -f "nsys profile.*$OUT/vllm" >/dev/null || break; done
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
