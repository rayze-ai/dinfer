#!/usr/bin/env bash
# K3 里程碑 C-6：全 93 层 tp8 真权重 e2e（B300-50，需 8 卡独占）。在 B300 上跑：
#   bash scripts/dev/k3_c6_tp8.sh prep     # 停 GPU 0/1 的 Qwen3.5 常驻（dinfer :8012 + vLLM 容器）；确认 8 卡空
#   [K3_BUDGET=2048] bash scripts/dev/k3_c6_tp8.sh serve    # 起 dinfer --features kimik3 --tp 8 全层 :8014（tmux k3），等 health
#   bash scripts/dev/k3_c6_tp8.sh smoke    # 一条流式 chat + 3 条 greedy 文本抽查（与 vLLM 基线文本对照需另起 vLLM，放 C-6b）
#   bash scripts/dev/k3_c6_tp8.sh bench <OUT>   # 四组闭环 bench（只 dinfer 端；vLLM 基线取 test/kimik3/2026-10-19/bench_b300_vllm1）
#   bash scripts/dev/k3_c6_tp8.sh stop     # 停 k3 serve
# 口径：vLLM tp8 基线 TPOT c1 8.70 / c8 13.09 / c32 19.15 / c8-long 13.11 ms（design-kimik3 §0.1）。
# 已知：预算默认 2048——每层各持按 max_tokens 的激活缓冲，16384 时 ~13 GB/层/卡必 OOM（k3-serve 3b；ActScratch 共享后再回 16384）。graph 默认关（K3_GRAPH=1 强开，LatentMoe host tile 计划在 capture 内读 0）；AttnRes host softmax；MLA 吸收每 head 小 GEMM。
set -uo pipefail
source /data/yaochen/infer/env.sh
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
cd /data/yaochen/infer/dinfer
R=/data/yaochen/infer; PORT=8014; M=/data/models/moonshotai/Kimi-K3
case "${1:-}" in
  prep)
    tmux kill-session -t q35 2>/dev/null; pkill -INT -f "dinfer serve .*--port 8012"; sleep 3
    sudo -n nerdctl -n k8s.io rm -f vllm-qwen35-b300 >/dev/null 2>&1; sleep 5
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " "; echo ;;
  serve)
    rm -f target/.rustc_info.json
    cargo build --release --features kimik3 -p dinfer -p dinfer-bench 2>&1 | grep -E "^error|Finished" | head -n 3
    tmux kill-session -t k3 2>/dev/null
    tmux new-session -d -s k3 -c "$PWD" "source $R/env.sh; unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RUST_LOG=info taskset -c 0-63 ./target/release/dinfer serve --model $M --served-model-name kimi-k3 --tp 8 --host 127.0.0.1 --port $PORT --max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens ${K3_BUDGET:-2048} 2>&1 | tee $R/bootstrap/k3_c6_serve.log"
    for _ in $(seq 1 1800); do sleep 1; curl -s -m 2 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break; pgrep -f "dinfer serve .*--port $PORT" >/dev/null || break; done
    sed "s/\x1b\[[0-9;]*m//g" $R/bootstrap/k3_c6_serve.log | grep -E "load|layers|startup|ERROR|Error|panic|GB|GiB" | tail -n 6 | cut -c1-160
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " "; echo ;;
  smoke)
    curl -s -N -m 120 http://127.0.0.1:$PORT/v1/chat/completions -H "content-type: application/json" -d '{"model":"kimi-k3","messages":[{"role":"user","content":"用一句话解释什么是傅里叶变换。"}],"max_tokens":64,"stream":true}' | grep -c "^data:"
    for p in "用一句话解释什么是傅里叶变换。" "Write a Python function that returns the n-th Fibonacci number, iteratively." "List three capitals in Europe and their countries."; do
      curl -s -m 180 http://127.0.0.1:$PORT/v1/chat/completions -H "content-type: application/json" -d "{\"model\":\"kimi-k3\",\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"max_tokens\":48,\"temperature\":0}" | python3 -c "import sys,json;d=json.load(sys.stdin);m=d['choices'][0]['message'];print(repr((m.get('reasoning_content') or '')[:80]), '|', repr((m.get('content') or '')[:80]))" 2>&1 | cut -c1-200
    done ;;
  bench)
    OUT=${2:?OUT}; mkdir -p "$OUT"; echo B300 > "$OUT/GPU"; echo "C-6 全 93 层 tp8 首轮（graph 关，预算 ${K3_BUDGET:-2048}）" > "$OUT/CHANGE"
    for g in "1 128 128" "8 128 128" "32 128 128" "8 1024 128"; do set -- $g; c=$1; p=$2; o=$3; n=$(( c*4 > 16 ? c*4 : 16 ))
      taskset -c 96-127 ./target/release/bench --url http://127.0.0.1:$PORT --model kimi-k3 --label dinfer --concurrency $c --num-requests $n --prompt-len $p --max-tokens $o --seed $(date +%s) --out "$OUT/dinfer_c${c}_p${p}_o${o}.json" 2>&1 | grep -E "throughput|TTFT |TPOT" | head -n 3 | sed "s/^/[c$c p$p] /" | cut -c1-110
    done
    cp test/kimik3/2026-10-19/bench_b300_vllm1/vllm_*.json "$OUT/" 2>/dev/null; echo "vllm baseline copied from bench_b300_vllm1" ;;
  stop) tmux kill-session -t k3 2>/dev/null; pkill -INT -f "dinfer serve .*--port $PORT"; sleep 3; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " "; echo ;;
  *) echo "usage: $0 prep|serve|smoke|bench <OUT>|stop"; exit 2 ;;
esac
