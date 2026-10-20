#!/usr/bin/env bash
# 抓 qwen35 单发长 prompt 的 prefill kernel 分解（docs/loops/engine-gap.md 8z / r11 第一步）。
#
# 之前两次没落 .nsys-rep（engine-gap 8b）：`--kill=sigint` 是非法选项；`pkill -INT <serve>` 只杀了 nsys
# spawn 的子进程，nsys 自己没收到信号就写不出报告。本脚本改为**把 SIGINT 发给 nsys 进程本身**。
#
# 用法（B300-50，远端直接跑）：
#   bash scripts/dev/q35_nsys_prefill.sh <tag> [gpu=2] [port=8021] [lengths="146 430 860"]
# 产物：/tmp/q35_<tag>.nsys-rep + /tmp/q35_<tag>_serve.log + 每长度一条 TTFT
# 之后：python3 -I scripts/dev/nsys_steps_sqlite.py /tmp/q35_<tag>.sqlite 30
set -uo pipefail
tag=${1:?tag}
gpu=${2:-2}
port=${3:-8021}
lengths=${4:-"146 430 860"}     # 词数：146≈1K token、430≈3K、860≈6K
M=${MODEL_DIR:-/data/models/Qwen/Qwen3.8-27B}
cd "$(dirname "$0")/../.." && source /data/yaochen/infer/env.sh 2>/dev/null
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# 只杀本端口的 serve：全局 `pkill -x dinfer` 会连 GPU 0 的常驻 :8012 一起杀（2026-10-20 误杀过）
pkill -f "dinfer serve .*--port $port\b" 2>/dev/null; sleep 2
rm -f /tmp/q35_$tag.nsys-rep /tmp/q35_$tag.sqlite

tmux kill-session -t nsprof 2>/dev/null
tmux new-session -d -s nsprof -c "$PWD" "source /data/yaochen/infer/env.sh; CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu \
  nsys profile -t cuda --cuda-flush-interval=1000 -o /tmp/q35_$tag --force-overwrite true \
  ./target/release/dinfer serve --model $M --served-model-name Qwen3.8-27B --host 127.0.0.1 --port $port \
  --max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens 16384 > /tmp/q35_${tag}_serve.log 2>&1"

for i in $(seq 1 240); do sleep 1; curl -s -m 2 127.0.0.1:$port/v1/models >/dev/null 2>&1 && break; done
curl -s -m 3 127.0.0.1:$port/v1/models | grep -oE '"id":"[^"]+"' | head -1 || { echo "serve 未起"; tail -5 /tmp/q35_${tag}_serve.log; exit 1; }

# 每个长度发 5 次（max_tokens=1 → 每请求恰好一个 prefill 步），先发一发预热（不计）。
# 只发 3 条时 CUPTI 活动缓冲填不满、进程被信号杀前不 flush → 报告 0 kernel（2026-10-20 四次空报告的根因候选）；
# 配合 --cuda-flush-interval=1000 周期 flush。
taskset -c 100-103 python3 -I scripts/dev/ttft_probe.py $port 1 146 >/dev/null 2>&1
for w in $lengths; do
  taskset -c 100-103 python3 -I scripts/dev/ttft_probe.py $port 10 $w 2>&1 | tail -1
done
sleep 20  # CUPTI 活动缓冲要填满或超时才 flush：3 条请求 0 kernel、5 条 6 K 只录 0.68 s（9b）；三长度各 10 次 + 20 s 稳妥

# SIGINT 发给 nsys 本身（不是 serve），它才会 flush 报告
np=$(pgrep -x nsys | head -1)
[ -z "$np" ] && np=$(pgrep -f '^nsys profile' | head -1)
echo "nsys pid=$np → SIGINT"
[ -n "$np" ] && kill -INT "$np"
# 等 nsys 进程**退出**（它先建 0 字节 .nsys-rep 再花几十秒写入；只等文件出现会拿到空文件 → 9d）
for i in $(seq 1 300); do sleep 1; pgrep -x nsys >/dev/null || break; done
sleep 2
ls -la /tmp/q35_$tag.nsys-rep 2>/dev/null | cut -c25- || { echo "报告未生成"; tmux capture-pane -pt nsprof 2>/dev/null | tail -8; }
pkill -f "dinfer serve .*--port $port\b" 2>/dev/null; tmux kill-session -t nsprof 2>/dev/null
