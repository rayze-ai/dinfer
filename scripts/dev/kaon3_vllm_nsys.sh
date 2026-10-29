#!/usr/bin/env bash
# Step A（loop kaon3-prefill-gemm）：抓 vLLM 同口径 c16 的 kernel 表。kaon3-full 容器里有 vllm 0.26.1rc1 + nsys；GPU3（基线 :28201 在 GPU2 不动）。
# 启动参数照基线：--max-model-len 16384 --max-num-batched-tokens 4096 --block-size 32 --gpu-memory-utilization 0.85 --chat-template …
# 结果 tmp_vnsys_RESULT.txt，标记 VNSYS_DONE。容器内：docker exec -d kaon3-full bash -c 'bash /w/vllm_nsys.sh > /w/dinfer/tmp_vnsys_run.log 2>&1'
R=/w/dinfer/tmp_vnsys_RESULT.txt; : > $R; date +%T >> $R
NSYS=$(ls /opt/nvidia/nsight-systems/2026*/target-linux-x64/nsys 2>/dev/null | head -1)
[ -z "$NSYS" ] && { echo "no nsys" >> $R; echo VNSYS_DONE >> $R; exit 1; }
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
PORT=28301; L=/w/dinfer/tmp_vnsys_serve.log
export CUDA_VISIBLE_DEVICES=3 VLLM_LOGGING_LEVEL=INFO
rm -f /w/dinfer/tmp_vnsys.nsys-rep /w/dinfer/tmp_vnsys.sqlite
taskset -c 32-59 "$NSYS" profile -o /w/dinfer/tmp_vnsys --force-overwrite true -t cuda --cuda-graph-trace=node --cuda-flush-interval=1000 --sample none --cpuctxsw none \
  vllm serve "$M" --port $PORT --served-model-name kaon-vllm --max-model-len 16384 --max-num-batched-tokens 4096 --block-size 32 --gpu-memory-utilization 0.85 --chat-template "$M/chat_template.jinja" > $L 2>&1 &
NP=$!
for i in $(seq 1 900); do curl -s -m2 http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q kaon-vllm && break; sleep 2; done
curl -s -m2 http://127.0.0.1:$PORT/v1/models | grep -q kaon-vllm || { echo "vllm NOTUP" >> $R; tail -5 $L | cut -c1-200 >> $R; kill -INT $NP; echo VNSYS_DONE >> $R; exit 1; }
echo "vllm up $(date +%T)" >> $R
cd /w/dinfer
taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:$PORT --concurrency 16 --num-requests 32 --prompt-len 400 --max-tokens 16 --prefix-repeat 0 --warmup 0 --seed 1 >/dev/null 2>&1
taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:$PORT --concurrency 16 --num-requests 128 --prompt-len 400 --max-tokens 64 --prefix-repeat 0 --warmup 2 --seed 4016 2>&1 | grep -E "throughput|TTFT |TPOT " | tr '\n' ' ' | cut -c1-300 >> $R; echo >> $R
# 停 vLLM：找 EngineCore / API server 的真 pid（容器里 vllm 子进程多），INT 给 nsys 启动的根进程树
VP=$(for d in /proc/[0-9]*; do tr "\0" " " < $d/cmdline 2>/dev/null | grep -qE "vllm serve .*--port $PORT" && basename $d; done | head -1)
echo "vllm_pid=$VP nsys_pid=$NP" >> $R
kill -INT $VP 2>/dev/null; for i in $(seq 1 120); do kill -0 $NP 2>/dev/null || break; sleep 2; done; kill -0 $NP 2>/dev/null && { kill -TERM $VP 2>/dev/null; sleep 20; }
ls -la /w/dinfer/tmp_vnsys.nsys-rep | cut -c1-100 >> $R 2>&1
"$NSYS" export --type sqlite --force-overwrite true -o /w/dinfer/tmp_vnsys.sqlite /w/dinfer/tmp_vnsys.nsys-rep >/dev/null 2>&1
echo "== decode graph kernels (nsys_kern --all)" >> $R
taskset -c 100-127 python3 /w/nsys_kern.py /w/dinfer/tmp_vnsys.sqlite --all 2>&1 | head -60 | cut -c1-200 >> $R
echo "== eager (prefill) steps" >> $R
taskset -c 100-127 python3 -I /w/nsys_prefill_steps.py /w/dinfer/tmp_vnsys.sqlite 100 30 2 2>&1 | head -60 | cut -c1-170 >> $R
echo VNSYS_DONE >> $R
