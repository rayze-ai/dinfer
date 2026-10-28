#!/usr/bin/env bash
# Step A（loop kaon3-decode-opt 第二轮）：5 轮 × c=1,2,4,8,16，dinfer 非 eager，参数对齐 vLLM :28201
# （max-model-len 16384 / max-num-batched-tokens 4096 / max-num-seqs 64）。每窗口：bench(dinfer) + bench(vLLM)
# + nvidia-smi 采样 + 解析 HOST_TIME 日志各段均值。结果：/w/dinfer/tmp_stepA3_RESULT.txt。
# 用法（容器内）：docker exec -d kaon3-full bash /w/stepA3.sh
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
TAG=${TAG:-stepA3}; PORT=${PORT:-8330}; ROUNDS=${ROUNDS:-5}; VPORT=28201
R=/w/dinfer/tmp_${TAG}_RESULT.txt; LOG=/w/dinfer/tmp_${TAG}.log; : > $R
strip() { sed "s/\x1b\[[0-9;]*m//g"; }
echo "bin=$(stat -c %y target/release/dinfer | cut -c1-16) probes=$(strings target/release/dinfer | grep -cE 'kaon3-host|begin-split|render-time')" >> $R
echo "vllm:$(curl -s -m5 http://127.0.0.1:$VPORT/v1/models | cut -c1-120)" >> $R
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=0 HOST_TIME=1 ${EXTRA_ENV:-}
T0=$(date +%s.%N)
./target/release/dinfer serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 >$LOG 2>&1 &
SP=$!
for i in $(seq 1 150); do strip < $LOG | grep -q "listening addr" && break; sleep 1; done
strip < $LOG | grep -q "listening addr" || { echo NOTUP >> $R; strip < $LOG | grep -iE "error|panic" | tail -3 >> $R; echo DONE >> $R; kill -9 $SP; exit 1; }
echo "startup: wall=$(awk -v a=$T0 -v b=$(date +%s.%N) "BEGIN{printf \"%.1f\", b-a}")s $(strip < $LOG | grep -oE 'model_load_ms=[0-9]+' | head -1) $(strip < $LOG | grep -oE 'num_blocks=[0-9]+|kv[_ a-z]*blocks[=: ]+[0-9]+' | head -1)" >> $R
# 预热：让每个 graph 桶都捕获过
taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:$PORT --concurrency 16 --num-requests 32 --prompt-len 400 --max-tokens 16 --prefix-repeat 0 --warmup 0 --seed 1 >/dev/null 2>&1
for r in $(seq 1 $ROUNDS); do
  for c in 1 2 4 8 16; do
    echo "=== round=$r concurrency=$c ===" >> $R
    LO=$(wc -l < $LOG)
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits -i 0 -lms 200 > /tmp/smi_$TAG.txt 2>/dev/null &
    SMI=$!
    echo -n "  [dinfer] " >> $R
    taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:$PORT --concurrency $c --num-requests $((c*8)) --prompt-len 400 --max-tokens 64 --prefix-repeat 0 --warmup 2 --seed 7 2>&1 | grep -E "throughput|TTFT |TPOT |client:" | tr "\n" " " | sed "s/  */ /g" | cut -c1-700 >> $R; echo >> $R
    kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
    HI=$(wc -l < $LOG)
    echo "  [gpu] $(awk -F, 'NR>3{s+=$1;m+=$2;n++} END{if(n) printf "sm=%.0f%% mem=%.0f%% samples=%d", s/n, m/n, n}' /tmp/smi_$TAG.txt)" >> $R
    echo "  [host] $(python3 -I /w/kaon3_hosttime_parse.py $LOG $LO $HI 2>&1 | cut -c1-900)" >> $R
    echo -n "  [vllm] " >> $R
    taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:$VPORT --concurrency $c --num-requests $((c*8)) --prompt-len 400 --max-tokens 64 --prefix-repeat 0 --warmup 2 --seed 7 2>&1 | grep -E "throughput|TTFT |TPOT " | tr "\n" " " | sed "s/  */ /g" | cut -c1-300 >> $R; echo >> $R
  done
done
kill -INT $SP 2>/dev/null; sleep 2; kill -KILL $SP 2>/dev/null
echo DONE >> $R
