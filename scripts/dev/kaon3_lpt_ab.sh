#!/usr/bin/env bash
# D203 Step B(b)：调度旋钮 A/B —— `--long-prefill-token-threshold` 0（A，现状）vs 512（B）vs 1024（C），同二进制，
# stepA4 口径 c ∈ {8,16,32,64}，每变体 2 轮交错（A B C A B C），每窗口换 seed；每变体抓固定 prompt 输出做一致性。
# 结果 tmp_lpt_RESULT.txt，结束标记 LPT_DONE。容器内：docker exec -d kaon3-full bash -c 'bash /w/lpt.sh > /w/dinfer/tmp_lpt_run.log 2>&1'
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
TAG=${TAG:-lpt}; PORT=${PORT:-8342}; CS=${CS:-"8 16 32 64"}; VARS=${VARS:-"0 512 1024"}
R=/w/dinfer/tmp_${TAG}_RESULT.txt; : > $R
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
echo "serve_taskset=0-31 bench_taskset=60-91 bin=$(stat -c %y target/release/dinfer | cut -c1-16) md5=$(md5sum target/release/dinfer | cut -c1-12) vars=[$VARS] $(date +%T)" >> $R
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=0 HOST_TIME=1
strip() { sed "s/\x1b\[[0-9;]*m//g"; }
run_var() {  # $1=round $2=threshold
  local r=$1 thr=$2 L=/w/dinfer/tmp_${TAG}_r${1}_t${2}.log
  taskset -c 0-31 ./target/release/dinfer serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 --long-prefill-token-threshold $thr >$L 2>&1 &
  local sp=$!
  for i in $(seq 1 180); do strip < $L | grep -q "listening addr" && break; sleep 1; done
  strip < $L | grep -q "listening addr" || { echo "r$r t$thr NOTUP" >> $R; strip < $L | grep -iE "error|panic" | tail -2 | cut -c1-200 >> $R; kill -KILL $sp 2>/dev/null; return 1; }
  taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:$PORT --concurrency 16 --num-requests 32 --prompt-len 400 --max-tokens 16 --prefix-repeat 0 --warmup 0 --seed 1 >/dev/null 2>&1
  for c in $CS; do
    local seed=$((7000 + 100 * r + c + thr / 7)) LO=$(wc -l < $L)
    echo -n "r$r t$thr c$c: " >> $R
    taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:$PORT --concurrency $c --num-requests $((c*8)) --prompt-len 400 --max-tokens 64 --prefix-repeat 0 --warmup 2 --seed $seed 2>&1 | grep -E "throughput|TTFT |TPOT " | tr '\n' ' ' | sed -E 's/ p90[^T]*TPOT/ TPOT/; s/ p90.*//' | cut -c1-200 >> $R
    local HI=$(wc -l < $L)
    echo " || $(python3 -I /w/kaon3_hosttime_parse.py $L $LO $HI 2>&1 | sed -E 's/render=[^|]*\|\| //; s/ publish=[0-9]+ total=[0-9]+ \| upd=[^|]*\| htod=[^|]*//g' | cut -c1-260)" >> $R
  done
  echo -n "r$r t$thr out: " >> $R
  curl -s -m30 http://127.0.0.1:$PORT/v1/completions -H "content-type: application/json" -d '{"model":"kaon3","prompt":"The capital of France is","max_tokens":24,"temperature":0}' | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'][:120].replace(chr(10),' '))" >> $R 2>&1
  kill -INT $sp 2>/dev/null; sleep 2; kill -KILL $sp 2>/dev/null; sleep 3
}
for r in 1 2; do for thr in $VARS; do run_var $r $thr; done; done
echo LPT_DONE >> $R
