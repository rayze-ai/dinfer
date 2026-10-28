#!/usr/bin/env bash
# D81：ttft8（HOST_TIME=1）c8 TTFT 75 ms vs bN 545 ms ——隔离 HOST_TIME。同一服务，c8×64 三跑：HOST_TIME 由启动决定 → 两个服务。
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
R=/w/dinfer/tmp_ttft8b_RESULT.txt; : > $R
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=0 FP8_GEMV=1 FP8_LT_TUNE=0
PORT=8396
one() { # $1 tag, $2.. env
  local tag=$1; shift
  LOG=/w/dinfer/tmp_ttft8b_$tag.log
  env "$@" taskset -c 0-31 ./target/release/dinfer serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 >$LOG 2>&1 &
  SP=$!
  for i in $(seq 1 150); do sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -q "listening addr" && break; sleep 1; done
  B="taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:$PORT --prompt-len 400 --prefix-repeat 0 --seed 7 --max-tokens 64 --warmup 2"
  echo "== $tag [$*] c1 x8" >> $R; $B --concurrency 1 --num-requests 8 2>&1 | grep -E "TTFT |TPOT" | cut -c1-120 >> $R
  for k in 1 2 3; do echo "== $tag [$*] c8 x64 #$k" >> $R; $B --concurrency 8 --num-requests 64 2>&1 | grep -E "req/s|TTFT |TPOT" | cut -c1-120 >> $R; done
  curl -s http://127.0.0.1:$PORT/metrics | grep -E "step_execute_(prefill|decode|mixed)_count|_count " | head -6 | cut -c1-120 >> $R
  kill -INT $SP 2>/dev/null; sleep 2; kill -KILL $SP 2>/dev/null; sleep 3
}
one plain
one hosttime HOST_TIME=1
one plain2
echo DONE >> $R
