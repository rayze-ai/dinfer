#!/usr/bin/env bash
# h1（loop kaon3-pd-balance）：开环泊松 A/B。A = E6 静态（控制器关）  B = PD_BALANCE_SLO_TTFT_MS=200 PD_BALANCE_SLO_TPOT_MS=12（控制器开，Q1 配额动态）。
# rate ∈ {12,16,20} req/s，每档 160 请求，prompt 混合 128:0.5,1024:0.35,2048:0.15，out 64；同 seed；bench 报 goodput（同 SLO）。
# 两轮交错 A B A B。结果 tmp_h1_RESULT.txt，标记 H1_DONE。前置：编 HEAD。
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LIBRARY_PATH=/tmp/stub LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=120 CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0
cd /w/dinfer
R=/w/dinfer/tmp_h1_RESULT.txt; : > $R; date +%T >> $R
cargo build --release -p dinfer --features kaon3 > /w/dinfer/tmp_h1_build.log 2>&1; brc=$?
echo "BUILD EXIT=$brc md5=$(md5sum target/release/dinfer | cut -c1-12) marker=$(strings target/release/dinfer | grep -c PD_BALANCE_SLO) $(date +%T)" >> $R
[ $brc -ne 0 ] && { grep -E "^error" -A8 /w/dinfer/tmp_h1_build.log | head -30 | cut -c1-200 >> $R; echo H1_DONE >> $R; exit 1; }
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 HOST_TIME=1
SLO_T=${SLO_T:-200}; SLO_P=${SLO_P:-12}; RATES=${RATES:-"12 16 20"}; NREQ=${NREQ:-160}; MIX=${MIX:-"128:0.5,1024:0.35,2048:0.15"}
strip() { sed "s/\x1b\[[0-9;]*m//g"; }
run_var() {  # $1=round $2=tag $3=env
  local r=$1 tag=$2 L=/w/dinfer/tmp_h1_r${1}_$2.log PORT=8371
  env $3 taskset -c 0-31 ./target/release/dinfer serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 >$L 2>&1 &
  local sp=$!
  for i in $(seq 1 240); do strip < $L | grep -q "listening addr" && break; sleep 1; done
  strip < $L | grep -q "listening addr" || { echo "r$r $tag NOTUP" >> $R; strip < $L | grep -iE "error|panic" | tail -2 | cut -c1-200 >> $R; kill -KILL $sp 2>/dev/null; return 1; }
  taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:$PORT --concurrency 16 --num-requests 32 --prompt-len 400 --max-tokens 16 --prefix-repeat 0 --warmup 0 --seed 1 >/dev/null 2>&1
  for rate in $RATES; do
    local seed=$((21000 + 100 * r + rate))
    echo -n "r$r $tag rate$rate: " >> $R
    taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:$PORT --request-rate $rate --num-requests $NREQ --prompt-len-mix "$MIX" --max-tokens 64 --slo-ttft-ms $SLO_T --slo-tpot-ms $SLO_P --seed $seed --warmup 4 2>&1 | grep -E "throughput|TTFT |TPOT |goodput" | tr '\n' ' ' | sed -E 's/ throughput: [0-9.]+ req.s +([0-9.]+) output tok.s +[0-9.]+ total tok.s +TTFT +mean +[0-9.]+ +p50 +([0-9.]+) +p90 +[0-9.]+ +p99 +([0-9.]+)[^T]*TPOT +mean +[0-9.]+ +p50 +([0-9.]+) +p90 +[0-9.]+ +p99 +([0-9.]+).*goodput[^0-9]*([0-9.]+).*/tps=\1 ttft_p50=\2 ttft_p99=\3 tpot_p50=\4 tpot_p99=\5 goodput=\6/' | cut -c1-220 >> $R
  done
  echo "r$r $tag pd-balance lines: $(strip < $L | grep -c 'pd-balance')  last: $(strip < $L | grep 'pd-balance' | tail -1 | cut -c1-120)" >> $R
  kill -INT $sp 2>/dev/null; sleep 2; kill -KILL $sp 2>/dev/null; sleep 3
}
for r in 1 2; do run_var $r A "PD_BALANCE_SLO_TTFT_MS=0"; run_var $r B "PD_BALANCE_SLO_TTFT_MS=$SLO_T PD_BALANCE_SLO_TPOT_MS=$SLO_P"; done
echo H1_DONE >> $R
