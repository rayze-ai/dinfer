#!/usr/bin/env bash
# 通用同二进制 ABAB（stepA4 口径，c ∈ CS，两轮交错，每窗口换 seed，固定 prompt 输出 + HOST_TIME 混批步统计）：
#   TAG=<名> AENV="K=V" BENV="K=V" [AARGS="--x 1" BARGS="--y 2"] [CS="8 16 32 64"] bash /w/envab_c.sh
# 结果 /w/dinfer/tmp_<TAG>_RESULT.txt，结束标记 <TAG>_DONE（大写）。容器内：docker exec -d kaon3-full bash -c '... > /w/dinfer/tmp_<TAG>_run.log 2>&1'
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib CUDA_VISIBLE_DEVICES=0
cd /w/dinfer
TAG=${TAG:-envab}; PORT=${PORT:-8345}; CS=${CS:-"8 16 32 64"}; DONE_MARK="$(echo $TAG | tr a-z A-Z)_DONE"
R=/w/dinfer/tmp_${TAG}_RESULT.txt; : > $R
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
echo "serve_taskset=0-31 bench_taskset=60-91 bin=$(stat -c %y target/release/dinfer | cut -c1-16) md5=$(md5sum target/release/dinfer | cut -c1-12) A=[$AENV $AARGS] B=[$BENV $BARGS] $(date +%T)" >> $R
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 HOST_TIME=1
strip() { sed "s/\x1b\[[0-9;]*m//g"; }
run_var() {  # $1=round $2=tag $3=env $4=args
  local r=$1 tag=$2 L=/w/dinfer/tmp_${TAG}_r${1}_$2.log
  env $3 taskset -c 0-31 ./target/release/dinfer serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 $4 >$L 2>&1 &
  local sp=$!
  for i in $(seq 1 240); do strip < $L | grep -q "listening addr" && break; sleep 1; done
  strip < $L | grep -q "listening addr" || { echo "r$r $tag NOTUP" >> $R; strip < $L | grep -iE "error|panic" | tail -2 | cut -c1-200 >> $R; kill -KILL $sp 2>/dev/null; return 1; }
  echo "r$r $tag startup: $(strip < $L | grep -oE 'model_load_ms=[0-9]+' | head -1)" >> $R
  taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:$PORT --concurrency 16 --num-requests 32 --prompt-len 400 --max-tokens 16 --prefix-repeat 0 --warmup 0 --seed 1 >/dev/null 2>&1
  local vi=0; [ $tag = B ] && vi=1
  for c in $CS; do
    local seed=$((13000 + 100 * r + c + vi * 13)) LO=$(wc -l < $L)
    echo -n "r$r $tag c$c: " >> $R
    taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:$PORT --concurrency $c --num-requests $((c*8)) --prompt-len 400 --max-tokens 64 --prefix-repeat 0 --warmup 2 --seed $seed 2>&1 | grep -E "throughput|TTFT |TPOT " | tr '\n' ' ' | sed -E 's/ throughput: [0-9.]+ req.s +([0-9.]+) output tok.s +[0-9.]+ total tok.s +TTFT +mean +[0-9.]+ +p50 +([0-9.]+) +p90 +[0-9.]+ +p99 +([0-9.]+)[^T]*TPOT +mean +[0-9.]+ +p50 +([0-9.]+) +p90 +[0-9.]+ +p99 +([0-9.]+).*/tps=\1 ttft_p50=\2 ttft_p99=\3 tpot_p50=\4 tpot_p99=\5/' | cut -c1-200 >> $R
    local HI=$(wc -l < $L)
    echo " || $(python3 -I /w/kaon3_hosttime_parse.py $L $LO $HI 2>&1 | sed -E 's/render=[^|]*\|\| //; s/ publish=[0-9]+ total=[0-9]+ \| upd=[^|]*\| htod=[^|]*//g' | cut -c1-200)" >> $R
  done
  echo -n "r$r $tag out: " >> $R
  curl -s -m30 http://127.0.0.1:$PORT/v1/completions -H "content-type: application/json" -d '{"model":"kaon3","prompt":"The capital of France is","max_tokens":24,"temperature":0}' | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'][:100].replace(chr(10),' '))" >> $R 2>&1
  echo -n "r$r $tag out2: " >> $R
  curl -s -m60 http://127.0.0.1:$PORT/v1/chat/completions -H "content-type: application/json" -d '{"model":"kaon3","messages":[{"role":"user","content":"用三句话介绍一下长城的历史。"}],"max_tokens":60,"temperature":0}' | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'][:100].replace(chr(10),' '))" >> $R 2>&1
  kill -INT $sp 2>/dev/null; sleep 2; kill -KILL $sp 2>/dev/null; sleep 3
}
for r in 1 2; do run_var $r A "$AENV" "$AARGS"; run_var $r B "$BENV" "$BARGS"; done
echo $DONE_MARK >> $R
