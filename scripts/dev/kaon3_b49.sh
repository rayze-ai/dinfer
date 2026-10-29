#!/usr/bin/env bash
# b49（D207）：B46 lm_head 只算采样行 + B47 整步 prefill 配额。编 HEAD → strings 验 marker → 三变体交错两轮：
#   A = LM_HEAD_ROWS_ONLY=0（= b48 行为）  B = 默认（B46 开）  C = B46 开 + --prefill-tokens-per-step 1024
# c ∈ {8,16,32,64}，stepA4 口径，每窗口换 seed，每变体固定 prompt 输出 + HOST_TIME 混批步统计。结果 tmp_b49_RESULT.txt，标记 B49_DONE。
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LIBRARY_PATH=/tmp/stub LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=120 CUDA_HOME=/usr/local/cuda
cd /w/dinfer
TAG=${TAG:-b49}; PORT=${PORT:-8343}; CS=${CS:-"8 16 32 64"}; PTS=${PTS:-1024}
R=/w/dinfer/tmp_${TAG}_RESULT.txt; : > $R; date +%T >> $R
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
if [ "${SKIP_BUILD:-0}" != "1" ]; then
  cp -f target/release/dinfer /w/dinfer_b48_head_bak 2>/dev/null
  cargo build --release -p dinfer --features kaon3 > /w/dinfer/tmp_${TAG}_build.log 2>&1; brc=$?
  echo "BUILD EXIT=$brc md5=$(md5sum target/release/dinfer | cut -c1-12) $(date +%T)" >> $R
  [ $brc -ne 0 ] && { grep -E "^error" -A8 /w/dinfer/tmp_${TAG}_build.log | head -30 | cut -c1-200 >> $R; echo B49_DONE >> $R; exit 1; }
fi
echo "marker LM_HEAD_ROWS_ONLY=$(strings target/release/dinfer | grep -c LM_HEAD_ROWS_ONLY) prefill-tokens-per-step=$(strings target/release/dinfer | grep -c prefill-tokens-per-step)" >> $R
echo "serve_taskset=0-31 bench_taskset=60-91 vars=A(LM_HEAD_ROWS_ONLY=0) B(default) C(--prefill-tokens-per-step $PTS)" >> $R
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=0 HOST_TIME=1
strip() { sed "s/\x1b\[[0-9;]*m//g"; }
run_var() {  # $1=round $2=tag $3=env "K=V" $4=extra args
  local r=$1 tag=$2 L=/w/dinfer/tmp_${TAG}_r${1}_$2.log
  env $3 taskset -c 0-31 ./target/release/dinfer serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 $4 >$L 2>&1 &
  local sp=$!
  for i in $(seq 1 180); do strip < $L | grep -q "listening addr" && break; sleep 1; done
  strip < $L | grep -q "listening addr" || { echo "r$r $tag NOTUP" >> $R; strip < $L | grep -iE "error|panic" | tail -2 | cut -c1-200 >> $R; kill -KILL $sp 2>/dev/null; return 1; }
  taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:$PORT --concurrency 16 --num-requests 32 --prompt-len 400 --max-tokens 16 --prefix-repeat 0 --warmup 0 --seed 1 >/dev/null 2>&1
  local vi=0; case $tag in A) vi=0;; B) vi=1;; C) vi=2;; esac
  for c in $CS; do
    local seed=$((9000 + 100 * r + c + vi * 13)) LO=$(wc -l < $L)
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
for r in 1 2; do
  run_var $r A "LM_HEAD_ROWS_ONLY=0" ""
  run_var $r B "LM_HEAD_ROWS_ONLY=1" ""
  run_var $r C "LM_HEAD_ROWS_ONLY=1" "--prefill-tokens-per-step $PTS"
done
echo B49_DONE >> $R
