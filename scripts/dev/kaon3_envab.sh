# 通用同二进制 env 开关 ABAB：TAG=<名> AENV="K=V" BENV="K=V" bash /w/envab.sh   （c1+c8 + 两条固定 prompt 输出）
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
TAG=${TAG:-envab}; PORT=${PORT:-8395}
R=/w/dinfer/tmp_${TAG}_RESULT.txt; : > $R
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
echo "bin=$(stat -c %y target/release/dinfer | cut -c1-16) A=[$AENV] B=[$BENV]" >> $R
run_one() {
  local tag=$1; shift
  export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=0
  setsid env $* ./target/release/dinfer serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 >/w/dinfer/tmp_${TAG}_$tag.log 2>&1 &
  local sp=$!
  for i in $(seq 1 150); do sed "s/\x1b\[[0-9;]*m//g" /w/dinfer/tmp_${TAG}_$tag.log | grep -q "listening addr" && break; sleep 1; done
  sed "s/\x1b\[[0-9;]*m//g" /w/dinfer/tmp_${TAG}_$tag.log | grep -q "listening addr" || { echo "$tag NOTUP" >> $R; sed "s/\x1b\[[0-9;]*m//g" /w/dinfer/tmp_${TAG}_$tag.log | grep -iE "error|panic" | tail -2 | cut -c1-200 >> $R; kill -9 $sp; return; }
  for c in 1 8; do
    echo -n "$tag c$c: " >> $R
    taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:$PORT --concurrency $c --num-requests $((c*8)) --prompt-len 400 --max-tokens 64 --prefix-repeat 0 --warmup 2 --seed 7 2>&1 | grep -E "throughput|TTFT |TPOT " | tr "\n" " " | sed "s/  */ /g; s/throughput: \([0-9.]* req\/s [0-9.]* output\) tok\/s [0-9.]* total tok\/s/\1/; s/ p90 [0-9.]* p99 [0-9.]* max [0-9.]* (ms)//g" | cut -c1-160 >> $R; echo >> $R
  done
  echo -n "$tag out: " >> $R
  curl -s -m30 http://127.0.0.1:$PORT/v1/completions -H "content-type: application/json" -d '{"model":"kaon3","prompt":"The capital of France is","max_tokens":24,"temperature":0}' | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'].replace(chr(10),' ')[:120])" >> $R 2>&1
  echo -n "$tag out2: " >> $R
  curl -s -m60 http://127.0.0.1:$PORT/v1/chat/completions -H "content-type: application/json" -d '{"model":"kaon3","messages":[{"role":"user","content":"用三句话介绍一下长城的历史。"}],"max_tokens":80,"temperature":0}' | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'].replace(chr(10),' ')[:160])" >> $R 2>&1
  kill -INT $sp 2>/dev/null; sleep 2; kill -KILL $sp 2>/dev/null; sleep 3
}
run_one A $AENV
run_one B $BENV
run_one A2 $AENV
run_one B2 $BENV
echo DONE >> $R
