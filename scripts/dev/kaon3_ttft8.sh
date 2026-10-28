#!/usr/bin/env bash
# D80 探针：c8 TTFT 545 ms 的量级问题。① max_tokens=1 纯 prefill c=1/2/4/8 标度；② c8×64 HOST_TIME 每步 mode/t/total 统计。
# 容器内：docker exec -d kaon3-full bash /w/ttft8.sh   → /w/dinfer/tmp_ttft8_RESULT.txt
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
R=/w/dinfer/tmp_ttft8_RESULT.txt; : > $R
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=0 FP8_GEMV=1 FP8_LT_TUNE=0 HOST_TIME=1
LOG=/w/dinfer/tmp_ttft8.log; PORT=8397
echo "bin=$(stat -c %y target/release/dinfer | cut -c1-16) $(date +%T)" >> $R
taskset -c 0-31 ./target/release/dinfer serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 >$LOG 2>&1 &
SP=$!
for i in $(seq 1 150); do sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -q "listening addr" && break; sleep 1; done
sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -q "listening addr" || { echo NOTUP >> $R; kill $SP; exit 1; }
B="taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:$PORT --prompt-len 400 --prefix-repeat 0 --seed 7"
for c in 1 2 4 8 16; do
  lo=$(wc -l < $LOG)
  echo "== pf c$c (max_tokens=1, n=$((c*4)))" >> $R
  $B --concurrency $c --num-requests $((c*4)) --max-tokens 1 --warmup 1 2>&1 | grep -E "req/s|TTFT" | cut -c1-160 >> $R
  sed -n "$((lo+1)),\$p" $LOG | grep -oE "EXIT mode=[A-Za-z]+ t=[0-9]+ total=[0-9]+" | awk '{split($2,m,"=");split($3,t,"=");split($4,u,"=");k=m[2];n[k]++;s[k]+=u[2];if(t[2]>mx[k])mx[k]=t[2]} END{for(k in n)printf "  %s steps=%d mean_total=%.0fus max_t=%d\n",k,n[k],s[k]/n[k],mx[k]}' >> $R
done
lo=$(wc -l < $LOG)
echo "== c8 x64 max_tokens=64 (bench 口径)" >> $R
$B --concurrency 8 --num-requests 64 --max-tokens 64 --warmup 2 2>&1 | grep -E "req/s|TTFT|TPOT" | cut -c1-160 >> $R
sed -n "$((lo+1)),\$p" $LOG | grep -oE "EXIT mode=[A-Za-z]+ t=[0-9]+ total=[0-9]+" | awk '{split($2,m,"=");split($3,t,"=");split($4,u,"=");k=m[2];n[k]++;s[k]+=u[2];if(t[2]>mx[k])mx[k]=t[2]} END{for(k in n)printf "  %s steps=%d mean_total=%.0fus max_t=%d\n",k,n[k],s[k]/n[k],mx[k]}' >> $R
echo "  -- Prefill/Paged 步按 t 分桶（t, 次数, 均值 us）" >> $R
sed -n "$((lo+1)),\$p" $LOG | grep -oE "EXIT mode=(Prefill|Paged) t=[0-9]+ total=[0-9]+" | awk '{split($2,m,"=");split($3,t,"=");split($4,u,"=");k=m[2]" t="t[2];n[k]++;s[k]+=u[2]} END{for(k in n)printf "  %s n=%d mean=%.0fus\n",k,n[k],s[k]/n[k]}' | sort -t= -k3 -n | head -40 >> $R
kill -INT $SP 2>/dev/null; sleep 2; kill -KILL $SP 2>/dev/null
echo DONE >> $R
