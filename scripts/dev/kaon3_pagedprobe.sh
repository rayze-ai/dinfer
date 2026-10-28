#!/usr/bin/env bash
# D85：B26 后 Paged 步耗时表。等 b26 DONE → 新服务（HOST_TIME=1）→ c1×8 预热 idx0-7 → c8×64 新 prompt（seed 11，避免 cache）→ 步序 + 按 t 分桶均值。
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
for i in $(seq 1 240); do grep -q "^DONE" /w/dinfer/tmp_b26_RESULT.txt 2>/dev/null && break; sleep 20; done
sleep 5
R=/w/dinfer/tmp_pagedprobe_RESULT.txt; : > $R
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
export MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU=4 MEGA_SPDN=4 ROUTER_GEMM_BI=0 CUDA_VISIBLE_DEVICES=0 FP8_GEMV=1 HOST_TIME=1
LOG=/w/dinfer/tmp_pagedprobe.log; PORT=8394
echo "bin=$(stat -c %y target/release/dinfer | cut -c1-16) $(date +%T)" >> $R
taskset -c 0-31 ./target/release/dinfer serve --model "$M" --tokenizer "$M" --port $PORT --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 4096 >$LOG 2>&1 &
SP=$!
for i in $(seq 1 150); do sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -q "listening addr" && break; sleep 1; done
B="taskset -c 60-91 ./target/release/bench --url http://127.0.0.1:$PORT --prompt-len 400 --prefix-repeat 0 --max-tokens 64"
$B --seed 7 --concurrency 1 --num-requests 8 --warmup 2 2>&1 | grep -E "TTFT |TPOT" | cut -c1-120 >> $R
lo=$(wc -l < $LOG)
echo "== c8 x64 seed 11 (fresh)" >> $R
$B --seed 11 --concurrency 8 --num-requests 64 --warmup 0 2>&1 | grep -E "req/s|TTFT |TPOT" | cut -c1-120 >> $R
sed -n "$((lo+1)),\$p" $LOG | grep -oE "EXIT mode=[A-Za-z]+ t=[0-9]+ total=[0-9]+" | awk '{split($2,m,"=");split($3,t,"=");split($4,u,"=");k=m[2];n[k]++;s[k]+=u[2];if(t[2]>mx[k])mx[k]=t[2]} END{for(k in n)printf "  %s steps=%d mean_total=%.0fus max_t=%d\n",k,n[k],s[k]/n[k],mx[k]}' >> $R
echo "  -- Paged/Prefill 步 (mode t total_us)，按出现顺序前 40" >> $R
sed -n "$((lo+1)),\$p" $LOG | grep -oE "EXIT mode=(Prefill|Paged) t=[0-9]+ total=[0-9]+" | sed 's/EXIT mode=//;s/ t=/ /;s/ total=/ /' | head -40 | tr '\n' ';' >> $R; echo >> $R
echo "  -- 步序 run-length" >> $R
sed -n "$((lo+1)),\$p" $LOG | grep -oE "EXIT mode=[A-Za-z]+ t=[0-9]+" | sed "s/EXIT mode=//;s/ t=/:/" | uniq -c | awk '{printf "%sx%s ", $2,$1}' | cut -c1-1200 >> $R; echo >> $R
kill -INT $SP 2>/dev/null; sleep 2; kill -KILL $SP 2>/dev/null
echo DONE >> $R
