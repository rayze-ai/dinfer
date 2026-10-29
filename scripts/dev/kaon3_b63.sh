#!/usr/bin/env bash
# b63（E4）：流优先级 A/B。编 HEAD → envab_c：A = PREFILL_ASYNC=1 XSTREAM=1（E1，= b62 B）  B = + MAIN_STREAM_PRIO=1（decode 主流最高优先级）。
# 结果 tmp_b63_RESULT.txt / tmp_b63ab_RESULT.txt，标记 B63_DONE。附 HOST_TIME decode 步分布（ht_mixed.py）两侧。
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LIBRARY_PATH=/tmp/stub LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=120 CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0
cd /w/dinfer
R=/w/dinfer/tmp_b63_RESULT.txt; : > $R; date +%T >> $R
cargo build --release -p dinfer --features kaon3 > /w/dinfer/tmp_b63_build.log 2>&1; brc=$?
echo "BUILD EXIT=$brc md5=$(md5sum target/release/dinfer | cut -c1-12) marker=$(strings target/release/dinfer | grep -c MAIN_STREAM_PRIO) $(date +%T)" >> $R
[ $brc -ne 0 ] && { grep -E "^error" -A8 /w/dinfer/tmp_b63_build.log | head -30 | cut -c1-200 >> $R; echo B63_DONE >> $R; exit 1; }
TAG=b63ab AENV="PREFILL_ASYNC=1 XSTREAM=1" BENV="PREFILL_ASYNC=1 XSTREAM=1 MAIN_STREAM_PRIO=1" PORT=8361 bash /w/envab_c.sh
grep -E "^r[12] [AB] (c|out|startup)" /w/dinfer/tmp_b63ab_RESULT.txt >> $R
for side in A B; do echo "== $side host-time" >> $R; sed "s/\x1b\[[0-9;]*m//g" /w/dinfer/tmp_b63ab_r1_$side.log | grep -E "stream prio|runner step mix" | tail -2 | cut -c1-200 >> $R; sed "s/\x1b\[[0-9;]*m//g" /w/dinfer/tmp_b63ab_r1_$side.log | taskset -c 100-127 python3 -I /w/ht_mixed.py /dev/stdin 2>&1 | head -5 >> $R; done
echo B63_DONE >> $R
