#!/usr/bin/env bash
# b62（E2）：prefill 异步收割 A/B。编 HEAD → envab_c：A = 现状（XSTREAM=0 单流混批）  B = PREFILL_ASYNC=1 XSTREAM=1（需要 AltSet；pair 路径被 late 路径取代）。
# 结果 tmp_b62_RESULT.txt / tmp_b62ab_RESULT.txt，标记 B62_DONE。B 侧 serve 日志的 "runner step mix" 行给 late_begin / late_harvest 计数。
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LIBRARY_PATH=/tmp/stub LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=120 CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0
cd /w/dinfer
R=/w/dinfer/tmp_b62_RESULT.txt; : > $R; date +%T >> $R
cargo build --release -p dinfer --features kaon3 > /w/dinfer/tmp_b62_build.log 2>&1; brc=$?
echo "BUILD EXIT=$brc md5=$(md5sum target/release/dinfer | cut -c1-12) marker=$(strings target/release/dinfer | grep -c PREFILL_ASYNC) $(date +%T)" >> $R
[ $brc -ne 0 ] && { grep -E "^error" -A8 /w/dinfer/tmp_b62_build.log | head -30 | cut -c1-200 >> $R; echo B62_DONE >> $R; exit 1; }
TAG=b62ab AENV="PREFILL_ASYNC=0 XSTREAM=0" BENV="PREFILL_ASYNC=1 XSTREAM=1" PORT=8360 bash /w/envab_c.sh
grep -E "^r[12] [AB] (c|out|startup)" /w/dinfer/tmp_b62ab_RESULT.txt >> $R
for f in /w/dinfer/tmp_b62ab_r1_B.log /w/dinfer/tmp_b62ab_r2_B.log; do sed "s/\x1b\[[0-9;]*m//g" $f | grep -E "runner step mix" | tail -1 | cut -c1-200 >> $R; done
echo B62_DONE >> $R
