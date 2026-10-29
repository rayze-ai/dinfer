#!/usr/bin/env bash
# b58（D228）：B51 env A/B —— A = 默认（gemv ≤ 16，Lt 自调优 m ≥ 64）；B = `FP8_GEMV_MAX_M=8 FP8_LT_TUNE_MIN_M=9`（M=9..16 走 cuBLASLt 自调优）。
# 编 HEAD（含 B50 XSTREAM 默认关）→ marker → envab_c。结果 tmp_b58_RESULT.txt / tmp_b58ab_RESULT.txt，标记 B58_DONE。
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LIBRARY_PATH=/tmp/stub LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=120 CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0
cd /w/dinfer
R=/w/dinfer/tmp_b58_RESULT.txt; : > $R; date +%T >> $R
cargo build --release -p dinfer --features kaon3 > /w/dinfer/tmp_b58_build.log 2>&1; brc=$?
echo "BUILD EXIT=$brc md5=$(md5sum target/release/dinfer | cut -c1-12) marker=$(strings target/release/dinfer | grep -c FP8_GEMV_MAX_M)/$(strings target/release/dinfer | grep -c FP8_LT_TUNE_MIN_M) $(date +%T)" >> $R
[ $brc -ne 0 ] && { grep -E "^error" -A8 /w/dinfer/tmp_b58_build.log | head -30 | cut -c1-200 >> $R; echo B58_DONE >> $R; exit 1; }
TAG=b58ab AENV="FP8_GEMV_MAX_M=16" BENV="FP8_GEMV_MAX_M=8 FP8_LT_TUNE_MIN_M=9 GEMM_FP8_DEBUG=1" PORT=8355 bash /w/envab_c.sh
grep -E "^r[12] [AB] (c|out|startup)" /w/dinfer/tmp_b58ab_RESULT.txt >> $R
sed "s/\x1b\[[0-9;]*m//g" /w/dinfer/tmp_b58ab_r1_B.log | grep -oE "gemm fp8 tune\] m=(9|1[0-6]) n=[0-9]+ k=[0-9]+ best=[0-9.]+us pick='[^']{0,40}" | sort -u | head -12 >> $R
echo B58_DONE >> $R
