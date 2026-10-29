#!/usr/bin/env bash
# b60（D230）：B53 dense MLP M=9..16 不走 gemv 走 cuBLASLt 自调优。编 HEAD → envab_c ABAB：A = DENSE_GEMV_MAX_M=16（现状）  B = DENSE_GEMV_MAX_M=8。
# 结果 tmp_b60_RESULT.txt / tmp_b60ab_RESULT.txt，标记 B60_DONE。
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LIBRARY_PATH=/tmp/stub LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=120 CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0
cd /w/dinfer
R=/w/dinfer/tmp_b60_RESULT.txt; : > $R; date +%T >> $R
cargo build --release -p dinfer --features kaon3 > /w/dinfer/tmp_b60_build.log 2>&1; brc=$?
echo "BUILD EXIT=$brc md5=$(md5sum target/release/dinfer | cut -c1-12) marker=$(strings target/release/dinfer | grep -c DENSE_GEMV_MAX_M) $(date +%T)" >> $R
[ $brc -ne 0 ] && { grep -E "^error" -A8 /w/dinfer/tmp_b60_build.log | head -30 | cut -c1-200 >> $R; echo B60_DONE >> $R; exit 1; }
TAG=b60ab AENV="DENSE_GEMV_MAX_M=16" BENV="DENSE_GEMV_MAX_M=8 GEMM_FP8_DEBUG=1" PORT=8357 bash /w/envab_c.sh
grep -E "^r[12] [AB] (c|out|startup)" /w/dinfer/tmp_b60ab_RESULT.txt >> $R
sed "s/\x1b\[[0-9;]*m//g" /w/dinfer/tmp_b60ab_r1_B.log | grep -oE "gemm fp8 tune\] m=16 n=(2112|2816) k=(2816|2112) best=[0-9.]+us pick='[^']{0,40}" | sort -u | head -6 >> $R
echo B60_DONE >> $R
