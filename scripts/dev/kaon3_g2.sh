#!/usr/bin/env bash
# g2（G3a）：fp8 Lt 自调优在 m 桶 64..2048 排除 128x128 tile（tile=20）。编 HEAD → envab_c ABAB：A = 默认  B = FP8_LT_PREFER_SMALL_TILE=1。
# 两侧 LDSM 取 ${LDSM_ENV:-}（g1 定稿后填 MOE_P64_LDSM=1 或空）。结果 tmp_g2_RESULT.txt，标记 G2_DONE。
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LIBRARY_PATH=/tmp/stub LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=120 CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0
cd /w/dinfer
R=/w/dinfer/tmp_g2_RESULT.txt; : > $R; date +%T >> $R
cargo build --release -p dinfer --features kaon3 > /w/dinfer/tmp_g2_build.log 2>&1; brc=$?
echo "BUILD EXIT=$brc md5=$(md5sum target/release/dinfer | cut -c1-12) marker=$(strings target/release/dinfer | grep -c FP8_LT_PREFER_SMALL_TILE) $(date +%T)" >> $R
[ $brc -ne 0 ] && { grep -E "^error" -A8 /w/dinfer/tmp_g2_build.log | head -30 | cut -c1-200 >> $R; echo G2_DONE >> $R; exit 1; }
TAG=g2ab AENV="${LDSM_ENV:-} GEMM_FP8_DEBUG=1" BENV="${LDSM_ENV:-} FP8_LT_PREFER_SMALL_TILE=1 GEMM_FP8_DEBUG=1" PORT=8367 bash /w/envab_c.sh
grep -E "^r[12] [AB] (c|out|startup)" /w/dinfer/tmp_g2ab_RESULT.txt >> $R
for s in A B; do echo "== $s tune picks (m>=64 buckets)" >> $R; sed "s/\x1b\[[0-9;]*m//g" /w/dinfer/tmp_g2ab_r1_$s.log | grep -oE "gemm fp8 tune\] m=[0-9]+ n=(8192|2816) k=(2816|4096) best=[0-9.]+us pick='id=[0-9]+ tile=[0-9]+ splitk=[0-9]+" | sort -t= -k2,2n -u | head -14 >> $R; done
echo G2_DONE >> $R
