#!/usr/bin/env bash
# b40：stage40/layer.rs（只加 MOE_EXPERT_STATS 诊断，默认关）入树 → 编译 → expstats（c1/c8 各 3 轮，数激活专家）。不跑 bN。
cd /w/dinfer
cp -f /w/stage40/layer.rs models/kaon3/src/layer.rs
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LIBRARY_PATH=/tmp/stub LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=120 CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0
[ -f /w/dinfer_prev_b40 ] || cp -f target/release/dinfer /w/dinfer_prev_b40   # = b39 二进制
R=/w/dinfer/tmp_b40_RESULT.txt; : > $R; date +%T >> $R
cargo build --release -p dinfer --features kaon3 > /w/dinfer/tmp_b40_build.log 2>&1; rc=$?; echo "BUILD EXIT=$rc marker=$(strings target/release/dinfer | grep -c MOE_EXPERT_STATS)" >> $R
[ $rc -ne 0 ] && { grep -E "^error" -A8 /w/dinfer/tmp_b40_build.log | head -30 | cut -c1-200 >> $R; echo DONE >> $R; exit 1; }
cd /w; BIN=./target/release/dinfer TAG=expstats bash /w/expstats.sh >/w/dinfer/tmp_expstats_run.log 2>&1
cat /w/dinfer/tmp_expstats_RESULT.txt >> $R; echo DONE >> $R
