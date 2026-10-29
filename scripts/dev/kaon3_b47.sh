#!/usr/bin/env bash
# b47 链（B45 修正：combine+norm 融合版 gather 改 uint4 向量读）：stage47 入树 → 编译 → parity → envab 同二进制 A=MOE_COMBINE_NORM=0（= b44 路径）/ B=默认（融合）。
# 不用 bN（b46 的 prev 二进制默认已是融合版）。唯一结束标记 B47_DONE。
cd /w/dinfer
cp -f /w/stage47/moe.cu crates/kernels/cuda/moe.cu
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LIBRARY_PATH=/tmp/stub LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=120 CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0
R=/w/dinfer/tmp_b47par_RESULT.txt; : > $R; date +%T >> $R
cargo test --release -p dinfer-kernels --test moe_combine_norm_parity -- --ignored --nocapture > /w/dinfer/tmp_b47par.log 2>&1; rc=$?; echo "EXIT=$rc" >> $R
grep -E "^error|error\[|combine\+norm|panicked|test result|失败|FAIL" /w/dinfer/tmp_b47par.log | cut -c1-200 | head -20 >> $R
[ $rc -ne 0 ] && { echo B47_DONE >> $R; exit 1; }
cargo build --release -p dinfer --features kaon3 > /w/dinfer/tmp_b47_build.log 2>&1; brc=$?; echo "BUILD EXIT=$brc md5=$(md5sum target/release/dinfer | cut -c1-12)" >> $R
[ $brc -ne 0 ] && { grep -E "^error" -A6 /w/dinfer/tmp_b47_build.log | head -20 | cut -c1-200 >> $R; echo B47_DONE >> $R; exit 1; }
cd /w; TAG=b47 AENV="MOE_COMBINE_NORM=0" BENV="" bash /w/envab.sh >/w/dinfer/tmp_b47_run.log 2>&1
echo B47_DONE >> $R
