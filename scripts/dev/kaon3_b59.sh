#!/usr/bin/env bash
# b59（D229）：B52 route 拆两段。编 → `moe_route_fused_parity`（tokens 33/64 自动走 split 路径，与旧链逐位比）→ envab_c ABAB
#   A = ROUTE_SPLIT_MIN_TOK=0（fused）  B = 默认（≥9 split）。两侧都带 B51 env（FP8_GEMV_MAX_M=8 FP8_LT_TUNE_MIN_M=9）。结果 tmp_b59_RESULT.txt，标记 B59_DONE。
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LIBRARY_PATH=/tmp/stub LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=120 CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0
cd /w/dinfer
R=/w/dinfer/tmp_b59_RESULT.txt; : > $R; date +%T >> $R
cargo test --release -p dinfer-kernels --test moe_route_fused_parity --no-run > /w/dinfer/tmp_b59_tbuild.log 2>&1; rc=$?; echo "TEST BUILD EXIT=$rc" >> $R
[ $rc -ne 0 ] && { grep -E "error" -A8 /w/dinfer/tmp_b59_tbuild.log | head -30 | cut -c1-200 >> $R; echo B59_DONE >> $R; exit 1; }
cargo test --release -p dinfer-kernels --test moe_route_fused_parity -- --ignored --nocapture 2>&1 | grep -E "test result|失败|panicked|ok" | tail -4 | cut -c1-200 >> $R
grep -q "test result: ok" $R || { echo "PARITY FAIL" >> $R; echo B59_DONE >> $R; exit 1; }
cargo build --release -p dinfer --features kaon3 > /w/dinfer/tmp_b59_build.log 2>&1; brc=$?
echo "BUILD EXIT=$brc md5=$(md5sum target/release/dinfer | cut -c1-12) marker=$(strings target/release/dinfer | grep -c ROUTE_SPLIT_MIN_TOK) $(date +%T)" >> $R
[ $brc -ne 0 ] && { grep -E "^error" -A8 /w/dinfer/tmp_b59_build.log | head -30 | cut -c1-200 >> $R; echo B59_DONE >> $R; exit 1; }
TAG=b59ab AENV="ROUTE_SPLIT_MIN_TOK=0 FP8_GEMV_MAX_M=8 FP8_LT_TUNE_MIN_M=9" BENV="ROUTE_SPLIT_MIN_TOK=9 FP8_GEMV_MAX_M=8 FP8_LT_TUNE_MIN_M=9" PORT=8356 bash /w/envab_c.sh
grep -E "^r[12] [AB] (c|out|startup)" /w/dinfer/tmp_b59ab_RESULT.txt >> $R
echo B59_DONE >> $R
