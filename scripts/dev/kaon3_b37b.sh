#!/usr/bin/env bash
# b37b 链（R1 v2 router gemv）：stage37 入树 → parity + 微基准 → bN TAG=b37b（A = prev_b37 = b36 二进制；B = ROUTER_GEMV 默认开 v2）
cd /w/dinfer
cp -f /w/stage37/moe.cu crates/kernels/cuda/moe.cu; cp -f /w/stage37/kernels.h crates/kernels/cuda/kernels.h
cp -f /w/stage37/ffi.rs crates/kernels/src/ffi.rs; cp -f /w/stage37/ops.rs crates/kernels/src/ops.rs
cp -f /w/stage37/router_gemv_parity.rs /w/stage37/router_gemv_timing.rs crates/kernels/tests/; cp -f /w/stage37/layer.rs models/kaon3/src/layer.rs
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LIBRARY_PATH=/tmp/stub LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=120 CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0
cp -f /w/dinfer_prev_b37 /w/dinfer_prev_b37b   # A = b36 二进制
R=/w/dinfer/tmp_b37bpar_RESULT.txt; : > $R; date +%T >> $R
cargo test --release -p dinfer-kernels --test router_gemv_parity --test router_gemv_timing --test moe_route_fused_parity -- --ignored --nocapture > /w/dinfer/tmp_b37bpar.log 2>&1; rc=$?; echo "EXIT=$rc" >> $R
grep -E "^error|error\[|router_gemv|panicked|test result|失败|FAIL" /w/dinfer/tmp_b37bpar.log | cut -c1-200 | head -40 >> $R
grep -E "^error|error\[" -A8 /w/dinfer/tmp_b37bpar.log | head -40 | cut -c1-200 >> $R
echo DONE >> $R
[ $rc -ne 0 ] && { echo "PARITY_FAIL" >> /w/dinfer/tmp_b37b_RESULT.txt; echo DONE >> /w/dinfer/tmp_b37b_RESULT.txt; exit 1; }
cd /w; TAG=b37b MARKER="router_gemv_bf16" BENV="" bash -x /w/bN_keep3.sh >/w/dinfer/tmp_b37b_bN_trace.log 2>&1
