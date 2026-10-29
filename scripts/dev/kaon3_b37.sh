#!/usr/bin/env bash
# b37 链（R1 router bf16 gemv）：stage37 入树 → parity（router_gemv + moe_route_fused）→ bN TAG=b37（A = 编译前 target = b36 二进制；B = ROUTER_GEMV 默认开）
cd /w/dinfer
cp -f /w/stage37/moe.cu crates/kernels/cuda/moe.cu; cp -f /w/stage37/kernels.h crates/kernels/cuda/kernels.h
cp -f /w/stage37/ffi.rs crates/kernels/src/ffi.rs; cp -f /w/stage37/ops.rs crates/kernels/src/ops.rs
cp -f /w/stage37/router_gemv_parity.rs crates/kernels/tests/; cp -f /w/stage37/layer.rs models/kaon3/src/layer.rs
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LIBRARY_PATH=/tmp/stub LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=120 CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0
[ -f /w/dinfer_prev_b37 ] || cp -f target/release/dinfer /w/dinfer_prev_b37   # A = b36 二进制（编译前的 target）
R=/w/dinfer/tmp_b37par_RESULT.txt; : > $R; date +%T >> $R
cargo test --release -p dinfer-kernels --test router_gemv_parity --test moe_route_fused_parity -- --ignored --nocapture > /w/dinfer/tmp_b37par.log 2>&1; rc=$?; echo "EXIT=$rc" >> $R
grep -E "^error|error\[|router_gemv:|panicked|test result|失败|FAIL" /w/dinfer/tmp_b37par.log | cut -c1-200 | head -40 >> $R
grep -E "^error|error\[" -A8 /w/dinfer/tmp_b37par.log | head -40 | cut -c1-200 >> $R
echo DONE >> $R
[ $rc -ne 0 ] && { echo "PARITY_FAIL" >> /w/dinfer/tmp_b37_RESULT.txt; echo DONE >> /w/dinfer/tmp_b37_RESULT.txt; exit 1; }
cd /w; TAG=b37 MARKER="router_gemv_bf16" BENV="" bash -x /w/bN_keep3.sh >/w/dinfer/tmp_b37_bN_trace.log 2>&1
