#!/usr/bin/env bash
# b43 链（B42b：fp8 gemv/gateup 权重预取提前到 load_a 前）：stage43 入树 → parity（mlp_fp8_decode_parity 默认 + FP8_GEMV_A8=0 两遍）
# + 微基准 fp8_gemv_timing → bN TAG=b43（A = 编译前 target = b38 二进制；B = A8 默认开）。
cd /w/dinfer
cp -f /w/stage43/mlp_fp8_decode.cu crates/kernels/cuda/mlp_fp8_decode.cu
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LIBRARY_PATH=/tmp/stub LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=120 CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0
[ -f /w/dinfer_prev_b43 ] || cp -f target/release/dinfer /w/dinfer_prev_b43   # A = b41 二进制（编译前 target）
R=/w/dinfer/tmp_b43par_RESULT.txt; : > $R; date +%T >> $R
echo "A_md5=$(md5sum /w/dinfer_prev_b43 | cut -c1-12)" >> $R
cargo test --release -p dinfer-kernels --test mlp_fp8_decode_parity --test fp8_gemv_timing -- --ignored --nocapture > /w/dinfer/tmp_b43par.log 2>&1; rc=$?; echo "EXIT(A8 on)=$rc" >> $R
FP8_GEMV_A8=0 cargo test --release -p dinfer-kernels --test mlp_fp8_decode_parity -- --ignored --nocapture > /w/dinfer/tmp_b43par_off.log 2>&1; rc2=$?; echo "EXIT(A8 off)=$rc2" >> $R
grep -E "^error|error\[|gemv N=|us|panicked|test result|失败|FAIL" /w/dinfer/tmp_b43par.log | grep -vE "warning" | cut -c1-200 | head -60 >> $R
grep -E "^error|error\[" -A8 /w/dinfer/tmp_b43par.log | head -40 | cut -c1-200 >> $R
echo DONE >> $R
[ $rc -ne 0 -o $rc2 -ne 0 ] && { echo "PARITY_FAIL" >> /w/dinfer/tmp_b43_RESULT.txt; echo DONE >> /w/dinfer/tmp_b43_RESULT.txt; exit 1; }
cd /w; TAG=b43 MARKER="fp8gemv_b42b_prefetch" BENV="" bash -x /w/bN_keep3.sh >/w/dinfer/tmp_b43_bN_trace.log 2>&1
