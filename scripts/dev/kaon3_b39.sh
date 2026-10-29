#!/usr/bin/env bash
# b39 链（A8：fp8 gemv 的 A 以 e4m3 驻留 smem，M=8/K=4096 的 o_proj 回到自写 gemv）：stage39 入树 → parity（mlp_fp8_decode_parity 默认 + FP8_GEMV_A8=0 两遍）
# + 微基准 fp8_gemv_timing → bN TAG=b39（A = 编译前 target = b38 二进制；B = A8 默认开）。
cd /w/dinfer
for f in mlp_fp8_decode.cu kernels.h; do cp -f /w/stage39/$f crates/kernels/cuda/$f; done
cp -f /w/stage39/ffi.rs crates/kernels/src/ffi.rs; cp -f /w/stage39/ops.rs crates/kernels/src/ops.rs
cp -f /w/stage39/layer.rs models/kaon3/src/layer.rs
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LIBRARY_PATH=/tmp/stub LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=120 CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0
[ -f /w/dinfer_prev_b39 ] || cp -f target/release/dinfer /w/dinfer_prev_b39   # A = b38 二进制（编译前 target）
R=/w/dinfer/tmp_b39par_RESULT.txt; : > $R; date +%T >> $R
echo "A_md5=$(md5sum /w/dinfer_prev_b39 | cut -c1-12)" >> $R
cargo test --release -p dinfer-kernels --test mlp_fp8_decode_parity --test fp8_gemv_timing -- --ignored --nocapture > /w/dinfer/tmp_b39par.log 2>&1; rc=$?; echo "EXIT(A8 on)=$rc" >> $R
FP8_GEMV_A8=0 cargo test --release -p dinfer-kernels --test mlp_fp8_decode_parity -- --ignored --nocapture > /w/dinfer/tmp_b39par_off.log 2>&1; rc2=$?; echo "EXIT(A8 off)=$rc2" >> $R
grep -E "^error|error\[|gemv N=|us|panicked|test result|失败|FAIL" /w/dinfer/tmp_b39par.log | grep -vE "warning" | cut -c1-200 | head -60 >> $R
grep -E "^error|error\[" -A8 /w/dinfer/tmp_b39par.log | head -40 | cut -c1-200 >> $R
echo DONE >> $R
[ $rc -ne 0 -o $rc2 -ne 0 ] && { echo "PARITY_FAIL" >> /w/dinfer/tmp_b39_RESULT.txt; echo DONE >> /w/dinfer/tmp_b39_RESULT.txt; exit 1; }
cd /w; TAG=b39 MARKER="FP8_GEMV_A8" BENV="" bash -x /w/bN_keep3.sh >/w/dinfer/tmp_b39_bN_trace.log 2>&1
