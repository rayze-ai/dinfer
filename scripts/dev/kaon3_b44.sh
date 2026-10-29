#!/usr/bin/env bash
# b44 链（B42c：fp8 gemv 预取只对 M_TILE≤2 + B43：route softmax 并行 exp / warp 扫描）：stage44 入树 → 门禁（moe_route_fused_parity + mlp_fp8_decode_parity）→ bN TAG=b44
# A = 编译前 target（b43 二进制 = B42b 全预取版）。**注意 A 侧是 b43（c1 4.73 / c8 7.54），对比基线用 b41 定稿（4.78 / 7.19）**。唯一结束标记 B44_DONE。
cd /w/dinfer
cp -f /w/stage44/mlp_fp8_decode.cu crates/kernels/cuda/mlp_fp8_decode.cu; cp -f /w/stage44/moe.cu crates/kernels/cuda/moe.cu
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LIBRARY_PATH=/tmp/stub LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=120 CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0
[ -f /w/dinfer_prev_b44 ] || cp -f target/release/dinfer /w/dinfer_prev_b44
R=/w/dinfer/tmp_b44par_RESULT.txt; : > $R; date +%T >> $R
echo "A_md5=$(md5sum /w/dinfer_prev_b44 | cut -c1-12)" >> $R
cargo test --release -p dinfer-kernels --test moe_route_fused_parity --test mlp_fp8_decode_parity -- --ignored --nocapture > /w/dinfer/tmp_b44par.log 2>&1; rc=$?; echo "EXIT=$rc" >> $R
ROUTE_SOFTMAX_WARP=0 cargo test --release -p dinfer-kernels --test moe_route_fused_parity -- --ignored --nocapture > /w/dinfer/tmp_b44par_sm0.log 2>&1; rc2=$?; echo "EXIT(ROUTE_SOFTMAX_WARP=0)=$rc2" >> $R
grep -E "^error|error\[|panicked|test result|失败|FAIL|differ|位差" /w/dinfer/tmp_b44par.log | cut -c1-200 | head -30 >> $R
grep -E "^error|error\[" -A8 /w/dinfer/tmp_b44par.log | head -30 | cut -c1-200 >> $R
[ $rc -ne 0 -o $rc2 -ne 0 ] && { echo B44_DONE >> $R; echo "PARITY_FAIL" >> /w/dinfer/tmp_b44_RESULT.txt; echo DONE >> /w/dinfer/tmp_b44_RESULT.txt; exit 1; }
cd /w; TAG=b44 MARKER="fp8gemv_b42c_prefetch_small" BENV="" bash -x /w/bN_keep3.sh >/w/dinfer/tmp_b44_bN_trace.log 2>&1
echo B44_DONE >> $R
