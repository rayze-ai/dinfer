#!/usr/bin/env bash
# b46 链（B45：combine + post_ff_norm_2 融合，`MOE_COMBINE_NORM` 默认开）：stage46 入树 → 门禁 moe_combine_norm_parity → bN TAG=b46
# A = 编译前 target（b45 二进制：split-K 默认关 ⇒ 行为 = b44）。唯一结束标记 B46_DONE。
cd /w/dinfer
cp -f /w/stage46/moe.cu crates/kernels/cuda/moe.cu; cp -f /w/stage46/kernels.h crates/kernels/cuda/kernels.h
cp -f /w/stage46/ffi.rs crates/kernels/src/ffi.rs; cp -f /w/stage46/ops.rs crates/kernels/src/ops.rs
cp -f /w/stage46/layer.rs models/kaon3/src/layer.rs; cp -f /w/stage46/moe_combine_norm_parity.rs crates/kernels/tests/
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LIBRARY_PATH=/tmp/stub LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=120 CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0
[ -f /w/dinfer_prev_b46 ] || cp -f target/release/dinfer /w/dinfer_prev_b46
R=/w/dinfer/tmp_b46par_RESULT.txt; : > $R; date +%T >> $R
echo "A_md5=$(md5sum /w/dinfer_prev_b46 | cut -c1-12)" >> $R
cargo test --release -p dinfer-kernels --test moe_combine_norm_parity --test moe_route_fused_parity -- --ignored --nocapture > /w/dinfer/tmp_b46par.log 2>&1; rc=$?; echo "EXIT=$rc" >> $R
grep -E "^error|error\[|combine\+norm|panicked|test result|失败|FAIL" /w/dinfer/tmp_b46par.log | cut -c1-200 | head -30 >> $R
grep -E "^error|error\[" -A8 /w/dinfer/tmp_b46par.log | head -30 | cut -c1-200 >> $R
[ $rc -ne 0 ] && { echo B46_DONE >> $R; echo "PARITY_FAIL" >> /w/dinfer/tmp_b46_RESULT.txt; echo DONE >> /w/dinfer/tmp_b46_RESULT.txt; exit 1; }
cd /w; TAG=b46 MARKER="MOE_COMBINE_NORM" BENV="" bash -x /w/bN_keep3.sh >/w/dinfer/tmp_b46_bN_trace.log 2>&1
echo B46_DONE >> $R
