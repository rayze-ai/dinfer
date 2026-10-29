#!/usr/bin/env bash
# b38 链（B34③ gelu·mul + down quant 融合）：stage38 入树 → parity（gelu_quant_fused + moe_stage）→ bN TAG=b38（A = 编译前 target；B = MOE_GELU_QUANT_FUSED 默认开）
cd /w/dinfer
cp -f /w/stage38/moe_grouped_sm120.cu crates/kernels/cuda/; cp -f /w/stage38/kernels.h crates/kernels/cuda/kernels.h
cp -f /w/stage38/ffi.rs crates/kernels/src/ffi.rs; cp -f /w/stage38/ops.rs crates/kernels/src/ops.rs
cp -f /w/stage38/moe_gelu_quant_fused_parity.rs crates/kernels/tests/; cp -f /w/stage38/layer.rs models/kaon3/src/layer.rs
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LIBRARY_PATH=/tmp/stub LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=120 CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0
[ -f /w/dinfer_prev_b38 ] || cp -f target/release/dinfer /w/dinfer_prev_b38   # A = 编译前 target（b37b 定稿与否由 B38_A 决定）
R=/w/dinfer/tmp_b38par_RESULT.txt; : > $R; date +%T >> $R
cargo test --release -p dinfer-kernels --test moe_gelu_quant_fused_parity --test moe_norm_quant_fused_parity -- --ignored --nocapture > /w/dinfer/tmp_b38par.log 2>&1; rc=$?; echo "EXIT=$rc" >> $R
grep -E "^error|error\[|gelu\+quant|panicked|test result|失败|FAIL" /w/dinfer/tmp_b38par.log | cut -c1-200 | head -40 >> $R
grep -E "^error|error\[" -A8 /w/dinfer/tmp_b38par.log | head -40 | cut -c1-200 >> $R
echo DONE >> $R
[ $rc -ne 0 ] && { echo "PARITY_FAIL" >> /w/dinfer/tmp_b38_RESULT.txt; echo DONE >> /w/dinfer/tmp_b38_RESULT.txt; exit 1; }
cd /w; TAG=b38 MARKER="moe_gelu_grouped_quant" BENV="${B38_BENV:-}" bash -x /w/bN_keep3.sh >/w/dinfer/tmp_b38_bN_trace.log 2>&1
