#!/usr/bin/env bash
# b45 链（B44 split-K：gate‖up dual split-K×2 写 fp32 部分和，gelu+down 量化顺带归约；`MOE_SPLITK` 默认 1=关）：
# stage45 入树 → 门禁 moe_splitk_parity（ksplit 1 位同 / 2,4 码差 ≤ 2%）+ moe_gelu_quant_fused_parity → bN TAG=b45，B 侧 env MOE_SPLITK=2
# A = 编译前 target（b44 二进制）。唯一结束标记 B45_DONE。
cd /w/dinfer
cp -f /w/stage45/moe_grouped_sm120.cu crates/kernels/cuda/; cp -f /w/stage45/ffi.rs crates/kernels/src/ffi.rs; cp -f /w/stage45/ops.rs crates/kernels/src/ops.rs
cp -f /w/stage45/layer.rs models/kaon3/src/layer.rs; cp -f /w/stage45/moe_splitk_parity.rs crates/kernels/tests/
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LIBRARY_PATH=/tmp/stub LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=120 CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0
[ -f /w/dinfer_prev_b45 ] || cp -f target/release/dinfer /w/dinfer_prev_b45
R=/w/dinfer/tmp_b45par_RESULT.txt; : > $R; date +%T >> $R
echo "A_md5=$(md5sum /w/dinfer_prev_b45 | cut -c1-12)" >> $R
cargo test --release -p dinfer-kernels --test moe_splitk_parity --test moe_gelu_quant_fused_parity --test moe_pipe_parity -- --ignored --nocapture > /w/dinfer/tmp_b45par.log 2>&1; rc=$?; echo "EXIT=$rc" >> $R
grep -E "^error|error\[|MOE_SPLITK_PARITY|panicked|test result|失败|FAIL" /w/dinfer/tmp_b45par.log | cut -c1-200 | head -30 >> $R
grep -E "^error|error\[" -A8 /w/dinfer/tmp_b45par.log | head -30 | cut -c1-200 >> $R
[ $rc -ne 0 ] && { echo B45_DONE >> $R; echo "PARITY_FAIL" >> /w/dinfer/tmp_b45_RESULT.txt; echo DONE >> /w/dinfer/tmp_b45_RESULT.txt; exit 1; }
cd /w; TAG=b45 MARKER="MOE_SPLITK" BENV="MOE_SPLITK=2" bash -x /w/bN_keep3.sh >/w/dinfer/tmp_b45_bN_trace.log 2>&1
echo B45_DONE >> $R
