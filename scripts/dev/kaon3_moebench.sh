#!/usr/bin/env bash
# B40 MoE grouped GEMM 微基准扫参（nsys 外）：每档一进程（env 为 static once）。结果 tmp_moebench_RESULT.txt
cd /w/dinfer
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub; ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so; ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LIBRARY_PATH=/tmp/stub LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=120 CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0
cp -f /w/stage41/moe_gemm_timing.rs crates/kernels/tests/
R=/w/dinfer/tmp_moebench_RESULT.txt; : > $R; date +%T >> $R
cargo test --release -p dinfer-kernels --test moe_gemm_timing --no-run > /w/dinfer/tmp_moebench_build.log 2>&1; rc=$?; echo "BUILD EXIT=$rc" >> $R
[ $rc -ne 0 ] && { grep -E "^error|error\[" -A8 /w/dinfer/tmp_moebench_build.log | head -30 | cut -c1-200 >> $R; echo MOEBENCH_DONE >> $R; exit 1; }
for cfg in "default" "MOE_GEMM_BK=128" "MOE_GEMM_BK=64"; do
  echo "== cfg: $cfg" >> $R
  if [ "$cfg" = default ]; then envs=""; else envs="$cfg"; fi
  env $envs cargo test --release -p dinfer-kernels --test moe_gemm_timing -- --ignored --nocapture 2>&1 | grep -E "MOE_GEMM_TIMING|panicked|error" | cut -c1-200 >> $R
done
echo MOEBENCH_DONE >> $R
