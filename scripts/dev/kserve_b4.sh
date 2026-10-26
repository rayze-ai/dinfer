#!/bin/bash
# B4 (cudagraph) serve：MOE_GROUPED=1 固定开（grouped MoE 去 D2H 是图捕获前置）。
# NO_GRAPH=1 关图跑 eager（A/B 对比文本一致性）。端口 KAON3_PORT（默认 8133）。
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub
ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so
ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
PORT=${KAON3_PORT:-8133}
export MOE_GROUPED=1
export KAON3_MEGA=0
CUDA_VISIBLE_DEVICES=0 exec ./target/release/dinfer serve --model "$M" --tokenizer "$M" --port "$PORT" --max-model-len 16384 "$@"
