#!/bin/bash
# 与 /hf_data/yaochen/kserve.sh 同口径，只是打开 dense MLP megakernel + 换端口/卡，用于 A/B。
# 用法：GPU=1 PORT=8132 bash kserve_mega.sh
NV=/usr/local/lib/python3.12/dist-packages/nvidia
mkdir -p /tmp/stub
ln -sf $NV/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so
ln -sf $NV/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cu13/lib
cd /w/dinfer
M=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478/
export KAON3_MEGA="${KAON3_MEGA:-1}"
export FP8_SELF_KERNEL="${FP8_SELF_KERNEL:-0}"
# `--gpus device=N` 已把目标卡映射成容器内 device 0，**不要**再设 CUDA_VISIBLE_DEVICES=N（会 NO_DEVICE）。
unset CUDA_VISIBLE_DEVICES
echo "[kserve_mega] KAON3_MEGA=$KAON3_MEGA FP8_SELF_KERNEL=$FP8_SELF_KERNEL PORT=${PORT:-8132}"
exec ./target/release/dinfer serve --model "$M" --tokenizer "$M" --port "${PORT:-8132}" --max-model-len 16384
