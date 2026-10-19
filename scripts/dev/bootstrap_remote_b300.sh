#!/usr/bin/env bash
# B300 机器首次 bootstrap（照 bootstrap_remote4090.sh）。用户约束：远程一切只放 /data/yaochen/infer（$ROOT）。
#   bash bootstrap_remote_b300.sh          # 在远端 shell 跑一次；全部在 tmux 会话 boot 里，断线不影响（幂等）
#   tmux attach -t boot                    # 看进度；日志 $ROOT/bootstrap/*.log；完成标志 $ROOT/bootstrap/done.log
# 布局：$ROOT/dinfer 仓库 | $ROOT/models 权重 | $ROOT/rustup + $ROOT/cargo 工具链 | $ROOT/pip（NCCL 等 pip 包）| $ROOT/bootstrap 日志
# 机器已有 /usr/local/cuda-13.1（driver 595，8 × B300 sm_103）→ 不装 toolkit；NCCL 若系统没有则 pip 装到 $ROOT/pip（同 4090 做法）。
# 用法（之后每个 shell）：source $ROOT/env.sh
set -euo pipefail
ROOT=/data/yaochen/infer
mkdir -p $ROOT/bootstrap $ROOT/models $ROOT/cargo $ROOT/rustup $ROOT/pip
cat > $ROOT/env.sh <<ENV
export RUSTUP_HOME=$ROOT/rustup CARGO_HOME=$ROOT/cargo
export RUSTUP_DIST_SERVER=https://rsproxy.cn RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup
export CUDA_HOME=/usr/local/cuda-13.1 CUDA_ARCHS=103   # B300 = sm_103（Blackwell Ultra）；cudarc 按 nvcc 自动探测版本
export PATH=\$CARGO_HOME/bin:\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:$ROOT/lib:\${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=$ROOT/lib:\${LIBRARY_PATH:-}
export NCCL_DEBUG=WARN
ENV
mkdir -p $ROOT/lib
S() { tmux send-keys -t boot "$1" Enter; }
tmux has-session -t boot 2>/dev/null || tmux new-session -d -s boot -c $ROOT
S "source $ROOT/env.sh"
S "[ -x \$CARGO_HOME/bin/cargo ] || (curl -sSf https://rsproxy.cn/rustup-init.sh | sh -s -- -y --default-toolchain 1.89.0 --profile minimal --no-modify-path) 2>&1 | tee $ROOT/bootstrap/rustup.log"
S "mkdir -p \$CARGO_HOME && printf '[source.crates-io]\nreplace-with = \"rsproxy-sparse\"\n[source.rsproxy-sparse]\nregistry = \"sparse+https://rsproxy.cn/index/\"\n[net]\ngit-fetch-with-cli = true\n' > \$CARGO_HOME/config.toml"
# NCCL：系统有 libnccl 就用系统的；没有则 pip 到 $ROOT/pip 并软链到 $ROOT/lib（cudarc nccl feature 需要 -lnccl）
S "ls /usr/lib/x86_64-linux-gnu/libnccl.so* \$CUDA_HOME/lib64/libnccl.so* 2>/dev/null || (python3 -m pip install --target $ROOT/pip nvidia-nccl-cu13 2>&1 | tail -2; ln -sf $ROOT/pip/nvidia/nccl/lib/libnccl.so.2 $ROOT/lib/libnccl.so.2; ln -sf $ROOT/lib/libnccl.so.2 $ROOT/lib/libnccl.so; ls -l $ROOT/lib) 2>&1 | tee $ROOT/bootstrap/nccl.log"
S "nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | tee $ROOT/bootstrap/gpus.log; nvcc --version | tail -1; cargo --version; echo done | tee $ROOT/bootstrap/done.log"
echo "started in tmux 'boot'; tail -f $ROOT/bootstrap/*.log"
