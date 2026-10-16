#!/usr/bin/env bash
# 4090 测试机首次 bootstrap。在远端 root shell 里跑一次；全部放进 tmux 会话 `boot`，断线不影响。
#   bash bootstrap_remote4090.sh            # 启动（幂等：已装的步骤会跳过）
#   tmux attach -t boot                     # 看进度；日志在 /root/bootstrap/*.log
# 装：rustup 1.89.0、CUDA 12.4.1 toolkit（驱动 580 兼容）、cmake/pkg-config/libssl-dev；探测 crates.io 直连。
# 仓库 / 权重 / golden 由本机 rsync 推（scripts/dev/remote4090.sh push），或 clone GitHub（见 README-remote4090.md）。
set -u
mkdir -p /root/bootstrap /data/code /data/ref/models
cd /root/bootstrap
command -v tmux >/dev/null || DEBIAN_FRONTEND=noninteractive apt-get install -y -q tmux >/dev/null
tmux has-session -t boot 2>/dev/null && { echo "tmux 会话 boot 已存在：tmux attach -t boot"; exit 0; }
tmux new-session -d -s boot -n main
tmux send-keys -t boot:main 'cd /root/bootstrap' C-m
tmux send-keys -t boot:main '[ -x ~/.cargo/bin/cargo ] || (curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.89.0 --profile minimal) 2>&1 | tee rustup.log | tail -2' C-m
tmux send-keys -t boot:main 'DEBIAN_FRONTEND=noninteractive apt-get install -y -q cmake pkg-config libssl-dev 2>&1 | tail -1 | tee apt.log' C-m
tmux send-keys -t boot:main '[ -x /usr/local/cuda/bin/nvcc ] || { [ -s cuda.run ] || wget -q --show-progress https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run -O cuda.run; sh cuda.run --silent --toolkit --no-drm --no-man-page --override 2>&1 | tee cuda.log | tail -3; }; /usr/local/cuda/bin/nvcc --version | tail -1' C-m
tmux send-keys -t boot:main 'for u in https://index.crates.io/config.json https://static.crates.io/crates/anyhow/anyhow-1.0.86.crate https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json https://github.com; do printf "%s -> " "$u"; curl -sS -o /dev/null -w "%{http_code}\n" --max-time 15 "$u"; done | tee net.log' C-m
tmux send-keys -t boot:main 'echo BOOTSTRAP_DONE $(date +%T) | tee -a done.log' C-m
echo "已在 tmux 会话 boot 中启动；tmux attach -t boot 查看，/root/bootstrap/*.log 看日志"
