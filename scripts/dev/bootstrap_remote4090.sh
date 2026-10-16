#!/usr/bin/env bash
# 4090 测试机首次 bootstrap。用户约束：远程一切只放 /data/yaochen/infer（$ROOT）。
#   bash bootstrap_remote4090.sh          # 在远端 root shell 跑一次；全部在 tmux 会话 boot 里，断线不影响（幂等）
#   tmux attach -t boot                   # 看进度；日志 $ROOT/bootstrap/*.log；完成标志 $ROOT/bootstrap/done.log
# 布局：$ROOT/dinfer 仓库 | $ROOT/models 权重 | $ROOT/rustup + $ROOT/cargo 工具链 | $ROOT/cuda-13.0 toolkit | $ROOT/bootstrap 日志
# 用法（之后每个 shell）：source $ROOT/env.sh
# 网络：static.rust-lang.org 只有 ~100 KB/s → rustup / crates 走 rsproxy.cn；huggingface 不通，权重由本机 rsync。
set -u
ROOT=/data/yaochen/infer
mkdir -p $ROOT/bootstrap $ROOT/models $ROOT/cargo $ROOT/rustup
cat > $ROOT/env.sh <<ENV
export RUSTUP_HOME=$ROOT/rustup CARGO_HOME=$ROOT/cargo
export RUSTUP_DIST_SERVER=https://rsproxy.cn RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup
export CUDA_HOME=$ROOT/cuda-13.0 CUDA_ARCHS=89   # cudarc 用 cuda-version-from-build-system 按 nvcc 自动探测
export PATH=\$CARGO_HOME/bin:\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}   # toolkit 不在 /usr/local，运行时要能找到 libcudart.so.13
export TMPDIR=$ROOT/tmp
ENV
mkdir -p $ROOT/tmp $ROOT/cargo
cat > $ROOT/cargo/config.toml <<'CFG'
[source.crates-io]
replace-with = 'rsproxy-sparse'
[source.rsproxy-sparse]
registry = "sparse+https://rsproxy.cn/index/"
[registries.rsproxy]
index = "https://rsproxy.cn/crates.io-index"
[net]
git-fetch-with-cli = true
CFG
command -v tmux >/dev/null || DEBIAN_FRONTEND=noninteractive apt-get install -y -q tmux >/dev/null
tmux kill-session -t boot 2>/dev/null || true
rm -f $ROOT/bootstrap/done.log
tmux new-session -d -s boot -n main -c $ROOT/bootstrap
S() { tmux send-keys -t boot:main "$1" C-m; }
S "source $ROOT/env.sh"
S "[ -x \$CARGO_HOME/bin/cargo ] || (curl -sSf https://rsproxy.cn/rustup-init.sh | sh -s -- -y --default-toolchain 1.89.0 --profile minimal --no-modify-path) 2>&1 | tee rustup.log | tail -2; cargo --version | tee -a rustup.log"
S "DEBIAN_FRONTEND=noninteractive apt-get install -y -q cmake pkg-config libssl-dev 2>&1 | tail -1 | tee apt.log"
S "[ -x \$CUDA_HOME/bin/nvcc ] || { [ -s cuda.run ] || wget -q https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda_13.0.2_580.95.05_linux.run -O cuda.run; sh cuda.run --silent --toolkit --toolkitpath=\$CUDA_HOME --no-drm --no-man-page --override --tmpdir=$ROOT/tmp 2>&1 | tee cuda.log | tail -3; rm -f /usr/local/cuda; }; nvcc --version | tail -1 | tee -a cuda.log"
S "echo BOOTSTRAP_DONE \$(date +%T) | tee done.log"
echo "已在 tmux 会话 boot 中启动；tmux attach -t boot 查看，$ROOT/bootstrap/*.log 看日志"
