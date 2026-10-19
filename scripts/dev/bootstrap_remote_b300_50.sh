#!/usr/bin/env bash
# B300-50（host-172-16-3-50）首次 bootstrap。与第一台的差别：/data/yaochen/infer 需先 `sudo mkdir && chown`（用户有 sudo）；
# 无直连外网，用户给的代理 172.16.1.35:7890 只通国内镜像 → rustup / crates 走 TUNA，权重走 hf-mirror。
# 不装 CUDA / NCCL / Python 包（系统已有 /usr/local/cuda-13.1 与 libnccl，用户 2026-10-19 明确要求不要装）。
#   bash scripts/dev/bootstrap_remote_b300_50.sh     # 远端跑；在 tmux 会话 boot 里，幂等；完成标志 $ROOT/bootstrap/done.log
set -euo pipefail
ROOT=/data/yaochen/infer
mkdir -p $ROOT/bootstrap $ROOT/models $ROOT/cargo $ROOT/rustup
cat > $ROOT/env.sh <<ENV
export http_proxy=http://10.96.0.106:1080 https_proxy=http://10.96.0.106:1080 HTTP_PROXY=http://10.96.0.106:1080 HTTPS_PROXY=http://10.96.0.106:1080
export no_proxy=localhost,127.0.0.1,::1,172.16.0.0/16,10.0.0.0/8 NO_PROXY=localhost,127.0.0.1,::1,172.16.0.0/16,10.0.0.0/8   # 大小写都给：curl 读小写，reqwest / 部分工具读大写（用户 2026-10-19：注意设置 NO_PROXY）
export RUSTUP_HOME=$ROOT/rustup CARGO_HOME=$ROOT/cargo
# 固定版本 1.89.0：TUNA 只镜像 stable 频道（2026-10-19 实测 channel-rust-1.89.0.toml 404），USTC 有完整 dist
export RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static RUSTUP_UPDATE_ROOT=https://mirrors.ustc.edu.cn/rust-static/rustup
export RUSTUP_TOOLCHAIN=1.89.0-x86_64-unknown-linux-gnu
export CUDA_HOME=/usr/local/cuda-13.1 CUDA_ARCHS=103
export PATH=$ROOT/cargo/bin:$ROOT/rustup/toolchains/1.89.0-x86_64-unknown-linux-gnu/bin:/usr/local/cuda-13.1/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:\${LD_LIBRARY_PATH:-}
ENV
S() { echo "+ $*" | tee -a $ROOT/bootstrap/boot.log; bash -c "source $ROOT/env.sh; $*" 2>&1 | tee -a $ROOT/bootstrap/boot.log | tail -3; }
S "[ -x \$CARGO_HOME/bin/cargo ] || (curl -sSfL https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup/dist/x86_64-unknown-linux-gnu/rustup-init -o /tmp/rustup-init && chmod +x /tmp/rustup-init && /tmp/rustup-init -y --default-toolchain 1.89.0 --profile minimal --no-modify-path)"
S "mkdir -p \$CARGO_HOME && printf '[source.crates-io]\nreplace-with = \"tuna\"\n[source.tuna]\nregistry = \"sparse+https://mirrors.tuna.tsinghua.edu.cn/crates.io-index/\"\n[net]\ngit-fetch-with-cli = true\n' > \$CARGO_HOME/config.toml"
# 权重：hf-mirror（Qwen3-0.6B 10 个文件，约 1.5 GB）
S "cd $ROOT/models && mkdir -p qwen3-0.6b && cd qwen3-0.6b && for f in config.json generation_config.json merges.txt model.safetensors tokenizer.json tokenizer_config.json vocab.json; do [ -s \$f ] || curl -sSfL --retry 3 -o \$f https://hf-mirror.com/Qwen/Qwen3-0.6B/resolve/main/\$f; done; ls -la model.safetensors | cut -c20-80"
S "nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | tee $ROOT/bootstrap/gpus.log | head -2; nvcc --version | tail -1; cargo --version; ldconfig -p | grep -m1 libnccl.so"
echo done > $ROOT/bootstrap/done.log; echo BOOTSTRAP_DONE
