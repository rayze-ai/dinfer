#!/usr/bin/env bash
# B300-50 环境收尾。代理 B 10.96.0.106:1080 通 crates.io / docker.io / github / huggingface.co（代理 A 172.16.1.35:7890 只通国内镜像）。
# 权重**不下载**：机器 /data/models 已有全部模型（用户 2026-10-19 两次强调），models/qwen3-0.6b 做软链接。不装 CUDA / NCCL。幂等。
set -uo pipefail
ROOT=/data/yaochen/infer
P=http://10.96.0.106:1080
sed -i "s#^export http_proxy=.*#export http_proxy=$P https_proxy=$P HTTP_PROXY=$P HTTPS_PROXY=$P no_proxy=localhost,127.0.0.1,172.16.0.0/16,10.0.0.0/8#" $ROOT/env.sh
sed -i '/RUSTUP_DIST_SERVER/d' $ROOT/env.sh
source $ROOT/env.sh
printf '[net]\ngit-fetch-with-cli = true\nretry = 5\n' > $CARGO_HOME/config.toml
rm -rf $ROOT/models/qwen3-0.6b && ln -s /data/models/Qwen/Qwen3-0.6B $ROOT/models/qwen3-0.6b
ls -la $ROOT/models/qwen3-0.6b/ | grep -c safetensors
cd $ROOT/dinfer && cargo build --release --workspace --features qwen3 2>&1 | grep -E "^error|error\[|Finished|failed to download" | head -5
ls -la target/release/dinfer 2>&1 | cut -c1-80
sudo -n env HTTP_PROXY=$P HTTPS_PROXY=$P nerdctl pull docker.io/vllm/vllm-openai:nightly 2>&1 | tail -2 | cut -c1-120
sudo -n nerdctl images 2>&1 | grep -i vllm | cut -c1-110
echo ENV_FINISH_DONE
