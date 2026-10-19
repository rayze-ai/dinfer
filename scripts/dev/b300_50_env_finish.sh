#!/usr/bin/env bash
# B300-50 环境收尾。用户 2026-10-19 给了第二个代理 10.96.0.106:1080，实测通 crates.io / docker.io / github / huggingface.co / pypi
# （第一个 172.16.1.35:7890 只通国内镜像）。→ env.sh 换代理 B，crates 用官方源，权重直接 huggingface.co，vLLM 镜像直接 docker.io。
# 不装 CUDA / NCCL（系统已有）。在远端跑，幂等。
set -uo pipefail
ROOT=/data/yaochen/infer
P=http://10.96.0.106:1080
sed -i "s#^export http_proxy=.*#export http_proxy=$P https_proxy=$P HTTP_PROXY=$P HTTPS_PROXY=$P no_proxy=localhost,127.0.0.1,172.16.0.0/16,10.0.0.0/8#" $ROOT/env.sh
sed -i '/RUSTUP_DIST_SERVER/d' $ROOT/env.sh   # rustup 已装好，不再需要镜像源
source $ROOT/env.sh
rm -f $CARGO_HOME/config.toml; printf '[net]\ngit-fetch-with-cli = true\nretry = 5\n' > $CARGO_HOME/config.toml
echo "proxy=$https_proxy; cargo=$(cargo --version)"
cd $ROOT/models/qwen3-0.6b
for f in config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json; do [ -s $f ] || curl -sSfL --retry 3 -o $f https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/$f; done
for i in $(seq 1 8); do
  curl -sSfL -C - --retry 5 --retry-all-errors -o model.safetensors https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors 2>&1 | tail -1 | cut -c1-100
  s=$(stat -c %s model.safetensors 2>/dev/null || echo 0); echo "weights try $i size=$s"; [ "$s" -ge 1503300000 ] && break; sleep 2
done
python3 -c "import json,struct;f=open('model.safetensors','rb');n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));print('safetensors header ok, tensors:',len(h))" 2>&1 | tail -1
cd $ROOT/dinfer && cargo build --release --workspace --features qwen3 2>&1 | grep -E "^error|error\[|Finished|failed to download" | head -5
ls -la target/release/dinfer 2>&1 | cut -c1-80
sudo -n env HTTP_PROXY=$P HTTPS_PROXY=$P nerdctl pull docker.io/vllm/vllm-openai:nightly 2>&1 | tail -2 | cut -c1-120
sudo -n nerdctl images 2>&1 | grep -i vllm | cut -c1-110
echo ENV_FINISH_DONE
