#!/usr/bin/env bash
# B300-50 环境收尾（bootstrap 之后）：① crates 源换成能真下载 .crate 的镜像（TUNA 的 dl 指回 static.crates.io，代理不通）；
# ② 续传 Qwen3-0.6B 权重（hf-mirror 经代理会 SSL EOF 中断，-C - 续传 + 重试）；③ release 构建。在远端跑，幂等。
set -uo pipefail
ROOT=/data/yaochen/infer; source $ROOT/env.sh
pick=""
for m in "ustc|https://mirrors.ustc.edu.cn/crates.io-index/|https://mirrors.ustc.edu.cn/crates.io/api/v1/crates" \
         "aliyun|https://mirrors.aliyun.com/crates.io-index/|https://mirrors.aliyun.com/crates/api/v1/crates"; do
  IFS="|" read -r name idx dl <<<"$m"
  code=$(timeout 20 curl -sL -o /tmp/probe.crate -w "%{http_code}" "$dl/serde/1.0.219/download")
  echo "crates mirror $name dl probe: $code $(stat -c %s /tmp/probe.crate 2>/dev/null)B"
  if [ "$code" = 200 ] && [ "$(stat -c %s /tmp/probe.crate)" -gt 10000 ]; then pick=$name; printf '[source.crates-io]\nreplace-with = "%s"\n[source.%s]\nregistry = "sparse+%s"\n[net]\ngit-fetch-with-cli = true\nretry = 5\n' "$name" "$name" "$idx" > $CARGO_HOME/config.toml; break; fi
done
echo "picked crates mirror: ${pick:-NONE}"
cd $ROOT/models/qwen3-0.6b
for i in $(seq 1 12); do
  curl -sSfL -C - --retry 5 --retry-all-errors --speed-limit 20000 --speed-time 30 -o model.safetensors https://hf-mirror.com/Qwen/Qwen3-0.6B/resolve/main/model.safetensors 2>&1 | tail -1 | cut -c1-100
  s=$(stat -c %s model.safetensors 2>/dev/null || echo 0); echo "weights try $i size=$s"
  [ "$s" -ge 1503300000 ] && break; sleep 3
done
python3 -c "import json,struct;f=open('model.safetensors','rb');n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));print('safetensors header ok, tensors:',len(h))" 2>&1 | tail -1
cd $ROOT/dinfer && cargo build --release --workspace --features qwen3 2>&1 | grep -E "^error|error\[|Finished|failed to download" | head -5
ls -la target/release/dinfer 2>&1 | cut -c1-80
echo ENV_FINISH_DONE
