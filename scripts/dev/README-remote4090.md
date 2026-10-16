# 4090 测试机（2026-10-16 起，编译 / GPU 测试迁到这里）

- 主机 `host-10-121-192-98`（10.121.192.98），经 JumpServer `jump.aicloud.paigod.work:2222` 登录，root。
- 账号是**临时 token**（用户名 `JMS-…`，有过期时间；首个 token 2026-10-16 11:46 +0800 过期）。过期后向用户要新 token，
  写到 `~/.ssh/jms_4090_user` / `~/.ssh/jms_4090_pass`（chmod 600），`scripts/dev/remote4090.sh` 即可继续用。
  实测：`yaochen#root#<asset-id>@jump` 的"指定资产"写法被拒（Permission denied），**token 用户名直连可用**，登录后直接是 root shell。
- 硬件：8 × RTX 4090 24 GB（sm_89），驱动 580.178.04；128 核 / 503 GB；`/` 1.8 TB（91 GB 已用）、`/data` 7 TB RAID（102 GB 已用）。
- 软件（2026-10-16 实测）：Ubuntu 22.04.5、Python 3.10.12、git / rsync / gcc 有；**无 nvcc、无 cargo、无 cmake**。
  网络：`curl -I https://github.com` 200；`https://crates.io` 返回 403（HEAD 被拒还是被墙未验证，cargo 实际拉取要再试；不行就本机 `cargo vendor` 后 rsync）。

## 首次 bootstrap（待新 token 后执行，按序）

```bash
R=scripts/dev/remote4090.sh
$R run 'curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.89.0 && ~/.cargo/bin/cargo --version'
$R run 'apt-get install -y cmake; wget -q https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run -O /tmp/cuda.run && sh /tmp/cuda.run --silent --toolkit --no-driver-check && /usr/local/cuda/bin/nvcc --version'
$R push                                                   # 仓库（不含 target/、golden）
$R run 'cd /data/code/dinfer && export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH CUDA_ARCHS=89 && cargo build --workspace --features qwen3 2>&1 | tail -3'
# 模型 / golden：rsync 本机 /data/ref/models/qwen3-0.6b（1.5 GB）与 test/qwen3_golden 到远端同路径，或远端直接从 HF 下载
```
CUDA 12.4 与驱动 580 兼容；`CUDA_ARCHS=89` 是 kernels crate 的编译架构变量（见 build.rs）。
