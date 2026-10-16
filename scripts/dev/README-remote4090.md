# 4090 测试机（2026-10-16 起，编译 / GPU 测试迁到这里）

- 主机 `host-10-121-192-98`（10.121.192.98），经 JumpServer `jump.aicloud.paigod.work:2222` 登录，root。
- 登录（2026-10-16 12:00 实测可用）：JumpServer 已登记本机公钥 `~/.ssh/id_ed25519_gpu4090.pub`（MD5 `25:e8:5b:83:…:07:20`），
  用 **token 用户名 + 该私钥** 登录，不再依赖 token 密码（密码登录对第 2–4 个 token 全部 `Permission denied`，原因未查明）。
  token 用户名仍会过期（`JMS-…`，写在 `~/.ssh/jms_4090_user`）；过期后向用户要新的，只改这个文件。
  `scripts/dev/remote4090.sh up` 建一条常驻 ControlMaster（keepalive 20 s），之后 `run / push / pull / ssh` 全部复用，不重新鉴权。
  "指定资产"写法 `yaochen#root#<asset-id>@jump` 被拒。登录后直接是 root shell（`/opt/nexus/login_tip.sh` 缺失的报错无害）。
- 硬件：8 × RTX 4090 24 GB（sm_89），驱动 580.178.04；128 核 / 503 GB；`/` 1.8 TB（91 GB 已用）、`/data` 7 TB RAID（102 GB 已用）。
- 软件（2026-10-16 实测）：Ubuntu 22.04.5、Python 3.10.12、git / rsync / gcc 有；**无 nvcc、无 cargo、无 cmake**。
  网络（GET 实测）：index.crates.io / static.crates.io / github.com / sh.rustup.rs / developer.download.nvidia.com 均可达（HEAD crates.io 的 403 是误报）；**huggingface.co 不通** → 模型权重从本机 rsync（`/data/ref/models/qwen3-0.6b` 1.5 GB）。

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
