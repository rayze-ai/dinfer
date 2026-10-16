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

- **目录约束（用户，2026-10-16）**：远程一切只放 `/data/yaochen/infer`：`dinfer/` 仓库、`models/` 权重、`rustup/` + `cargo/` 工具链、
  `cuda-12.4/` toolkit、`bootstrap/` 日志、`env.sh`（每个 shell 先 `source /data/yaochen/infer/env.sh`）。不要往 `/root`、`/data/code`、`/usr/local` 放东西。
- 代理（用户给，2026-10-16）：远端 `http://127.0.0.1:1081`（HTTP 代理，不支持 socks5）。实测经代理 huggingface 可达（307）、github 0.5 MB/s、nvidia 0.8 MB/s。`source env.sh` 后 `proxy_on` / `proxy_off` 切换；cargo / rustup 走 rsproxy 直连更快，不要套代理。
- 镜像：`static.rust-lang.org` 实测 ~100 KB/s，`rsproxy.cn` 14 MB/s → rustup 与 crates registry 都走 rsproxy（`cargo/config.toml` 已写）。

## 首次 bootstrap

```bash
scripts/dev/remote4090.sh up && scripts/dev/remote4090.sh push          # 仓库 → /data/yaochen/infer/dinfer
scripts/dev/remote4090.sh run 'bash /data/yaochen/infer/dinfer/scripts/dev/bootstrap_remote4090.sh'   # tmux boot：rustup / cmake / CUDA 12.4
rsync 本机 /data/ref/models/qwen3-0.6b → 远端 /data/yaochen/infer/models/；test/{qwen3_golden,kernel_refs} → 远端 dinfer/test/
scripts/dev/remote4090.sh run 'source /data/yaochen/infer/env.sh && cd /data/yaochen/infer/dinfer && cargo build --workspace --features qwen3 2>&1 | tail -3'
```
CUDA 12.4 与驱动 580 兼容；`CUDA_ARCHS=89` 是 kernels crate 的编译架构变量（见 build.rs）。
