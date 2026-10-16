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
  `cuda-13.0/` toolkit、`bootstrap/` 日志、`env.sh`（每个 shell 先 `source /data/yaochen/infer/env.sh`）。不要往 `/root`、`/data/code`、`/usr/local` 放东西。
- 代理（用户给，2026-10-16）：远端 `http://127.0.0.1:1081`（HTTP 代理，不支持 socks5）。实测经代理 huggingface 可达（307）、github 0.5 MB/s、nvidia 0.8 MB/s。`source env.sh` 后 `proxy_on` / `proxy_off` 切换；cargo / rustup 走 rsproxy 直连更快，不要套代理。
- 镜像：`static.rust-lang.org` 实测 ~100 KB/s，`rsproxy.cn` 14 MB/s → rustup 与 crates registry 都走 rsproxy（`cargo/config.toml` 已写）。

## 首次 bootstrap

```bash
scripts/dev/remote4090.sh up && scripts/dev/remote4090.sh push          # 仓库 → /data/yaochen/infer/dinfer
scripts/dev/remote4090.sh run 'bash /data/yaochen/infer/dinfer/scripts/dev/bootstrap_remote4090.sh'   # tmux boot：rustup / cmake / CUDA 13.0.2
rsync 本机 /data/ref/models/qwen3-0.6b → 远端 /data/yaochen/infer/models/；test/{qwen3_golden,kernel_refs} → 远端 dinfer/test/
scripts/dev/remote4090.sh run 'source /data/yaochen/infer/env.sh && cd /data/yaochen/infer/dinfer && cargo build --workspace --features qwen3 2>&1 | tail -3'
```
**CUDA toolkit 用 13.0.2**（用户要求；驱动 580.178 支持 13.0，cudarc 0.18.2 有 `cuda-13000`）。注意 `nvidia-smi` 的 `CUDA Version: 13.0` 只是驱动能力，宿主机本身没装 toolkit（containerd 镜像层里的 cuda-13.0 / 12.9 是容器内的）。`.run` 安装器会自建 `/usr/local/cuda` 软链，装完要 `rm -f`（目录约束）。首次误装的 12.4 在 `cuda-12.4/`，13.0 验证通过后删。
  cudarc 版本：`Cargo.toml` 用 `cuda-version-from-build-system`（cudarc 在 build 时跑 `nvcc --version` 探测）→ 本机 3090 得 12.4、远端得 13.0，一份配置两边用。
  **不要**再加 `CUDARC_CUDA_VERSION`：它与固定 feature 同时生效时 cudarc 会把 `CUDA_VERSION` 定义两次（E0428，2026-10-16 踩过）。
  `env.sh` 由 bootstrap 脚本生成，改了脚本要在远端重新生成或手改。

## 日常

```bash
scripts/dev/remote4090.sh status || scripts/dev/remote4090.sh up     # 连接
scripts/dev/remote4090.sh push                                           # 同步代码（不含 target/ 与 golden）
scripts/dev/remote4090.sh run 'source /data/yaochen/infer/env.sh && cd /data/yaochen/infer/dinfer && cargo build --workspace --features qwen3 2>&1 | tail -2'
scripts/dev/remote4090.sh run 'cd /data/yaochen/infer/dinfer && bash scripts/verify/remote4090_baseline.sh'   # 迁移基线：kernels + real/sampling parity + 单测 + e2e
```
远端 kernel_refs / qwen3_golden 由本机（有 torch 的机器）生成后 rsync；远端没有 Python torch 环境。

## 常驻服务（2026-10-16 起）

- dinfer Qwen3-0.6B：tmux 会话 `dinfer`，GPU 0，`0.0.0.0:8012`，日志 `/data/yaochen/infer/bootstrap/serve.log`。启动命令见 tmux 窗口（`--max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens 4096`，num_blocks 自动）。
  重启：`tmux kill-session -t dinfer` 后按同样命令再起；代码更新后先 `remote4090.sh push` + 远端 `cargo build --features qwen3`。
- **JumpServer 不放行任何端口转发**（2026-10-16 实测：`-L` 本地监听建立但连接被对端 reset，`-D` SOCKS 同样，`-W` 直接 `Session open refused by peer`）。所以本机不能直接 curl 到 8012；访问方式只有两种：① 在 4090 上 `scripts/dev/chat.sh`（或任何能到 10.121.192.98 内网的机器）；② `remote4090.sh run 'curl …'`。
- 机器内网地址 10.121.192.98；端口 8012 是否对内网开放取决于机房策略（未验证）。服务**无鉴权**，不要暴露到公网。
- vLLM 对照（Q3 基线，用户要求 Docker + 最新版）：`docker.m.daocloud.io/vllm/vllm-openai:v0.31.0`（PyPI / Docker Hub 最新 release，与 `/data/ref/vllm` `de8291e` 同版本；Docker Hub 直连不通、经代理 6 KB/s，daocloud 镜像站直连可拉，首次一次 502 重试即好；镜像 9 GB 落在 `/var/lib/docker`，docker 不支持按镜像指定目录），容器 `vllm-qwen3`，GPU 1，端口 8013，`--max-model-len 8192 --gpu-memory-utilization 0.9 --enable-prefix-caching`，模型目录只读挂 `/models`，`HF_HUB_OFFLINE=1`。启动 ~40 s，KV 181k token。脚本 `scripts/dev/vllm_docker.sh {pull,run,logs,smoke,stop}`。
- 发请求：`scripts/dev/chat.sh http://127.0.0.1:8012 "问题" [--stream] [--think]`（8013 = vLLM）。测试都在 4090 上做，结果拿回来；不要试图把端口转发到本机（JumpServer 不放行，用户也不要）。
